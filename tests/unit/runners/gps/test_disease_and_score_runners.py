import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.gps import build_disease_signature as disease_runner
from lipid_screening_agent.runners.gps import score_compounds as score_runner
from lipid_screening_agent.runtime import RunContext, sha256_file

pytest.importorskip(
    "statsmodels", reason="GPS disease runner unit tests require the optional statsmodels extra"
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


def _context(tmp_path: Path, run_id: str) -> tuple[RunContext, Path, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    prepared = project / "runs" / run_id / "inputs/prepared"
    output = project / "runs" / run_id / "artifacts/gps"
    project.mkdir()
    resources.mkdir()
    context = RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        input_dir=prepared,
        output_dir=output,
    )
    return context, prepared, resources


def _write_expression_inputs(context: RunContext, prepared: Path) -> Path:
    original = context.resolve_run_relative("inputs/original")
    original.mkdir(parents=True)
    control = ["c1", "c2", "c3", "c4"]
    disease = ["d1", "d2", "d3", "d4"]
    samples = control + disease
    tpm = pd.DataFrame(
        [
            ["1", 0, 0, 0, 0, 30, 35, 40, 32],
            ["2", 30, 35, 40, 32, 0, 0, 0, 0],
            ["3", 5, 6, 5, 6, 5, 6, 5, 6],
        ],
        columns=["GeneID", *samples],
    )
    tpm_path = original / "TPM_matrix_1.tsv"
    metadata_path = original / "metadata_1.tsv"
    tpm.to_csv(tpm_path, sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": samples,
            "group": ["control"] * 4 + ["disease"] * 4,
        }
    ).to_csv(metadata_path, sep="\t", index=False)
    manifest = prepared / "expression_comparisons.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "comparisons": [
                    {
                        "comparison_id": "comparison_1",
                        "tpm_path": context.relative_path(tpm_path),
                        "metadata_path": context.relative_path(metadata_path),
                        "sample_counts": {
                            "total": 8,
                            "control": 4,
                            "disease": 4,
                        },
                        "sha256": {
                            "tpm": sha256_file(tpm_path),
                            "metadata": sha256_file(metadata_path),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_gene_info(resources: Path) -> Path:
    path = resources / "gene_info.tsv"
    path.write_text(
        "GeneID\tSymbol\tSynonyms\n1\tUPGENE\tUP-ALIAS\n2\tDOWNGENE\tDOWN-ALIAS\n3\tFLATGENE\t-\n",
        encoding="utf-8",
    )
    return path


def _write_raw_drug_profile(context: RunContext) -> Path:
    path = context.resolve_run_relative("artifacts/gps/Drug_GPS.csv.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cmp-reverse": [-1, 1, 0],
            "cmp-mimic": [1, -1, 1],
        },
        index=pd.Index(["UPGENE", "DOWNGENE", "UNMAPPED"], name="GeneSymbol"),
    ).to_csv(path, compression="gzip")
    return path


def _manifests_by_type(context: RunContext, node_id: str, result) -> dict[str, object]:
    output = {}
    for artifact_id in result.outputs:
        path = context.run_dir / "artifacts/manifests" / node_id / "main" / f"{artifact_id}.json"
        manifest = load_artifact_manifest(path, run_root=context.run_dir)
        output[manifest.artifact_type] = manifest
    return output


def test_disease_runner_preserves_raw_drug_profile_and_writes_two_schemas(
    tmp_path: Path,
) -> None:
    context, prepared, resources = _context(tmp_path, "disease-run")
    expression_manifest = _write_expression_inputs(context, prepared)
    gene_info = _write_gene_info(resources)
    raw_drug = _write_raw_drug_profile(context)
    raw_hash = sha256_file(raw_drug)

    result = disease_runner.gps_build_disease_signature(
        context=context,
        expression_status="prepared",
        expression_manifest_path=expression_manifest,
        drug_gps_path=raw_drug,
        gene_info_path=gene_info,
        settings=CONFIG.gps.disease_signature,
        config_hash=CONFIG_HASH,
        code_version="test-version",
        input_artifact_ids=("expression-artifact", "raw-drug-artifact"),
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert sha256_file(raw_drug) == raw_hash
    assert result.metrics["comparison_count"] == 1
    assert result.metrics["direction_consistent_deg_count"] == 2
    assert result.metrics["disease_profile_gene_count"] == 2
    assert result.metrics["drug_profile_unmapped_gene_count"] == 1
    assert result.metrics["device_actual"] == "cpu"

    disease_path = context.resolve_run_relative(
        disease_runner.DISEASE_OUTPUT_RELATIVE_PATH, must_exist=True
    )
    disease = pd.read_csv(disease_path, dtype={"GeneID": str})
    assert disease.columns.tolist() == [
        "GeneID",
        "disease_log2FC_mean",
        "disease_direction",
    ]
    assert set(disease["GeneID"]) == {"1", "2"}
    assert set(disease["disease_direction"]) == {"up", "down"}

    aligned_path = context.resolve_run_relative(
        disease_runner.ENTREZ_DRUG_OUTPUT_RELATIVE_PATH, must_exist=True
    )
    aligned = pd.read_csv(aligned_path, compression="gzip", index_col=0)
    assert aligned.index.name == "GeneID"
    assert aligned.index.astype(str).tolist() == ["1", "2"]
    assert aligned.columns.tolist() == ["cmp-reverse", "cmp-mimic"]
    assert set(np.unique(aligned.to_numpy())) <= {-1, 0, 1}

    manifests = _manifests_by_type(context, disease_runner.NODE_ID, result)
    assert set(manifests) == {
        "gps_drug_profiles_entrez",
        "gps_disease_signature",
    }
    assert manifests["gps_drug_profiles_entrez"].relative_path == (
        disease_runner.ENTREZ_DRUG_OUTPUT_RELATIVE_PATH
    )


def test_disease_runner_returns_skipped_without_imports_or_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, _, _ = _context(tmp_path, "skip-run")

    def unexpected_import():
        raise AssertionError("skipped execution must not import DEG dependencies")

    monkeypatch.setattr(disease_runner, "load_disease_dependencies", unexpected_import)

    result = disease_runner.gps_build_disease_signature(
        context=context,
        expression_status="skipped",
        expression_manifest_path=None,
        drug_gps_path=None,
        gene_info_path=None,
        settings=CONFIG.gps.disease_signature,
        config_hash=CONFIG_HASH,
        code_version="test-version",
    )

    assert result.status is NodeStatus.SKIPPED
    assert result.outputs == ()
    assert result.metrics["comparison_count"] == 0
    assert result.metrics["elapsed_seconds"] >= 0
    assert not context.resolve_run_relative(disease_runner.DISEASE_OUTPUT_RELATIVE_PATH).exists()
    assert not context.resolve_run_relative(
        disease_runner.ENTREZ_DRUG_OUTPUT_RELATIVE_PATH
    ).exists()


def test_score_runner_writes_sorted_unique_output_schema(tmp_path: Path) -> None:
    context, _, _ = _context(tmp_path, "score-run")
    output_root = context.resolve_run_relative("artifacts/gps")
    output_root.mkdir(parents=True, exist_ok=True)
    drug_path = output_root / "Drug_GPS.entrez.csv.gz"
    disease_path = output_root / "Disease_GPS.csv"
    pd.DataFrame(
        {
            "cmp-reverse": [-1, 1, 0],
            "cmp-mimic": [1, -1, 1],
        },
        index=pd.Index(["1", "2", "3"], name="GeneID"),
    ).to_csv(drug_path, compression="gzip")
    pd.DataFrame(
        {
            "GeneID": ["1", "2", "3"],
            "disease_log2FC_mean": [2.0, -2.0, 0.5],
            "disease_direction": ["up", "down", "up"],
        }
    ).to_csv(disease_path, index=False)

    result = score_runner.gps_score_compounds(
        context=context,
        drug_gps_path=drug_path,
        disease_gps_path=disease_path,
        settings=CONFIG.gps.scoring,
        config_hash=CONFIG_HASH,
        code_version="test-version",
        input_artifact_ids=("aligned-drug", "disease-signature"),
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["random_background_samples"] == 1500
    assert result.metrics["seed"] == 42
    assert result.metrics["compound_count"] == 2
    scores_path = context.resolve_run_relative(score_runner.OUTPUT_RELATIVE_PATH, must_exist=True)
    scores = pd.read_csv(scores_path, dtype={"ID": str})
    assert scores.columns.tolist() == [
        "ID",
        "GPS_score_zRGES_like_lower_better",
    ]
    assert scores["ID"].is_unique
    assert scores["GPS_score_zRGES_like_lower_better"].is_monotonic_increasing
    manifests = _manifests_by_type(context, score_runner.NODE_ID, result)
    assert set(manifests) == {"gps_scores"}


def test_cli_parsers_expose_explicit_inputs_and_device(tmp_path: Path) -> None:
    common = [
        "--run-dir",
        str(tmp_path / "run"),
        "--input-dir",
        str(tmp_path / "run/inputs/prepared"),
        "--resource-dir",
        str(tmp_path / "resources"),
        "--output-dir",
        str(tmp_path / "run/artifacts/gps"),
        "--config",
        str(PROJECT_ROOT / "configs/workflow.yaml"),
    ]
    predict = __import__(
        "lipid_screening_agent.runners.gps.predict_drug_profiles",
        fromlist=["build_parser"],
    )
    namespace = predict.build_parser().parse_args(
        [*common, "--compounds", str(tmp_path / "compounds.csv"), "--device", "cpu"]
    )
    assert namespace.device == "cpu"

    skipped = disease_runner.build_parser().parse_args([*common, "--expression-status", "skipped"])
    assert skipped.expression_status == "skipped"
