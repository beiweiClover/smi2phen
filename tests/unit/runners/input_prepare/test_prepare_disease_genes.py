import csv
import gzip
import json
import shutil
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.runners.input_prepare.prepare_disease_genes import (
    main,
    prepare_disease_genes,
)
from lipid_screening_agent.runtime import RunContext
from lipid_screening_agent.runtime.hashing import sha256_bytes, sha256_file

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "small_inputs"
CONFIG = Path(__file__).resolve().parents[4] / "configs" / "workflow.yaml"


def _context(tmp_path: Path, *, run_id: str = "run-genes") -> tuple[RunContext, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    resources.mkdir()
    context = RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        output_dir=project / "runs" / run_id / "inputs" / "prepared",
    )
    original = context.input_dir / "original"
    original.mkdir()
    with gzip.open(resources / "Homo_sapiens.gene_info.gz", "wb") as handle:
        handle.write((FIXTURES / "gene_info.tsv").read_bytes())
    return context, resources


def _copy_input(context: RunContext, name: str) -> Path:
    destination = context.input_dir / "original" / name
    shutil.copyfile(FIXTURES / name, destination)
    return destination


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _run(context: RunContext, input_path: Path, resources: Path):
    return prepare_disease_genes(
        context=context,
        input_path=input_path,
        gene_info_path=resources / "Homo_sapiens.gene_info.gz",
        config_hash=sha256_bytes(b"config"),
        code_version="test-version",
        input_artifact_ids=("a-input",),
    )


def test_symbol_synonym_unmapped_dedup_and_formal_symbol_priority(
    tmp_path: Path,
) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, "genes.symbols.txt"), resources)

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics == {
        "input_count": 5,
        "valid_count": 3,
        "unmapped_count": 1,
        "duplicate_count": 1,
        "skipped_count": 2,
        "mapped_by_symbol_count": 3,
        "mapped_by_synonym_count": 0,
        "mapped_by_entrez_count": 0,
    }
    assert _tsv(context.run_dir / "inputs/prepared/disease_genes.normalized.tsv") == [
        {"symbol": "TP53", "entrez_id": "7157"},
        {"symbol": "APOE", "entrez_id": "348"},
        {"symbol": "OFFICIAL_ALIAS", "entrez_id": "1234"},
    ]
    assert _tsv(context.run_dir / "inputs/prepared/unmapped_genes.tsv") == [
        {"input_value": "UNKNOWN", "reason": "unknown_symbol"}
    ]


def test_single_synonym_maps_to_official_symbol_and_entrez(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, "genes.synonym.txt"), resources)

    assert result.status is NodeStatus.SUCCEEDED
    assert _tsv(context.run_dir / "inputs/prepared/disease_genes.normalized.tsv") == [
        {"symbol": "TP53", "entrez_id": "7157"}
    ]
    assert result.metrics["mapped_by_synonym_count"] == 1


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        (
            "genes.entrez.tsv",
            [
                {"symbol": "TP53", "entrez_id": "7157"},
                {"symbol": "EGFR", "entrez_id": "1956"},
            ],
        ),
        (
            "genes.numeric_no_header.txt",
            [
                {"symbol": "TP53", "entrez_id": "7157"},
                {"symbol": "EGFR", "entrez_id": "1956"},
            ],
        ),
    ],
)
def test_entrez_only_with_header_or_numeric_inference(
    tmp_path: Path,
    fixture_name: str,
    expected: list[dict[str, str]],
) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, fixture_name), resources)

    assert result.status is NodeStatus.SUCCEEDED
    assert _tsv(context.run_dir / "inputs/prepared/disease_genes.normalized.tsv") == expected
    assert result.metrics["mapped_by_entrez_count"] == 2
    assert result.metrics["unmapped_count"] == (1 if fixture_name.endswith(".tsv") else 0)


def test_symbol_entrez_alias_columns_report_mismatch_and_ambiguous_synonym(
    tmp_path: Path,
) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, "genes.paired.csv"), resources)

    assert result.status is NodeStatus.SUCCEEDED
    assert _tsv(context.run_dir / "inputs/prepared/disease_genes.normalized.tsv") == [
        {"symbol": "TP53", "entrez_id": "7157"},
        {"symbol": "EGFR", "entrez_id": "1956"},
    ]
    assert _tsv(context.run_dir / "inputs/prepared/unmapped_genes.tsv") == [
        {
            "input_value": "symbol=APOE;entrez_id=7157",
            "reason": "symbol_entrez_mismatch",
        },
        {
            "input_value": "symbol=SHARED;entrez_id=9999",
            "reason": "ambiguous_synonym",
        },
    ]


def test_output_manifests_metrics_and_resource_hash(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, "genes.symbols.txt"), resources)

    assert len(result.outputs) == 2
    manifests = [
        load_artifact_manifest(
            context.run_dir
            / "artifacts/manifests/prepare_disease_genes/main"
            / f"{artifact_id}.json",
            run_root=context.run_dir,
        )
        for artifact_id in result.outputs
    ]
    assert {manifest.artifact_type for manifest in manifests} == {
        "disease_genes_normalized",
        "unmapped_genes",
    }
    assert all(manifest.input_artifact_ids == ("a-input",) for manifest in manifests)
    assert all(
        manifest.resource_hashes["resources.gps.human_gene_info"]
        == sha256_file(resources / "Homo_sapiens.gene_info.gz")
        for manifest in manifests
    )
    node_result = json.loads(
        (context.run_dir / "artifacts/node_results/prepare_disease_genes/main.json").read_text(
            encoding="utf-8"
        )
    )
    assert node_result["metrics"] == result.to_dict()["metrics"]


def test_all_unmapped_is_a_structured_input_failure(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    result = _run(context, _copy_input(context, "genes.unmapped.txt"), resources)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert result.error.details["unmapped_count"] == 2
    assert result.metrics["input_count"] == 2
    assert result.metrics["valid_count"] == 0
    assert result.metrics["unmapped_count"] == 2
    assert _tsv(context.run_dir / "inputs/prepared/unmapped_genes.tsv") == [
        {"input_value": "UNKNOWN", "reason": "unknown_symbol"},
        {"input_value": "SHARED", "reason": "ambiguous_synonym"},
    ]
    assert not (context.run_dir / "inputs/prepared/disease_genes.normalized.tsv").exists()


def test_missing_gene_info_is_a_structured_resource_failure(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    result = prepare_disease_genes(
        context=context,
        input_path=_copy_input(context, "genes.symbols.txt"),
        gene_info_path=resources / "missing.gene_info.gz",
        config_hash=sha256_bytes(b"config"),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "resource"


def test_cli_uses_common_paths_and_configured_gene_info(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, resources = _context(tmp_path)
    input_path = _copy_input(context, "genes.numeric_no_header.txt")
    monkeypatch.setenv("LIPID_AGENT_GPS_RESOURCE_DIR", str(resources))

    exit_code = main(
        [
            "--run-dir",
            str(context.run_dir),
            "--input-dir",
            str(context.input_dir),
            "--resource-dir",
            str(resources),
            "--output-dir",
            str(context.output_dir),
            "--config",
            str(CONFIG),
            "--input-file",
            str(input_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_cli_reports_missing_configured_resource_root_structurally(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, resources = _context(tmp_path)
    input_path = _copy_input(context, "genes.numeric_no_header.txt")
    monkeypatch.delenv("LIPID_AGENT_GPS_RESOURCE_DIR", raising=False)

    exit_code = main(
        [
            "--run-dir",
            str(context.run_dir),
            "--input-dir",
            str(context.input_dir),
            "--resource-dir",
            str(resources),
            "--output-dir",
            str(context.output_dir),
            "--config",
            str(CONFIG),
            "--input-file",
            str(input_path),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error"]["category"] == "configuration"
    assert payload["metrics"]["valid_count"] == 0


def test_headered_gene_table_rejects_malformed_row_width(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    source = context.input_dir / "original" / "malformed.tsv"
    source.write_text(
        "symbol\tentrez_id\nTP53\t7157\textra\n",
        encoding="utf-8",
    )

    result = _run(context, source, resources)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.details["row_number"] == 2
