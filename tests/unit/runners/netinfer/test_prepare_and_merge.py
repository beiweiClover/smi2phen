import builtins
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.netinfer._dependencies import (
    load_prepare_dependencies,
)
from lipid_screening_agent.runners.netinfer.merge_targets import (
    netinfer_merge_targets,
)
from lipid_screening_agent.runners.netinfer.predict_known import (
    netinfer_predict_known,
)
from lipid_screening_agent.runners.netinfer.prepare_inputs import (
    netinfer_prepare_inputs,
)
from lipid_screening_agent.runtime import EnvironmentError, RunContext, sha256_file

openpyxl = pytest.importorskip(
    "openpyxl", reason="NetInfer preparation unit tests require the optional openpyxl extra"
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


def test_prepare_dependency_failure_is_environment_error(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "openpyxl":
            raise ImportError("fixture missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    try:
        load_prepare_dependencies()
    except EnvironmentError as error:
        assert error.category == "environment"
        assert error.retryable is False
    else:  # pragma: no cover - the injected import failure must be classified
        raise AssertionError("expected EnvironmentError")


def _context(tmp_path: Path, run_id: str) -> tuple[RunContext, Path, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "netinfer-resources"
    prepared = project / "runs" / run_id / "inputs/prepared"
    output = project / "runs" / run_id / "artifacts/netinfer"
    project.mkdir(parents=True)
    resources.mkdir(parents=True)
    return (
        RunContext.create(
            runs_root=project / "runs",
            run_id=run_id,
            project_root=project,
            resource_dir=resources,
            input_dir=prepared,
            output_dir=output,
        ),
        prepared,
        resources,
    )


def _workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    drugs = workbook.active
    drugs.title = "Drug information"
    drugs.append(["Drug ID", "SMILES (standardized)"])
    drugs.append(["D1", "CCO"])
    targets = workbook.create_sheet("Target information")
    targets.append(["UniProt AC", "Gene symbol"])
    targets.append(["P1", "GENE1"])
    targets.append(["P2", "GENE2"])
    targets.append(["P3", "GENE3"])
    workbook.save(path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_prepare_matches_known_batches_novel_and_is_repeatable(tmp_path: Path) -> None:
    context, prepared, resources = _context(tmp_path, "prepare")
    compounds = prepared / "compounds.normalized.csv"
    compounds.parent.mkdir(parents=True, exist_ok=True)
    compounds.write_text(
        "ID,SMILES\nknown-user,CCO\nnovel-a,C1CC1\nnovel-b,CCN\n",
        encoding="utf-8",
    )
    workbook = resources / "supplementary.xlsx"
    _workbook(workbook)
    settings = replace(CONFIG.netinfer, batch_size=1)

    first = netinfer_prepare_inputs(
        context=context,
        compounds_path=compounds,
        supplementary_workbook_path=workbook,
        settings=settings,
        config_hash=CONFIG_HASH,
        code_version="test",
    )

    assert first.status is NodeStatus.SUCCEEDED, first.to_json()
    mapping_path = context.resolve_run_relative(
        "artifacts/netinfer/input_mapping.tsv", must_exist=True
    )
    mapping = _read_tsv(mapping_path)
    assert [row["netinfer_input_type"] for row in mapping] == [
        "DRUG",
        "COMPOUND",
        "COMPOUND",
    ]
    assert mapping[0]["netinfer_input_id"] == "D1"
    manifest_path = context.resolve_run_relative(
        "artifacts/netinfer/batch_manifest.json", must_exist=True
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["batch_size"] == 1
    assert manifest["batch_count"] == 2
    assert [batch["batch_id"] for batch in manifest["batches"]] == [
        "batch_0001",
        "batch_0002",
    ]
    assert all(
        set(batch)
        == {
            "batch_id",
            "task_id",
            "compound_count",
            "compound_ids",
            "input_path",
            "input_sha256",
            "prediction_path",
        }
        for batch in manifest["batches"]
    )
    first_hashes = {
        "mapping": sha256_file(mapping_path),
        "manifest": sha256_file(manifest_path),
        **{
            batch["batch_id"]: sha256_file(
                context.resolve_run_relative(batch["input_path"], must_exist=True)
            )
            for batch in manifest["batches"]
        },
    }

    second = netinfer_prepare_inputs(
        context=context,
        compounds_path=compounds,
        supplementary_workbook_path=workbook,
        settings=settings,
        config_hash=CONFIG_HASH,
        code_version="test",
        attempt=2,
    )

    assert second.status is NodeStatus.SUCCEEDED, second.to_json()
    assert first_hashes["mapping"] == sha256_file(mapping_path)
    assert first_hashes["manifest"] == sha256_file(manifest_path)
    for batch in manifest["batches"]:
        assert first_hashes[batch["batch_id"]] == sha256_file(
            context.resolve_run_relative(batch["input_path"], must_exist=True)
        )


def test_prepare_no_novel_and_known_predict_no_op_without_known(tmp_path: Path) -> None:
    known_context, prepared, resources = _context(tmp_path / "known", "known-only")
    compounds = prepared / "compounds.normalized.csv"
    compounds.parent.mkdir(parents=True, exist_ok=True)
    compounds.write_text("ID,SMILES\nknown-user,CCO\n", encoding="utf-8")
    workbook = resources / "supplementary.xlsx"
    _workbook(workbook)
    result = netinfer_prepare_inputs(
        context=known_context,
        compounds_path=compounds,
        supplementary_workbook_path=workbook,
        settings=CONFIG.netinfer,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    manifest = json.loads(
        known_context.resolve_run_relative(
            "artifacts/netinfer/batch_manifest.json", must_exist=True
        ).read_text(encoding="utf-8")
    )
    assert result.status is NodeStatus.SUCCEEDED
    assert manifest["batches"] == []

    novel_context, prepared, _ = _context(tmp_path / "novel", "novel-only")
    mapping = prepared / "mapping.tsv"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text(
        "ID\tSMILES\tmatch_key\tofficial_drug_id\tnetinfer_input_type\t"
        "netinfer_input_id\tbatch_id\n"
        "novel\tC1CC1\tC1CC1\t\tCOMPOUND\tnovel\tbatch_0001\n",
        encoding="utf-8",
    )
    no_op = netinfer_predict_known(
        context=novel_context,
        mapping_path=mapping,
        drug_target_network_path="missing-DT.tsv",
        drug_substructure_network_path="missing-DS.tsv",
        settings=CONFIG.netinfer,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert no_op.status is NodeStatus.SUCCEEDED
    assert no_op.metrics["no_op"] is True
    assert no_op.metrics["device_actual"] == "not_used"
    assert no_op.outputs == ()


def _write_merge_fixture(context: RunContext) -> tuple[Path, Path, Path, Path, Path]:
    root = context.output_dir
    root.mkdir(parents=True, exist_ok=True)
    mapping = root / "input_mapping.tsv"
    mapping.write_text(
        "ID\tSMILES\tmatch_key\tofficial_drug_id\tnetinfer_input_type\t"
        "netinfer_input_id\tbatch_id\n"
        "known-user\tCCO\tCCO\tD1\tDRUG\tD1\t\n"
        "novel\tC1CC1\tC1CC1\t\tCOMPOUND\tnovel\tbatch_0001\n",
        encoding="utf-8",
    )
    target_map = root / "target_uniprot_to_symbol.tsv"
    target_map.write_text(
        "uniprot_id\tgene_symbol\nP1\tGENE1\nP2\tGENE2\nP3\tGENE3\n",
        encoding="utf-8",
    )
    batch_dir = root / "batches/batch_0001"
    batch_dir.mkdir(parents=True)
    manifest = root / "batch_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batch_size": 5000,
                "novel_compound_count": 1,
                "batch_count": 1,
                "batches": [
                    {
                        "batch_id": "batch_0001",
                        "task_id": "batch_0001",
                        "compound_count": 1,
                        "compound_ids": ["novel"],
                        "input_path": "artifacts/netinfer/batches/batch_0001/CS.tsv",
                        "input_sha256": "0" * 64,
                        "prediction_path": (
                            "artifacts/netinfer/batches/batch_0001/predictions.tsv"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    known = root / "known_predictions.tsv"
    known.write_text(
        "DRUG\tD2\tTARGET\tP3\t0.99\t-\n"
        "DRUG\tD1\tTARGET\tP2\t0.80\t1\n"
        "DRUG\tD1\tTARGET\tP1\t1.00\t-\n",
        encoding="utf-8",
    )
    batch = batch_dir / "predictions.tsv"
    batch.write_text(
        "COMPOUND\tnovel\tTARGET\tP3\t0.70\t1\n",
        encoding="utf-8",
    )
    return mapping, target_map, manifest, known, batch


def test_merge_writes_unified_schema_and_requires_every_batch(tmp_path: Path) -> None:
    context, _, _ = _context(tmp_path, "merge")
    mapping, target_map, manifest, known, batch = _write_merge_fixture(context)

    missing_batch = netinfer_merge_targets(
        context=context,
        mapping_path=mapping,
        target_map_path=target_map,
        batch_manifest_path=manifest,
        known_predictions_path=known,
        batch_prediction_paths={},
        settings=CONFIG.netinfer,
        config_hash=CONFIG_HASH,
        code_version="test",
    )
    assert missing_batch.status is NodeStatus.FAILED
    assert missing_batch.error is not None
    assert missing_batch.error.category == "input"
    assert missing_batch.error.details["missing_batch_ids"] == ("batch_0001",)

    result = netinfer_merge_targets(
        context=context,
        mapping_path=mapping,
        target_map_path=target_map,
        batch_manifest_path=manifest,
        known_predictions_path=known,
        batch_prediction_paths={"batch_0001": batch},
        settings=CONFIG.netinfer,
        config_hash=CONFIG_HASH,
        code_version="test",
        attempt=2,
    )

    assert result.status is NodeStatus.SUCCEEDED, result.to_json()
    payload = json.loads(
        context.resolve_run_relative(
            "artifacts/netinfer/drug_targets.json", must_exist=True
        ).read_text(encoding="utf-8")
    )
    assert list(payload) == ["known-user", "novel"]
    assert [item["gene_symbol"] for item in payload["known-user"]["targets"]] == [
        "GENE1",
        "GENE2",
    ]
    assert [item["evidence"] for item in payload["known-user"]["targets"]] == [
        "known",
        "predicted",
    ]
    assert payload["novel"]["targets"][0]["gene_symbol"] == "GENE3"
    assert (
        _read_tsv(
            context.resolve_run_relative(
                "artifacts/netinfer/missing_predictions.tsv", must_exist=True
            )
        )
        == []
    )
