from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lipid_screening_agent.artifacts import NodeStatus
from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runners.netinfer.algorithms import read_raw_predictions
from lipid_screening_agent.runners.netinfer.predict_batch import netinfer_predict_batch
from lipid_screening_agent.runners.netinfer.wsdtnbi import (
    WSDTNBIConfig,
    WSDTNBIEngine,
)
from lipid_screening_agent.runtime import RunContext, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG = load_workflow_config(PROJECT_ROOT / "configs/workflow.yaml")
CONFIG_HASH = hash_workflow_config(CONFIG)


def _networks(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    dt = root / "DT.tsv"
    ds = root / "DS.tsv"
    dt.write_text(
        "DRUG\tD1\tTARGET\tP1\t2\n"
        "DRUG\tD1\tTARGET\tP2\t1\n"
        "DRUG\tD2\tTARGET\tP2\t3\n"
        "DRUG\tD2\tTARGET\tP3\t4\n",
        encoding="utf-8",
    )
    ds.write_text(
        "DRUG\tD1\tSUB\tS1\t1\n"
        "DRUG\tD1\tSUB\tS2\t1\n"
        "DRUG\tD2\tSUB\tS2\t1\n"
        "DRUG\tD2\tSUB\tS3\t1\n",
        encoding="utf-8",
    )
    return dt, ds


def _context(tmp_path: Path) -> tuple[RunContext, Path]:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    _networks(resources)
    return (
        RunContext.create(
            runs_root=project / "runs",
            run_id="run",
            project_root=project,
            resource_dir=resources,
            input_dir=project / "runs/run/inputs/prepared",
            output_dir=project / "runs/run/artifacts/netinfer",
        ),
        resources,
    )


def test_python_engine_predicts_positive_targets_and_filters_zero_overlap(
    tmp_path: Path,
) -> None:
    dt, ds = _networks(tmp_path / "resources")
    cs = tmp_path / "CS.tsv"
    cs.write_text(
        "COMPOUND\tquery\tSUB\tS1\t1\n"
        "COMPOUND\tquery\tSUB\tS2\t1\n"
        "COMPOUND\tunknown\tSUB\tS999\t1\n",
        encoding="utf-8",
    )
    engine = WSDTNBIEngine(
        dt,
        ds,
        WSDTNBIConfig(device="cpu", batch_size=2, top_n=10),
    )

    known = engine.predict_official_drugs(["D1"])["D1"]
    assert {item["target"] for item in known}.isdisjoint({"P1", "P2"})
    assert all(float(item["score"]) > 0 for item in known)
    novel = engine.predict_compounds(["query", "unknown"], cs)
    assert novel["query"]
    assert all(float(item["score"]) > 0 for item in novel["query"])
    assert novel["unknown"] == []
    assert engine.summary()["device_actual"] == "cpu"


def test_batch_runner_writes_the_existing_six_column_contract(tmp_path: Path) -> None:
    context, resources = _context(tmp_path)
    batch_dir = context.output_dir / "batches/batch_0001"
    batch_dir.mkdir(parents=True)
    batch_input = batch_dir / "CS.tsv"
    batch_input.write_text(
        "COMPOUND\tnovel\tSUB\tS1\t1\n"
        "COMPOUND\tnovel\tSUB\tS2\t1\n",
        encoding="utf-8",
    )
    manifest = context.output_dir / "batch_manifest.json"
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
                        "input_sha256": sha256_file(batch_input),
                        "prediction_path": (
                            "artifacts/netinfer/batches/batch_0001/predictions.tsv"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    settings = replace(
        CONFIG.netinfer,
        device="cpu",
        inference_batch_size=2,
    )

    result = netinfer_predict_batch(
        context=context,
        batch_id="batch_0001",
        batch_manifest_path=manifest,
        batch_input_path=batch_input,
        drug_target_network_path=resources / "DT.tsv",
        drug_substructure_network_path=resources / "DS.tsv",
        settings=settings,
        config_hash=CONFIG_HASH,
        code_version="test",
    )

    output = batch_dir / "predictions.tsv"
    rows = read_raw_predictions(output)
    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["prediction_backend"] == "python_torch"
    assert result.metrics["device_actual"] == "cpu"
    assert rows
    assert {row.source_id for row in rows} == {"novel"}
    assert all(row.source_type == "COMPOUND" and row.target_type == "TARGET" for row in rows)
