"""Run one independently retryable Python wSDTNBI novel-compound batch."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import NetInferConfig
from lipid_screening_agent.runtime import InputError, RunContext, sha256_file
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import resolve_netinfer_output, resolve_run_input_file
from ._io import batch_index, read_batch_manifest
from ._settings import validate_netinfer_settings
from .python_prediction import execute_python_prediction

NODE_ID = "netinfer_predict_batch"


def _validate_batch_input(path: Path, expected_ids: set[str]) -> int:
    seen_ids: set[str] = set()
    edge_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 5:
                    raise InputError(
                        "NetInfer batch CS row must contain five fields",
                        details={"row": row_number, "field_count": len(row)},
                    )
                source_type, compound_id, sub_type, sub_id, weight = (
                    value.strip() for value in row
                )
                if (
                    source_type != "COMPOUND"
                    or sub_type != "SUB"
                    or not compound_id
                    or not sub_id
                    or weight != "1"
                ):
                    raise InputError(
                        "NetInfer batch CS row violates the COMPOUND-SUB schema",
                        details={"row": row_number},
                    )
                if compound_id not in expected_ids:
                    raise InputError(
                        "NetInfer batch CS contains a compound outside its manifest item",
                        details={"compound_id": compound_id},
                    )
                seen_ids.add(compound_id)
                edge_count += 1
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "NetInfer batch CS could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    missing = sorted(expected_ids - seen_ids)
    if missing:
        raise InputError(
            "NetInfer batch CS has no edges for one or more manifest compounds",
            details={"missing_compound_ids": missing[:50]},
        )
    return edge_count


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    batch_id: str,
    batch_manifest_path: str | Path,
    batch_input_path: str | Path,
    drug_target_network_path: str | Path,
    drug_substructure_network_path: str | Path,
    settings: NetInferConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            "prediction_backend": "python_torch",
            "device_requested": settings.device,
            "batch_id": batch_id,
            "batch_compound_count": 0,
            "batch_cs_edge_count": 0,
            "top_n_predicted_targets": settings.top_n_predicted_targets,
            "raw_prediction_row_count": 0,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        validate_netinfer_settings(settings)
        manifest_file = resolve_run_input_file(
            context, batch_manifest_path, label="NetInfer batch manifest"
        )
        input_file = resolve_run_input_file(
            context, batch_input_path, label=f"NetInfer {batch_id} CS input"
        )
        batch = batch_index(read_batch_manifest(manifest_file)).get(batch_id)
        if batch is None:
            raise InputError(
                "requested NetInfer batch ID is absent from the batch manifest",
                details={"batch_id": batch_id},
            )
        expected_input_path = str(batch["input_path"])
        observed_input_path = context.relative_path(input_file)
        if observed_input_path != expected_input_path:
            raise InputError(
                "explicit NetInfer batch input does not match the manifest path",
                details={
                    "batch_id": batch_id,
                    "expected": expected_input_path,
                    "observed": observed_input_path,
                },
            )
        observed_hash = sha256_file(input_file)
        if observed_hash != batch["input_sha256"]:
            raise InputError(
                "NetInfer batch input no longer matches its manifest SHA-256",
                details={
                    "batch_id": batch_id,
                    "expected": batch["input_sha256"],
                    "observed": observed_hash,
                },
            )
        expected_ids = list(batch["compound_ids"])
        edge_count = _validate_batch_input(input_file, set(expected_ids))
        execution.update_metrics(
            {
                "batch_compound_count": len(expected_ids),
                "batch_cs_edge_count": edge_count,
            }
        )
        expected_prediction_path = (
            f"artifacts/netinfer/batches/{batch_id}/predictions.tsv"
        )
        if batch["prediction_path"] != expected_prediction_path:
            raise InputError(
                "NetInfer batch prediction path does not match the contract",
                details={"batch_id": batch_id},
            )

        output = resolve_netinfer_output(
            context, f"batches/{batch_id}/predictions.tsv"
        )
        execution.update_metrics(
            execute_python_prediction(
                execution,
                context=context,
                settings=settings,
                source_type="COMPOUND",
                source_ids=expected_ids,
                drug_target_network_path=drug_target_network_path,
                drug_substructure_network_path=drug_substructure_network_path,
                compound_substructure_path=input_file,
                output_path=output,
            )
        )
        execution.add_output(
            "netinfer_batch_predictions", output, instance_key=batch_id
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def netinfer_predict_batch(
    *,
    context: RunContext,
    batch_id: str,
    batch_manifest_path: str | Path,
    batch_input_path: str | Path,
    drug_target_network_path: str | Path,
    drug_substructure_network_path: str | Path,
    settings: NetInferConfig,
    config_hash: str,
    code_version: str,
    task_id: str | None = None,
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            batch_id=batch_id,
            batch_manifest_path=batch_manifest_path,
            batch_input_path=batch_input_path,
            drug_target_network_path=drug_target_network_path,
            drug_substructure_network_path=drug_substructure_network_path,
            settings=settings,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id or batch_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one independently retryable Python wSDTNBI batch."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--batch-input", required=True, type=Path)
    parser.add_argument("--drug-target-network", type=Path)
    parser.add_argument("--drug-substructure-network", type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    if task_id == "main":
        task_id = namespace.batch_id
    resources = environment.config.resources.netinfer
    result = netinfer_predict_batch(
        context=environment.context,
        batch_id=namespace.batch_id,
        batch_manifest_path=namespace.batch_manifest,
        batch_input_path=namespace.batch_input,
        drug_target_network_path=(
            namespace.drug_target_network or resources.drug_target_network.raw
        ),
        drug_substructure_network_path=(
            namespace.drug_substructure_network
            or resources.drug_substructure_network.raw
        ),
        settings=environment.config.netinfer,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "netinfer_predict_batch"]
