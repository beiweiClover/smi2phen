"""Run the Python wSDTNBI predictor for mapped official DRUG nodes."""

from __future__ import annotations

import argparse
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
from lipid_screening_agent.runtime import RunContext
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import resolve_netinfer_output, resolve_run_input_file
from ._io import read_mapping
from ._settings import validate_netinfer_settings
from .python_prediction import execute_python_prediction

NODE_ID = "netinfer_predict_known"
OUTPUT_RELATIVE_PATH = "known_predictions.tsv"


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    mapping_path: str | Path,
    drug_target_network_path: str | Path,
    drug_substructure_network_path: str | Path,
    settings: NetInferConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            "prediction_backend": "python_torch",
            "device_requested": settings.device,
            "input_compound_count": 0,
            "known_compound_count": 0,
            "known_drug_node_count": 0,
            "no_op": False,
            "top_n_predicted_targets": settings.top_n_predicted_targets,
            "raw_prediction_row_count": 0,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        validate_netinfer_settings(settings)
        mapping_file = resolve_run_input_file(
            context, mapping_path, label="NetInfer input mapping"
        )
        mapping = read_mapping(mapping_file)
        known_rows = [row for row in mapping if row["netinfer_input_type"] == "DRUG"]
        known_nodes = list(
            dict.fromkeys(row["netinfer_input_id"] for row in known_rows)
        )
        execution.update_metrics(
            {
                "input_compound_count": len(mapping),
                "known_compound_count": len(known_rows),
                "known_drug_node_count": len(known_nodes),
            }
        )
        if not known_nodes:
            execution.metric("device_actual", "not_used")
            execution.metric("no_op", True)
            execution.logger.info(
                "known_prediction_no_op",
                "No mapped official DRUG inputs; Python wSDTNBI was not started",
            )
            return

        output = resolve_netinfer_output(context, OUTPUT_RELATIVE_PATH)
        execution.update_metrics(
            execute_python_prediction(
                execution,
                context=context,
                settings=settings,
                source_type="DRUG",
                source_ids=known_nodes,
                drug_target_network_path=drug_target_network_path,
                drug_substructure_network_path=drug_substructure_network_path,
                compound_substructure_path=None,
                output_path=output,
            )
        )
        execution.add_output("netinfer_known_predictions", output)
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def netinfer_predict_known(
    *,
    context: RunContext,
    mapping_path: str | Path,
    drug_target_network_path: str | Path,
    drug_substructure_network_path: str | Path,
    settings: NetInferConfig,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            mapping_path=mapping_path,
            drug_target_network_path=drug_target_network_path,
            drug_substructure_network_path=drug_substructure_network_path,
            settings=settings,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Python wSDTNBI prediction for mapped official DRUG nodes."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--mapping", required=True, type=Path)
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
    resources = environment.config.resources.netinfer
    result = netinfer_predict_known(
        context=environment.context,
        mapping_path=namespace.mapping,
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


__all__ = [
    "OUTPUT_RELATIVE_PATH",
    "build_parser",
    "main",
    "netinfer_predict_known",
]
