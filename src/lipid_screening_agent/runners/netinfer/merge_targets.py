"""Merge complete NetInfer raw outputs into the unified known-first target artifact."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import NetInferConfig
from lipid_screening_agent.runtime import InputError, RunContext, atomic_write_json
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import (
    add_execution_identity_arguments,
    execution_identity,
)
from ._common import (
    atomic_write_delimited,
    resolve_netinfer_output,
    resolve_run_input_file,
)
from ._io import batch_index, read_batch_manifest, read_mapping
from ._settings import validate_netinfer_settings
from .algorithms import (
    RawPrediction,
    merge_compound_targets,
    read_raw_predictions,
    validate_raw_predictions,
)

NODE_ID = "netinfer_merge_targets"
TARGETS_RELATIVE_PATH = "drug_targets.json"
MISSING_RELATIVE_PATH = "missing_predictions.tsv"


def read_target_map(path: Path) -> dict[str, str]:
    """Read the normalized UniProt-to-symbol map, retaining its first occurrence."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or not {
                "uniprot_id",
                "gene_symbol",
            }.issubset(reader.fieldnames):
                raise InputError("NetInfer target map must contain uniprot_id and gene_symbol")
            result: dict[str, str] = {}
            for row_number, row in enumerate(reader, start=2):
                uniprot_id = str(row.get("uniprot_id") or "").strip()
                gene_symbol = str(row.get("gene_symbol") or "").strip()
                if not uniprot_id or not gene_symbol:
                    raise InputError(
                        "NetInfer target map contains an empty value",
                        details={"row": row_number},
                    )
                result.setdefault(uniprot_id, gene_symbol)
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "NetInfer target map could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not result:
        raise InputError("NetInfer target map contains no target identifiers")
    return result


def _validate_prediction_inputs(
    *,
    mapping_rows: Sequence[Mapping[str, str]],
    manifest: Mapping[str, object],
    known_predictions_path: str | Path | None,
    batch_prediction_paths: Mapping[str, str | Path],
) -> tuple[bool, dict[str, dict[str, object]]]:
    has_known = any(row["netinfer_input_type"] == "DRUG" for row in mapping_rows)
    if has_known and known_predictions_path is None:
        raise InputError("known NetInfer mappings exist but the known raw output was not supplied")
    if not has_known and known_predictions_path is not None:
        raise InputError(
            "a known raw output was supplied even though the mapping has no known DRUG inputs"
        )

    batches = batch_index(manifest)
    expected = set(batches)
    provided = set(batch_prediction_paths)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise InputError(
            "NetInfer merge requires exactly one successful raw output for every batch",
            details={"missing_batch_ids": missing, "unexpected_batch_ids": extra},
        )
    return has_known, batches


def _load_predictions(
    *,
    context: RunContext,
    known_predictions_path: str | Path | None,
    batch_prediction_paths: Mapping[str, str | Path],
    batches: Mapping[str, Mapping[str, object]],
    execution: NodeExecution,
) -> tuple[list[RawPrediction], int, int]:
    predictions: list[RawPrediction] = []
    known_count = 0
    batch_count = 0
    if known_predictions_path is not None:
        known_file = resolve_run_input_file(
            context, known_predictions_path, label="NetInfer known raw output"
        )
        known = read_raw_predictions(known_file)
        validate_raw_predictions(
            known,
            expected_source_type="DRUG",
            # The legacy known command predicts the complete official DRUG network.
            # Mapping back to the submitted structures happens after loading.
            allowed_source_ids=None,
        )
        predictions.extend(known)
        known_count = len(known)
        execution.logger.info(
            "known_raw_loaded",
            "NetInfer known raw output was validated",
            raw_row_count=known_count,
        )

    total_batches = len(batches)
    for completed, batch_id in enumerate(sorted(batches), start=1):
        item = batches[batch_id]
        batch_file = resolve_run_input_file(
            context,
            batch_prediction_paths[batch_id],
            label=f"NetInfer {batch_id} raw output",
        )
        expected_path = str(item["prediction_path"])
        observed_path = context.relative_path(batch_file)
        if observed_path != expected_path:
            raise InputError(
                "NetInfer batch raw output does not match the manifest path",
                details={
                    "batch_id": batch_id,
                    "expected": expected_path,
                    "observed": observed_path,
                },
            )
        batch_predictions = read_raw_predictions(batch_file)
        validate_raw_predictions(
            batch_predictions,
            expected_source_type="COMPOUND",
            allowed_source_ids=set(item["compound_ids"]),
        )
        predictions.extend(batch_predictions)
        batch_count += len(batch_predictions)
        execution.logger.info(
            "batch_raw_loaded",
            "one NetInfer batch raw output was validated",
            batch_id=batch_id,
            raw_row_count=len(batch_predictions),
            completed=completed,
            total=total_batches,
        )
    return predictions, known_count, batch_count


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    mapping_path: str | Path,
    target_map_path: str | Path,
    batch_manifest_path: str | Path,
    known_predictions_path: str | Path | None,
    batch_prediction_paths: Mapping[str, str | Path],
    settings: NetInferConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            "device_actual": "cpu",
            "torch_version": "not_used",
            "rdkit_version": "not_used",
            "input_compound_count": 0,
            "known_compound_count": 0,
            "novel_compound_count": 0,
            "standardization_failed_count": 0,
            "batch_count": 0,
            "known_raw_row_count": 0,
            "batch_raw_row_count": 0,
            "raw_prediction_row_count": 0,
            "top_n_predicted_targets": settings.top_n_predicted_targets,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        validate_netinfer_settings(settings)
        mapping_file = resolve_run_input_file(context, mapping_path, label="NetInfer input mapping")
        target_map_file = resolve_run_input_file(
            context, target_map_path, label="NetInfer target map"
        )
        manifest_file = resolve_run_input_file(
            context, batch_manifest_path, label="NetInfer batch manifest"
        )
        mapping_rows = read_mapping(mapping_file)
        manifest = read_batch_manifest(manifest_file)
        has_known, batches = _validate_prediction_inputs(
            mapping_rows=mapping_rows,
            manifest=manifest,
            known_predictions_path=known_predictions_path,
            batch_prediction_paths=batch_prediction_paths,
        )
        known_source_ids = {
            row["netinfer_input_id"] for row in mapping_rows if row["netinfer_input_type"] == "DRUG"
        }
        novel_ids = {
            row["netinfer_input_id"]
            for row in mapping_rows
            if row["netinfer_input_type"] == "COMPOUND" and row["match_key"]
        }
        standardization_failed_ids = {
            row["netinfer_input_id"]
            for row in mapping_rows
            if row["netinfer_input_type"] == "COMPOUND" and not row["match_key"]
        }
        manifest_ids = {
            compound_id for batch in batches.values() for compound_id in batch["compound_ids"]
        }
        if novel_ids != manifest_ids:
            raise InputError(
                "NetInfer mapping and batch manifest disagree about novel compounds",
                details={
                    "missing_from_manifest": sorted(novel_ids - manifest_ids)[:50],
                    "unexpected_in_manifest": sorted(manifest_ids - novel_ids)[:50],
                },
            )
        execution.update_metrics(
            {
                "input_compound_count": len(mapping_rows),
                "known_compound_count": sum(
                    row["netinfer_input_type"] == "DRUG" for row in mapping_rows
                ),
                "known_drug_node_count": len(known_source_ids),
                "novel_compound_count": len(novel_ids),
                "standardization_failed_count": len(standardization_failed_ids),
                "batch_count": len(batches),
                "known_input_present": has_known,
            }
        )

        predictions, known_raw_count, batch_raw_count = _load_predictions(
            context=context,
            known_predictions_path=known_predictions_path,
            batch_prediction_paths=batch_prediction_paths,
            batches=batches,
            execution=execution,
        )
        uniprot_to_symbol = read_target_map(target_map_file)
        targets, missing, merge_metrics = merge_compound_targets(
            mapping_rows,
            predictions,
            uniprot_to_symbol,
            top_n_predicted=settings.top_n_predicted_targets,
        )
        targets_output = resolve_netinfer_output(context, TARGETS_RELATIVE_PATH)
        missing_output = resolve_netinfer_output(context, MISSING_RELATIVE_PATH)
        atomic_write_json(targets_output, targets, allowed_root=context.output_dir)
        atomic_write_delimited(
            missing_output,
            missing,
            allowed_root=context.output_dir,
            delimiter="\t",
            header=("ID", "reason"),
        )
        execution.update_metrics(
            {
                **merge_metrics,
                "target_map_entry_count": len(uniprot_to_symbol),
                "known_raw_row_count": known_raw_count,
                "batch_raw_row_count": batch_raw_count,
                "raw_prediction_row_count": len(predictions),
            }
        )
        execution.add_output("drug_targets", targets_output)
        execution.add_output("netinfer_missing_predictions", missing_output)
        if missing:
            execution.warn(f"{len(missing)} compound(s) had no usable NetInfer prediction rows.")
        if merge_metrics["unmapped_uniprot_count"]:
            execution.warn(
                f"{merge_metrics['unmapped_uniprot_count']} target entry/entries had no "
                "gene symbol and retained the UniProt ID as the symbol."
            )
        execution.logger.info(
            "netinfer_targets_merged",
            "NetInfer raw outputs were mapped to unified known-first gene targets",
            input_compound_count=len(mapping_rows),
            missing_prediction_count=len(missing),
            raw_prediction_row_count=len(predictions),
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def netinfer_merge_targets(
    *,
    context: RunContext,
    mapping_path: str | Path,
    target_map_path: str | Path,
    batch_manifest_path: str | Path,
    known_predictions_path: str | Path | None,
    batch_prediction_paths: Mapping[str, str | Path],
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
            target_map_path=target_map_path,
            batch_manifest_path=batch_manifest_path,
            known_predictions_path=known_predictions_path,
            batch_prediction_paths=batch_prediction_paths,
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


def parse_batch_prediction_arguments(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``BATCH_ID=PATH`` values without guessing paths."""

    result: dict[str, Path] = {}
    for value in values:
        batch_id, separator, path = value.partition("=")
        batch_id = batch_id.strip()
        path = path.strip()
        if not separator or not batch_id or not path:
            raise InputError(
                "--batch-prediction must use BATCH_ID=PATH",
                details={"value": value},
            )
        if batch_id in result:
            raise InputError(
                "a NetInfer batch prediction was supplied more than once",
                details={"batch_id": batch_id},
            )
        result[batch_id] = Path(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge complete NetInfer raw outputs into drug_targets.json."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--target-map", required=True, type=Path)
    parser.add_argument("--batch-manifest", required=True, type=Path)
    parser.add_argument("--known-predictions", type=Path)
    parser.add_argument(
        "--batch-prediction",
        action="append",
        default=[],
        metavar="BATCH_ID=PATH",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    try:
        batch_predictions = parse_batch_prediction_arguments(namespace.batch_prediction)
    except InputError as exc:
        build_parser().error(exc.message)
    result = netinfer_merge_targets(
        context=environment.context,
        mapping_path=namespace.mapping,
        target_map_path=namespace.target_map,
        batch_manifest_path=namespace.batch_manifest,
        known_predictions_path=namespace.known_predictions,
        batch_prediction_paths=batch_predictions,
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
    "MISSING_RELATIVE_PATH",
    "TARGETS_RELATIVE_PATH",
    "build_parser",
    "main",
    "netinfer_merge_targets",
    "parse_batch_prediction_arguments",
    "read_target_map",
]
