"""Validate provided compound targets behind the future NetInfer/Python provider boundary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.runtime import (
    InputError,
    PathSafetyError,
    RunContext,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ._common import add_execution_identity_arguments, execution_identity

NODE_ID = "import_drug_targets"
TARGETS_PATH = "artifacts/targets/drug_targets.json"
MAPPING_PATH = "artifacts/targets/target_mapping.tsv"


def _input_file(context: RunContext, value: str | Path, *, label: str) -> Path:
    try:
        path = ensure_within(
            Path(value).resolve(strict=True),
            context.resolve_run_relative("inputs/original", must_exist=True),
        )
    except (OSError, PathSafetyError, ValueError) as exc:
        raise InputError(f"{label} must be a registered input file") from exc
    if not path.is_file():
        raise InputError(f"{label} must be a regular file")
    return path


def _prepared_file(context: RunContext, value: str | Path, *, label: str) -> Path:
    try:
        path = ensure_within(Path(value).resolve(strict=True), context.run_dir)
    except (OSError, PathSafetyError, ValueError) as exc:
        raise InputError(f"{label} must be a committed run artifact") from exc
    if not path.is_file():
        raise InputError(f"{label} must be a regular file")
    return path


def _compounds(path: Path) -> tuple[list[str], dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError("normalized compounds could not be read") from exc
    if not rows or not {"ID", "SMILES"}.issubset(rows[0]):
        raise InputError("normalized compounds require ID and SMILES columns")
    order: list[str] = []
    smiles: dict[str, str] = {}
    for row in rows:
        compound_id = str(row.get("ID") or "").strip()
        value = str(row.get("SMILES") or "").strip()
        if not compound_id or not value or compound_id in smiles:
            raise InputError("normalized compounds contain an invalid or duplicate ID")
        order.append(compound_id)
        smiles[compound_id] = value
    return order, smiles


def _mapping(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["gene_symbol", "entrez_id"]:
                raise InputError("target mapping requires gene_symbol and entrez_id columns")
            rows = list(reader)
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError("target mapping could not be read") from exc
    output: list[tuple[str, str]] = []
    mapping: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        symbol = str(row.get("gene_symbol") or "").strip().upper()
        entrez_id = str(row.get("entrez_id") or "").strip()
        if not symbol or not entrez_id.isdigit():
            raise InputError(
                "target mapping contains an invalid row", details={"row": row_number}
            )
        previous = mapping.get(symbol)
        if previous is not None and previous != entrez_id:
            raise InputError(
                "target mapping assigns one symbol to multiple Entrez IDs",
                details={"gene_symbol": symbol},
            )
        if previous is None:
            mapping[symbol] = entrez_id
            output.append((symbol, entrez_id))
    if not output:
        raise InputError("target mapping contains no usable records")
    return output, mapping


def _target_item(
    value: Any,
    *,
    compound_id: str,
    position: int,
    mapping: Mapping[str, str],
    predicted_started: bool,
    previous_rank: int,
) -> tuple[dict[str, Any], bool, int]:
    if not isinstance(value, Mapping):
        raise InputError("drug target entries must be objects")
    symbol = str(value.get("gene_symbol") or "").strip().upper()
    uniprot_id = str(value.get("uniprot_id") or "").strip()
    evidence = str(value.get("evidence") or "").strip().lower()
    if not symbol or symbol not in mapping:
        raise InputError(
            "every target symbol must occur in target_mapping.tsv",
            details={"ID": compound_id, "target_index": position, "gene_symbol": symbol},
        )
    if not uniprot_id:
        raise InputError(
            "every target requires a non-empty uniprot_id",
            details={"ID": compound_id, "gene_symbol": symbol},
        )
    if evidence not in {"known", "predicted"}:
        raise InputError("target evidence must be known or predicted")
    item: dict[str, Any] = {
        "gene_symbol": symbol,
        "uniprot_id": uniprot_id,
        "evidence": evidence,
    }
    if evidence == "known":
        if predicted_started:
            raise InputError("known targets must precede predicted targets")
    else:
        predicted_started = True
        rank = value.get("prediction_rank")
        if isinstance(rank, bool) or not isinstance(rank, (int, float)):
            raise InputError("predicted targets require an integer prediction_rank")
        rank_value = int(rank)
        if float(rank) != rank_value or not 1 <= rank_value <= 10:
            raise InputError("prediction_rank must be an integer from 1 to 10")
        if rank_value < previous_rank:
            raise InputError("predicted targets must be ordered by prediction_rank")
        previous_rank = rank_value
        item["prediction_rank"] = rank_value
    if value.get("score") is not None:
        score = value["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InputError("target score must be numeric")
        score_value = float(score)
        if not math.isfinite(score_value):
            raise InputError("target score must be finite")
        item["score"] = score_value
    return item, predicted_started, previous_rank


def _targets(
    path: Path,
    *,
    compound_order: Sequence[str],
    compound_smiles: Mapping[str, str],
    mapping: Mapping[str, str],
) -> tuple[dict[str, Any], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError("drug_targets.json could not be read") from exc
    if not isinstance(payload, Mapping):
        raise InputError("drug_targets.json must be keyed by compound ID")
    missing = [compound_id for compound_id in compound_order if compound_id not in payload]
    extra = sorted(set(payload) - set(compound_order))
    if missing or extra:
        raise InputError(
            "drug target compound IDs must exactly match normalized compounds",
            details={"missing": missing[:20], "extra": extra[:20]},
        )

    normalized: dict[str, Any] = {}
    target_count = 0
    for compound_id in compound_order:
        info = payload[compound_id]
        if not isinstance(info, Mapping) or not isinstance(info.get("targets"), list):
            raise InputError("each compound must contain a targets list")
        smiles = str(info.get("smiles") or "").strip()
        if smiles != compound_smiles[compound_id]:
            raise InputError(
                "drug target SMILES must match the normalized compound library",
                details={"ID": compound_id},
            )
        targets: list[dict[str, Any]] = []
        seen: set[str] = set()
        predicted_started = False
        previous_rank = 0
        for position, raw_target in enumerate(info["targets"]):
            item, predicted_started, previous_rank = _target_item(
                raw_target,
                compound_id=compound_id,
                position=position,
                mapping=mapping,
                predicted_started=predicted_started,
                previous_rank=previous_rank,
            )
            if item["gene_symbol"] in seen:
                raise InputError(
                    "targets must be unique by gene_symbol",
                    details={"ID": compound_id, "gene_symbol": item["gene_symbol"]},
                )
            seen.add(item["gene_symbol"])
            targets.append(item)
        target_count += len(targets)
        normalized[compound_id] = {"smiles": smiles, "targets": targets}
    if target_count == 0:
        raise InputError("drug_targets.json contains no target records")
    return normalized, target_count


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    drug_targets_path: str | Path,
    target_mapping_path: str | Path,
    compounds_path: str | Path,
) -> None:
    source_targets = _input_file(context, drug_targets_path, label="drug targets")
    source_mapping = _input_file(context, target_mapping_path, label="target mapping")
    compounds = _prepared_file(context, compounds_path, label="normalized compounds")
    compound_order, compound_smiles = _compounds(compounds)
    mapping_rows, mapping = _mapping(source_mapping)
    targets, target_count = _targets(
        source_targets,
        compound_order=compound_order,
        compound_smiles=compound_smiles,
        mapping=mapping,
    )

    targets_output = context.resolve_run_relative(TARGETS_PATH)
    mapping_output = context.resolve_run_relative(MAPPING_PATH)
    atomic_write_json(targets_output, targets, allowed_root=context.run_dir)
    mapping_text = "gene_symbol\tentrez_id\n" + "".join(
        f"{symbol}\t{entrez_id}\n" for symbol, entrez_id in mapping_rows
    )
    atomic_write_text(mapping_output, mapping_text, allowed_root=context.run_dir)
    execution.update_metrics(
        {
            "compound_count": len(compound_order),
            "target_count": target_count,
            "mapping_count": len(mapping_rows),
            "target_source": "provided",
        }
    )
    execution.add_output("drug_targets", targets_output)
    execution.add_output("target_mapping", mapping_output)


def import_drug_targets(
    *,
    context: RunContext,
    drug_targets_path: str | Path,
    target_mapping_path: str | Path,
    compounds_path: str | Path,
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
            drug_targets_path=drug_targets_path,
            target_mapping_path=target_mapping_path,
            compounds_path=compounds_path,
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
    parser = argparse.ArgumentParser(description="Validate provided compound targets")
    add_common_runner_arguments(parser)
    parser.add_argument("--drug-targets", required=True, type=Path)
    parser.add_argument("--target-mapping", required=True, type=Path)
    parser.add_argument("--compounds", required=True, type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    environment = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(namespace),
        project_root=Path(__file__).resolve().parents[4],
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = import_drug_targets(
        context=environment.context,
        drug_targets_path=namespace.drug_targets,
        target_mapping_path=namespace.target_mapping,
        compounds_path=namespace.compounds,
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


__all__ = ["build_parser", "import_drug_targets", "main"]
