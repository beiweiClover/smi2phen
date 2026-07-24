"""Deterministic two- or three-evidence candidate ranking runner."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import RankingConfig
from lipid_screening_agent.runtime import (
    ConfigurationError,
    InputError,
    OutputContractError,
    RunContext,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity

NODE_ID = "rank_candidates"
FINAL_CANDIDATES_PATH = "artifacts/final/final_candidates.tsv"
RANKING_SUMMARY_PATH = "artifacts/final/ranking_summary.json"
FINAL_COLUMNS = (
    "final_rank",
    "compound_id",
    "compound_name",
    "evidence_mode",
    "evidence_count",
    "consensus_rank_percentile_mean",
    "kg_rank_mean",
    "kg_score_mean",
    "proximity_z",
    "gps_score",
)

_USER_LIBRARY_ID = re.compile(r"UserLibrary:([^;]+)")
_SUCCESSFUL_GPS_STATUSES = frozenset({NodeStatus.SUCCEEDED, NodeStatus.CACHED})


def _read_rows(path: Path, *, label: str) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise InputError(f"{label} has no header")
            fields = tuple(str(field).strip() for field in reader.fieldnames)
            if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
                raise InputError(f"{label} has an invalid or duplicate header")
            return [dict(row) for row in reader], fields
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(f"{label} could not be read", details={"path": str(path)}) from exc


def _require_columns(fields: Iterable[str], required: set[str], *, label: str) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise InputError(
            f"{label} is missing required columns",
            details={"missing_columns": missing},
        )


def _finite_number(value: object, *, label: str, row_number: int) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"{label} contains a non-numeric value",
            details={"row_number": row_number, "value": value},
        ) from exc
    if not math.isfinite(number):
        raise InputError(
            f"{label} contains a non-finite value",
            details={"row_number": row_number, "value": value},
        )
    return number


def _direction_key(value: float, direction: str) -> float:
    if direction == "ascending":
        return value
    if direction == "descending":
        return -value
    raise ConfigurationError(f"unsupported score direction: {direction!r}")


def _passes_filter(value: float, operator: str, threshold: float) -> bool:
    operations = {
        "lt": lambda: value < threshold,
        "le": lambda: value <= threshold,
        "gt": lambda: value > threshold,
        "ge": lambda: value >= threshold,
    }
    try:
        return operations[operator]()
    except KeyError as exc:
        raise ConfigurationError(f"unsupported ranking filter operator: {operator!r}") from exc


def _compound_ids(row: Mapping[str, str], *, row_number: int) -> list[str]:
    values: list[object]
    encoded = str(row.get("compound_ids") or "").strip()
    if encoded:
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise InputError(
                "KG compound_ids is not valid JSON",
                details={"row_number": row_number},
            ) from exc
        if not isinstance(decoded, list):
            raise InputError(
                "KG compound_ids must encode a JSON array",
                details={"row_number": row_number},
            )
        values = decoded
    else:
        source_ids = str(row.get("source_ids") or "")
        values = _USER_LIBRARY_ID.findall(source_ids)

    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            raise InputError(
                "KG compound_ids contains an empty or non-string ID",
                details={"row_number": row_number},
            )
        compound_id = raw.strip()
        if compound_id not in seen:
            seen.add(compound_id)
            result.append(compound_id)
    return result


def _prepare_kg(
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    settings: RankingConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    metric = settings.kg.metric
    _require_columns(fields, {"node_id", "node_name", metric, "score_mean"}, label="KG ranking")
    if "compound_ids" not in fields and "source_ids" not in fields:
        raise InputError("KG ranking requires compound_ids or source_ids for user-ID mapping")

    normalized: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        node_id = str(row.get("node_id") or "").strip()
        if not node_id or node_id in seen_nodes:
            raise InputError(
                "KG ranking contains an empty or duplicate node_id",
                details={"row_number": row_number, "node_id": node_id},
            )
        seen_nodes.add(node_id)
        primary = _finite_number(row.get(metric), label=f"KG {metric}", row_number=row_number)
        score_mean = _finite_number(
            row.get("score_mean"), label="KG score_mean", row_number=row_number
        )
        rank_median = (
            _finite_number(row.get("rank_median"), label="KG rank_median", row_number=row_number)
            if "rank_median" in fields
            else primary
        )
        rank_std = (
            _finite_number(row.get("rank_std"), label="KG rank_std", row_number=row_number)
            if "rank_std" in fields
            else 0.0
        )
        normalized.append(
            {
                "node_id": node_id,
                "compound_name": str(row.get("node_name") or "").strip(),
                "kg_rank_mean": primary,
                "kg_score_mean": score_mean,
                "rank_median": rank_median,
                "rank_std": rank_std,
                "compound_ids": _compound_ids(row, row_number=row_number),
            }
        )

    normalized.sort(
        key=lambda row: (
            _direction_key(row["kg_rank_mean"], settings.kg.direction),
            _direction_key(row["rank_median"], settings.kg.direction),
            row["rank_std"],
            -row["kg_score_mean"],
            row["node_id"],
        )
    )
    selected = normalized[: settings.kg.top_n]
    expanded: dict[str, dict[str, Any]] = {}
    duplicate_mappings = 0
    expanded_rows = 0
    for selection_position, row in enumerate(selected, start=1):
        for compound_id in row["compound_ids"]:
            expanded_rows += 1
            if compound_id in expanded:
                duplicate_mappings += 1
                continue
            expanded[compound_id] = {
                "compound_id": compound_id,
                "compound_name": row["compound_name"],
                "kg_rank_mean": row["kg_rank_mean"],
                "kg_score_mean": row["kg_score_mean"],
                "kg_selection_position": selection_position,
            }
    return expanded, {
        "kg_input_nodes": len(normalized),
        "kg_top_n_requested": settings.kg.top_n,
        "kg_top_n_nodes": len(selected),
        "kg_expanded_rows": expanded_rows,
        "kg_top_n_unique_compounds": len(expanded),
        "kg_duplicate_compound_mappings_resolved": duplicate_mappings,
    }


def _prepare_score_table(
    rows: Sequence[Mapping[str, str]],
    fields: Sequence[str],
    *,
    metric: str,
    label: str,
) -> dict[str, float]:
    _require_columns(fields, {"ID", metric}, label=label)
    scores: dict[str, float] = {}
    for row_number, row in enumerate(rows, start=2):
        compound_id = str(row.get("ID") or "").strip()
        if not compound_id or compound_id in scores:
            raise InputError(
                f"{label} contains an empty or duplicate compound ID",
                details={"row_number": row_number, "compound_id": compound_id},
            )
        scores[compound_id] = _finite_number(
            row.get(metric), label=f"{label} {metric}", row_number=row_number
        )
    return scores


def rank_percentiles(values: Mapping[str, float], *, direction: str) -> dict[str, float]:
    """Return pandas-compatible ``rank(method='min') / n`` percentiles."""

    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (_direction_key(item[1], direction), item[0]))
    result: dict[str, float] = {}
    previous: float | None = None
    tied_rank = 0
    for position, (identifier, value) in enumerate(ordered, start=1):
        directed_value = _direction_key(value, direction)
        if previous is None or directed_value != previous:
            tied_rank = position
            previous = directed_value
        result[identifier] = tied_rank / len(ordered)
    return result


def compute_candidate_ranking(
    *,
    kg_rows: Sequence[Mapping[str, str]],
    kg_fields: Sequence[str],
    proximity_rows: Sequence[Mapping[str, str]],
    proximity_fields: Sequence[str],
    settings: RankingConfig,
    gps_rows: Sequence[Mapping[str, str]] | None = None,
    gps_fields: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply configured evidence gates and equal-weight within-intersection percentiles."""

    enhanced = gps_rows is not None
    evidence_mode = "kg_proximity_gps" if enhanced else "kg_proximity"
    evidence_count = 3 if enhanced else 2
    kg, counts = _prepare_kg(kg_rows, kg_fields, settings)
    proximity = _prepare_score_table(
        proximity_rows,
        proximity_fields,
        metric=settings.proximity.metric,
        label="proximity scores",
    )
    proximity_pass = {
        compound_id: score
        for compound_id, score in proximity.items()
        if _passes_filter(
            score, settings.proximity.filter.operator, settings.proximity.filter.value
        )
    }
    counts.update(
        {
            "proximity_input_compounds": len(proximity),
            "proximity_threshold_pass": len(proximity_pass),
        }
    )

    intersection = set(kg) & set(proximity_pass)
    gps: dict[str, float] = {}
    gps_pass: dict[str, float] = {}
    if enhanced:
        gps = _prepare_score_table(
            gps_rows or (), gps_fields, metric=settings.gps.metric, label="GPS scores"
        )
        gps_pass = {
            compound_id: score
            for compound_id, score in gps.items()
            if _passes_filter(score, settings.gps.filter.operator, settings.gps.filter.value)
        }
        intersection &= set(gps_pass)
        counts.update(
            {
                "gps_input_compounds": len(gps),
                "gps_threshold_pass": len(gps_pass),
            }
        )
    else:
        counts.update({"gps_input_compounds": 0, "gps_threshold_pass": 0})

    identifiers = sorted(intersection)
    # The legacy final-ranking notebook ranks the deterministic KG Top-N selection
    # positions inside the final intersection.  Those positions preserve the
    # ensemble ordering (rank_mean, rank_median, rank_std, score_mean, node_id)
    # when rank_mean is tied; ranking raw rank_mean again would discard those
    # configured tie-breakers.
    kg_values = {identifier: kg[identifier]["kg_selection_position"] for identifier in identifiers}
    proximity_values = {identifier: proximity_pass[identifier] for identifier in identifiers}
    gps_values = (
        {identifier: gps_pass[identifier] for identifier in identifiers} if enhanced else {}
    )
    # Selection position 1 is always best because the configured KG direction
    # has already been applied while constructing the deterministic Top-N.
    kg_percentiles = rank_percentiles(kg_values, direction="ascending")
    proximity_percentiles = rank_percentiles(
        proximity_values, direction=settings.proximity.direction
    )
    gps_percentiles = (
        rank_percentiles(gps_values, direction=settings.gps.direction) if enhanced else {}
    )

    ranked: list[dict[str, Any]] = []
    sort_records: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for identifier in identifiers:
        component_percentiles = [kg_percentiles[identifier], proximity_percentiles[identifier]]
        if enhanced:
            component_percentiles.append(gps_percentiles[identifier])
        consensus = sum(component_percentiles) / evidence_count
        row = {
            "compound_id": identifier,
            "compound_name": kg[identifier]["compound_name"],
            "evidence_mode": evidence_mode,
            "evidence_count": evidence_count,
            "consensus_rank_percentile_mean": consensus,
            "kg_rank_mean": kg[identifier]["kg_rank_mean"],
            "kg_score_mean": kg[identifier]["kg_score_mean"],
            "proximity_z": proximity_pass[identifier],
            "gps_score": gps_pass[identifier] if enhanced else None,
        }
        sort_key = (
            consensus,
            kg_percentiles[identifier],
            proximity_percentiles[identifier],
            gps_percentiles.get(identifier, -1.0),
            _direction_key(row["kg_rank_mean"], settings.kg.direction),
            _direction_key(row["proximity_z"], settings.proximity.direction),
            _direction_key(row["gps_score"], settings.gps.direction) if enhanced else 0.0,
            identifier,
        )
        sort_records.append((sort_key, row))
    sort_records.sort(key=lambda item: item[0])
    for final_rank, (_, row) in enumerate(sort_records, start=1):
        ranked.append({"final_rank": final_rank, **row})

    counts["intersection_candidates"] = len(ranked)
    counts["final_candidates"] = len(ranked)
    summary = {
        "schema_version": "1.0",
        "status": "succeeded" if ranked else settings.empty_intersection.status,
        "evidence_mode": evidence_mode,
        "evidence_count": evidence_count,
        "method": settings.consensus.method,
        "thresholds": {
            "kg": {
                "metric": settings.kg.metric,
                "direction": settings.kg.direction,
                "top_n": settings.kg.top_n,
            },
            "proximity": {
                "metric": settings.proximity.metric,
                "direction": settings.proximity.direction,
                "operator": settings.proximity.filter.operator,
                "value": settings.proximity.filter.value,
            },
            "gps": {
                "applied": enhanced,
                "metric": settings.gps.metric,
                "direction": settings.gps.direction,
                "operator": settings.gps.filter.operator,
                "value": settings.gps.filter.value,
            },
            "auto_relax_thresholds": settings.empty_intersection.auto_relax_thresholds,
        },
        "percentile_definition": "minimum_tied_rank_divided_by_intersection_size",
        "component_rank_sources": {
            "kg": "deterministic_kg_top_n_selection_position",
            "proximity": settings.proximity.metric,
            "gps": settings.gps.metric if enhanced else None,
        },
        "weights": settings.consensus.weights,
        "tie_breakers": [
            "consensus_rank_percentile_mean ascending",
            "KG percentile ascending",
            "proximity percentile ascending",
            *(["GPS percentile ascending"] if enhanced else []),
            "configured raw evidence directions",
            "compound_id ascending",
        ],
        "stage_counts": counts,
        "recommendation": (
            "Review input evidence coverage and the configured scientific thresholds before any "
            "user-authorized rerun; thresholds were not changed automatically."
            if not ranked
            else None
        ),
    }
    return ranked, summary


def _write_candidates(path: Path, rows: Sequence[Mapping[str, Any]], *, allowed_root: Path) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FINAL_COLUMNS, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                column: "" if row.get(column) is None else row.get(column, "")
                for column in FINAL_COLUMNS
            }
        )
    atomic_write_text(path, buffer.getvalue(), allowed_root=allowed_root)


def _resolve_input(context: RunContext, value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = context.resolve_run_relative(path)
    try:
        path = ensure_within(path, context.run_dir)
    except Exception as exc:
        raise InputError(f"{label} must be inside the run workspace") from exc
    if not path.is_file():
        raise InputError(f"{label} does not exist", details={"path": str(path)})
    return path


def _resolve_outputs(context: RunContext) -> tuple[Path, Path]:
    expected = context.resolve_run_relative("artifacts/final")
    if context.output_dir != expected:
        raise OutputContractError(
            "rank_candidates output_dir must be artifacts/final",
            details={"expected": str(expected), "observed": str(context.output_dir)},
        )
    return (
        ensure_within(context.resolve_run_relative(FINAL_CANDIDATES_PATH), context.output_dir),
        ensure_within(context.resolve_run_relative(RANKING_SUMMARY_PATH), context.output_dir),
    )


def _gps_status(value: NodeStatus | str) -> NodeStatus:
    try:
        return value if isinstance(value, NodeStatus) else NodeStatus(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"unsupported GPS node status: {value!r}") from exc


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    kg_ranking_path: str | Path,
    proximity_scores_path: str | Path,
    gps_scores_path: str | Path | None,
    gps_status: NodeStatus | str,
    settings: RankingConfig,
) -> None:
    started = time.perf_counter()
    status = _gps_status(gps_status)
    execution.update_metrics(
        {
            "gps_upstream_status": status.value,
            "kg_top_n": settings.kg.top_n,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        if status not in _SUCCESSFUL_GPS_STATUSES | {NodeStatus.SKIPPED}:
            execution.metric("ranking_status", "blocked")
            execution.mark_blocked(
                f"GPS was planned but its terminal status is {status.value}; automatic downgrade is forbidden"
            )
            return
        if status in _SUCCESSFUL_GPS_STATUSES and gps_scores_path is None:
            raise InputError("successful GPS status requires a GPS score artifact")

        kg_path = _resolve_input(context, kg_ranking_path, label="KG ensemble ranking")
        proximity_path = _resolve_input(context, proximity_scores_path, label="proximity scores")
        gps_path = (
            _resolve_input(context, gps_scores_path, label="GPS scores")
            if status in _SUCCESSFUL_GPS_STATUSES and gps_scores_path is not None
            else None
        )
        kg_rows, kg_fields = _read_rows(kg_path, label="KG ensemble ranking")
        proximity_rows, proximity_fields = _read_rows(proximity_path, label="proximity scores")
        gps_rows: list[dict[str, str]] | None = None
        gps_fields: tuple[str, ...] = ()
        if gps_path is not None:
            gps_rows, gps_fields = _read_rows(gps_path, label="GPS scores")

        candidates, summary = compute_candidate_ranking(
            kg_rows=kg_rows,
            kg_fields=kg_fields,
            proximity_rows=proximity_rows,
            proximity_fields=proximity_fields,
            gps_rows=gps_rows,
            gps_fields=gps_fields,
            settings=settings,
        )
        candidates_path, summary_path = _resolve_outputs(context)
        summary["input_sources"] = {
            "kg_ensemble_ranking": context.relative_path(kg_path),
            "proximity_scores": context.relative_path(proximity_path),
            "gps_scores": None if gps_path is None else context.relative_path(gps_path),
            "gps_status": status.value,
        }
        summary["outputs"] = {
            "final_candidates": context.relative_path(candidates_path),
            "ranking_summary": context.relative_path(summary_path),
        }
        _write_candidates(candidates_path, candidates, allowed_root=context.output_dir)
        atomic_write_json(summary_path, summary, allowed_root=context.output_dir)
        execution.add_output("final_candidates", candidates_path)
        execution.add_output("ranking_summary", summary_path)
        execution.update_metrics(
            {
                **summary["stage_counts"],
                "ranking_status": summary["status"],
                "evidence_mode": summary["evidence_mode"],
                "evidence_count": summary["evidence_count"],
            }
        )
        if summary["status"] == "no_candidates_passed":
            execution.warn("no candidate passed the unchanged configured evidence thresholds")
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def rank_candidates(
    *,
    context: RunContext,
    kg_ranking_path: str | Path,
    proximity_scores_path: str | Path,
    gps_status: NodeStatus | str,
    settings: RankingConfig,
    config_hash: str,
    code_version: str,
    gps_scores_path: str | Path | None = None,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Run the final evidence intersection without implicit GPS downgrade."""

    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            kg_ranking_path=kg_ranking_path,
            proximity_scores_path=proximity_scores_path,
            gps_scores_path=gps_scores_path,
            gps_status=gps_status,
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
        description="Rank candidates using configured KG, proximity, and optional GPS evidence."
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--kg-ranking", required=True, type=Path)
    parser.add_argument("--proximity-scores", required=True, type=Path)
    parser.add_argument("--gps-scores", type=Path)
    parser.add_argument(
        "--gps-status",
        required=True,
        choices=[status.value for status in NodeStatus],
        help="Upstream GPS plan/result status; skipped selects core mode, failed blocks.",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    environment = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(namespace),
        project_root=Path(__file__).resolve().parents[4],
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = rank_candidates(
        context=environment.context,
        kg_ranking_path=namespace.kg_ranking,
        proximity_scores_path=namespace.proximity_scores,
        gps_scores_path=namespace.gps_scores,
        gps_status=namespace.gps_status,
        settings=environment.config.ranking,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.CACHED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FINAL_CANDIDATES_PATH",
    "FINAL_COLUMNS",
    "RANKING_SUMMARY_PATH",
    "build_parser",
    "compute_candidate_ranking",
    "main",
    "rank_candidates",
    "rank_percentiles",
]
