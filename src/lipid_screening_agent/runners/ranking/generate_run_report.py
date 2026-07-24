"""Structured, computation-only run report generator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import WorkflowConfig
from lipid_screening_agent.runtime import (
    InputError,
    OutputContractError,
    RunContext,
    atomic_write_json,
    atomic_write_text,
    ensure_within,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from .rank_candidates import FINAL_COLUMNS

NODE_ID = "generate_run_report"
REPORT_JSON_PATH = "reports/run_report.json"
REPORT_MARKDOWN_PATH = "reports/run_report.md"
_DISCLAIMER = (
    "This report describes computational evidence integration only and does not establish "
    "clinical efficacy, safety, indication, or treatment recommendations."
)
_DEFAULT_SKIP_FILES = {
    "invalid_smiles": "inputs/prepared/invalid_smiles.tsv",
    "unmapped_genes": "inputs/prepared/unmapped_genes.tsv",
    "netinfer_missing_predictions": "artifacts/netinfer/missing_predictions.tsv",
    "proximity_skipped_compounds": "artifacts/proximity/skipped_compounds.tsv",
    "proximity_unmapped_targets": "artifacts/proximity/unmapped_targets.tsv",
    "kg_invalid_smiles": "artifacts/kg/construction/invalid_smiles.tsv",
    "kg_unmapped_targets": "artifacts/kg/construction/unmapped_target_symbols.tsv",
}


def _sanitize(value: Any) -> Any:
    """Drop toxicity-named fields defensively; Stage 08 has no toxicity contract."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if "toxicity" not in str(key).casefold()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value


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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} could not be read as JSON") from exc
    if not isinstance(value, dict):
        raise InputError(f"{label} must contain a JSON object")
    return value


def _candidate_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FINAL_COLUMNS:
                raise InputError(
                    "final candidates header does not match the Stage 08 contract",
                    details={"observed": reader.fieldnames, "expected": list(FINAL_COLUMNS)},
                )
            count = 0
            for row_number, row in enumerate(reader, start=2):
                if not str(row.get("compound_id") or "").strip():
                    raise InputError(
                        "final candidates contains an empty compound_id",
                        details={"row_number": row_number},
                    )
                count += 1
            return count
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError("final candidates could not be read") from exc


def _read_node_results(
    context: RunContext, paths: Sequence[str | Path]
) -> list[tuple[Path, NodeResult]]:
    resolved: list[Path]
    if paths:
        resolved = [_resolve_input(context, path, label="node result") for path in paths]
    else:
        root = context.resolve_run_relative("artifacts/node_results")
        resolved = sorted(root.rglob("*.json")) if root.is_dir() else []
    unique = sorted(set(resolved), key=lambda path: context.relative_path(path))
    results: list[tuple[Path, NodeResult]] = []
    for path in unique:
        payload = _read_json_object(path, label="node result")
        try:
            result = NodeResult.from_dict(payload)
        except Exception as exc:
            raise InputError(
                "node result does not satisfy the runtime contract",
                details={"path": context.relative_path(path)},
            ) from exc
        if result.node_id == NODE_ID:
            continue
        results.append((path, result))
    results.sort(key=lambda item: (item[1].started_at, item[1].node_id, item[1].task_id))
    return results


def _skip_summary(path: Path, *, context: RunContext, category: str) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise InputError("skip-record artifact has no header")
            rows = list(reader)
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "skip-record artifact could not be read", details={"category": category}
        ) from exc
    reasons = Counter(
        str(row.get("reason") or "unspecified").strip() or "unspecified" for row in rows
    )
    return {
        "category": category,
        "path": context.relative_path(path),
        "record_count": len(rows),
        "reason_counts": dict(sorted(reasons.items())),
    }


def _observed_environment(results: Sequence[tuple[Path, NodeResult]]) -> dict[str, Any]:
    software: dict[str, set[str]] = {}
    hardware: dict[str, set[str]] = {}
    for _, result in results:
        for key, value in result.metrics.items():
            lowered = key.casefold()
            if "toxicity" in lowered or value is None:
                continue
            rendered = str(value)
            if lowered.endswith("_version") or lowered in {"python_version", "code_version"}:
                software.setdefault(key, set()).add(rendered)
            if any(token in lowered for token in ("device", "gpu", "cuda", "cpu", "memory")):
                hardware.setdefault(key, set()).add(rendered)
    return {
        "host": {
            "operating_system": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_logical_count": os.cpu_count(),
        },
        "software_observed_in_nodes": {
            key: sorted(values) for key, values in sorted(software.items())
        },
        "hardware_observed_in_nodes": {
            key: sorted(values) for key, values in sorted(hardware.items())
        },
    }


def _node_payload(path: Path, result: NodeResult, *, context: RunContext) -> dict[str, Any]:
    return {
        "node_id": result.node_id,
        "task_id": result.task_id,
        "status": result.status.value,
        "attempt": result.attempt,
        "started_at": result.to_dict()["started_at"],
        "finished_at": result.to_dict()["finished_at"],
        "duration_seconds": result.duration_seconds,
        "outputs": list(result.outputs),
        "metrics": _sanitize(dict(result.metrics)),
        "warnings": list(result.warnings),
        "error": None if result.error is None else _sanitize(result.error.to_dict()),
        "node_result_path": context.relative_path(path),
    }


def build_report_payload(
    *,
    context: RunContext,
    config: WorkflowConfig,
    run_manifest: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    ranking_summary: Mapping[str, Any],
    final_candidate_count: int,
    node_results: Sequence[tuple[Path, NodeResult]],
    skipped_records: Sequence[Mapping[str, Any]],
    environment_report: Mapping[str, Any] | None,
    output_paths: Mapping[str, str],
) -> dict[str, Any]:
    expected_count = (ranking_summary.get("stage_counts") or {}).get("final_candidates")
    if expected_count is not None and int(expected_count) != final_candidate_count:
        raise InputError(
            "ranking summary candidate count disagrees with final_candidates.tsv",
            details={"summary": expected_count, "table": final_candidate_count},
        )
    nodes = [_node_payload(path, result, context=context) for path, result in node_results]
    status_counts = Counter(node["status"] for node in nodes)
    started = [result.started_at for _, result in node_results]
    finished = [result.finished_at for _, result in node_results]
    timing = {
        "workflow_started_at": min(started).isoformat() if started else None,
        "workflow_finished_at": max(finished).isoformat() if finished else None,
        "workflow_wall_seconds": (
            (max(finished) - min(started)).total_seconds() if started and finished else 0.0
        ),
        "node_duration_sum_seconds": sum(result.duration_seconds for _, result in node_results),
    }
    executed_skips = [
        {"node_id": node["node_id"], "task_id": node["task_id"], "warnings": node["warnings"]}
        for node in nodes
        if node["status"] == NodeStatus.SKIPPED.value
    ]
    planned_skips = (run_manifest.get("planning") or {}).get("skipped_nodes") or []
    skipped_nodes_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in planned_skips:
        if not isinstance(raw, Mapping):
            continue
        node_id = str(raw.get("node_id") or "").strip()
        task_id = str(raw.get("task_id") or "").strip()
        if node_id and task_id:
            skipped_nodes_by_key[(node_id, task_id)] = {
                "node_id": node_id,
                "task_id": task_id,
                "reason": str(raw.get("reason") or "").strip() or None,
                "source": "plan",
            }
    for record in executed_skips:
        skipped_nodes_by_key[(record["node_id"], record["task_id"])] = {
            **record,
            "source": "node_result",
        }
    skipped_nodes = [skipped_nodes_by_key[key] for key in sorted(skipped_nodes_by_key)]
    payload = {
        "schema_version": "1.0",
        "report_type": "computational_evidence_run_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "workflow": {
            "id": config.workflow.id,
            "version": config.workflow.version,
            "evidence_mode": ranking_summary.get("evidence_mode"),
            "evidence_count": ranking_summary.get("evidence_count"),
            "ranking_status": ranking_summary.get("status"),
        },
        "configuration": config.to_dict(),
        "run_manifest": _sanitize(dict(run_manifest)),
        "input_sources": {
            "declared": _sanitize(run_manifest.get("input_sources", [])),
            "registered": _sanitize(input_manifest.get("inputs", [])),
        },
        "skips": {
            "node_records": skipped_nodes,
            "artifact_records": [_sanitize(dict(record)) for record in skipped_records],
            "artifact_record_total": sum(
                int(record.get("record_count", 0)) for record in skipped_records
            ),
        },
        "nodes": nodes,
        "node_status_counts": dict(sorted(status_counts.items())),
        "timing": timing,
        "environment": _observed_environment(node_results),
        "provided_environment_report": (
            None if environment_report is None else _sanitize(dict(environment_report))
        ),
        "ranking": _sanitize(dict(ranking_summary)),
        "candidate_count": final_candidate_count,
        "output_paths": dict(output_paths),
        "interpretation_scope": _DISCLAIMER,
    }
    return _sanitize(payload)


def render_markdown(report: Mapping[str, Any]) -> str:
    workflow = report["workflow"]
    timing = report["timing"]
    skip_info = report["skips"]
    lines = [
        f"# Run report: {report['run_id']}",
        "",
        "> 说明：本报告只描述计算证据整合，不构成临床疗效、安全性、适应证或治疗建议。",
        "",
        "## Result",
        "",
        f"- Evidence mode: `{workflow.get('evidence_mode')}`",
        f"- Evidence count: {workflow.get('evidence_count')}",
        f"- Ranking status: `{workflow.get('ranking_status')}`",
        f"- Final candidate count: {report.get('candidate_count', 0)}",
        "- Scientific thresholds were applied exactly as configured; no automatic relaxation was performed.",
        "",
        "## Inputs and skips",
        "",
        f"- Registered inputs: {len(report['input_sources'].get('registered') or [])}",
        f"- Skipped artifact records: {skip_info.get('artifact_record_total', 0)}",
        f"- Skipped nodes (planned or executed): {len(skip_info.get('node_records') or [])}",
        "",
        "## Node status",
        "",
        "| Node | Task | Status | Seconds |",
        "|---|---|---:|---:|",
    ]
    for node in report.get("nodes", []):
        lines.append(
            f"| `{node['node_id']}` | `{node['task_id']}` | `{node['status']}` | "
            f"{float(node['duration_seconds']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Timing and environment",
            "",
            f"- Workflow wall time: {float(timing.get('workflow_wall_seconds', 0.0)):.3f} seconds",
            f"- Operating system: {report['environment']['host'].get('operating_system')}",
            f"- Machine: {report['environment']['host'].get('machine')}",
            f"- Python: {report['environment']['host'].get('python_version')}",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, path in report.get("output_paths", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _resolve_outputs(context: RunContext) -> tuple[Path, Path]:
    expected = context.resolve_run_relative("reports")
    if context.output_dir != expected:
        raise OutputContractError(
            "generate_run_report output_dir must be reports",
            details={"expected": str(expected), "observed": str(context.output_dir)},
        )
    return (
        ensure_within(context.resolve_run_relative(REPORT_JSON_PATH), context.output_dir),
        ensure_within(context.resolve_run_relative(REPORT_MARKDOWN_PATH), context.output_dir),
    )


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    config: WorkflowConfig,
    run_manifest_path: str | Path,
    input_manifest_path: str | Path,
    ranking_summary_path: str | Path,
    final_candidates_path: str | Path,
    node_result_paths: Sequence[str | Path],
    skipped_record_paths: Mapping[str, str | Path],
    environment_report_path: str | Path | None,
) -> None:
    started = time.perf_counter()
    try:
        run_path = _resolve_input(context, run_manifest_path, label="run manifest")
        inputs_path = _resolve_input(context, input_manifest_path, label="input manifest")
        ranking_path = _resolve_input(context, ranking_summary_path, label="ranking summary")
        candidates_path = _resolve_input(context, final_candidates_path, label="final candidates")
        run_manifest = _read_json_object(run_path, label="run manifest")
        input_manifest = _read_json_object(inputs_path, label="input manifest")
        ranking_summary = _read_json_object(ranking_path, label="ranking summary")
        candidate_count = _candidate_count(candidates_path)
        node_results = _read_node_results(context, node_result_paths)

        skip_paths = dict(skipped_record_paths)
        for category, relative in _DEFAULT_SKIP_FILES.items():
            candidate = context.resolve_run_relative(relative)
            if category not in skip_paths and candidate.is_file():
                skip_paths[category] = candidate
        skip_records = []
        for category, raw_path in sorted(skip_paths.items()):
            if "toxicity" in category.casefold() or "toxicity" in str(raw_path).casefold():
                raise InputError("toxicity artifacts are outside the Stage 08 report contract")
            path = _resolve_input(context, raw_path, label=f"skip record {category}")
            skip_records.append(_skip_summary(path, context=context, category=category))

        environment_report = None
        if environment_report_path is not None:
            environment_path = _resolve_input(
                context, environment_report_path, label="environment report"
            )
            environment_report = _read_json_object(environment_path, label="environment report")
        report_json_path, report_md_path = _resolve_outputs(context)
        output_paths = {
            "final_candidates": context.relative_path(candidates_path),
            "ranking_summary": context.relative_path(ranking_path),
            "run_report_json": context.relative_path(report_json_path),
            "run_report_markdown": context.relative_path(report_md_path),
        }
        report = build_report_payload(
            context=context,
            config=config,
            run_manifest=run_manifest,
            input_manifest=input_manifest,
            ranking_summary=ranking_summary,
            final_candidate_count=candidate_count,
            node_results=node_results,
            skipped_records=skip_records,
            environment_report=environment_report,
            output_paths=output_paths,
        )
        atomic_write_json(report_json_path, report, allowed_root=context.output_dir)
        atomic_write_text(report_md_path, render_markdown(report), allowed_root=context.output_dir)
        execution.add_output("run_report_json", report_json_path)
        execution.add_output("run_report_markdown", report_md_path)
        execution.update_metrics(
            {
                "evidence_mode": report["workflow"]["evidence_mode"],
                "candidate_count": candidate_count,
                "reported_node_count": len(node_results),
                "skipped_record_count": report["skips"]["artifact_record_total"],
            }
        )
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def generate_run_report(
    *,
    context: RunContext,
    config: WorkflowConfig,
    run_manifest_path: str | Path,
    input_manifest_path: str | Path,
    ranking_summary_path: str | Path,
    final_candidates_path: str | Path,
    config_hash: str,
    code_version: str,
    node_result_paths: Sequence[str | Path] = (),
    skipped_record_paths: Mapping[str, str | Path] | None = None,
    environment_report_path: str | Path | None = None,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            config=config,
            run_manifest_path=run_manifest_path,
            input_manifest_path=input_manifest_path,
            ranking_summary_path=ranking_summary_path,
            final_candidates_path=final_candidates_path,
            node_result_paths=node_result_paths,
            skipped_record_paths=dict(skipped_record_paths or {}),
            environment_report_path=environment_report_path,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def _keyed_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key.strip() or not raw_path.strip() or key in result:
            raise InputError("--skipped-record must be a unique CATEGORY=PATH value")
        result[key.strip()] = Path(raw_path.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate JSON and Markdown run reports.")
    add_common_runner_arguments(parser)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--ranking-summary", required=True, type=Path)
    parser.add_argument("--final-candidates", required=True, type=Path)
    parser.add_argument("--node-result", action="append", default=[], type=Path)
    parser.add_argument(
        "--skipped-record",
        action="append",
        default=[],
        metavar="CATEGORY=PATH",
    )
    parser.add_argument("--environment-report", type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    environment = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(namespace),
        project_root=Path(__file__).resolve().parents[4],
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = generate_run_report(
        context=environment.context,
        config=environment.config,
        run_manifest_path=namespace.run_manifest,
        input_manifest_path=namespace.input_manifest,
        ranking_summary_path=namespace.ranking_summary,
        final_candidates_path=namespace.final_candidates,
        node_result_paths=namespace.node_result,
        skipped_record_paths=_keyed_paths(namespace.skipped_record),
        environment_report_path=namespace.environment_report,
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
    "REPORT_JSON_PATH",
    "REPORT_MARKDOWN_PATH",
    "build_parser",
    "build_report_payload",
    "generate_run_report",
    "main",
    "render_markdown",
]
