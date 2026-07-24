"""Aggregate all configured KG fine-tune seeds; RRA is deliberately optional and unused."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import KGAggregationConfig, KGFinetuneConfig
from lipid_screening_agent.runtime import InputError, RunContext, atomic_write_json
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from ._training_common import (
    atomic_dataframe_csv,
    check_unique_seeds,
    load_training_dependencies,
    output_path,
    resolve_committed_artifact,
)

NODE_ID = "kg_aggregate_seeds"


def aggregate_rankings(rankings, *, seeds, pandas, numpy):
    combined = pandas.concat(rankings, ignore_index=True)
    required = {"seed", "node_id", "rank", "score"}
    if missing := required - set(combined.columns):
        raise InputError(
            "seed rankings are missing required columns", details={"missing": sorted(missing)}
        )
    combined["node_id"] = combined.node_id.astype(str)
    combined["rank"] = pandas.to_numeric(combined["rank"], errors="raise").astype(float)
    combined["score"] = pandas.to_numeric(combined["score"], errors="raise").astype(float)
    expected_nodes = None
    for seed in seeds:
        frame = combined[combined.seed == seed]
        if frame.node_id.duplicated().any():
            raise InputError("a seed ranking contains duplicate node_id", details={"seed": seed})
        observed = set(frame.node_id)
        if expected_nodes is None:
            expected_nodes = observed
        elif observed != expected_nodes:
            raise InputError(
                "candidate node set differs between seed rankings",
                details={"seed": seed, "expected": len(expected_nodes), "observed": len(observed)},
            )
    metadata_columns = [
        column
        for column in (
            "node_id",
            "node_name",
            "source_ids",
            "compound_ids",
            "is_custom_disease_train_positive",
        )
        if column in combined.columns
    ]
    metadata = (
        combined.sort_values(["node_id", "seed"])
        .drop_duplicates("node_id")[metadata_columns]
        .copy()
    )
    grouped = combined.groupby("node_id", sort=False)
    result = grouped.agg(
        n_seeds=("seed", "nunique"),
        rank_mean=("rank", "mean"),
        rank_median=("rank", "median"),
        rank_std=("rank", "std"),
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
        score_mean=("score", "mean"),
        score_std=("score", "std"),
    ).reset_index()
    frequency = (
        grouped["rank"]
        .agg(
            top100_freq=lambda values: float(numpy.mean(values.to_numpy() <= 100)),
            top200_freq=lambda values: float(numpy.mean(values.to_numpy() <= 200)),
        )
        .reset_index()
    )
    result = result.merge(frequency, on="node_id", how="inner")
    result[["rank_std", "score_std"]] = result[["rank_std", "score_std"]].fillna(0.0)
    if not (result.n_seeds == len(seeds)).all():
        raise InputError("one or more candidates are missing a configured seed")
    result = metadata.merge(result, on="node_id", how="inner")
    result = result.sort_values(
        ["rank_mean", "rank_median", "rank_std", "score_mean", "node_id"],
        ascending=[True, True, True, False, True],
    ).reset_index(drop=True)
    result.insert(0, "ensemble_rank", numpy.arange(1, len(result) + 1))
    return result


def _flatten_summary(summary):
    row = {
        "seed": int(summary["seed"]),
        "status": summary.get("status"),
        "seed_configuration_hash": summary.get("seed_configuration_hash"),
        "best_epoch": summary.get("best_epoch"),
        "best_valid_metric": summary.get("best_valid_metric"),
        "training_seconds": summary.get("training_seconds"),
    }
    for prefix in ("best_valid_metrics", "test"):
        for key, value in (summary.get(prefix) or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"{prefix}_{key}"] = value
    for key, value in (summary.get("custom_disease_train_positive_rank") or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"custom_{key}"] = value
    return row


def _load_raw_results(seed_result_dirs, configured, pandas):
    rankings, summaries, missing = [], [], []
    for seed in configured:
        directory = Path(seed_result_dirs.get(seed, "")) if seed in seed_result_dirs else None
        ranking = None if directory is None else directory / "custom_disease_drug_ranking.csv"
        summary = None if directory is None else directory / "summary.json"
        if ranking is None or summary is None or not ranking.is_file() or not summary.is_file():
            missing.append(seed)
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("status") != "succeeded" or int(payload.get("seed", -1)) != seed:
            missing.append(seed)
            continue
        frame = pandas.read_csv(ranking, low_memory=False)
        frame["seed"] = seed
        rankings.append(frame)
        summaries.append(payload)
    return rankings, summaries, missing


def _operation(
    execution: NodeExecution,
    *,
    context,
    finetune_settings,
    aggregation_settings,
    ranking_manifest_paths,
    summary_manifest_paths,
    seed_result_dirs,
):
    started = time.perf_counter()
    configured = check_unique_seeds(finetune_settings.seeds)
    deps = load_training_dependencies(require_dgl=False, require_torch=False)
    pandas, numpy = deps["pandas"], deps["numpy"]
    rankings, summaries, missing = [], [], []
    if ranking_manifest_paths or summary_manifest_paths:
        ranking_by_task = {}
        summary_by_task = {}
        upstream = []
        for path in ranking_manifest_paths:
            file_path, manifest = resolve_committed_artifact(
                context,
                path,
                artifact_type="kg_seed_ranking",
                producer_node_id="kg_finetune_seed",
            )
            ranking_by_task[manifest.producer_task_id] = (file_path, manifest)
            upstream.append(manifest)
        for path in summary_manifest_paths:
            file_path, manifest = resolve_committed_artifact(
                context,
                path,
                artifact_type="kg_seed_summary",
                producer_node_id="kg_finetune_seed",
            )
            summary_by_task[manifest.producer_task_id] = (file_path, manifest)
            upstream.append(manifest)
        execution.input_artifact_ids = tuple(
            dict.fromkeys((*execution.input_artifact_ids, *(item.artifact_id for item in upstream)))
        )
        observed = {}
        for task_id in sorted(set(ranking_by_task) & set(summary_by_task)):
            payload = json.loads(summary_by_task[task_id][0].read_text(encoding="utf-8"))
            seed = int(payload.get("seed", -1))
            if payload.get("status") != "succeeded" or seed not in configured or seed in observed:
                continue
            frame = pandas.read_csv(ranking_by_task[task_id][0], low_memory=False)
            frame["seed"] = seed
            observed[seed] = (frame, payload)
        for seed in configured:
            if seed not in observed:
                missing.append(seed)
            else:
                rankings.append(observed[seed][0])
                summaries.append(observed[seed][1])
    else:
        rankings, summaries, missing = _load_raw_results(seed_result_dirs, configured, pandas)
    if missing:
        execution.metric("configured_seeds", list(configured))
        execution.metric("missing_seeds", missing)
        execution.metric("completed_seed_count", len(configured) - len(missing))
        execution.mark_blocked(
            "aggregation requires every configured seed; missing/unsuccessful seeds: "
            + ", ".join(str(seed) for seed in missing)
        )
        return
    hashes = {str(summary.get("seed_configuration_hash")) for summary in summaries}
    if len(hashes) != 1 or None in hashes or "None" in hashes:
        raise InputError(
            "seed summaries do not share one seed configuration hash",
            details={"hashes": sorted(hashes)},
        )
    seed_hash = hashes.pop()
    ensemble = aggregate_rankings(rankings, seeds=configured, pandas=pandas, numpy=numpy)
    base = seed_hash
    ranking_path = output_path(context, f"{base}/kg_ensemble_ranking.csv")
    seed_summaries_path = output_path(context, f"{base}/ensemble_seed_summaries.csv")
    summary_path = output_path(context, f"{base}/ensemble_summary.json")
    atomic_dataframe_csv(ensemble, ranking_path, allowed_root=context.output_dir, index=False)
    flat = pandas.DataFrame([_flatten_summary(summary) for summary in summaries]).sort_values(
        "seed"
    )
    atomic_dataframe_csv(flat, seed_summaries_path, allowed_root=context.output_dir, index=False)
    summary_payload = {
        "status": "succeeded",
        "method": "rank_mean_5_seed_consensus",
        "primary_sort": aggregation_settings.primary_sort,
        "configured_seeds": list(configured),
        "num_seeds": len(configured),
        "seed_configuration_hash": seed_hash,
        "num_candidates": int(len(ensemble)),
        "required_metrics": list(aggregation_settings.required_metrics),
        "rra": {
            "executed": False,
            "enabled_by_default": aggregation_settings.rra.enabled_by_default,
            "role": aggregation_settings.rra.role,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "files": {
            "kg_ensemble_ranking": context.relative_path(ranking_path),
            "ensemble_seed_summaries": context.relative_path(seed_summaries_path),
        },
    }
    atomic_write_json(summary_path, summary_payload, allowed_root=context.output_dir)
    execution.add_output("kg_ensemble_ranking", ranking_path)
    execution.add_output("kg_ensemble_seed_summaries", seed_summaries_path)
    execution.add_output("kg_ensemble_summary", summary_path)
    execution.update_metrics(
        {
            "seed_configuration_hash": seed_hash,
            "configured_seeds": list(configured),
            "num_seeds": len(configured),
            "num_candidates": len(ensemble),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )


def kg_aggregate_seeds(
    *,
    context: RunContext,
    finetune_settings: KGFinetuneConfig,
    aggregation_settings: KGAggregationConfig,
    config_hash: str,
    code_version: str,
    ranking_manifest_paths=(),
    summary_manifest_paths=(),
    seed_result_dirs=None,
    task_id="main",
    attempt=1,
    input_artifact_ids=(),
) -> NodeResult:
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            finetune_settings=finetune_settings,
            aggregation_settings=aggregation_settings,
            ranking_manifest_paths=tuple(ranking_manifest_paths),
            summary_manifest_paths=tuple(summary_manifest_paths),
            seed_result_dirs=dict(seed_result_dirs or {}),
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate every configured KG fine-tune seed.")
    add_common_runner_arguments(parser)
    parser.add_argument("--seed-ranking-manifest", action="append", default=[], type=Path)
    parser.add_argument("--seed-summary-manifest", action="append", default=[], type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv=None):
    ns = build_parser().parse_args(argv)
    env = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(ns), project_root=Path(__file__).resolve().parents[4]
    )
    task_id, attempt, artifact_ids = execution_identity(ns)
    result = kg_aggregate_seeds(
        context=env.context,
        finetune_settings=env.config.kg.finetune,
        aggregation_settings=env.config.kg.aggregation,
        config_hash=env.config_hash,
        code_version=__version__,
        ranking_manifest_paths=ns.seed_ranking_manifest,
        summary_manifest_paths=ns.seed_summary_manifest,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.CACHED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["aggregate_rankings", "build_parser", "kg_aggregate_seeds", "main"]
