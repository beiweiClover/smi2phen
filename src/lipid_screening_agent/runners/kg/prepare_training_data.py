"""Prepare canonical typed KG edges and deterministic train/valid/test splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import DiseaseConfig, KGTrainingDataConfig
from lipid_screening_agent.runtime import InputError, RunContext, atomic_write_json, sha256_file
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from ._training_common import (
    atomic_dataframe_csv,
    load_training_dependencies,
    output_path,
    resolve_committed_artifact,
    resolve_run_file,
)

NODE_ID = "kg_prepare_training_data"
SCHEMA_VERSION = "1.0"
OUTPUT_FILES = {
    "kg_training_nodes": "node_typed.csv",
    "kg_training_base_edges": "kg_base.csv.gz",
    "kg_training_edges": "kg_directed.csv.gz",
    "kg_training_manifest": "manifest.json",
}


def compute_holdout_sizes(n_rows: int, valid_fraction: float, test_fraction: float):
    n_test = int(round(n_rows * test_fraction))
    n_valid = int(round(n_rows * valid_fraction))
    if n_rows >= 3:
        if test_fraction > 0 and n_test == 0:
            n_test = 1
        if valid_fraction > 0 and n_valid == 0:
            n_valid = 1
    maximum = max(n_rows - 1, 0)
    while n_test + n_valid > maximum and n_valid > 0:
        n_valid -= 1
    while n_test + n_valid > maximum and n_test > 0:
        n_test -= 1
    return n_valid, n_test


def _swap(frame, mask, pairs):
    for left, right in pairs:
        values = frame.loc[mask, left].copy()
        frame.loc[mask, left] = frame.loc[mask, right].values
        frame.loc[mask, right] = values.values
    return frame


def _add_reverse_edges(frame, pairs, pandas):
    frames = [frame.copy()]
    hetero = frame[frame["x_type"] != frame["y_type"]].copy()
    if not hetero.empty:
        _swap(hetero, pandas.Series(True, index=hetero.index), pairs)
        hetero["relation"] = "rev_" + hetero["relation"].astype(str)
        frames.append(hetero)
    homogeneous = frame[frame["x_type"] == frame["y_type"]].copy()
    if not homogeneous.empty:
        reverse = homogeneous.copy()
        _swap(reverse, pandas.Series(True, index=reverse.index), pairs)
        frames.append(reverse)
    directed = pandas.concat(frames, ignore_index=True)
    directed = directed.drop_duplicates(subset=["x_id", "relation", "y_id", "split"]).reset_index(
        drop=True
    )
    directed["directed_edge_id"] = [
        f"directed_edge:{index:09d}" for index in range(1, len(directed) + 1)
    ]
    return directed


def prepare_frames(nodes, graph, *, settings, custom_disease_id, numpy, pandas):
    required_nodes = {"node_index", "node_id", "node_type"}
    required_graph = {"x_type", "x_id", "x_index", "relation", "y_type", "y_id", "y_index"}
    if missing := required_nodes - set(nodes.columns):
        raise InputError(
            "kg_nodes is missing required columns", details={"missing": sorted(missing)}
        )
    if missing := required_graph - set(graph.columns):
        raise InputError(
            "kg_graph is missing required columns", details={"missing": sorted(missing)}
        )
    if nodes["node_id"].astype(str).duplicated().any():
        raise InputError("kg_nodes contains duplicate node_id values")
    if nodes["node_index"].duplicated().any():
        raise InputError("kg_nodes contains duplicate global node_index values")

    nodes = nodes.copy()
    graph = graph.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)
    graph["x_id"] = graph["x_id"].astype(str)
    graph["y_id"] = graph["y_id"].astype(str)
    nodes["node_index"] = pandas.to_numeric(nodes["node_index"], errors="raise").astype(int)
    graph["x_index"] = pandas.to_numeric(graph["x_index"], errors="raise").astype(int)
    graph["y_index"] = pandas.to_numeric(graph["y_index"], errors="raise").astype(int)
    nodes = nodes.sort_values("node_index", kind="stable").reset_index(drop=True)
    nodes["type_local_index"] = nodes.groupby("node_type", sort=False).cumcount().astype(int)
    local = nodes.set_index("node_id")["type_local_index"].to_dict()
    node_types = nodes.set_index("node_id")["node_type"].astype(str).to_dict()
    missing_endpoints = sorted((set(graph["x_id"]) | set(graph["y_id"])) - set(local))
    if missing_endpoints:
        raise InputError(
            "kg_graph references endpoints absent from kg_nodes",
            details={"count": len(missing_endpoints), "examples": missing_endpoints[:20]},
        )
    if any(node_types[node] != str(kind) for node, kind in zip(graph.x_id, graph.x_type)):
        raise InputError("kg_graph x_type disagrees with kg_nodes")
    if any(node_types[node] != str(kind) for node, kind in zip(graph.y_id, graph.y_type)):
        raise InputError("kg_graph y_type disagrees with kg_nodes")
    graph["x_idx"] = graph["x_id"].map(local).astype(int)
    graph["y_idx"] = graph["y_id"].map(local).astype(int)

    pairs = [
        ("x_type", "y_type"),
        ("x_id", "y_id"),
        ("x_index", "y_index"),
        ("x_idx", "y_idx"),
    ]
    for optional in (("x_name", "y_name"), ("x_source", "y_source")):
        if set(optional) <= set(graph.columns):
            pairs.append(optional)
    base = graph.copy()
    homogeneous = base["x_type"] == base["y_type"]
    _swap(base, homogeneous & (base["x_index"] > base["y_index"]), pairs)
    before = len(base)
    base = base.drop_duplicates(subset=["x_id", "relation", "y_id"]).reset_index(drop=True)
    base["base_edge_id"] = [f"base_edge:{index:09d}" for index in range(1, len(base) + 1)]
    base["split"] = pandas.NA
    pinned = (
        bool(settings.pin_custom_disease_positive_edges_to_train)
        & base["relation"].eq("drug_disease")
        & base["y_id"].eq(custom_disease_id)
    )
    base.loc[pinned, "split"] = "train"
    rng = numpy.random.RandomState(settings.seed)
    for relation in sorted(base["relation"].astype(str).unique()):
        indices = base.index[base["relation"].eq(relation) & base["split"].isna()].to_numpy()
        if not len(indices):
            continue
        indices = rng.permutation(indices)
        n_valid, n_test = compute_holdout_sizes(
            len(indices), settings.valid_fraction, settings.test_fraction
        )
        base.loc[indices[:n_test], "split"] = "test"
        base.loc[indices[n_test : n_test + n_valid], "split"] = "valid"
        base.loc[indices[n_test + n_valid :], "split"] = "train"
    if base["split"].isna().any():
        raise InputError("some KG edges did not receive a split")
    split_frames = []
    for split in ("train", "valid", "test"):
        split_frames.append(_add_reverse_edges(base[base.split == split].copy(), pairs, pandas))
    directed = pandas.concat(split_frames, ignore_index=True)
    metrics = {
        "num_rows_before_dedup": int(before),
        "num_base_edges": int(len(base)),
        "num_directed_edges": int(len(directed)),
        "num_nodes": int(len(nodes)),
        "pinned_edge_count": int(pinned.sum()),
        "base_split_counts": {str(k): int(v) for k, v in base.split.value_counts().items()},
        "directed_split_counts": {str(k): int(v) for k, v in directed.split.value_counts().items()},
        "node_type_counts": {str(k): int(v) for k, v in nodes.node_type.value_counts().items()},
        "canonical_etypes": [
            list(row)
            for row in directed[["x_type", "relation", "y_type"]]
            .drop_duplicates()
            .sort_values(["x_type", "relation", "y_type"])
            .itertuples(index=False, name=None)
        ],
    }
    return nodes, base, directed, metrics


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    settings: KGTrainingDataConfig,
    disease: DiseaseConfig,
    nodes_path: Path | None,
    graph_path: Path | None,
    construction_path: Path | None,
    nodes_manifest_path: Path | None,
    graph_manifest_path: Path | None,
    construction_manifest_path: Path | None,
) -> None:
    started = time.perf_counter()
    manifests = []
    if nodes_manifest_path is not None:
        nodes_path, manifest = resolve_committed_artifact(
            context,
            nodes_manifest_path,
            artifact_type="kg_nodes",
            producer_node_id="kg_construct_graph",
            expected_relative_path="artifacts/kg/construction/node.csv",
        )
        manifests.append(manifest)
        graph_path, manifest = resolve_committed_artifact(
            context,
            graph_manifest_path,
            artifact_type="kg_graph",
            producer_node_id="kg_construct_graph",
            expected_relative_path="artifacts/kg/construction/kg.csv",
        )
        manifests.append(manifest)
        construction_path, manifest = resolve_committed_artifact(
            context,
            construction_manifest_path,
            artifact_type="kg_construction_manifest",
            producer_node_id="kg_construct_graph",
            expected_relative_path="artifacts/kg/construction/manifest.json",
        )
        manifests.append(manifest)
        execution.input_artifact_ids = tuple(
            dict.fromkeys(
                (*execution.input_artifact_ids, *(item.artifact_id for item in manifests))
            )
        )
    else:
        nodes_path = resolve_run_file(context, nodes_path, label="kg_nodes")
        graph_path = resolve_run_file(context, graph_path, label="kg_graph")
        construction_path = resolve_run_file(
            context, construction_path, label="kg_construction_manifest"
        )
    dependencies = load_training_dependencies(require_dgl=False, require_torch=False)
    pandas, numpy = dependencies["pandas"], dependencies["numpy"]
    try:
        construction = json.loads(construction_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InputError("kg_construction_manifest is invalid JSON") from exc
    custom_disease_id = str(
        construction.get("disease", {}).get("custom_node_id") or disease.custom_node_id
    )
    candidate_source_tag = str(
        construction.get("configuration", {}).get("candidate_source_tag") or "UserLibrary:"
    )
    nodes = pandas.read_csv(nodes_path, sep="\t", low_memory=False)
    graph = pandas.read_csv(graph_path, low_memory=False)
    nodes, base, directed, metrics = prepare_frames(
        nodes,
        graph,
        settings=settings,
        custom_disease_id=custom_disease_id,
        numpy=numpy,
        pandas=pandas,
    )
    paths = {name: output_path(context, filename) for name, filename in OUTPUT_FILES.items()}
    atomic_dataframe_csv(
        nodes, paths["kg_training_nodes"], allowed_root=context.output_dir, index=False
    )
    atomic_dataframe_csv(
        base,
        paths["kg_training_base_edges"],
        allowed_root=context.output_dir,
        index=False,
        compression="gzip",
    )
    atomic_dataframe_csv(
        directed,
        paths["kg_training_edges"],
        allowed_root=context.output_dir,
        index=False,
        compression="gzip",
    )
    scientific_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "legacy_txgnn_relation_stratified_split",
        "configuration": asdict(settings),
        "custom_disease_id": custom_disease_id,
        "candidate_source_tag": candidate_source_tag,
        "inputs": {
            "kg_nodes_sha256": sha256_file(nodes_path),
            "kg_graph_sha256": sha256_file(graph_path),
            "kg_construction_manifest_sha256": sha256_file(construction_path),
            "artifact_ids": [item.artifact_id for item in manifests],
        },
        "metrics": metrics,
        "files": {
            key: context.relative_path(path)
            for key, path in paths.items()
            if key != "kg_training_manifest"
        },
        "file_sha256": {
            key: sha256_file(path) for key, path in paths.items() if key != "kg_training_manifest"
        },
    }
    atomic_write_json(
        paths["kg_training_manifest"], scientific_manifest, allowed_root=context.output_dir
    )
    for artifact_type, path in paths.items():
        execution.add_output(artifact_type, path)
    execution.update_metrics(metrics)
    execution.metric("elapsed_seconds", time.perf_counter() - started)


def kg_prepare_training_data(
    *,
    context: RunContext,
    settings: KGTrainingDataConfig,
    disease: DiseaseConfig,
    config_hash: str,
    code_version: str,
    nodes_path: str | Path | None = None,
    graph_path: str | Path | None = None,
    construction_path: str | Path | None = None,
    nodes_manifest_path: str | Path | None = None,
    graph_manifest_path: str | Path | None = None,
    construction_manifest_path: str | Path | None = None,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids=(),
) -> NodeResult:
    manifest_mode = nodes_manifest_path is not None
    if manifest_mode != (
        graph_manifest_path is not None and construction_manifest_path is not None
    ):
        raise InputError("all three Stage 06 artifact manifests must be supplied together")
    if not manifest_mode and None in (nodes_path, graph_path, construction_path):
        raise InputError("raw Stage 06 paths must be supplied together")
    return execute_node(
        lambda execution: _operation(
            execution,
            context=context,
            settings=settings,
            disease=disease,
            nodes_path=None if nodes_path is None else Path(nodes_path),
            graph_path=None if graph_path is None else Path(graph_path),
            construction_path=None if construction_path is None else Path(construction_path),
            nodes_manifest_path=None if nodes_manifest_path is None else Path(nodes_manifest_path),
            graph_manifest_path=None if graph_manifest_path is None else Path(graph_manifest_path),
            construction_manifest_path=(
                None if construction_manifest_path is None else Path(construction_manifest_path)
            ),
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
    parser = argparse.ArgumentParser(description="Prepare typed KG training splits.")
    add_common_runner_arguments(parser)
    parser.add_argument("--kg-nodes-manifest", required=True, type=Path)
    parser.add_argument("--kg-graph-manifest", required=True, type=Path)
    parser.add_argument("--kg-construction-manifest", required=True, type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv=None) -> int:
    namespace = build_parser().parse_args(argv)
    environment = load_common_runner_environment(
        CommonRunnerArguments.from_namespace(namespace),
        project_root=Path(__file__).resolve().parents[4],
    )
    task_id, attempt, artifact_ids = execution_identity(namespace)
    result = kg_prepare_training_data(
        context=environment.context,
        settings=environment.config.kg.training_data,
        disease=environment.config.disease,
        config_hash=environment.config_hash,
        code_version=__version__,
        nodes_manifest_path=namespace.kg_nodes_manifest,
        graph_manifest_path=namespace.kg_graph_manifest,
        construction_manifest_path=namespace.kg_construction_manifest,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.CACHED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "compute_holdout_sizes",
    "kg_prepare_training_data",
    "main",
    "prepare_frames",
]
