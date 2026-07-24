"""Score prepared compound target sets with legacy network-proximity z-scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.config.models import ProximityConfig
from lipid_screening_agent.runtime import InputError, RunContext, sha256_file
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ..input_prepare._common import add_execution_identity_arguments, execution_identity
from ._common import (
    atomic_write_delimited,
    resolve_proximity_output,
    resolve_run_input_file,
    resolve_successful_preparation,
)
from .algorithms import (
    CACHE_FORMAT_VERSION,
    build_node_to_equivalent_indices,
    get_degree_binning,
    score_target_sets,
)

NODE_ID = "proximity_score_compounds"
SCORES_PATH = "proximity_scores.csv"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} could not be read") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must contain a JSON object")
    return payload


def _manifest_file(
    context: RunContext,
    manifest: Mapping[str, Any],
    key: str,
) -> Path:
    files = manifest.get("files")
    hashes = manifest.get("file_sha256")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise InputError("preparation manifest is missing files/file_sha256 mappings")
    relative_path = files.get(key)
    expected_hash = hashes.get(key)
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise InputError(
            "preparation manifest is missing a required prepared file",
            details={"file_key": key},
        )
    path = resolve_run_input_file(context, relative_path, label=f"prepared {key}")
    if sha256_file(path) != expected_hash:
        raise InputError(
            "prepared proximity file does not match its manifest",
            details={"file_key": key, "path": relative_path},
        )
    return path


def _validate_settings(manifest: Mapping[str, Any], settings: ProximityConfig) -> None:
    parameters = manifest.get("parameters")
    expected = {
        "randomizations": settings.randomizations,
        "minimum_degree_bin_size": settings.minimum_degree_bin_size,
        "seed": settings.seed,
        "job_batch_size": settings.job_batch_size,
        "lower_is_better": settings.lower_is_better,
    }
    if parameters != expected:
        raise InputError(
            "proximity scoring configuration does not match preparation",
            details={"expected": expected, "prepared": parameters},
        )
    if not settings.lower_is_better:
        raise InputError("proximity lower_is_better must remain true")


def _load_network(path: Path) -> tuple[nx.Graph[int], np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {"node_ids", "edge_source_indices", "edge_target_indices"}
            if not required.issubset(payload.files):
                raise InputError("prepared network NPZ is missing required arrays")
            node_ids = np.asarray(payload["node_ids"])
            sources = np.asarray(payload["edge_source_indices"], dtype=np.int64)
            targets = np.asarray(payload["edge_target_indices"], dtype=np.int64)
    except InputError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise InputError("prepared network NPZ could not be loaded") from exc
    if node_ids.ndim != 1 or len(node_ids) == 0:
        raise InputError("prepared network contains no nodes")
    if sources.ndim != 1 or targets.ndim != 1 or sources.shape != targets.shape:
        raise InputError("prepared network edge arrays have invalid shapes")
    if (
        (sources < 0).any()
        or (targets < 0).any()
        or (sources >= len(node_ids)).any()
        or (targets >= len(node_ids)).any()
    ):
        raise InputError("prepared network edge index is out of bounds")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_ids)))
    graph.add_edges_from(zip(sources.tolist(), targets.tolist(), strict=True))
    if not nx.is_connected(graph):
        raise InputError("prepared PPI background is not connected")
    return graph, node_ids


def _load_distance_cache(
    path: Path,
    *,
    node_ids: np.ndarray,
    manifest: Mapping[str, Any],
    n_random: int,
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "node_ids",
                "disease_node_ids",
                "real_distances",
                "random_distances",
                "cache_key",
                "cache_format_version",
            }
            if not required.issubset(payload.files):
                raise InputError("disease distance cache is missing required arrays")
            cached_nodes = np.asarray(payload["node_ids"])
            disease_nodes = np.asarray(payload["disease_node_ids"])
            real = np.asarray(payload["real_distances"], dtype=np.int32)
            random_distances = np.asarray(payload["random_distances"], dtype=np.int32)
            cache_key = str(payload["cache_key"].item())
            format_version = str(payload["cache_format_version"].item())
    except InputError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise InputError("disease distance cache could not be loaded") from exc
    if not np.array_equal(cached_nodes, node_ids):
        raise InputError("distance-cache node order does not match prepared network")
    if disease_nodes.ndim != 1 or len(disease_nodes) == 0:
        raise InputError("distance cache contains an empty disease module")
    if real.shape != (len(node_ids),) or random_distances.shape != (
        n_random,
        len(node_ids),
    ):
        raise InputError("distance cache dimensions do not match prepared parameters")
    if (real < 0).any() or (random_distances < 0).any():
        raise InputError("distance cache contains unreachable-node distances")
    if cache_key != manifest.get("cache_key"):
        raise InputError("distance cache key does not match preparation manifest")
    if format_version != CACHE_FORMAT_VERSION:
        raise InputError("distance cache format version is incompatible")
    return np.concatenate((real[np.newaxis, :], random_distances), axis=0)


def _load_target_jobs(
    path: Path, *, node_count: int
) -> tuple[
    list[tuple[int, tuple[int, ...]]],
    dict[int, list[tuple[int, str]]],
    int,
]:
    jobs: list[tuple[int, tuple[int, ...]]] = []
    entries: dict[int, list[tuple[int, str]]] = defaultdict(list)
    target_set_to_job: dict[tuple[int, ...], int] = {}
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"ID", "target_lcc_indices"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise InputError("prepared targets table is missing required columns")
            row_count = 0
            for row_index, row in enumerate(reader):
                compound_id = str(row.get("ID") or "").strip()
                if not compound_id or compound_id in seen_ids:
                    raise InputError(
                        "prepared targets contain an empty or duplicate compound ID",
                        details={"ID": compound_id},
                    )
                seen_ids.add(compound_id)
                try:
                    raw_indices = json.loads(row.get("target_lcc_indices") or "")
                except json.JSONDecodeError as exc:
                    raise InputError(
                        "prepared target indices are not valid JSON",
                        details={"ID": compound_id},
                    ) from exc
                if not isinstance(raw_indices, list) or not raw_indices:
                    raise InputError(
                        "prepared compound has no LCC target indices",
                        details={"ID": compound_id},
                    )
                indices: list[int] = []
                for value in raw_indices:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise InputError("target LCC indices must be integers")
                    if value < 0 or value >= node_count:
                        raise InputError("target LCC index is out of bounds")
                    indices.append(value)
                target_set = tuple(sorted(set(indices)))
                job_index = target_set_to_job.get(target_set)
                if job_index is None:
                    job_index = len(jobs)
                    target_set_to_job[target_set] = job_index
                    jobs.append((job_index, target_set))
                entries[job_index].append((row_index, compound_id))
                row_count += 1
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError("prepared targets table could not be read") from exc
    if not jobs:
        raise InputError("prepared targets contain no scorable compounds")
    return jobs, entries, row_count


def _operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    preparation_manifest_path: str | Path,
    settings: ProximityConfig,
) -> None:
    started = time.perf_counter()
    execution.update_metrics(
        {
            "n_random": settings.randomizations,
            "minimum_degree_bin_size": settings.minimum_degree_bin_size,
            "seed": settings.seed,
            "job_batch_size": settings.job_batch_size,
            "lower_is_better": settings.lower_is_better,
            "elapsed_seconds": 0.0,
        }
    )
    try:
        (
            scientific_path,
            preparation_artifact,
            preparation_output_ids,
        ) = resolve_successful_preparation(context, preparation_manifest_path)
        execution.input_artifact_ids = tuple(
            dict.fromkeys(
                (
                    *execution.input_artifact_ids,
                    preparation_artifact.artifact_id,
                    *preparation_output_ids,
                )
            )
        )
        manifest = _load_json_object(scientific_path, label="proximity preparation manifest")
        _validate_settings(manifest, settings)
        network_path = _manifest_file(context, manifest, "prepared_network")
        targets_path = _manifest_file(context, manifest, "prepared_targets")
        cache_path = _manifest_file(context, manifest, "disease_distance_cache")
        graph, node_ids = _load_network(network_path)
        distance_matrix = _load_distance_cache(
            cache_path,
            node_ids=node_ids,
            manifest=manifest,
            n_random=settings.randomizations,
        )
        jobs, entries, compound_count = _load_target_jobs(targets_path, node_count=len(node_ids))
        try:
            bins = get_degree_binning(graph, settings.minimum_degree_bin_size)
            equivalents = build_node_to_equivalent_indices(
                bins, {node: node for node in graph.nodes()}
            )
            scores, execution_mode = score_target_sets(
                jobs,
                distance_matrix=distance_matrix,
                node_to_equivalent=equivalents,
                n_random=settings.randomizations,
                seed=settings.seed,
                job_batch_size=settings.job_batch_size,
            )
        except ValueError as exc:
            raise InputError(
                "prepared proximity data cannot support degree-matched scoring",
                details={"reason": str(exc)},
            ) from exc

        result_rows: list[tuple[int, dict[str, Any]]] = []
        for job_index, job_entries in entries.items():
            if job_index not in scores:
                raise InputError("a prepared target set did not receive a proximity score")
            score = float(scores[job_index])
            if not math.isfinite(score):
                raise InputError("proximity scoring produced a non-finite z-score")
            for row_index, compound_id in job_entries:
                result_rows.append((row_index, {"ID": compound_id, "z": score}))
        result_rows.sort(key=lambda item: item[0])
        stable_rows = [row for _, row in result_rows]
        stable_rows.sort(key=lambda row: row["z"])
        output_path = resolve_proximity_output(context, SCORES_PATH)
        atomic_write_delimited(
            output_path,
            stable_rows,
            fieldnames=("ID", "z"),
            delimiter=",",
            allowed_root=context.output_dir,
        )
        execution.update_metrics(
            {
                "input_compound_count": compound_count,
                "unique_target_set_count": len(jobs),
                "output_compound_count": len(stable_rows),
                "degree_bin_count": len(bins),
                "execution_mode": execution_mode,
                "result_order": "z_ascending_stable",
            }
        )
        execution.add_output("proximity_scores", output_path)
    finally:
        execution.metric("elapsed_seconds", time.perf_counter() - started)


def proximity_score_compounds(
    *,
    context: RunContext,
    preparation_manifest_path: str | Path,
    settings: ProximityConfig,
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
            preparation_manifest_path=preparation_manifest_path,
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
        description="Score compounds from a committed proximity preparation artifact"
    )
    add_common_runner_arguments(parser)
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(arguments, project_root=project_root)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = proximity_score_compounds(
        context=environment.context,
        preparation_manifest_path=namespace.prepared_manifest,
        settings=environment.config.proximity,
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


__all__ = ["SCORES_PATH", "build_parser", "main", "proximity_score_compounds"]
