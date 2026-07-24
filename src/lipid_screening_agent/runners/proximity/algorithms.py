"""Legacy-equivalent network-proximity algorithms.

The functions in this module preserve the two degree-matched randomizations
from ``模块_proximity/proximity.ipynb``: one randomizes the disease module
before shortest-path caching and the other randomizes each drug target set
while calculating its z-score.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import networkx as nx
import numpy as np

ALGORITHM_VERSION = "legacy-degree-matched-proximity-v1"
CACHE_FORMAT_VERSION = "proximity-distance-npz-v1"


def get_degree_binning(graph: nx.Graph[Any], bin_size: int) -> list[tuple[int, int, list[Any]]]:
    """Exactly reproduce the legacy toolbox ``get_degree_binning`` behavior."""

    degree_to_nodes: dict[int, list[Any]] = {}
    for node, degree in graph.degree():
        degree_to_nodes.setdefault(degree, []).append(node)

    values = sorted(degree_to_nodes)
    bins: list[tuple[int, int, list[Any]]] = []
    i = 0
    while i < len(values):
        low = values[i]
        members = list(degree_to_nodes[values[i]])
        while len(members) < bin_size:
            i += 1
            if i == len(values):
                break
            members.extend(degree_to_nodes[values[i]])
        if i == len(values):
            i -= 1
        high = values[i]
        i += 1

        if len(members) < bin_size and bins:
            previous_low, _, previous_members = bins[-1]
            bins[-1] = (previous_low, high, previous_members + members)
        else:
            bins.append((low, high, members))
    return bins


def get_degree_equivalents(
    selected_nodes: Sequence[Any],
    bins: Sequence[tuple[int, int, Sequence[Any]]],
    graph: nx.Graph[Any],
) -> dict[Any, list[Any]]:
    equivalents: dict[Any, list[Any]] = {}
    for selected in selected_nodes:
        degree = graph.degree(selected)
        for low, high, bin_nodes in bins:
            if low <= degree <= high:
                candidates = list(bin_nodes)
                if selected in candidates:
                    candidates.remove(selected)
                if not candidates:
                    raise ValueError(f"node {selected!r} has no degree-matched random candidate")
                equivalents[selected] = candidates
                break
        if selected not in equivalents:
            raise ValueError(f"node {selected!r} has no matching degree bin")
    return equivalents


def pick_random_nodes_matching_selected(
    graph: nx.Graph[Any],
    bins: Sequence[tuple[int, int, Sequence[Any]]],
    selected_nodes: Sequence[Any],
    n_random: int,
    *,
    seed: int,
) -> list[list[Any]]:
    """Reproduce the legacy degree-aware sampling, including 20 de-dup retries."""

    rng = random.Random(seed)
    equivalents = get_degree_equivalents(selected_nodes, bins, graph)
    values: list[list[Any]] = []
    for _ in range(n_random):
        random_nodes: set[Any] = set()
        for candidates in equivalents.values():
            chosen = rng.choice(candidates)
            for _ in range(20):
                if chosen in random_nodes:
                    chosen = rng.choice(candidates)
            random_nodes.add(chosen)
        values.append(list(random_nodes))
    return values


def build_node_to_equivalent_indices(
    bins: Sequence[tuple[int, int, Sequence[Any]]],
    node_to_index: dict[Any, int],
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for _, _, bin_nodes in bins:
        bin_indices = [node_to_index[node] for node in bin_nodes]
        for node in bin_nodes:
            node_index = node_to_index[node]
            candidates = tuple(index for index in bin_indices if index != node_index)
            if not candidates:
                raise ValueError(f"node {node!r} has no degree-matched random candidate")
            result[node_index] = candidates
    if len(result) != len(node_to_index):
        raise ValueError("one or more PPI nodes were not assigned to a degree bin")
    return result


def adjacency_from_graph(
    graph: nx.Graph[Any], node_list: Sequence[Any]
) -> tuple[tuple[int, ...], ...]:
    node_to_index = {node: index for index, node in enumerate(node_list)}
    return tuple(
        tuple(node_to_index[neighbor] for neighbor in graph.neighbors(node)) for node in node_list
    )


def multi_source_bfs_distance(
    source_indices: Sequence[int], adjacency: Sequence[Sequence[int]]
) -> np.ndarray:
    distances = np.full(len(adjacency), -1, dtype=np.int32)
    queue: deque[int] = deque()
    for source_index in source_indices:
        source_index = int(source_index)
        if distances[source_index] == 0:
            continue
        distances[source_index] = 0
        queue.append(source_index)

    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1
        for neighbor in adjacency[current]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    if (distances < 0).any():
        raise ValueError("the PPI background is not connected")
    return distances


_DISTANCE_ADJACENCY: tuple[tuple[int, ...], ...] | None = None


def _distance_worker(job: tuple[int, tuple[int, ...]]) -> tuple[int, np.ndarray]:
    if _DISTANCE_ADJACENCY is None:  # pragma: no cover - worker invariant
        raise RuntimeError("distance worker was not initialized")
    job_index, sources = job
    return job_index, multi_source_bfs_distance(sources, _DISTANCE_ADJACENCY)


def _fork_available() -> bool:
    return "fork" in mp.get_all_start_methods()


def build_disease_distance_matrix(
    *,
    graph: nx.Graph[Any],
    node_list: Sequence[Any],
    disease_nodes: Sequence[Any],
    n_random: int,
    minimum_bin_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[int, tuple[int, ...]], int, str]:
    """Build real/random disease-to-all-node distances in deterministic row order."""

    node_to_index = {node: index for index, node in enumerate(node_list)}
    bins = get_degree_binning(graph, minimum_bin_size)
    node_to_equivalent = build_node_to_equivalent_indices(bins, node_to_index)
    random_modules = pick_random_nodes_matching_selected(
        graph,
        bins,
        disease_nodes,
        n_random,
        seed=seed,
    )
    jobs: list[tuple[int, tuple[int, ...]]] = [
        (0, tuple(sorted(node_to_index[node] for node in disease_nodes)))
    ]
    for row_index, random_nodes in enumerate(random_modules, start=1):
        jobs.append(
            (
                row_index,
                tuple(sorted({node_to_index[node] for node in random_nodes})),
            )
        )

    adjacency = adjacency_from_graph(graph, node_list)
    matrix = np.empty((len(jobs), len(node_list)), dtype=np.int32)
    execution_mode = "serial"
    if _fork_available() and len(jobs) > 1:
        global _DISTANCE_ADJACENCY
        _DISTANCE_ADJACENCY = adjacency
        context = mp.get_context("fork")
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            futures = [executor.submit(_distance_worker, job) for job in jobs]
            for future in as_completed(futures):
                row_index, distances = future.result()
                matrix[row_index] = distances
        execution_mode = "fork"
    else:
        for row_index, sources in jobs:
            matrix[row_index] = multi_source_bfs_distance(sources, adjacency)
    return matrix, node_to_equivalent, len(bins), execution_mode


def calculate_target_z(
    target_indices: Sequence[int],
    *,
    real_disease_distance: np.ndarray,
    random_disease_distances: np.ndarray,
    node_to_equivalent: dict[int, tuple[int, ...]],
    n_random: int,
    seed: int,
) -> float:
    target_array = np.fromiter(target_indices, dtype=np.int32, count=len(target_indices))
    real_distance = float(real_disease_distance[target_array].mean())

    rng = random.Random(seed)
    equivalent_lists = [node_to_equivalent[index] for index in target_indices]
    random_values = np.empty(n_random, dtype=np.float64)
    for random_index in range(n_random):
        sampled_nodes: set[int] = set()
        for candidates in equivalent_lists:
            chosen = rng.choice(candidates)
            for _ in range(20):
                if chosen in sampled_nodes:
                    chosen = rng.choice(candidates)
            sampled_nodes.add(chosen)
        sampled_array = np.fromiter(sampled_nodes, dtype=np.int32, count=len(sampled_nodes))
        random_values[random_index] = float(
            random_disease_distances[random_index, sampled_array].mean()
        )

    random_mean = float(random_values.mean())
    random_standard_deviation = float(random_values.std())
    if random_standard_deviation == 0.0:
        return 0.0
    return (real_distance - random_mean) / random_standard_deviation


_SCORE_STATE: (
    tuple[
        np.ndarray,
        np.ndarray,
        dict[int, tuple[int, ...]],
        int,
        int,
    ]
    | None
) = None


def _score_worker(
    batch: Sequence[tuple[int, tuple[int, ...]]],
) -> list[tuple[int, float]]:
    if _SCORE_STATE is None:  # pragma: no cover - worker invariant
        raise RuntimeError("score worker was not initialized")
    real, random_distances, equivalents, n_random, seed = _SCORE_STATE
    return [
        (
            job_index,
            calculate_target_z(
                target_indices,
                real_disease_distance=real,
                random_disease_distances=random_distances,
                node_to_equivalent=equivalents,
                n_random=n_random,
                seed=seed,
            ),
        )
        for job_index, target_indices in batch
    ]


def _chunks(
    records: Sequence[tuple[int, tuple[int, ...]]], chunk_size: int
) -> list[Sequence[tuple[int, tuple[int, ...]]]]:
    return [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]


def score_target_sets(
    jobs: Sequence[tuple[int, tuple[int, ...]]],
    *,
    distance_matrix: np.ndarray,
    node_to_equivalent: dict[int, tuple[int, ...]],
    n_random: int,
    seed: int,
    job_batch_size: int,
) -> tuple[dict[int, float], str]:
    """Score unique target sets, using fork only where it is natively safe."""

    global _SCORE_STATE
    if distance_matrix.shape[0] != n_random + 1:
        raise ValueError("disease distance matrix row count does not match n_random")
    real = distance_matrix[0]
    random_distances = distance_matrix[1:]
    batches = _chunks(jobs, job_batch_size)
    scores: dict[int, float] = {}
    execution_mode = "serial"
    if _fork_available() and len(jobs) > 1:
        _SCORE_STATE = (real, random_distances, node_to_equivalent, n_random, seed)
        context = mp.get_context("fork")
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
            futures = [executor.submit(_score_worker, batch) for batch in batches]
            for future in as_completed(futures):
                for job_index, score in future.result():
                    scores[job_index] = score
        execution_mode = "fork"
    else:
        state = (real, random_distances, node_to_equivalent, n_random, seed)
        _SCORE_STATE = state
        for batch in batches:
            for job_index, score in _score_worker(batch):
                scores[job_index] = score
    return scores, execution_mode


__all__ = [
    "ALGORITHM_VERSION",
    "CACHE_FORMAT_VERSION",
    "adjacency_from_graph",
    "build_disease_distance_matrix",
    "build_node_to_equivalent_indices",
    "calculate_target_z",
    "get_degree_binning",
    "get_degree_equivalents",
    "multi_source_bfs_distance",
    "pick_random_nodes_matching_selected",
    "score_target_sets",
]
