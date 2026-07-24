"""History-backed interval ETA estimates; never manufactures point precision."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from statistics import median

from .models import SUCCESS_STATUSES, ETAEstimate, NodeRecord
from .store import WorkflowStore


class ETAEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        *,
        nodes: Sequence[NodeRecord],
        hardware_fingerprint: str | None,
        input_scale: Mapping[str, float],
    ) -> ETAEstimate:
        pass


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scaled_duration(
    history: Mapping[str, object], current_scale: Mapping[str, float]
) -> tuple[float, bool]:
    """Scale a historical duration by comparable input dimensions, conservatively clamped."""

    historical_scale = history.get("input_scale", {})
    if not isinstance(historical_scale, Mapping):
        historical_scale = {}
    ratios: list[float] = []
    for key, current in current_scale.items():
        previous = historical_scale.get(key)
        if (
            isinstance(current, (int, float))
            and isinstance(previous, (int, float))
            and current > 0
            and previous > 0
        ):
            ratios.append(float(current) / float(previous))
    # Runner metrics are a secondary source when older history lacks an explicit scale vector.
    metrics = history.get("metrics", {})
    if not ratios and isinstance(metrics, Mapping):
        previous_count = metrics.get("input_count")
        current_count = current_scale.get("compound_count")
        if (
            isinstance(previous_count, (int, float))
            and isinstance(current_count, (int, float))
            and previous_count > 0
            and current_count > 0
        ):
            ratios.append(float(current_count) / float(previous_count))
    scale = 1.0
    if ratios:
        # A geometric mean avoids allowing one high-dimensional matrix measure to dominate.
        scale = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        scale = min(4.0, max(0.25, scale))
    return float(history["duration_seconds"]) * scale, bool(ratios)


class HistoricalETAEstimator(ETAEstimator):
    def __init__(
        self,
        store: WorkflowStore,
        *,
        conservative_reference_ranges: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.store = store
        self.reference_ranges = dict(conservative_reference_ranges or {})

    def estimate(
        self,
        *,
        nodes: Sequence[NodeRecord],
        hardware_fingerprint: str | None,
        input_scale: Mapping[str, float],
    ) -> ETAEstimate:
        remaining = [
            node
            for node in nodes
            if node.status not in SUCCESS_STATUSES
            and node.status.value not in {"skipped", "cancelled"}
        ]
        if not remaining:
            return ETAEstimate(
                status="estimated", lower_seconds=0.0, upper_seconds=0.0, basis="complete"
            )
        lower_total = 0.0
        upper_total = 0.0
        samples = 0
        bases: set[str] = set()
        for node in remaining:
            history = self.store.history(node.node_id, hardware_fingerprint)
            if not history and hardware_fingerprint is not None:
                history = self.store.history(node.node_id, None)
                if history:
                    bases.add("cross_hardware_history")
            elif history:
                bases.add("matching_hardware_history")
            scaled = [_scaled_duration(item, input_scale) for item in history]
            durations = [value for value, _used_scale in scaled]
            if any(used_scale for _value, used_scale in scaled):
                bases.add("input_scale_adjusted")
            if durations:
                samples += len(durations)
                if len(durations) >= 4:
                    lower = _percentile(durations, 0.20)
                    upper = _percentile(durations, 0.80)
                else:
                    center = median(durations)
                    lower, upper = center * 0.7, center * 1.5
                lower_total += max(0.0, lower)
                upper_total += max(lower, upper)
                continue
            reference = self.reference_ranges.get(node.node_id)
            if reference is None:
                return ETAEstimate(
                    status="unknown",
                    basis=f"no history for {node.node_id}",
                    sample_count=samples,
                )
            bases.add("conservative_reference")
            lower_total += reference[0]
            upper_total += reference[1]
        return ETAEstimate(
            status="estimated",
            lower_seconds=round(lower_total, 1),
            upper_seconds=round(upper_total, 1),
            basis="+".join(sorted(bases)) or "history",
            sample_count=samples,
        )


class UnknownETAEstimator(ETAEstimator):
    def estimate(
        self,
        *,
        nodes: Sequence[NodeRecord],
        hardware_fingerprint: str | None,
        input_scale: Mapping[str, float],
    ) -> ETAEstimate:
        return ETAEstimate(status="unknown", basis="no estimator history")


__all__ = ["ETAEstimator", "HistoricalETAEstimator", "UnknownETAEstimator"]
