"""Serializable workflow state models shared by the planner, store, and service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CACHED = "cached"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.CACHED,
        WorkflowStatus.SKIPPED,
        WorkflowStatus.CANCELLED,
    }
)
SUCCESS_STATUSES = frozenset({WorkflowStatus.SUCCEEDED, WorkflowStatus.CACHED})


class InputAvailability(str, Enum):
    MISSING = "missing"
    AVAILABLE = "available"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class InputState:
    compounds: InputAvailability | str
    disease_genes: InputAvailability | str
    expression_pairs: InputAvailability | str = InputAvailability.MISSING
    drug_targets: InputAvailability | str = InputAvailability.MISSING
    target_mapping: InputAvailability | str = InputAvailability.MISSING
    netinfer_batch_ids: tuple[str, ...] | Sequence[str] | None = None
    input_artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    input_scale: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "compounds",
            "disease_genes",
            "expression_pairs",
            "drug_targets",
            "target_mapping",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, InputAvailability(value))
        if self.netinfer_batch_ids is not None:
            items = tuple(str(item) for item in self.netinfer_batch_ids)
            if len(items) != len(set(items)):
                raise ValueError("netinfer_batch_ids must be unique")
            object.__setattr__(self, "netinfer_batch_ids", items)
        object.__setattr__(self, "input_artifact_hashes", dict(self.input_artifact_hashes))
        object.__setattr__(self, "input_scale", dict(self.input_scale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "compounds": self.compounds.value,
            "disease_genes": self.disease_genes.value,
            "expression_pairs": self.expression_pairs.value,
            "drug_targets": self.drug_targets.value,
            "target_mapping": self.target_mapping.value,
            "netinfer_batch_ids": (
                None if self.netinfer_batch_ids is None else list(self.netinfer_batch_ids)
            ),
            "input_artifact_hashes": dict(self.input_artifact_hashes),
            "input_scale": dict(self.input_scale),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputState:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class DependencyRef:
    node_id: str
    task_id: str = "main"

    def to_dict(self) -> dict[str, str]:
        return {"node_id": self.node_id, "task_id": self.task_id}


@dataclass(frozen=True, slots=True)
class PlannedNode:
    node_id: str
    task_id: str = "main"
    stage: str = "workflow"
    dependencies: tuple[DependencyRef, ...] | Sequence[DependencyRef] = ()
    initial_status: WorkflowStatus | str = WorkflowStatus.PENDING
    skip_reason: str | None = None
    resource_class: str = "cpu"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "initial_status", WorkflowStatus(self.initial_status))
        object.__setattr__(self, "parameters", dict(self.parameters))

    @property
    def key(self) -> str:
        return f"{self.node_id}:{self.task_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task_id": self.task_id,
            "stage": self.stage,
            "dependencies": [item.to_dict() for item in self.dependencies],
            "initial_status": self.initial_status.value,
            "skip_reason": self.skip_reason,
            "resource_class": self.resource_class,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class FanoutPlan:
    source_node_id: str
    target_node_id: str
    consumer_node_id: str
    item_key: str
    items: tuple[str, ...] | Sequence[str] | None

    def __post_init__(self) -> None:
        if self.items is not None:
            object.__setattr__(self, "items", tuple(self.items))

    @property
    def resolved(self) -> bool:
        return self.items is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "consumer_node_id": self.consumer_node_id,
            "item_key": self.item_key,
            "items": None if self.items is None else list(self.items),
            "resolved": self.resolved,
        }


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    run_id: str
    workflow_id: str
    workflow_version: str
    mode: str
    evidence_mode: str
    nodes: tuple[PlannedNode, ...] | Sequence[PlannedNode]
    fanouts: tuple[FanoutPlan, ...] | Sequence[FanoutPlan] = ()
    module_readiness: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "fanouts", tuple(self.fanouts))
        object.__setattr__(
            self,
            "module_readiness",
            {name: dict(value) for name, value in self.module_readiness.items()},
        )
        keys = [node.key for node in self.nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("workflow plan contains duplicate node task keys")

    def to_dict(self) -> dict[str, Any]:
        edges = [
            {
                "from": f"{dependency.node_id}:{dependency.task_id}",
                "to": node.key,
            }
            for node in self.nodes
            for dependency in node.dependencies
        ]
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "mode": self.mode,
            "evidence_mode": self.evidence_mode,
            "created_at": isoformat(self.created_at),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": edges,
            "fanouts": [fanout.to_dict() for fanout in self.fanouts],
            "module_readiness": {
                name: dict(value) for name, value in self.module_readiness.items()
            },
            "scheduling": {
                "kg_training_resource_class": "gpu_training",
                "maximum_concurrent_tasks_per_gpu": 1,
            },
        }


@dataclass(frozen=True, slots=True)
class NodeRecord:
    run_id: str
    node_id: str
    task_id: str
    stage: str
    status: WorkflowStatus
    attempt: int
    progress: float | None
    created_at: str
    queued_at: str | None
    started_at: str | None
    finished_at: str | None
    heartbeat_at: str | None
    worker_id: str | None
    queue: str | None
    error: Mapping[str, Any] | None
    artifacts: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    warnings: tuple[str, ...]
    dependencies: tuple[DependencyRef, ...]
    resource_class: str
    parameters: Mapping[str, Any]
    cache_key: str | None
    version: int

    @property
    def key(self) -> str:
        return f"{self.node_id}:{self.task_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task_id": self.task_id,
            "stage": self.stage,
            "status": self.status.value,
            "attempt": self.attempt,
            "progress": self.progress,
            "created_at": self.created_at,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "heartbeat_at": self.heartbeat_at,
            "worker_id": self.worker_id,
            "queue": self.queue,
            "error": None if self.error is None else dict(self.error),
            "artifacts": [dict(value) for value in self.artifacts],
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "dependencies": [value.to_dict() for value in self.dependencies],
            "resource_class": self.resource_class,
            "parameters": dict(self.parameters),
            "cache_key": self.cache_key,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class RunnerOutcome:
    status: WorkflowStatus | str
    artifacts: tuple[Mapping[str, Any], ...] | Sequence[Mapping[str, Any]] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] | Sequence[str] = ()
    fanout_items: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = WorkflowStatus(self.status)
        if status not in TERMINAL_STATUSES:
            raise ValueError("runner outcome must be terminal")
        if status is WorkflowStatus.FAILED and self.error is None:
            raise ValueError("failed runner outcome requires error")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(dict(item) for item in self.artifacts))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "fanout_items",
            {name: tuple(items) for name, items in self.fanout_items.items()},
        )


@dataclass(frozen=True, slots=True)
class ETAEstimate:
    status: str
    lower_seconds: float | None = None
    upper_seconds: float | None = None
    basis: str | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lower_seconds": self.lower_seconds,
            "upper_seconds": self.upper_seconds,
            "basis": self.basis,
            "sample_count": self.sample_count,
        }
