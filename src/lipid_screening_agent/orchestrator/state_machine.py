"""The single authority for workflow state transitions."""

from __future__ import annotations

from .models import TERMINAL_STATUSES, WorkflowStatus


class InvalidStateTransition(ValueError):
    """Raised when a caller tries to bypass the deterministic state machine."""


_ALLOWED: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.PENDING: frozenset(
        {
            WorkflowStatus.READY,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.SKIPPED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.READY: frozenset(
        {
            WorkflowStatus.QUEUED,
            WorkflowStatus.CACHED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.SKIPPED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.QUEUED: frozenset(
        {WorkflowStatus.RUNNING, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.RUNNING: frozenset(
        {
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CACHED,
            WorkflowStatus.SKIPPED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.FAILED: frozenset({WorkflowStatus.READY, WorkflowStatus.CANCELLED}),
    WorkflowStatus.BLOCKED: frozenset(
        {WorkflowStatus.PENDING, WorkflowStatus.READY, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.SUCCEEDED: frozenset(),
    WorkflowStatus.CACHED: frozenset(),
    WorkflowStatus.SKIPPED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
}


def validate_transition(
    current: WorkflowStatus | str,
    target: WorkflowStatus | str,
    *,
    invalidation: bool = False,
) -> tuple[WorkflowStatus, WorkflowStatus]:
    source = WorkflowStatus(current)
    destination = WorkflowStatus(target)
    if source == destination:
        raise InvalidStateTransition(f"state is already {source.value}")
    if invalidation:
        if destination is not WorkflowStatus.PENDING or source not in TERMINAL_STATUSES:
            raise InvalidStateTransition(
                f"invalidation transition {source.value} -> {destination.value} is not allowed"
            )
        return source, destination
    if destination not in _ALLOWED[source]:
        raise InvalidStateTransition(
            f"illegal workflow transition: {source.value} -> {destination.value}"
        )
    return source, destination


def is_terminal(status: WorkflowStatus | str) -> bool:
    return WorkflowStatus(status) in TERMINAL_STATUSES


_RUN_ALLOWED: dict[WorkflowStatus, frozenset[WorkflowStatus]] = {
    WorkflowStatus.PENDING: frozenset(
        {WorkflowStatus.READY, WorkflowStatus.BLOCKED, WorkflowStatus.CANCELLED}
    ),
    WorkflowStatus.READY: frozenset(
        {
            WorkflowStatus.QUEUED,
            WorkflowStatus.RUNNING,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.QUEUED: frozenset(
        {
            WorkflowStatus.READY,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.RUNNING: frozenset(
        {
            WorkflowStatus.READY,
            WorkflowStatus.QUEUED,
            WorkflowStatus.SUCCEEDED,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.CANCELLED,
        }
    ),
    WorkflowStatus.FAILED: frozenset({WorkflowStatus.READY, WorkflowStatus.CANCELLED}),
    WorkflowStatus.BLOCKED: frozenset({WorkflowStatus.READY, WorkflowStatus.CANCELLED}),
    WorkflowStatus.SUCCEEDED: frozenset(),
    WorkflowStatus.CACHED: frozenset(),
    WorkflowStatus.SKIPPED: frozenset(),
    WorkflowStatus.CANCELLED: frozenset(),
}


def validate_run_transition(
    current: WorkflowStatus | str, target: WorkflowStatus | str
) -> tuple[WorkflowStatus, WorkflowStatus]:
    source = WorkflowStatus(current)
    destination = WorkflowStatus(target)
    if source == destination:
        return source, destination
    if destination not in _RUN_ALLOWED[source]:
        raise InvalidStateTransition(
            f"illegal run transition: {source.value} -> {destination.value}"
        )
    return source, destination


__all__ = [
    "InvalidStateTransition",
    "is_terminal",
    "validate_run_transition",
    "validate_transition",
]
