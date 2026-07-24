"""Common node-completion wrapper used by CLI runners and future executors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from lipid_screening_agent.artifacts import (
    MAX_MESSAGE_LENGTH,
    ArtifactManifest,
    ErrorInfo,
    NodeResult,
    NodeStatus,
    create_artifact_manifest,
)

from .atomic import atomic_write_json
from .context import RunContext
from .errors import ExecutionError, OutputContractError
from .hashing import normalize_json_value
from .logging import NodeLogger, create_node_logger
from .paths import validate_portable_segment
from .time import utc_now

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """One output file registered by a runner before completion is committed."""

    artifact_type: str
    path: Path
    artifact_id: str | None = None
    instance_key: str | None = None


class NodeExecution:
    """Collect runner outputs and commit manifests followed by one terminal result.

    A successful ``NodeResult`` is the commit marker. If manifest creation fails, the wrapper
    writes a failed result; any already-written manifest is an uncommitted side file that a future
    Workflow Engine must ignore.
    """

    def __init__(
        self,
        *,
        context: RunContext,
        node_id: str,
        task_id: str = "main",
        attempt: int = 1,
        config_hash: str,
        code_version: str,
        input_artifact_ids: Sequence[str] = (),
        resource_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.context = context
        self.node_id = validate_portable_segment(node_id, label="node_id")
        self.task_id = validate_portable_segment(task_id, label="task_id")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise OutputContractError("attempt must be a positive integer")
        self.attempt = attempt
        self.config_hash = config_hash
        self.code_version = code_version
        self.input_artifact_ids = tuple(input_artifact_ids)
        self.resource_hashes = dict(resource_hashes or {})
        self._pending: list[PendingArtifact] = []
        self._metrics: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._requested_status = NodeStatus.SUCCEEDED
        self._started_at = None
        self._result: NodeResult | None = None
        self._manifests: tuple[ArtifactManifest, ...] = ()
        self._logger: NodeLogger | None = None

    @property
    def logger(self) -> NodeLogger:
        if self._logger is None:
            raise ExecutionError("node logger is available only while an execution is running")
        return self._logger

    @property
    def result(self) -> NodeResult:
        if self._result is None:
            raise ExecutionError("node execution has not reached a terminal result")
        return self._result

    @property
    def manifests(self) -> tuple[ArtifactManifest, ...]:
        return self._manifests

    def add_output(
        self,
        artifact_type: str,
        path: str | Path,
        *,
        artifact_id: str | None = None,
        instance_key: str | None = None,
    ) -> None:
        """Register an actual output file; optional contract outputs need not be registered."""

        if self._requested_status is NodeStatus.SKIPPED:
            raise OutputContractError("a skipped execution cannot register outputs")
        validate_portable_segment(artifact_type, label="artifact_type")
        if artifact_id is not None:
            validate_portable_segment(artifact_id, label="artifact_id")
        if instance_key is not None:
            validate_portable_segment(instance_key, label="instance_key")
        self._pending.append(
            PendingArtifact(
                artifact_type=artifact_type,
                path=Path(path),
                artifact_id=artifact_id,
                instance_key=instance_key,
            )
        )

    def metric(self, name: str, value: Any) -> None:
        if not isinstance(name, str) or not name:
            raise OutputContractError("metric name must be a non-empty string")
        try:
            normalized = normalize_json_value(value, _location=f"$.metrics.{name}")
        except (TypeError, ValueError) as exc:
            raise OutputContractError(f"metric {name!r} is not JSON-compatible: {exc}") from exc
        self._metrics[name] = normalized

    def update_metrics(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            self.metric(name, value)

    def warn(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise OutputContractError("warning must be a non-empty string")
        if any(ord(character) < 32 for character in message):
            raise OutputContractError("warning cannot contain control characters")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise OutputContractError(f"warning exceeds {MAX_MESSAGE_LENGTH} characters")
        self._warnings.append(message)
        if self._logger is not None:
            self._logger.warning("runner_warning", message)

    def mark_skipped(self, reason: str) -> None:
        if self._pending:
            raise OutputContractError("an execution with registered outputs cannot be skipped")
        self._requested_status = NodeStatus.SKIPPED
        self.warn(reason)

    def mark_blocked(self, reason: str) -> None:
        """Commit a dependency-blocked result without manufacturing a failure."""

        if self._pending:
            raise OutputContractError("an execution with registered outputs cannot be blocked")
        self._requested_status = NodeStatus.BLOCKED
        self.warn(reason)

    def mark_cached(self, reason: str) -> None:
        """Commit verified existing outputs with the terminal ``cached`` status."""

        self._requested_status = NodeStatus.CACHED
        self.warn(reason)

    def mark_cancelled(self, reason: str = "cancel requested") -> None:
        """Persist a contract-correct cancelled NodeResult without output artifacts."""

        if self._pending:
            raise OutputContractError("a cancelled execution cannot register outputs")
        self._requested_status = NodeStatus.CANCELLED
        self.warn(reason)

    def _runtime_path(self, *parts: str) -> Path:
        relative = PurePosixPath(*parts).as_posix()
        return self.context.resolve_run_relative(relative)

    def _node_result_path(self) -> Path:
        return self._runtime_path("artifacts", "node_results", self.node_id, f"{self.task_id}.json")

    def _manifest_path(self, artifact_id: str) -> Path:
        validate_portable_segment(artifact_id, label="artifact_id")
        return self._runtime_path(
            "artifacts",
            "manifests",
            self.node_id,
            self.task_id,
            f"{artifact_id}.json",
        )

    def _open_logger(self) -> NodeLogger:
        jsonl_path = self._runtime_path("logs", self.node_id, f"{self.task_id}.jsonl")
        human_path = self._runtime_path("logs", self.node_id, f"{self.task_id}.log")
        return create_node_logger(
            run_id=self.context.run_id,
            node_id=self.node_id,
            task_id=self.task_id,
            jsonl_path=jsonl_path,
            human_path=human_path,
            allowed_root=self.context.run_dir,
        )

    def _write_result(self, result: NodeResult) -> None:
        atomic_write_json(
            self._node_result_path(),
            result.to_dict(),
            allowed_root=self.context.run_dir,
        )

    def _commit_success(self) -> NodeResult:
        manifests: list[ArtifactManifest] = []
        seen_ids: set[str] = set()
        for output in self._pending:
            manifest = create_artifact_manifest(
                output.path,
                run_root=self.context.run_dir,
                artifact_type=output.artifact_type,
                producer_node_id=self.node_id,
                producer_task_id=self.task_id,
                config_hash=self.config_hash,
                code_version=self.code_version,
                input_artifact_ids=self.input_artifact_ids,
                resource_hashes=self.resource_hashes,
                artifact_id=output.artifact_id,
                instance_key=output.instance_key,
            )
            if manifest.artifact_id in seen_ids:
                raise OutputContractError(f"duplicate artifact instance ID: {manifest.artifact_id}")
            seen_ids.add(manifest.artifact_id)
            manifests.append(manifest)

        for manifest in manifests:
            atomic_write_json(
                self._manifest_path(manifest.artifact_id),
                manifest.to_dict(),
                allowed_root=self.context.run_dir,
            )

        finished_at = utc_now()
        result = NodeResult(
            node_id=self.node_id,
            task_id=self.task_id,
            status=self._requested_status,
            started_at=self._started_at,
            finished_at=finished_at,
            attempt=self.attempt,
            outputs=tuple(manifest.artifact_id for manifest in manifests),
            metrics=self._metrics,
            warnings=self._warnings,
            error=None,
        )
        self._write_result(result)
        self._manifests = tuple(manifests)
        self._result = result
        return result

    def _commit_failure(self, exception: Exception) -> NodeResult:
        result = NodeResult(
            node_id=self.node_id,
            task_id=self.task_id,
            status=NodeStatus.FAILED,
            started_at=self._started_at,
            finished_at=utc_now(),
            attempt=self.attempt,
            outputs=(),
            metrics=self._metrics,
            warnings=self._warnings,
            error=ErrorInfo.from_exception(exception),
        )
        self._write_result(result)
        self._result = result
        return result

    def run(
        self,
        operation: Callable[[NodeExecution], T],
        *,
        raise_on_error: bool = False,
    ) -> NodeResult:
        """Run one callback and always attempt to persist a terminal ``NodeResult``."""

        if self._started_at is not None:
            raise ExecutionError("a NodeExecution instance can only run once", retryable=False)
        self._started_at = utc_now()
        try:
            self._logger = self._open_logger()
            self._logger.info("node_started", "node execution started", attempt=self.attempt)
            operation(self)
            result = self._commit_success()
            self._logger.info(
                "node_finished",
                "node execution reached a terminal status",
                status=result.status.value,
                output_count=len(result.outputs),
            )
            return result
        except Exception as exc:
            if self._logger is not None:
                try:
                    self._logger.exception(
                        "node_failed",
                        "node execution failed",
                        error_type=type(exc).__name__,
                    )
                except Exception:
                    pass
            try:
                result = self._commit_failure(exc)
            except Exception as commit_exc:
                raise ExecutionError(
                    "node failed and its failure result could not be persisted",
                    details={
                        "node_id": self.node_id,
                        "task_id": self.task_id,
                        "original_error": str(exc),
                        "result_error": str(commit_exc),
                    },
                    retryable=False,
                ) from commit_exc
            if raise_on_error:
                raise
            return result
        finally:
            if self._logger is not None:
                try:
                    self._logger.close()
                except Exception:
                    pass
                self._logger = None


def execute_node(
    operation: Callable[[NodeExecution], Any],
    *,
    context: RunContext,
    node_id: str,
    task_id: str = "main",
    attempt: int = 1,
    config_hash: str,
    code_version: str,
    input_artifact_ids: Sequence[str] = (),
    resource_hashes: Mapping[str, str] | None = None,
    raise_on_error: bool = False,
) -> NodeResult:
    """Functional facade for :class:`NodeExecution`, suitable for runner ``main`` functions."""

    execution = NodeExecution(
        context=context,
        node_id=node_id,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
        resource_hashes=resource_hashes,
    )
    return execution.run(operation, raise_on_error=raise_on_error)


__all__ = ["NodeExecution", "PendingArtifact", "execute_node"]
