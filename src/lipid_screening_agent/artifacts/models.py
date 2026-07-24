"""Serializable runtime records for artifacts and node completion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, NoReturn

from lipid_screening_agent.runtime.errors import OutputContractError
from lipid_screening_agent.runtime.hashing import (
    JsonValue,
    hash_json,
    normalize_json_value,
    stable_json_dumps,
)
from lipid_screening_agent.runtime.paths import parse_run_relative_path

PORTABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_PORTABLE_ID_LENGTH = 180
MAX_MESSAGE_LENGTH = 4096
MAX_EXCEPTION_TYPE_LENGTH = 512
MAX_DETAILS_JSON_BYTES = 16 * 1024
MAX_FALLBACK_DETAILS_LENGTH = 4096

_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _fail(message: str) -> NoReturn:
    raise OutputContractError(message)


def _validate_portable_id(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field_name} must be a non-empty string")
    if len(value) > MAX_PORTABLE_ID_LENGTH:
        _fail(f"{field_name} exceeds {MAX_PORTABLE_ID_LENGTH} characters")
    if PORTABLE_ID_PATTERN.fullmatch(value) is None:
        _fail(f"{field_name} is not a portable identifier: {value!r}")
    if value.endswith("."):
        _fail(f"{field_name} cannot end with a period")
    if value.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        _fail(f"{field_name} uses a Windows-reserved filename: {value!r}")
    return value


def _validate_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{field_name} must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_nonempty_text(value: Any, *, field_name: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field_name} must be a non-empty string")
    if len(value) > maximum:
        _fail(f"{field_name} exceeds {maximum} characters")
    if "\x00" in value:
        _fail(f"{field_name} contains a NUL character")
    return value


def _bounded_text(value: str, *, maximum: int) -> str:
    """Return non-empty text no longer than *maximum*, marking truncation."""

    if len(value) <= maximum:
        return value
    if maximum == 1:
        return "~"
    return f"{value[: maximum - 1]}~"


def _safe_render(value: Any, *, maximum: int) -> str:
    """Render diagnostic-only values without letting a broken ``repr`` mask an error."""

    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<unprintable {type(value).__name__}>"
    rendered = re.sub(r"[\x00-\x1f\x7f]+", " ", rendered).strip()
    return _bounded_text(rendered or "<empty>", maximum=maximum)


def _safe_exception_attribute(exception: BaseException, name: str, default: Any) -> Any:
    try:
        return getattr(exception, name)
    except Exception:
        return default


def _portable_error_code(value: Any, *, fallback: str) -> str:
    try:
        rendered = str(value)
    except Exception:
        rendered = fallback
    code = re.sub(r"[^A-Za-z0-9._-]+", "_", rendered).strip("._-") or "execution_error"
    if code.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        code = f"error_{code}"
    return code[:MAX_PORTABLE_ID_LENGTH].rstrip(".") or "execution_error"


def _bounded_exception_details(value: Any) -> Mapping[str, Any]:
    """Normalize exception details, falling back to a small serializable diagnostic."""

    if isinstance(value, Mapping):
        try:
            normalized = normalize_json_value(value, _location="$.error.details")
            if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
                raise TypeError("exception details did not normalize to an object")
            encoded_size = len(stable_json_dumps(normalized).encode("utf-8"))
            if encoded_size <= MAX_DETAILS_JSON_BYTES:
                return normalized
        except Exception:
            pass
    return {
        "reported_details": _safe_render(value, maximum=MAX_FALLBACK_DETAILS_LENGTH),
        "truncated_or_invalid": True,
    }


def _normalize_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{field_name} must be an ISO 8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise OutputContractError(f"{field_name} is not valid ISO 8601: {value!r}") from exc
    return _normalize_timestamp(parsed, field_name=field_name)


def _freeze_json(value: JsonValue) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalize_json_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field_name} must be a mapping")
    try:
        normalized = normalize_json_value(value, _location=f"$.{field_name}")
    except (TypeError, ValueError) as exc:
        raise OutputContractError(f"{field_name} is not JSON-compatible: {exc}") from exc
    if not isinstance(normalized, dict):  # pragma: no cover - guarded by Mapping above
        _fail(f"{field_name} must normalize to an object")
    return _freeze_json(normalized)


def _require_fields(data: Mapping[str, Any], expected: set[str], *, model: str) -> None:
    if not isinstance(data, Mapping):
        _fail(f"{model} must be a mapping")
    missing = expected - set(data)
    extra = set(data) - expected
    if missing:
        _fail(f"{model} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        _fail(f"{model} contains unknown fields: {', '.join(sorted(extra))}")


def make_artifact_id(
    producer_node_id: str,
    producer_task_id: str,
    artifact_type: str,
    *,
    instance_key: str | None = None,
) -> str:
    """Build a deterministic, portable artifact instance identifier.

    The logical name remains in ``artifact_type``.  The instance ID uses a
    fixed-width 128-bit digest so the deeply nested manifest path remains safe
    on Windows hosts that still enforce legacy path-length limits.
    """

    node_id = _validate_portable_id(producer_node_id, field_name="producer_node_id")
    task_id = _validate_portable_id(producer_task_id, field_name="producer_task_id")
    logical_type = _validate_portable_id(artifact_type, field_name="artifact_type")
    if instance_key is not None:
        instance_key = _validate_portable_id(instance_key, field_name="instance_key")
    identity = {
        "artifact_type": logical_type,
        "instance_key": instance_key,
        "producer_node_id": node_id,
        "producer_task_id": task_id,
    }
    artifact_id = f"a-{hash_json(identity)[:32]}"
    return _validate_portable_id(artifact_id, field_name="artifact_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactManifest:
    """Provenance record for one concrete file in a run workspace."""

    artifact_id: str
    artifact_type: str
    relative_path: str
    size_bytes: int
    sha256: str
    created_at: datetime
    producer_node_id: str
    producer_task_id: str
    input_artifact_ids: tuple[str, ...] | Sequence[str]
    config_hash: str
    code_version: str
    resource_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _validate_portable_id(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _validate_portable_id(self.artifact_type, field_name="artifact_type"),
        )

        try:
            relative_path = parse_run_relative_path(self.relative_path).as_posix()
        except OutputContractError:
            raise
        except Exception as exc:
            raise OutputContractError(
                f"invalid artifact relative_path: {self.relative_path!r}"
            ) from exc
        object.__setattr__(self, "relative_path", relative_path)

        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            _fail("size_bytes must be an integer")
        if self.size_bytes < 0:
            _fail("size_bytes cannot be negative")
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256, field_name="sha256"))
        object.__setattr__(
            self,
            "created_at",
            _normalize_timestamp(self.created_at, field_name="created_at"),
        )
        object.__setattr__(
            self,
            "producer_node_id",
            _validate_portable_id(self.producer_node_id, field_name="producer_node_id"),
        )
        object.__setattr__(
            self,
            "producer_task_id",
            _validate_portable_id(self.producer_task_id, field_name="producer_task_id"),
        )

        if not isinstance(self.input_artifact_ids, Sequence) or isinstance(
            self.input_artifact_ids, (str, bytes)
        ):
            _fail("input_artifact_ids must be a sequence of artifact instance IDs")
        input_ids = tuple(
            _validate_portable_id(value, field_name="input_artifact_ids item")
            for value in self.input_artifact_ids
        )
        if len(input_ids) != len(set(input_ids)):
            _fail("input_artifact_ids cannot contain duplicates")
        object.__setattr__(self, "input_artifact_ids", input_ids)

        object.__setattr__(
            self,
            "config_hash",
            _validate_sha256(self.config_hash, field_name="config_hash"),
        )
        object.__setattr__(
            self,
            "code_version",
            _validate_nonempty_text(self.code_version, field_name="code_version"),
        )

        if not isinstance(self.resource_hashes, Mapping):
            _fail("resource_hashes must be a mapping")
        resource_hashes: dict[str, str] = {}
        for resource_id, digest in self.resource_hashes.items():
            resource_key = _validate_nonempty_text(
                resource_id, field_name="resource_hashes key", maximum=180
            )
            resource_hashes[resource_key] = _validate_sha256(
                digest, field_name=f"resource_hashes[{resource_key!r}]"
            )
        object.__setattr__(
            self,
            "resource_hashes",
            MappingProxyType(dict(sorted(resource_hashes.items()))),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": _format_timestamp(self.created_at),
            "producer_node_id": self.producer_node_id,
            "producer_task_id": self.producer_task_id,
            "input_artifact_ids": list(self.input_artifact_ids),
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "resource_hashes": dict(self.resource_hashes),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return stable_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactManifest:
        expected = {
            "artifact_id",
            "artifact_type",
            "relative_path",
            "size_bytes",
            "sha256",
            "created_at",
            "producer_node_id",
            "producer_task_id",
            "input_artifact_ids",
            "config_hash",
            "code_version",
            "resource_hashes",
        }
        _require_fields(data, expected, model="ArtifactManifest")
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            relative_path=data["relative_path"],
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            created_at=_parse_timestamp(data["created_at"], field_name="created_at"),
            producer_node_id=data["producer_node_id"],
            producer_task_id=data["producer_task_id"],
            input_artifact_ids=data["input_artifact_ids"],
            config_hash=data["config_hash"],
            code_version=data["code_version"],
            resource_hashes=data["resource_hashes"],
        )


class ErrorCategory(str, Enum):
    CONFIGURATION = "configuration"
    INPUT = "input"
    RESOURCE = "resource"
    ENVIRONMENT = "environment"
    EXECUTION = "execution"
    OUTPUT_CONTRACT = "output_contract"


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorInfo:
    """Stable, non-traceback error payload stored in a :class:`NodeResult`."""

    category: ErrorCategory | str
    code: str
    message: str
    exception_type: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            category = (
                self.category
                if isinstance(self.category, ErrorCategory)
                else ErrorCategory(self.category)
            )
        except (TypeError, ValueError) as exc:
            raise OutputContractError(f"unsupported error category: {self.category!r}") from exc
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "code", _validate_portable_id(self.code, field_name="error.code"))
        object.__setattr__(
            self,
            "message",
            _validate_nonempty_text(
                self.message,
                field_name="error.message",
                maximum=MAX_MESSAGE_LENGTH,
            ),
        )
        object.__setattr__(
            self,
            "exception_type",
            _validate_nonempty_text(
                self.exception_type,
                field_name="error.exception_type",
                maximum=MAX_EXCEPTION_TYPE_LENGTH,
            ),
        )
        if not isinstance(self.retryable, bool):
            _fail("error.retryable must be boolean")
        object.__setattr__(
            self,
            "details",
            _normalize_json_mapping(self.details, field_name="error.details"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "exception_type": self.exception_type,
            "retryable": self.retryable,
            "details": normalize_json_value(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ErrorInfo:
        expected = {
            "category",
            "code",
            "message",
            "exception_type",
            "retryable",
            "details",
        }
        _require_fields(data, expected, model="ErrorInfo")
        return cls(**{key: data[key] for key in expected})

    @classmethod
    def from_exception(cls, exception: BaseException) -> ErrorInfo:
        exception_class = type(exception)
        fallback_name = getattr(exception_class, "__name__", "Exception")
        category_value = _safe_exception_attribute(
            exception, "category", ErrorCategory.EXECUTION.value
        )
        if isinstance(category_value, Enum):
            try:
                category_value = category_value.value
            except Exception:
                category_value = ErrorCategory.EXECUTION.value
        if isinstance(category_value, str) and category_value == "path_safety":
            category_value = ErrorCategory.OUTPUT_CONTRACT.value
        if not isinstance(category_value, str):
            category = ErrorCategory.EXECUTION
        else:
            try:
                category = ErrorCategory(category_value)
            except ValueError:
                category = ErrorCategory.EXECUTION

        raw_code = _safe_exception_attribute(exception, "code", None)
        if raw_code is None:
            raw_code = fallback_name
        code = _portable_error_code(raw_code, fallback=fallback_name)
        details = _bounded_exception_details(_safe_exception_attribute(exception, "details", {}))
        try:
            message = str(exception)
        except Exception:
            message = fallback_name
        message = re.sub(r"[\x00-\x1f\x7f]+", " ", message).strip() or fallback_name
        message = _bounded_text(message, maximum=MAX_MESSAGE_LENGTH)

        module = getattr(exception_class, "__module__", "builtins")
        qualname = getattr(exception_class, "__qualname__", fallback_name)
        exception_type = re.sub(r"[\x00-\x1f\x7f]+", " ", f"{module}.{qualname}").strip()
        exception_type = _bounded_text(
            exception_type or fallback_name,
            maximum=MAX_EXCEPTION_TYPE_LENGTH,
        )
        retryable_value = _safe_exception_attribute(exception, "retryable", False)
        try:
            retryable = bool(retryable_value)
        except Exception:
            retryable = False
        return cls(
            category=category,
            code=code,
            message=message,
            exception_type=exception_type,
            retryable=retryable,
            details=details,
        )


class NodeStatus(str, Enum):
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


TERMINAL_NODE_STATUSES = frozenset(
    {
        NodeStatus.SUCCEEDED,
        NodeStatus.FAILED,
        NodeStatus.BLOCKED,
        NodeStatus.CACHED,
        NodeStatus.SKIPPED,
        NodeStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeResult:
    """Terminal result of one node/task execution instance."""

    node_id: str
    task_id: str
    status: NodeStatus | str
    started_at: datetime
    finished_at: datetime
    attempt: int
    outputs: tuple[str, ...] | Sequence[str] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] | Sequence[str] = ()
    error: ErrorInfo | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", _validate_portable_id(self.node_id, field_name="node_id")
        )
        object.__setattr__(
            self, "task_id", _validate_portable_id(self.task_id, field_name="task_id")
        )
        try:
            status = self.status if isinstance(self.status, NodeStatus) else NodeStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise OutputContractError(f"unsupported node status: {self.status!r}") from exc
        if status not in TERMINAL_NODE_STATUSES:
            _fail(f"NodeResult requires a terminal status, got {status.value!r}")
        object.__setattr__(self, "status", status)

        started_at = _normalize_timestamp(self.started_at, field_name="started_at")
        finished_at = _normalize_timestamp(self.finished_at, field_name="finished_at")
        if finished_at < started_at:
            _fail("finished_at cannot be earlier than started_at")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "finished_at", finished_at)

        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            _fail("attempt must be a positive integer")

        if not isinstance(self.outputs, Sequence) or isinstance(self.outputs, (str, bytes)):
            _fail("outputs must be a sequence of artifact instance IDs")
        outputs = tuple(
            _validate_portable_id(output, field_name="outputs item") for output in self.outputs
        )
        if len(outputs) != len(set(outputs)):
            _fail("outputs cannot contain duplicate artifact IDs")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(
            self, "metrics", _normalize_json_mapping(self.metrics, field_name="metrics")
        )

        if not isinstance(self.warnings, Sequence) or isinstance(self.warnings, (str, bytes)):
            _fail("warnings must be a sequence of strings")
        warnings = tuple(
            _validate_nonempty_text(
                warning,
                field_name="warnings item",
                maximum=MAX_MESSAGE_LENGTH,
            )
            for warning in self.warnings
        )
        object.__setattr__(self, "warnings", warnings)

        error = self.error
        if isinstance(error, Mapping):
            error = ErrorInfo.from_dict(error)
        elif error is not None and not isinstance(error, ErrorInfo):
            _fail("error must be ErrorInfo, a compatible mapping, or null")
        if status is NodeStatus.FAILED and error is None:
            _fail("failed NodeResult requires structured error information")
        non_error_statuses = {
            NodeStatus.SUCCEEDED,
            NodeStatus.CACHED,
            NodeStatus.SKIPPED,
        }
        if status in non_error_statuses and error is not None:
            _fail(f"{status.value} NodeResult cannot contain an error")
        object.__setattr__(self, "error", error)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "node_id": self.node_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at": _format_timestamp(self.started_at),
            "finished_at": _format_timestamp(self.finished_at),
            "attempt": self.attempt,
            "outputs": list(self.outputs),
            "metrics": normalize_json_value(self.metrics),
            "warnings": list(self.warnings),
            "error": None if self.error is None else self.error.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return stable_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NodeResult:
        expected = {
            "node_id",
            "task_id",
            "status",
            "started_at",
            "finished_at",
            "attempt",
            "outputs",
            "metrics",
            "warnings",
            "error",
        }
        _require_fields(data, expected, model="NodeResult")
        return cls(
            node_id=data["node_id"],
            task_id=data["task_id"],
            status=data["status"],
            started_at=_parse_timestamp(data["started_at"], field_name="started_at"),
            finished_at=_parse_timestamp(data["finished_at"], field_name="finished_at"),
            attempt=data["attempt"],
            outputs=data["outputs"],
            metrics=data["metrics"],
            warnings=data["warnings"],
            error=data["error"],
        )


__all__ = [
    "ArtifactManifest",
    "ErrorCategory",
    "ErrorInfo",
    "MAX_DETAILS_JSON_BYTES",
    "MAX_EXCEPTION_TYPE_LENGTH",
    "MAX_FALLBACK_DETAILS_LENGTH",
    "MAX_MESSAGE_LENGTH",
    "MAX_PORTABLE_ID_LENGTH",
    "NodeResult",
    "NodeStatus",
    "PORTABLE_ID_PATTERN",
    "SHA256_PATTERN",
    "TERMINAL_NODE_STATUSES",
    "make_artifact_id",
]
