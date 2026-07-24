"""Per-node structured JSONL and human-readable logging."""

from __future__ import annotations

import json
import logging as stdlib_logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .paths import ensure_within
from .time import isoformat_utc

PathLike = str | os.PathLike[str]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("log fields cannot contain NaN or Infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return isoformat_utc(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("log field mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported structured log value: {type(value).__name__}")


def _record_timestamp(record: stdlib_logging.LogRecord) -> str:
    return isoformat_utc(datetime.fromtimestamp(record.created, timezone.utc))


class JsonLineFormatter(stdlib_logging.Formatter):
    """Render each record as one self-contained JSON object."""

    def format(self, record: stdlib_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _record_timestamp(record),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "node_id": getattr(record, "node_id", None),
            "task_id": getattr(record, "task_id", None),
            "event": getattr(record, "event", "message"),
            "fields": getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class HumanFormatter(stdlib_logging.Formatter):
    """Render compact records intended for people inspecting a run directory."""

    def format(self, record: stdlib_logging.LogRecord) -> str:
        context = "/".join(
            str(value)
            for value in (
                getattr(record, "run_id", None),
                getattr(record, "node_id", None),
                getattr(record, "task_id", None),
            )
            if value
        )
        event = getattr(record, "event", "message")
        fields = getattr(record, "fields", {})
        suffix = ""
        if fields:
            suffix = " " + json.dumps(
                _json_value(fields),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        rendered = (
            f"{_record_timestamp(record)} {record.levelname:<8} "
            f"[{context}] {event}: {record.getMessage()}{suffix}"
        )
        if record.exc_info:
            rendered += "\n" + self.formatException(record.exc_info)
        return rendered


@dataclass
class NodeLogger:
    """Small runner-facing facade over a pair of standard-library log handlers."""

    _logger: stdlib_logging.Logger
    run_id: str
    node_id: str
    task_id: str
    jsonl_path: Path
    human_path: Path

    def event(
        self,
        level: int,
        event: str,
        message: str,
        **fields: Any,
    ) -> None:
        self._logger.log(
            level,
            message,
            extra={
                "run_id": self.run_id,
                "node_id": self.node_id,
                "task_id": self.task_id,
                "event": event,
                "fields": _json_value(fields),
            },
        )

    def info(self, event: str, message: str, **fields: Any) -> None:
        self.event(stdlib_logging.INFO, event, message, **fields)

    def warning(self, event: str, message: str, **fields: Any) -> None:
        self.event(stdlib_logging.WARNING, event, message, **fields)

    def error(self, event: str, message: str, **fields: Any) -> None:
        self.event(stdlib_logging.ERROR, event, message, **fields)

    def exception(self, event: str, message: str, **fields: Any) -> None:
        self._logger.exception(
            message,
            extra={
                "run_id": self.run_id,
                "node_id": self.node_id,
                "task_id": self.task_id,
                "event": event,
                "fields": _json_value(fields),
            },
        )

    def close(self) -> None:
        for handler in tuple(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


def create_node_logger(
    *,
    run_id: str,
    node_id: str,
    task_id: str,
    jsonl_path: PathLike,
    human_path: PathLike,
    allowed_root: PathLike,
    level: int = stdlib_logging.INFO,
) -> NodeLogger:
    """Create isolated JSONL and text log handlers without handler accumulation."""

    root = Path(allowed_root).resolve(strict=False)
    json_path = ensure_within(Path(jsonl_path), root, allow_equal=False)
    text_path = ensure_within(Path(human_path), root, allow_equal=False)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = ensure_within(json_path, root, allow_equal=False)
    text_path = ensure_within(text_path, root, allow_equal=False)

    logger = stdlib_logging.getLogger(
        f"lipid_screening_agent.node.{node_id}.{task_id}.{uuid4().hex}"
    )
    logger.setLevel(level)
    logger.propagate = False

    json_handler = stdlib_logging.FileHandler(json_path, encoding="utf-8")
    json_handler.setFormatter(JsonLineFormatter())
    text_handler = stdlib_logging.FileHandler(text_path, encoding="utf-8")
    text_handler.setFormatter(HumanFormatter())
    logger.addHandler(json_handler)
    logger.addHandler(text_handler)

    return NodeLogger(logger, run_id, node_id, task_id, json_path, text_path)
