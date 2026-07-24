"""Atomic, UTF-8 file writers for runtime metadata and runner outputs."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .hashing import stable_json_dumps
from .paths import (
    canonical_path,
    ensure_within,
    resolve_run_relative,
    to_run_relative_posix,
)

PathLike = str | os.PathLike[str]


def _target_path(path: PathLike, allowed_root: PathLike | None) -> Path:
    target = Path(path)
    if allowed_root is None:
        if not target.is_absolute():
            raise ValueError("an absolute target or allowed_root is required")
        return target.resolve(strict=False)

    root = canonical_path(allowed_root, must_exist=True, label="allowed root")
    if not target.is_absolute():
        return resolve_run_relative(root, os.fspath(path))
    target = ensure_within(target, root, allow_equal=False)
    to_run_relative_posix(target, root)
    return target


def atomic_write_bytes(
    path: PathLike,
    data: bytes,
    *,
    allowed_root: PathLike | None = None,
) -> Path:
    """Write bytes through a same-directory temporary file and ``os.replace``."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    target = _target_path(path, allowed_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if allowed_root is not None:
        target = ensure_within(target, Path(allowed_root), allow_equal=False)
        to_run_relative_posix(target, Path(allowed_root))

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if allowed_root is not None:
            ensure_within(target, Path(allowed_root), allow_equal=False)
            to_run_relative_posix(target, Path(allowed_root))
        os.replace(temporary_path, target)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    allowed_root: PathLike | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write text without platform-dependent newline conversion."""

    if not isinstance(text, str):
        raise TypeError("text must be str")
    return atomic_write_bytes(path, text.encode(encoding), allowed_root=allowed_root)


def atomic_write_json(
    path: PathLike,
    value: Any,
    *,
    allowed_root: PathLike | None = None,
) -> Path:
    """Atomically write stable UTF-8 JSON with one terminating newline."""

    return atomic_write_text(
        path,
        stable_json_dumps(value, indent=2) + "\n",
        allowed_root=allowed_root,
    )


def atomic_write_yaml(
    path: PathLike,
    value: Mapping[str, Any],
    *,
    allowed_root: PathLike | None = None,
) -> Path:
    """Atomically write deterministic safe YAML with Unicode preserved."""

    if not isinstance(value, Mapping):
        raise TypeError("YAML document must be a mapping")
    rendered = yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    return atomic_write_text(path, rendered, allowed_root=allowed_root)
