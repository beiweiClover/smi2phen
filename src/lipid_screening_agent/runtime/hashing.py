"""Deterministic JSON and SHA-256 helpers for runtime provenance.

This module intentionally depends only on the Python standard library.  In
particular, hashing a configuration means hashing its parsed JSON-compatible
value, not its YAML spelling, comments, or key order.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileDigest:
    """A file's byte length and lowercase SHA-256 digest."""

    size_bytes: int
    sha256: str


def normalize_json_value(value: Any, *, _location: str = "$") -> JsonValue:
    """Return a detached, strictly JSON-compatible representation of *value*.

    Mapping keys must be strings, floating-point values must be finite, and
    arbitrary Python objects are rejected instead of being silently converted
    with ``repr`` or ``str``.  Tuples are accepted and normalized to JSON
    arrays because runtime models expose immutable tuples in Python.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {_location} is not valid JSON")
        return value

    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"mapping key at {_location} must be a string, got {type(key).__name__}"
                )
            normalized[key] = normalize_json_value(item, _location=f"{_location}.{key}")
        return normalized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            normalize_json_value(item, _location=f"{_location}[{index}]")
            for index, item in enumerate(value)
        ]

    raise TypeError(f"value at {_location} is not JSON-compatible: {type(value).__name__}")


def stable_json_dumps(value: Any, *, indent: int | None = None) -> str:
    """Serialize *value* deterministically without accepting NaN or infinity."""

    normalized = normalize_json_value(value)
    separators = (",", ":") if indent is None else (",", ": ")
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=separators,
        sort_keys=True,
    )


def stable_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes suitable for content hashing."""

    return stable_json_dumps(value).encode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 digest for an in-memory byte sequence."""

    return hashlib.sha256(data).hexdigest()


def hash_json(value: Any) -> str:
    """Hash a JSON-compatible value using its canonical serialization."""

    return sha256_bytes(stable_json_bytes(value))


def hash_config(config: Any) -> str:
    """Return the canonical SHA-256 of a mapping or typed configuration model."""

    serializer = getattr(config, "to_dict", None)
    if callable(serializer):
        config = serializer()
    return hash_json(config)


def file_digest(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> FileDigest:
    """Read a regular file once and return its size and SHA-256.

    A before/after stat check rejects a file that changes during hashing so a
    manifest cannot accidentally combine metadata from different versions.
    """

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    file_path = Path(path)
    before = file_path.stat()
    if not file_path.is_file():
        raise IsADirectoryError(f"not a regular file: {file_path}")

    digest = hashlib.sha256()
    size_bytes = 0
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size_bytes += len(chunk)

    after = file_path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size_bytes != after.st_size
    ):
        raise RuntimeError(f"file changed while it was being hashed: {file_path}")

    return FileDigest(size_bytes=size_bytes, sha256=digest.hexdigest())


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> str:
    """Return only the SHA-256 component of :func:`file_digest`."""

    return file_digest(path, chunk_size=chunk_size).sha256


__all__ = [
    "DEFAULT_HASH_CHUNK_SIZE",
    "FileDigest",
    "JsonScalar",
    "JsonValue",
    "file_digest",
    "hash_config",
    "hash_json",
    "normalize_json_value",
    "sha256_bytes",
    "sha256_file",
    "stable_json_bytes",
    "stable_json_dumps",
]
