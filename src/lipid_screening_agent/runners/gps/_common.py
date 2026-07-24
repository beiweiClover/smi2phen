"""Shared path, dependency-version, and atomic-output helpers for GPS runners."""

from __future__ import annotations

import gzip
import io
import os
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime import (
    InputError,
    OutputContractError,
    PathSafetyError,
    ResourceError,
    RunContext,
    ensure_within,
    resolve_run_relative,
)

GPS_OUTPUT_ROOT = "artifacts/gps"


def package_version(distribution: str) -> str:
    """Return an installed distribution version without importing the package."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def base_environment_metrics() -> dict[str, str]:
    """Return lightweight version fields shared by all three GPS nodes."""

    return {
        "device_actual": "unresolved",
        "torch_version": package_version("torch"),
        "rdkit_version": package_version("rdkit"),
        "numpy_version": package_version("numpy"),
        "pandas_version": package_version("pandas"),
    }


def resolve_gps_output(context: RunContext, filename: str) -> Path:
    """Resolve one fixed GPS artifact and enforce the contracted output boundary."""

    expected_root = context.resolve_run_relative(GPS_OUTPUT_ROOT)
    if context.output_dir != expected_root:
        raise OutputContractError(
            "GPS output_dir must be the contracted artifacts/gps directory",
            details={
                "configured_output_dir": str(context.output_dir),
                "expected_output_dir": str(expected_root),
            },
        )
    try:
        return ensure_within(expected_root / filename, expected_root)
    except PathSafetyError as exc:
        raise OutputContractError(f"unsafe GPS output filename: {filename!r}") from exc


def resolve_run_input_file(
    context: RunContext,
    value: str | Path,
    *,
    label: str,
    require_input_boundary: bool = False,
) -> Path:
    """Resolve an explicit regular input file inside the run workspace."""

    candidate = Path(value)
    try:
        if candidate.is_absolute():
            resolved = ensure_within(candidate, context.run_dir)
        else:
            resolved = context.resolve_run_relative(candidate.as_posix(), must_exist=True)
        if require_input_boundary:
            resolved = ensure_within(resolved, context.input_dir)
    except (OSError, PathSafetyError) as exc:
        boundary = "runner input boundary" if require_input_boundary else "run workspace"
        raise InputError(
            f"{label} is missing or outside the {boundary}",
            details={"path": str(value)},
        ) from exc
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise InputError(
            f"{label} must be an existing regular non-symlink file",
            details={"path": str(resolved)},
        )
    return resolved


def resolve_resource_path(
    context: RunContext,
    value: str | Path,
    *,
    label: str,
    kind: str = "file",
) -> Path:
    """Resolve an explicit path below the read-only GPS resource directory."""

    if context.resource_dir is None:
        raise ResourceError("a GPS resource directory is required")
    candidate = Path(value)
    try:
        if candidate.is_absolute():
            resolved = ensure_within(candidate, context.resource_dir)
        else:
            resolved = resolve_run_relative(
                context.resource_dir,
                candidate.as_posix(),
                must_exist=True,
            )
    except (OSError, PathSafetyError) as exc:
        raise ResourceError(
            f"{label} is missing or outside the GPS resource directory",
            details={"path": str(value), "resource_dir": str(context.resource_dir)},
        ) from exc

    valid = resolved.is_file() if kind == "file" else resolved.is_dir()
    if not valid or resolved.is_symlink():
        raise ResourceError(
            f"{label} must be an existing non-symlink {kind}",
            details={"path": str(resolved)},
        )
    return resolved


def atomic_dataframe_to_csv(
    frame: Any,
    path: str | Path,
    *,
    allowed_root: str | Path,
    index: bool,
    index_label: str | None = None,
    gzip_compression: bool = False,
) -> Path:
    """Stream a pandas DataFrame to a same-directory temporary file and replace atomically.

    Gzip output uses ``mtime=0`` so retries with identical data produce identical bytes.
    This changes no matrix values or ordering and avoids holding a second full CSV in memory.
    """

    root = Path(allowed_root)
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = ensure_within(target, root)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            if gzip_compression:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    mtime=0,
                ) as compressed:
                    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
                    try:
                        frame.to_csv(text, index=index, index_label=index_label)
                        text.flush()
                    finally:
                        text.detach()
            else:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                try:
                    frame.to_csv(text, index=index, index_label=index_label)
                    text.flush()
                finally:
                    text.detach()
            raw.flush()
            os.fsync(raw.fileno())
        ensure_within(target, root)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "GPS_OUTPUT_ROOT",
    "atomic_dataframe_to_csv",
    "base_environment_metrics",
    "package_version",
    "resolve_gps_output",
    "resolve_resource_path",
    "resolve_run_input_file",
]
