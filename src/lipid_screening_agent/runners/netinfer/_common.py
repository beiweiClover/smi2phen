"""Shared path and atomic-output helpers for the Stage 04 NetInfer runners."""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable, Sequence
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

NETINFER_OUTPUT_ROOT = "artifacts/netinfer"


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def resolve_netinfer_output(context: RunContext, relative_path: str) -> Path:
    """Resolve a contracted path below ``artifacts/netinfer``."""

    expected_root = context.resolve_run_relative(NETINFER_OUTPUT_ROOT)
    if context.output_dir != expected_root:
        raise OutputContractError(
            "NetInfer output_dir must be the contracted artifacts/netinfer directory",
            details={
                "configured_output_dir": str(context.output_dir),
                "expected_output_dir": str(expected_root),
            },
        )
    try:
        return resolve_run_relative(expected_root, relative_path)
    except (OSError, PathSafetyError) as exc:
        raise OutputContractError(
            "unsafe NetInfer output path", details={"relative_path": relative_path}
        ) from exc


def resolve_run_input_file(
    context: RunContext,
    value: str | Path,
    *,
    label: str,
    require_input_boundary: bool = False,
) -> Path:
    """Resolve an explicit regular, non-symlink input inside the run."""

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
) -> Path:
    """Resolve one explicit file below the read-only NetInfer resource root."""

    if context.resource_dir is None:
        raise ResourceError("a NetInfer resource directory is required")
    candidate = Path(value)
    try:
        if candidate.is_absolute():
            resolved = ensure_within(candidate, context.resource_dir)
        else:
            resolved = resolve_run_relative(
                context.resource_dir, candidate.as_posix(), must_exist=True
            )
    except (OSError, PathSafetyError) as exc:
        raise ResourceError(
            f"{label} is missing or outside the NetInfer resource directory",
            details={"path": str(value), "resource_dir": str(context.resource_dir)},
        ) from exc
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise ResourceError(
            f"{label} must be an existing regular non-symlink file",
            details={"path": str(resolved)},
        )
    return resolved


def atomic_write_delimited(
    path: str | Path,
    rows: Iterable[Sequence[Any]],
    *,
    allowed_root: str | Path,
    delimiter: str,
    header: Sequence[Any] | None = None,
) -> Path:
    """Stream a delimited table to a same-directory temp file and replace atomically."""

    root = Path(allowed_root)
    target = ensure_within(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = ensure_within(target, root)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        ensure_within(target, root)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def subprocess_log_paths(
    context: RunContext,
    node_id: str,
    task_id: str,
    attempt: int,
) -> tuple[Path, Path]:
    """Return dedicated full stdout/stderr paths alongside the structured node log."""

    log_root = context.resolve_run_relative(f"logs/{node_id}")
    prefix = f"{task_id}.attempt-{attempt:04d}.subprocess"
    return (
        ensure_within(log_root / f"{prefix}.stdout.log", context.run_dir),
        ensure_within(log_root / f"{prefix}.stderr.log", context.run_dir),
    )


__all__ = [
    "NETINFER_OUTPUT_ROOT",
    "atomic_write_delimited",
    "package_version",
    "resolve_netinfer_output",
    "resolve_resource_path",
    "resolve_run_input_file",
    "subprocess_log_paths",
]
