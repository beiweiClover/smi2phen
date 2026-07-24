"""Path, manifest, and atomic-I/O helpers for proximity runners."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lipid_screening_agent.artifacts import (
    ArtifactManifest,
    NodeResult,
    NodeStatus,
    load_artifact_manifest,
    verify_artifact_manifest,
)
from lipid_screening_agent.runtime import (
    InputError,
    OutputContractError,
    PathSafetyError,
    ResourceError,
    RunContext,
    ensure_within,
    resolve_run_relative,
)

PROXIMITY_OUTPUT_ROOT = "artifacts/proximity"
DRUG_TARGET_PROVIDERS = frozenset(
    {
        ("netinfer_merge_targets", "artifacts/netinfer/drug_targets.json"),
        ("import_drug_targets", "artifacts/targets/drug_targets.json"),
    }
)


def resolve_proximity_output(context: RunContext, relative_path: str) -> Path:
    expected_root = context.resolve_run_relative(PROXIMITY_OUTPUT_ROOT)
    if context.output_dir != expected_root:
        raise OutputContractError(
            "proximity output_dir must be artifacts/proximity",
            details={
                "configured_output_dir": str(context.output_dir),
                "expected_output_dir": str(expected_root),
            },
        )
    try:
        return resolve_run_relative(expected_root, relative_path)
    except (OSError, PathSafetyError) as exc:
        raise OutputContractError(
            "unsafe proximity output path", details={"relative_path": relative_path}
        ) from exc


def resolve_run_input_file(context: RunContext, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    try:
        if candidate.is_absolute():
            resolved = ensure_within(candidate, context.run_dir)
        else:
            resolved = context.resolve_run_relative(candidate.as_posix(), must_exist=True)
    except (OSError, PathSafetyError) as exc:
        raise InputError(
            f"{label} is missing or outside the run workspace",
            details={"path": str(value)},
        ) from exc
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise InputError(
            f"{label} must be an existing regular non-symlink file",
            details={"path": str(resolved)},
        )
    return resolved


def resolve_resource_file(context: RunContext, value: str | Path, *, label: str) -> Path:
    if context.resource_dir is None:
        raise ResourceError("a resource directory is required for proximity")
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
            f"{label} is missing or outside the resource directory",
            details={"path": str(value)},
        ) from exc
    if not resolved.exists() or not resolved.is_file() or resolved.is_symlink():
        raise ResourceError(
            f"{label} must be an existing regular non-symlink file",
            details={"path": str(resolved)},
        )
    return resolved


def resolve_successful_drug_targets(
    context: RunContext, manifest_path: str | Path
) -> tuple[Path, ArtifactManifest]:
    """Resolve drug_targets only through a committed successful artifact manifest."""

    manifest_file = resolve_run_input_file(
        context, manifest_path, label="drug_targets artifact manifest"
    )
    try:
        manifest = load_artifact_manifest(manifest_file, run_root=context.run_dir)
        verify_artifact_manifest(manifest, run_root=context.run_dir)
    except Exception as exc:
        raise InputError(
            "drug_targets artifact manifest is invalid or stale",
            details={"manifest_path": str(manifest_file)},
        ) from exc

    provider = (manifest.producer_node_id, manifest.relative_path)
    if manifest.artifact_type != "drug_targets" or provider not in DRUG_TARGET_PROVIDERS:
        raise InputError(
            "artifact manifest is not from an approved drug_targets provider",
            details={
                "approved_providers": sorted(DRUG_TARGET_PROVIDERS),
                "observed": {
                    "artifact_type": manifest.artifact_type,
                    "producer_node_id": manifest.producer_node_id,
                    "relative_path": manifest.relative_path,
                },
            },
        )

    result_path = context.resolve_run_relative(
        f"artifacts/node_results/{manifest.producer_node_id}/{manifest.producer_task_id}.json",
        must_exist=True,
    )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = NodeResult.from_dict(payload)
    except Exception as exc:
        raise InputError(
            "drug_targets producer NodeResult is missing or invalid",
            details={"node_result_path": str(result_path)},
        ) from exc
    if result.status not in {NodeStatus.SUCCEEDED, NodeStatus.CACHED}:
        raise InputError(
            "drug_targets producer did not finish successfully",
            details={"status": result.status.value, "node_result_path": str(result_path)},
        )
    if manifest.artifact_id not in result.outputs:
        raise InputError(
            "drug_targets manifest was not committed by the successful NodeResult",
            details={"artifact_id": manifest.artifact_id},
        )
    return context.resolve_run_relative(manifest.relative_path, must_exist=True), manifest


def resolve_successful_preparation(
    context: RunContext, manifest_path: str | Path
) -> tuple[Path, ArtifactManifest, tuple[str, ...]]:
    """Resolve the scientific preparation manifest through its runtime manifest."""

    manifest_file = resolve_run_input_file(
        context, manifest_path, label="proximity preparation artifact manifest"
    )
    try:
        manifest = load_artifact_manifest(manifest_file, run_root=context.run_dir)
        verify_artifact_manifest(manifest, run_root=context.run_dir)
    except Exception as exc:
        raise InputError(
            "proximity preparation artifact manifest is invalid or stale",
            details={"manifest_path": str(manifest_file)},
        ) from exc
    expected = {
        "artifact_type": "proximity_network_manifest",
        "producer_node_id": "proximity_prepare_network",
        "relative_path": "artifacts/proximity/network_manifest.json",
    }
    observed = {
        "artifact_type": manifest.artifact_type,
        "producer_node_id": manifest.producer_node_id,
        "relative_path": manifest.relative_path,
    }
    if observed != expected:
        raise InputError(
            "artifact manifest is not the committed proximity preparation manifest",
            details={"expected": expected, "observed": observed},
        )
    result_path = context.resolve_run_relative(
        f"artifacts/node_results/{manifest.producer_node_id}/{manifest.producer_task_id}.json",
        must_exist=True,
    )
    try:
        result = NodeResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise InputError("proximity prepare NodeResult is missing or invalid") from exc
    if result.status not in {NodeStatus.SUCCEEDED, NodeStatus.CACHED}:
        raise InputError(
            "proximity preparation did not finish successfully",
            details={"status": result.status.value},
        )
    if manifest.artifact_id not in result.outputs:
        raise InputError(
            "preparation manifest was not committed by the successful NodeResult",
            details={"artifact_id": manifest.artifact_id},
        )
    return (
        context.resolve_run_relative(manifest.relative_path, must_exist=True),
        manifest,
        result.outputs,
    )


def resolve_cache_directories(
    context: RunContext, shared_cache_dir: str | Path | None
) -> tuple[Path, Path | None]:
    run_cache = context.resolve_run_relative("cache/proximity")
    run_cache.mkdir(parents=True, exist_ok=True)
    if shared_cache_dir is None:
        return run_cache, None

    shared = Path(shared_cache_dir)
    if not shared.is_absolute():
        raise OutputContractError("shared cache directory must be absolute")
    shared = shared.resolve(strict=False)
    if shared.parent == shared:
        raise OutputContractError("filesystem root cannot be used as a shared cache")
    if context.resource_dir is not None:
        resource = context.resource_dir.resolve(strict=False)
        try:
            shared.relative_to(resource)
        except ValueError:
            pass
        else:
            raise OutputContractError(
                "shared cache cannot be inside the read-only resource directory"
            )
        try:
            resource.relative_to(shared)
        except ValueError:
            pass
        else:
            raise OutputContractError(
                "shared cache cannot contain the read-only resource directory"
            )
    shared.mkdir(parents=True, exist_ok=True)
    return run_cache, shared


def atomic_write_delimited(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
    delimiter: str,
    allowed_root: str | Path,
) -> Path:
    target = ensure_within(path, allowed_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter=delimiter,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_npz(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    allowed_root: str | Path,
) -> Path:
    target = ensure_within(path, allowed_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "DRUG_TARGET_PROVIDERS",
    "PROXIMITY_OUTPUT_ROOT",
    "atomic_write_delimited",
    "atomic_write_npz",
    "resolve_cache_directories",
    "resolve_proximity_output",
    "resolve_resource_file",
    "resolve_run_input_file",
    "resolve_successful_drug_targets",
    "resolve_successful_preparation",
]
