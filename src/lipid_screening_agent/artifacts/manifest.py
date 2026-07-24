"""Build and verify file-backed artifact manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime.errors import OutputContractError
from lipid_screening_agent.runtime.hashing import file_digest
from lipid_screening_agent.runtime.paths import ensure_within, to_run_relative_posix

from .models import ArtifactManifest, make_artifact_id


def _artifact_file(path: str | Path, run_root: str | Path) -> Path:
    try:
        to_run_relative_posix(path, run_root)
        candidate = ensure_within(path, run_root, allow_equal=False)
    except OutputContractError:
        raise
    except Exception as exc:
        raise OutputContractError(f"artifact path is outside the run workspace: {path}") from exc

    if not candidate.exists():
        raise OutputContractError(f"artifact file does not exist: {candidate}")
    if candidate.is_symlink():
        raise OutputContractError(f"artifact file cannot be a symbolic link: {candidate}")
    if not candidate.is_file():
        raise OutputContractError(f"artifact path is not a regular file: {candidate}")
    return candidate


def create_artifact_manifest(
    path: str | Path,
    *,
    run_root: str | Path,
    artifact_type: str,
    producer_node_id: str,
    producer_task_id: str,
    config_hash: str,
    code_version: str,
    input_artifact_ids: Sequence[str] = (),
    resource_hashes: Mapping[str, str] | None = None,
    artifact_id: str | None = None,
    instance_key: str | None = None,
    created_at: datetime | None = None,
) -> ArtifactManifest:
    """Create a manifest from one completed file inside *run_root*.

    ``instance_key`` is required by callers whenever a node/task produces more
    than one file of the same logical ``artifact_type``.  Supplying an explicit
    ``artifact_id`` is supported for restored executions, but it cannot be
    combined with ``instance_key``.
    """

    if artifact_id is not None and instance_key is not None:
        raise OutputContractError("artifact_id and instance_key cannot both be supplied")

    artifact_path = _artifact_file(path, run_root)
    try:
        digest = file_digest(artifact_path)
        relative_path = to_run_relative_posix(artifact_path, run_root)
    except OutputContractError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputContractError(f"could not fingerprint artifact file: {artifact_path}") from exc

    instance_id = artifact_id or make_artifact_id(
        producer_node_id,
        producer_task_id,
        artifact_type,
        instance_key=instance_key,
    )
    return ArtifactManifest(
        artifact_id=instance_id,
        artifact_type=artifact_type,
        relative_path=relative_path,
        size_bytes=digest.size_bytes,
        sha256=digest.sha256,
        created_at=created_at or datetime.now(timezone.utc),
        producer_node_id=producer_node_id,
        producer_task_id=producer_task_id,
        input_artifact_ids=tuple(input_artifact_ids),
        config_hash=config_hash,
        code_version=code_version,
        resource_hashes={} if resource_hashes is None else resource_hashes,
    )


def verify_artifact_manifest(
    manifest: ArtifactManifest,
    *,
    run_root: str | Path,
) -> None:
    """Raise :class:`OutputContractError` unless the recorded file still matches."""

    from lipid_screening_agent.runtime.paths import resolve_run_relative

    try:
        artifact_path = resolve_run_relative(run_root, manifest.relative_path, must_exist=True)
        artifact_path = _artifact_file(artifact_path, run_root)
        digest = file_digest(artifact_path)
    except OutputContractError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputContractError(
            f"could not verify artifact {manifest.artifact_id}: {manifest.relative_path}"
        ) from exc

    mismatches: list[str] = []
    if digest.size_bytes != manifest.size_bytes:
        mismatches.append(
            f"size_bytes expected {manifest.size_bytes}, observed {digest.size_bytes}"
        )
    if digest.sha256 != manifest.sha256:
        mismatches.append(f"sha256 expected {manifest.sha256}, observed {digest.sha256}")
    if mismatches:
        raise OutputContractError(
            f"artifact {manifest.artifact_id} failed verification: {'; '.join(mismatches)}"
        )


def artifact_matches_manifest(
    manifest: ArtifactManifest,
    *,
    run_root: str | Path,
) -> bool:
    """Return a boolean form of :func:`verify_artifact_manifest`."""

    try:
        verify_artifact_manifest(manifest, run_root=run_root)
    except (OutputContractError, OSError, RuntimeError, ValueError):
        return False
    return True


def load_artifact_manifest(
    path: str | Path,
    *,
    run_root: str | Path | None = None,
) -> ArtifactManifest:
    """Load and strictly validate a manifest JSON file."""

    manifest_path = Path(path)
    if run_root is not None:
        try:
            to_run_relative_posix(manifest_path, run_root)
            manifest_path = ensure_within(manifest_path, run_root, allow_equal=False)
        except OutputContractError:
            raise
        except Exception as exc:
            raise OutputContractError(
                f"manifest path is outside the run workspace: {manifest_path}"
            ) from exc
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutputContractError(f"could not read artifact manifest: {manifest_path}") from exc
    if not isinstance(data, Mapping):
        raise OutputContractError("artifact manifest JSON must contain an object")
    return ArtifactManifest.from_dict(data)


__all__ = [
    "artifact_matches_manifest",
    "create_artifact_manifest",
    "load_artifact_manifest",
    "verify_artifact_manifest",
]
