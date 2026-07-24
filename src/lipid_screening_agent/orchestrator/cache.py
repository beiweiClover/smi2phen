"""Content-addressed workflow cache identity and conservative validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime import hash_json, sha256_file


def build_cache_key(
    *,
    input_artifact_hashes: Mapping[str, str],
    config_hash: str,
    code_version: str,
    resource_hashes: Mapping[str, str],
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """Hash every provenance dimension required by the Stage 09 contract."""

    return hash_json(
        {
            "input_artifacts": dict(sorted(input_artifact_hashes.items())),
            "config_hash": config_hash,
            "code_version": code_version,
            "resource_hashes": dict(sorted(resource_hashes.items())),
            "parameters": dict(parameters or {}),
            "cache_schema": "workflow-cache-v1",
        }
    )


def cached_artifacts_are_valid(artifacts: Sequence[Mapping[str, Any]]) -> bool:
    """Reject cache records whose concrete files disappeared or changed.

    Artifact-only test doubles may omit paths. Production artifact records that include a path
    and digest are always verified before the cached status is committed.
    """

    for artifact in artifacts:
        path_value = artifact.get("path")
        expected = artifact.get("sha256")
        if path_value is None:
            continue
        path = Path(str(path_value))
        if not path.is_file():
            return False
        if expected is not None and sha256_file(path) != expected:
            return False
    return True


__all__ = ["build_cache_key", "cached_artifacts_are_valid"]
