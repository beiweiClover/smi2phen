"""Machine-readable component and queue registry shared by planner and workers."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REDIS_PROTOCOL_VERSION = "1.0"
KNOWN_QUEUES = frozenset({"cpu", "gps", "kg", "netinfer"})


def project_source_digest(project_root: str | Path) -> str:
    """Hash the source/config/contract inputs baked into both current-source images."""

    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    candidates: list[Path] = []
    for relative in ("src", "configs", "contracts"):
        base = root / relative
        if base.is_dir():
            candidates.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
                and path.suffix != ".pyc"
            )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        candidates.append(pyproject)
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def load_component_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    value = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("redis_protocol_version") != REDIS_PROTOCOL_VERSION:
        raise ValueError("unsupported or malformed component registry")
    queues = set(value.get("queues") or ())
    if queues != set(KNOWN_QUEUES):
        raise ValueError("component registry queues must be cpu/gps/kg/netinfer")
    for node_id, entry in (value.get("nodes") or {}).items():
        if not isinstance(entry, dict) or entry.get("queue") not in KNOWN_QUEUES:
            raise ValueError(f"invalid component registry entry: {node_id}")
        if not entry.get("runner_module") or not entry.get("image"):
            raise ValueError(f"incomplete component registry entry: {node_id}")
    return value


def queue_for_node(node_id: str, *, registry_path: str | Path | None = None) -> str:
    if registry_path is None:
        root = Path(
            os.environ.get("LIPID_AGENT_PROJECT_ROOT", Path(__file__).resolve().parents[3])
        )
        registry_path = root / "configs" / "component_registry.yaml"
    registry = load_component_registry(registry_path)
    try:
        return str(registry["nodes"][node_id]["queue"])
    except KeyError as exc:
        raise KeyError(f"workflow node is absent from component registry: {node_id}") from exc


__all__ = [
    "KNOWN_QUEUES",
    "REDIS_PROTOCOL_VERSION",
    "load_component_registry",
    "project_source_digest",
    "queue_for_node",
]
