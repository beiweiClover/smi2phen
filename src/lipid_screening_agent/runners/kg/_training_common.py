"""Shared Stage 07 provenance, environment, and graph helpers."""

from __future__ import annotations

import importlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent.artifacts import (
    ArtifactManifest,
    NodeResult,
    NodeStatus,
    load_artifact_manifest,
    verify_artifact_manifest,
)
from lipid_screening_agent.runtime import (
    EnvironmentError,
    InputError,
    RunContext,
    ensure_within,
    hash_json,
    sha256_file,
)


def load_training_dependencies(*, require_dgl: bool = True, require_torch: bool = True):
    """Import heavy modules lazily and report one structured environment failure."""

    modules: dict[str, Any] = {}
    missing: dict[str, str] = {}
    names = ["numpy", "pandas"]
    if require_torch:
        names.extend(["torch", "sklearn"])
    if require_dgl:
        if "torch" not in names:
            names.extend(["torch", "sklearn"])
        names.append("dgl")
    for name in names:
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:  # binary import errors matter as much as ImportError
            missing[name] = f"{type(exc).__name__}: {exc}"
    if missing:
        raise EnvironmentError(
            "KG training dependencies could not be imported",
            details={
                "missing_or_incompatible": missing,
                "python": sys.version,
                "platform": platform.platform(),
                "cpu_small_graph_testing_only": True,
            },
            retryable=False,
        )
    return modules


def resolve_device(torch, requested: str, *, allow_cpu: bool) -> Any:
    requested = str(requested).strip().lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "CUDA was requested for KG training but is unavailable",
                details=environment_snapshot(torch=torch, dgl=None),
                retryable=False,
            )
        try:
            device = torch.device(requested)
            torch.empty(1, device=device)
        except Exception as exc:
            raise EnvironmentError(
                "CUDA device readiness check failed",
                details={
                    **environment_snapshot(torch=torch, dgl=None),
                    "requested_device": requested,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                retryable=False,
            ) from exc
        return device
    if requested != "cpu":
        raise EnvironmentError(
            "Unsupported KG training device",
            details={"requested_device": requested},
            retryable=False,
        )
    if not allow_cpu:
        raise EnvironmentError(
            "CPU KG training is disabled for production-sized runs",
            details={
                "requested_device": requested,
                "hint": "CPU is supported only for explicit small-graph tests.",
            },
            retryable=False,
        )
    return torch.device("cpu")


def environment_snapshot(*, torch, dgl) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "dgl": getattr(dgl, "__version__", None) if dgl is not None else None,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cpu_full_training_supported": False,
        "cpu_scope": "unit tests and tiny smoke graphs only",
    }
    if cuda_available:
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes = total_bytes = None
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except Exception:
                total_bytes = int(props.total_memory)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(total_bytes or props.total_memory),
                    "free_memory_bytes": None if free_bytes is None else int(free_bytes),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        snapshot["gpus"] = devices
    return snapshot


def set_random_seed(seed: int, *, numpy, torch, dgl=None) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if dgl is not None:
        try:
            dgl.seed(seed)
            dgl.random.seed(seed)
        except Exception:
            pass


def resolve_committed_artifact(
    context: RunContext,
    manifest_path: str | Path,
    *,
    artifact_type: str,
    producer_node_id: str,
    expected_relative_path: str | None = None,
) -> tuple[Path, ArtifactManifest]:
    """Resolve a hash-verified artifact committed by a successful producer result."""

    try:
        manifest = load_artifact_manifest(manifest_path, run_root=context.run_dir)
        verify_artifact_manifest(manifest, run_root=context.run_dir)
    except Exception as exc:
        raise InputError(
            f"{artifact_type} artifact manifest is invalid or stale",
            details={"manifest_path": str(manifest_path)},
        ) from exc
    expected = {
        "artifact_type": artifact_type,
        "producer_node_id": producer_node_id,
    }
    observed = {
        "artifact_type": manifest.artifact_type,
        "producer_node_id": manifest.producer_node_id,
    }
    if observed != expected or (
        expected_relative_path is not None and manifest.relative_path != expected_relative_path
    ):
        raise InputError(
            "artifact manifest does not satisfy the Stage 07 handoff contract",
            details={
                "expected": {**expected, "relative_path": expected_relative_path},
                "observed": {**observed, "relative_path": manifest.relative_path},
            },
        )
    result_path = context.resolve_run_relative(
        f"artifacts/node_results/{manifest.producer_node_id}/{manifest.producer_task_id}.json",
        must_exist=True,
    )
    try:
        result = NodeResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise InputError("producer NodeResult is missing or invalid") from exc
    if result.status not in {NodeStatus.SUCCEEDED, NodeStatus.CACHED}:
        raise InputError(
            "producer did not finish successfully",
            details={"status": result.status.value, "producer": producer_node_id},
        )
    if manifest.artifact_id not in result.outputs:
        raise InputError(
            "artifact was not committed by the successful producer NodeResult",
            details={"artifact_id": manifest.artifact_id},
        )
    return context.resolve_run_relative(manifest.relative_path, must_exist=True), manifest


def resolve_run_file(context: RunContext, path: str | Path, *, label: str) -> Path:
    try:
        candidate = context.resolve_run_relative(context.relative_path(path), must_exist=True)
    except Exception as exc:
        raise InputError(f"{label} must be an existing file in the run workspace") from exc
    if not candidate.is_file():
        raise InputError(f"{label} is not a regular file")
    return candidate


def output_path(context: RunContext, relative: str) -> Path:
    path = ensure_within(context.output_dir / relative, context.output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_dataframe_csv(frame, path: Path, *, allowed_root: Path, **kwargs: Any) -> None:
    ensure_within(path, allowed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, **kwargs)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_torch_save(torch, value: Any, path: Path, *, allowed_root: Path) -> None:
    """Write a Torch artifact beside its destination and atomically replace it."""

    ensure_within(path, allowed_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def file_identity(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def seed_configuration_hash(
    *,
    seeds: Sequence[int],
    finetune_config: Mapping[str, Any],
    checkpoint_sha256: str,
    training_manifest_sha256: str,
) -> str:
    return hash_json(
        {
            "seeds": list(seeds),
            "finetune": dict(finetune_config),
            "checkpoint_sha256": checkpoint_sha256,
            "training_manifest_sha256": training_manifest_sha256,
        }
    )[:16]


def check_unique_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise InputError("configured KG fine-tune seeds must not be empty")
    if len(set(normalized)) != len(normalized):
        duplicates = sorted({seed for seed in normalized if normalized.count(seed) > 1})
        raise InputError(
            "configured KG fine-tune seeds must be unique",
            details={"duplicate_seeds": duplicates},
        )
    return normalized


def elapsed_record(started: float) -> float:
    return float(time.perf_counter() - started)


__all__ = [
    "atomic_dataframe_csv",
    "atomic_torch_save",
    "check_unique_seeds",
    "elapsed_record",
    "environment_snapshot",
    "file_identity",
    "load_training_dependencies",
    "output_path",
    "resolve_committed_artifact",
    "resolve_device",
    "resolve_run_file",
    "seed_configuration_hash",
    "set_random_seed",
]
