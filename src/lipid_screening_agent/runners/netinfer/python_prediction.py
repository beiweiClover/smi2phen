"""Shared in-process wSDTNBI execution for known and novel NetInfer nodes."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import torch

from lipid_screening_agent.config.models import NetInferConfig
from lipid_screening_agent.runtime import (
    ExecutionError,
    ResourceError,
    RunContext,
    atomic_write_text,
    sha256_file,
)
from lipid_screening_agent.runtime.execution import NodeExecution

from ._common import resolve_resource_path
from .algorithms import parse_raw_prediction_rows, validate_raw_predictions
from .wsdtnbi import WSDTNBIConfig, WSDTNBIEngine


def _engine_config(settings: NetInferConfig) -> WSDTNBIConfig:
    parameters = settings.parameters
    return WSDTNBIConfig(
        alpha=parameters.alpha,
        beta=parameters.beta,
        gamma=parameters.gamma,
        delta=int(parameters.delta),
        epsilon=int(parameters.epsilon),
        k=parameters.k,
        top_n=settings.top_n_predicted_targets,
        batch_size=settings.inference_batch_size,
        device=settings.device,
        dtype=settings.dtype,
    )


def execute_python_prediction(
    execution: NodeExecution,
    *,
    context: RunContext,
    settings: NetInferConfig,
    source_type: str,
    source_ids: Sequence[str],
    drug_target_network_path: str | Path,
    drug_substructure_network_path: str | Path,
    compound_substructure_path: Path | None,
    output_path: Path,
) -> dict[str, object]:
    """Build the scientific network, predict requested IDs, and atomically emit raw rows."""

    drug_target = resolve_resource_path(
        context, drug_target_network_path, label="NetInfer DT.tsv"
    )
    drug_substructure = resolve_resource_path(
        context, drug_substructure_network_path, label="NetInfer DS.tsv"
    )
    for resource_id, path in {
        "resources.netinfer.drug_target_network": drug_target,
        "resources.netinfer.drug_substructure_network": drug_substructure,
    }.items():
        try:
            execution.resource_hashes[resource_id] = sha256_file(path)
        except (OSError, RuntimeError) as exc:
            raise ResourceError(
                "NetInfer resource could not be hashed", details={"path": str(path)}
            ) from exc

    load_started = time.perf_counter()
    try:
        engine = WSDTNBIEngine(
            drug_target,
            drug_substructure,
            _engine_config(settings),
        )
        load_seconds = time.perf_counter() - load_started
        prediction_started = time.perf_counter()
        if source_type == "DRUG":
            predictions = engine.predict_official_drugs(source_ids)
        elif source_type == "COMPOUND" and compound_substructure_path is not None:
            predictions = engine.predict_compounds(source_ids, compound_substructure_path)
        else:
            raise ValueError(f"unsupported NetInfer source type: {source_type}")
        prediction_seconds = time.perf_counter() - prediction_started
    except torch.cuda.OutOfMemoryError as exc:
        raise ExecutionError(
            "NetInfer CUDA memory was exhausted",
            details={"device": settings.device},
            retryable=True,
        ) from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).casefold():
            raise ExecutionError(
                "NetInfer device memory was exhausted",
                details={"device": settings.device},
                retryable=True,
            ) from exc
        raise

    rows: list[tuple[str, str, str, str, str, str]] = []
    for source_id in source_ids:
        for prediction in predictions.get(str(source_id), ()):
            rows.append(
                (
                    source_type,
                    str(source_id),
                    "TARGET",
                    str(prediction["target"]),
                    format(float(prediction["score"]), ".9g"),
                    str(prediction["rank"]),
                )
            )
        if source_type == "DRUG":
            for target, weight in engine.known_targets_for_drug(str(source_id)):
                rows.append(
                    (
                        "DRUG",
                        str(source_id),
                        "TARGET",
                        target,
                        format(float(weight), ".9g"),
                        "-",
                    )
                )

    parsed = parse_raw_prediction_rows(rows, source_label="python-wSDTNBI")
    validate_raw_predictions(
        parsed,
        expected_source_type=source_type,
        allowed_source_ids=set(source_ids),
    )
    rendered = "".join("\t".join(row) + "\n" for row in rows)
    atomic_write_text(output_path, rendered, allowed_root=context.output_dir)
    metrics: dict[str, object] = {
        "prediction_backend": "python_torch",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "not_available",
        "engine_load_seconds": load_seconds,
        "prediction_seconds": prediction_seconds,
        "raw_prediction_row_count": len(rows),
    }
    metrics.update(engine.summary())
    execution.logger.info(
        "netinfer_python_prediction_finished",
        "Python wSDTNBI prediction completed",
        source_type=source_type,
        source_count=len(source_ids),
        **metrics,
    )
    return metrics


__all__ = ["execute_python_prediction"]
