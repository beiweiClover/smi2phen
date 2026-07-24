"""Frozen scientific parameter validation for the Python wSDTNBI implementation."""

from __future__ import annotations

from lipid_screening_agent.config.models import NetInferConfig, NetInferNetworkTypes
from lipid_screening_agent.runtime import ConfigurationError

LEGACY_PARAMETERS = {
    "k": 2,
    "alpha": 0.4,
    "beta": 0.2,
    "gamma": -0.5,
    "delta": 20.0,
    "epsilon": 4.0,
}


def validate_netinfer_settings(settings: NetInferConfig) -> None:
    observed_parameters = {
        "k": settings.parameters.k,
        "alpha": settings.parameters.alpha,
        "beta": settings.parameters.beta,
        "gamma": settings.parameters.gamma,
        "delta": settings.parameters.delta,
        "epsilon": settings.parameters.epsilon,
    }
    expected_networks = {
        "known": (("SUB", "DRUG", "TARGET"), ("DRUG", "TARGET")),
        "novel": (("SUB", "COMPOUND+DRUG", "TARGET"), ("COMPOUND", "TARGET")),
    }
    observed_networks = {
        "known": (settings.known_drug.nbi_types, settings.known_drug.predict_types),
        "novel": (
            settings.novel_compound.nbi_types,
            settings.novel_compound.predict_types,
        ),
    }
    if settings.method != "wnbi":
        raise ConfigurationError("NetInfer method must remain wnbi")
    if settings.top_n_predicted_targets != 10:
        raise ConfigurationError("NetInfer predicted target count must remain top10")
    if observed_parameters != LEGACY_PARAMETERS:
        raise ConfigurationError(
            "NetInfer wSDTNBI parameters must preserve the legacy values",
            details={"expected": LEGACY_PARAMETERS, "configured": observed_parameters},
        )
    if observed_networks != expected_networks:
        raise ConfigurationError(
            "NetInfer network/prediction types must preserve the legacy command",
            details={"expected": expected_networks, "configured": observed_networks},
        )
    fingerprint = settings.novel_compound_fingerprint
    if (
        fingerprint.algorithm != "morgan_feature_count"
        or fingerprint.radius != 2
        or not fingerprint.use_features
    ):
        raise ConfigurationError(
            "NetInfer novel compounds require Morgan feature-count radius=2 use_features=true"
        )
    if settings.device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(
            "NetInfer device must be auto, cpu, or cuda"
        )
    if settings.dtype not in {"float32", "float64"}:
        raise ConfigurationError("NetInfer dtype must be float32 or float64")
    if settings.inference_batch_size < 1:
        raise ConfigurationError("NetInfer inference batch size must be positive")


def format_parameter(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def network_types(settings: NetInferConfig, source_type: str) -> NetInferNetworkTypes:
    if source_type == "DRUG":
        return settings.known_drug
    if source_type == "COMPOUND":
        return settings.novel_compound
    raise ConfigurationError(f"unsupported NetInfer source type: {source_type}")


__all__ = [
    "LEGACY_PARAMETERS",
    "format_parameter",
    "network_types",
    "validate_netinfer_settings",
]
