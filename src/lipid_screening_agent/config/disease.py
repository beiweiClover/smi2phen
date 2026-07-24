"""Disease configuration normalization and declared module compatibility."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from lipid_screening_agent.runtime.errors import ConfigurationError

from .models import DiseaseConfig, WorkflowConfig

_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9_]{0,62}[a-z0-9])?$")
_CUSTOM_NODE_ID = re.compile(r"^disease:custom:[a-z0-9](?:[a-z0-9._-]{0,94}[a-z0-9])?$")
SUPPORTED_DISEASE_SPECIES = ("human",)


class UnsupportedDiseaseError(ConfigurationError):
    """The disease is validly described but outside this release's scientific scope."""


@dataclass(frozen=True, slots=True)
class ModuleReadiness:
    status: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def normalize_disease_slug(value: str) -> str:
    """Normalize a user-facing ASCII label into a portable lower-snake slug."""

    raw = str(value).strip()
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ConfigurationError("disease.slug: path syntax and parent traversal are not allowed")
    normalized = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)[:64].rstrip("_")
    if not normalized or not _SLUG.fullmatch(normalized):
        raise ConfigurationError(
            "disease.slug: provide a portable ASCII slug using lowercase letters, digits, and underscores"
        )
    return normalized


def normalize_custom_node_id(value: str | None, *, slug: str) -> str:
    """Derive a safe custom node ID, or strictly validate an explicit one."""

    node_id = f"disease:custom:{slug}" if value is None else str(value).strip().casefold()
    if not _CUSTOM_NODE_ID.fullmatch(node_id):
        raise ConfigurationError(
            "disease.custom_node_id: expected disease:custom:<portable-id> with no path separators"
        )
    return node_id


def normalize_disease_config(
    value: DiseaseConfig | Mapping[str, Any],
    *,
    defaults: DiseaseConfig | None = None,
) -> DiseaseConfig:
    """Build a complete, safe disease config from YAML or Agent session values."""

    raw = asdict(value) if isinstance(value, DiseaseConfig) else dict(value)
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ConfigurationError("disease.name: string must not be empty")
    explicit_slug = raw.get("slug")
    if explicit_slug is None:
        slug = normalize_disease_slug(name)
    else:
        slug = str(explicit_slug).strip()
        if not _SLUG.fullmatch(slug):
            raise ConfigurationError(
                "disease.slug: expected canonical lowercase letters, digits, and underscores"
            )
    species = str(raw.get("species") or "human").strip().casefold()
    if species in {"homo sapiens", "homo_sapiens", "9606"}:
        species = "human"
    if species not in SUPPORTED_DISEASE_SPECIES:
        raise UnsupportedDiseaseError(
            f"unsupported disease species {species!r}; this release supports human only"
        )

    def optional(key: str) -> str | None:
        item = raw.get(key)
        if item is None:
            return None
        text = str(item).strip()
        return text or None

    return DiseaseConfig(
        name=name,
        slug=slug,
        identifier=optional("identifier"),
        custom_node_id=normalize_custom_node_id(raw.get("custom_node_id"), slug=slug),
        species=species,
        tissue=optional("tissue"),
        description=optional("description"),
        source_tag=str(
            raw.get("source_tag") or (defaults.source_tag if defaults else "custom")
        ).strip(),
    )


def workflow_config_for_disease(
    config: WorkflowConfig, disease: DiseaseConfig | Mapping[str, Any]
) -> WorkflowConfig:
    """Freeze one normalized disease into an otherwise unchanged workflow config."""

    return replace(config, disease=normalize_disease_config(disease, defaults=config.disease))


def assess_module_readiness(
    config: WorkflowConfig,
    *,
    disease_genes_available: bool,
    expression_available: bool,
) -> dict[str, ModuleReadiness]:
    """Explain material and declared species compatibility for each evidence branch."""

    species = config.disease.species
    compatibility = {
        "gene_mapping": config.resources.gps.gene_mapping_supported_species,
        "netinfer": config.resources.netinfer.supported_species,
        "proximity": config.resources.proximity.supported_species,
        "kg": config.resources.kg.supported_species,
        "gps": config.resources.gps.expression_supported_species,
    }
    result: dict[str, ModuleReadiness] = {}
    for module, supported in compatibility.items():
        if species not in supported:
            result[module] = ModuleReadiness(
                "unsupported",
                f"configured {module} resources do not declare compatibility with species {species!r}",
            )
        else:
            result[module] = ModuleReadiness(
                "available", f"configured resources declare {species} compatibility"
            )

    if not disease_genes_available:
        for module in ("gene_mapping", "proximity", "kg"):
            if result[module].status == "available":
                result[module] = ModuleReadiness(
                    "blocked", "a sourced disease gene set is required for the core workflow"
                )
    if not expression_available:
        result["gps"] = ModuleReadiness(
            "skipped",
            "expression TPM/metadata are unavailable; core KG + proximity mode remains usable",
        )
    return result


def core_compatibility_errors(config: WorkflowConfig) -> tuple[str, ...]:
    modules = assess_module_readiness(
        config, disease_genes_available=True, expression_available=True
    )
    return tuple(
        item.reason
        for name, item in modules.items()
        if name in {"gene_mapping", "netinfer", "proximity", "kg"} and item.status == "unsupported"
    )


__all__ = [
    "ModuleReadiness",
    "SUPPORTED_DISEASE_SPECIES",
    "UnsupportedDiseaseError",
    "assess_module_readiness",
    "core_compatibility_errors",
    "normalize_custom_node_id",
    "normalize_disease_config",
    "normalize_disease_slug",
    "workflow_config_for_disease",
]
