"""Strict YAML loading and explicit path resolution for workflow configuration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from lipid_screening_agent.runtime.errors import ConfigurationError
from lipid_screening_agent.runtime.paths import canonical_path

from .disease import normalize_disease_config
from .models import (
    PathTemplate,
    ResolvedConfiguredPaths,
    ResolvedGPSResources,
    ResolvedKGResources,
    ResolvedNetInferResources,
    ResolvedProximityResources,
    ResolvedResources,
    WorkflowConfig,
)

_ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_workflow_config(path: str | os.PathLike[str]) -> WorkflowConfig:
    """Load and strictly validate a workflow YAML file.

    Environment-variable references deliberately remain unexpanded. This keeps
    parsing independent of a particular machine and makes configuration hashes
    portable between Windows development and Linux containers.
    """

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load workflow config {config_path}: {exc}") from exc
    return parse_workflow_config(document)


def parse_workflow_config(document: object) -> WorkflowConfig:
    """Build a typed configuration from already decoded YAML-compatible data."""

    config = _convert(document, WorkflowConfig, "config")
    if not isinstance(config, WorkflowConfig):  # pragma: no cover - converter invariant
        raise ConfigurationError("Workflow configuration has an invalid root model")
    config = replace(config, disease=normalize_disease_config(config.disease))
    _validate_semantics(config)
    return config


def canonical_config_dict(config: WorkflowConfig) -> dict[str, Any]:
    """Return the canonical, unexpanded configuration mapping."""

    return config.to_dict()


def canonical_config_bytes(config: WorkflowConfig) -> bytes:
    """Return deterministic UTF-8 JSON bytes suitable for hashing."""

    try:
        payload = json.dumps(
            canonical_config_dict(config),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - typed model invariant
        raise ConfigurationError(f"Workflow config is not canonically serializable: {exc}") from exc
    return payload.encode("utf-8")


def hash_workflow_config(config: WorkflowConfig) -> str:
    """Compute SHA-256 over the unexpanded canonical configuration."""

    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def resolve_run_workspace_parent(
    config: WorkflowConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve only the configured run-workspace parent."""

    values = os.environ if environ is None else environ
    return _resolve_absolute_template(
        config.run_workspace.parent,
        values,
        "run_workspace.parent",
    )


def resolve_resource_path(
    config: WorkflowConfig,
    reference: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve one dotted resource reference without touching unrelated resources."""

    values = os.environ if environ is None else environ
    gps = config.resources.gps
    netinfer = config.resources.netinfer
    proximity = config.resources.proximity
    kg = config.resources.kg

    if reference == "resources.gps.root":
        return _resolve_absolute_template(gps.root, values, reference)
    if reference in {
        "resources.gps.model_code",
        "resources.gps.model_data",
        "resources.gps.human_gene_info",
    }:
        root = _resolve_absolute_template(gps.root, values, "resources.gps.root")
        child = {
            "resources.gps.model_code": gps.model_code,
            "resources.gps.model_data": gps.model_data,
            "resources.gps.human_gene_info": gps.human_gene_info,
        }[reference]
        return _resolve_child(root, child, values, reference)

    if reference == "resources.netinfer.root":
        return _resolve_absolute_template(netinfer.root, values, reference)
    if reference in {
        "resources.netinfer.drug_target_network",
        "resources.netinfer.drug_substructure_network",
        "resources.netinfer.supplementary_workbook",
    }:
        root = _resolve_absolute_template(netinfer.root, values, "resources.netinfer.root")
        child = {
            "resources.netinfer.drug_target_network": netinfer.drug_target_network,
            "resources.netinfer.drug_substructure_network": (netinfer.drug_substructure_network),
            "resources.netinfer.supplementary_workbook": netinfer.supplementary_workbook,
        }[reference]
        return _resolve_child(root, child, values, reference)

    if reference == "resources.proximity.interactome":
        return _resolve_absolute_template(proximity.interactome, values, reference)

    if reference == "resources.kg.base_graph_root":
        return _resolve_absolute_template(kg.base_graph_root, values, reference)
    if reference in {
        "resources.kg.node_table",
        "resources.kg.edge_table",
        "resources.kg.manifest",
        "resources.kg.drug_smiles",
    }:
        root = _resolve_absolute_template(
            kg.base_graph_root,
            values,
            "resources.kg.base_graph_root",
        )
        child = {
            "resources.kg.node_table": kg.node_table,
            "resources.kg.edge_table": kg.edge_table,
            "resources.kg.manifest": kg.manifest,
            "resources.kg.drug_smiles": kg.drug_smiles,
        }[reference]
        return _resolve_child(root, child, values, reference)

    raise ConfigurationError(f"Unknown resource reference: {reference!r}")


def resolve_resource_paths(
    config: WorkflowConfig,
    references: tuple[str, ...] | list[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> Mapping[str, Path]:
    """Resolve a selected set of resource references into an immutable mapping."""

    values = os.environ if environ is None else environ
    resolved = {
        reference: resolve_resource_path(config, reference, environ=values)
        for reference in references
    }
    return MappingProxyType(resolved)


def resolve_configured_paths(
    config: WorkflowConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedConfiguredPaths:
    """Resolve only modeled path fields using an explicit environment mapping.

    This intentionally does not walk arbitrary strings in the configuration, so
    the explanatory ``${VARIABLE_NAME}`` value is never treated as a real input.
    Resolution is lexical and does not require resources to exist; runners perform
    their own existence and readability checks for the resources they consume.
    """

    values = os.environ if environ is None else environ
    workspace_parent = resolve_run_workspace_parent(config, environ=values)

    gps_root = _resolve_absolute_template(
        config.resources.gps.root,
        values,
        "resources.gps.root",
    )
    gps = ResolvedGPSResources(
        root=gps_root,
        model_code=_resolve_child(
            gps_root,
            config.resources.gps.model_code,
            values,
            "resources.gps.model_code",
        ),
        model_data=_resolve_child(
            gps_root,
            config.resources.gps.model_data,
            values,
            "resources.gps.model_data",
        ),
        human_gene_info=_resolve_child(
            gps_root,
            config.resources.gps.human_gene_info,
            values,
            "resources.gps.human_gene_info",
        ),
    )

    netinfer_root = _resolve_absolute_template(
        config.resources.netinfer.root,
        values,
        "resources.netinfer.root",
    )
    netinfer = ResolvedNetInferResources(
        root=netinfer_root,
        drug_target_network=_resolve_child(
            netinfer_root,
            config.resources.netinfer.drug_target_network,
            values,
            "resources.netinfer.drug_target_network",
        ),
        drug_substructure_network=_resolve_child(
            netinfer_root,
            config.resources.netinfer.drug_substructure_network,
            values,
            "resources.netinfer.drug_substructure_network",
        ),
        supplementary_workbook=_resolve_child(
            netinfer_root,
            config.resources.netinfer.supplementary_workbook,
            values,
            "resources.netinfer.supplementary_workbook",
        ),
    )

    proximity = ResolvedProximityResources(
        interactome=_resolve_absolute_template(
            config.resources.proximity.interactome,
            values,
            "resources.proximity.interactome",
        )
    )

    kg_root = _resolve_absolute_template(
        config.resources.kg.base_graph_root,
        values,
        "resources.kg.base_graph_root",
    )
    kg = ResolvedKGResources(
        base_graph_root=kg_root,
        node_table=_resolve_child(
            kg_root,
            config.resources.kg.node_table,
            values,
            "resources.kg.node_table",
        ),
        edge_table=_resolve_child(
            kg_root,
            config.resources.kg.edge_table,
            values,
            "resources.kg.edge_table",
        ),
        manifest=_resolve_child(
            kg_root,
            config.resources.kg.manifest,
            values,
            "resources.kg.manifest",
        ),
        drug_smiles=_resolve_child(
            kg_root,
            config.resources.kg.drug_smiles,
            values,
            "resources.kg.drug_smiles",
        ),
    )

    return ResolvedConfiguredPaths(
        run_workspace_parent=workspace_parent,
        resources=ResolvedResources(
            gps=gps,
            netinfer=netinfer,
            proximity=proximity,
            kg=kg,
        ),
    )


def _convert(value: object, expected: Any, path: str) -> Any:
    if expected is PathTemplate:
        return PathTemplate(_require_string(value, path))

    if is_dataclass(expected):
        mapping = _require_mapping(value, path)
        model_fields = {field.name: field for field in fields(expected)}
        missing = sorted(set(model_fields) - set(mapping))
        extra = sorted((key for key in mapping if key not in model_fields), key=repr)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing keys: {', '.join(missing)}")
            if extra:
                parts.append(f"unexpected keys: {', '.join(map(repr, extra))}")
            raise ConfigurationError(f"{path}: {'; '.join(parts)}")
        type_hints = get_type_hints(expected)
        values = {
            name: _convert(mapping[name], type_hints[name], f"{path}.{name}")
            for name in model_fields
        }
        return expected(**values)

    origin = get_origin(expected)
    arguments = get_args(expected)
    if origin in (types.UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        errors: list[str] = []
        for option in arguments:
            if option is type(None):
                continue
            try:
                return _convert(value, option, path)
            except ConfigurationError as exc:
                errors.append(str(exc))
        raise ConfigurationError(f"{path}: value does not match any permitted type")

    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigurationError(f"{path}: expected a YAML sequence")
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise ConfigurationError(f"{path}: unsupported tuple annotation")
        return tuple(
            _convert(item, arguments[0], f"{path}[{index}]") for index, item in enumerate(value)
        )

    if origin is Mapping:
        mapping = _require_mapping(value, path)
        key_type, value_type = arguments
        converted: dict[Any, Any] = {}
        for key, item in mapping.items():
            converted_key = _convert(key, key_type, f"{path}.<key>")
            converted[converted_key] = _convert(item, value_type, f"{path}.{key}")
        return MappingProxyType(converted)

    if expected is str:
        return _require_string(value, path)
    if expected is bool:
        if type(value) is not bool:
            raise ConfigurationError(f"{path}: expected a boolean")
        return value
    if expected is int:
        if type(value) is not int:
            raise ConfigurationError(f"{path}: expected an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{path}: expected a number")
        return float(value)
    raise ConfigurationError(f"{path}: unsupported configuration type {expected!r}")


def _require_mapping(value: object, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path}: expected a mapping")
    return value


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{path}: expected a string")
    if not value:
        raise ConfigurationError(f"{path}: string must not be empty")
    return value


def _validate_semantics(config: WorkflowConfig) -> None:
    _assert(config.schema_version == "1.0", "schema_version: only version 1.0 is supported")
    _assert(
        set(config.workflow.modes) == {"core", "enhanced"},
        "workflow.modes: expected exactly core and enhanced",
    )
    _assert(
        config.workflow.default_mode in config.workflow.modes,
        "workflow.default_mode: must reference a configured mode",
    )
    _assert(
        config.workflow.legacy_generated_results_as_default_inputs is False,
        "workflow.legacy_generated_results_as_default_inputs: must remain false",
    )
    _assert(
        config.workflow.artifact_path_root == "run_workspace",
        "workflow.artifact_path_root: must be run_workspace",
    )
    _assert(
        config.resource_reference_semantics.access == "read_only",
        "resource_reference_semantics.access: must be read_only",
    )
    _assert(
        config.resource_reference_semantics.copied_into_run_workspace is False,
        "resource_reference_semantics.copied_into_run_workspace: must remain false",
    )
    _assert(
        config.run_workspace.isolate_each_run,
        "run_workspace.isolate_each_run: must be true",
    )
    _assert(
        config.run_workspace.allow_outputs_outside_workspace is False,
        "run_workspace.allow_outputs_outside_workspace: must remain false",
    )
    _assert(
        config.run_workspace.child_template == "{run_id}",
        "run_workspace.child_template: must be {run_id}",
    )

    for path, supported in (
        (
            "resources.gps.gene_mapping_supported_species",
            config.resources.gps.gene_mapping_supported_species,
        ),
        (
            "resources.gps.expression_supported_species",
            config.resources.gps.expression_supported_species,
        ),
        ("resources.netinfer.supported_species", config.resources.netinfer.supported_species),
        ("resources.proximity.supported_species", config.resources.proximity.supported_species),
        ("resources.kg.supported_species", config.resources.kg.supported_species),
    ):
        _assert(len(set(supported)) == len(supported), f"{path}: values must be unique")
        _assert(
            set(supported) <= {"human"},
            f"{path}: Stage 15 supports only the human resource space",
        )

    for mode_name, mode in config.workflow.modes.items():
        _assert(mode.evidence, f"workflow.modes.{mode_name}.evidence: must not be empty")
        _assert(
            len(set(mode.evidence)) == len(mode.evidence),
            f"workflow.modes.{mode_name}.evidence: values must be unique",
        )
        _assert(
            set(mode.evidence) <= {"kg", "proximity", "gps"},
            f"workflow.modes.{mode_name}.evidence: contains unsupported evidence",
        )
        _assert(
            mode.gps_branch in {"required", "skipped"},
            f"workflow.modes.{mode_name}.gps_branch: expected required or skipped",
        )
    _assert(
        config.workflow.modes["core"].gps_branch == "skipped",
        "workflow.modes.core.gps_branch: must be skipped",
    )
    _assert(
        config.workflow.modes["enhanced"].gps_branch == "required",
        "workflow.modes.enhanced.gps_branch: must be required",
    )

    _positive(config.gps.drug_profiles.batch_size, "gps.drug_profiles.batch_size")
    _between(
        config.gps.drug_profiles.probability_threshold,
        "gps.drug_profiles.probability_threshold",
    )
    _assert(
        config.gps.drug_profiles.output_cell_line in config.gps.drug_profiles.cell_lines,
        "gps.drug_profiles.output_cell_line: must occur in cell_lines",
    )
    _positive(config.gps.drug_profiles.fingerprint.bits, "gps.drug_profiles.fingerprint.bits")
    _non_negative(
        config.gps.drug_profiles.fingerprint.radius,
        "gps.drug_profiles.fingerprint.radius",
    )
    _between(config.gps.disease_signature.fdr_cutoff, "gps.disease_signature.fdr_cutoff")
    _between(
        config.gps.disease_signature.minimum_group_fraction_expressed,
        "gps.disease_signature.minimum_group_fraction_expressed",
    )
    _positive(
        config.gps.scoring.random_background_samples,
        "gps.scoring.random_background_samples",
    )

    _positive(config.netinfer.top_n_predicted_targets, "netinfer.top_n_predicted_targets")
    _positive(config.netinfer.batch_size, "netinfer.batch_size")
    _positive(config.netinfer.inference_batch_size, "netinfer.inference_batch_size")
    _assert(
        config.netinfer.device in {"auto", "cpu", "cuda"},
        "netinfer.device: must be auto, cpu, or cuda",
    )
    _assert(
        config.netinfer.dtype in {"float32", "float64"},
        "netinfer.dtype: must be float32 or float64",
    )
    _non_negative(
        config.netinfer.novel_compound_fingerprint.radius,
        "netinfer.novel_compound_fingerprint.radius",
    )
    _positive(config.proximity.randomizations, "proximity.randomizations")
    _positive(config.proximity.minimum_degree_bin_size, "proximity.minimum_degree_bin_size")
    _positive(config.proximity.job_batch_size, "proximity.job_batch_size")

    _positive(config.kg.construction.netinfer_dti_top_n, "kg.construction.netinfer_dti_top_n")
    _positive(
        config.kg.construction.pubchem_fingerprint_chunk_size,
        "kg.construction.pubchem_fingerprint_chunk_size",
    )
    _between(config.kg.training_data.valid_fraction, "kg.training_data.valid_fraction")
    _between(config.kg.training_data.test_fraction, "kg.training_data.test_fraction")
    _assert(
        config.kg.training_data.valid_fraction + config.kg.training_data.test_fraction < 1.0,
        "kg.training_data: valid_fraction + test_fraction must be less than 1",
    )
    _positive(config.kg.pretrain.epochs, "kg.pretrain.epochs")
    _positive(config.kg.pretrain.batch_size, "kg.pretrain.batch_size")
    _positive(config.kg.pretrain.learning_rate, "kg.pretrain.learning_rate")
    _positive(config.kg.finetune.epochs, "kg.finetune.epochs")
    _positive(config.kg.finetune.learning_rate, "kg.finetune.learning_rate")
    _assert(config.kg.finetune.seeds, "kg.finetune.seeds: must not be empty")
    _assert(
        len(set(config.kg.finetune.seeds)) == len(config.kg.finetune.seeds),
        "kg.finetune.seeds: values must be unique",
    )
    _positive(
        config.kg.finetune.scheduling.maximum_concurrent_tasks_per_gpu,
        "kg.finetune.scheduling.maximum_concurrent_tasks_per_gpu",
    )
    _assert(
        config.kg.aggregation.required_metrics,
        "kg.aggregation.required_metrics: must not be empty",
    )
    _positive(config.ranking.kg.top_n, "ranking.kg.top_n")
    _assert(
        config.ranking.kg.metric == "rank_mean",
        "ranking.kg.metric: Stage 08 requires rank_mean",
    )
    _assert(
        config.ranking.proximity.metric == "z",
        "ranking.proximity.metric: Stage 08 requires z",
    )
    _assert(
        config.ranking.gps.metric == "GPS_score_zRGES_like_lower_better",
        "ranking.gps.metric: Stage 08 requires GPS_score_zRGES_like_lower_better",
    )
    for path, direction in (
        ("ranking.kg.direction", config.ranking.kg.direction),
        ("ranking.proximity.direction", config.ranking.proximity.direction),
        ("ranking.gps.direction", config.ranking.gps.direction),
    ):
        _assert(direction in {"ascending", "descending"}, f"{path}: unsupported direction")
    for path, operator in (
        ("ranking.proximity.filter.operator", config.ranking.proximity.filter.operator),
        ("ranking.gps.filter.operator", config.ranking.gps.filter.operator),
    ):
        _assert(operator in {"lt", "le", "gt", "ge"}, f"{path}: unsupported operator")
    _assert(
        config.ranking.consensus.method == "within_intersection_rank_percentile_mean",
        "ranking.consensus.method: unsupported method",
    )
    _assert(
        config.ranking.consensus.weights == "equal_over_available_evidence",
        "ranking.consensus.weights: unsupported weighting",
    )
    _assert(
        config.ranking.empty_intersection.status == "no_candidates_passed",
        "ranking.empty_intersection.status: must be no_candidates_passed",
    )
    _assert(
        config.ranking.empty_intersection.auto_relax_thresholds is False,
        "ranking.empty_intersection.auto_relax_thresholds: must remain false",
    )

    relative_templates = {
        "resources.gps.model_code": config.resources.gps.model_code,
        "resources.gps.model_data": config.resources.gps.model_data,
        "resources.gps.human_gene_info": config.resources.gps.human_gene_info,
        "resources.netinfer.drug_target_network": config.resources.netinfer.drug_target_network,
        "resources.netinfer.drug_substructure_network": (
            config.resources.netinfer.drug_substructure_network
        ),
        "resources.netinfer.supplementary_workbook": (
            config.resources.netinfer.supplementary_workbook
        ),
        "resources.kg.node_table": config.resources.kg.node_table,
        "resources.kg.edge_table": config.resources.kg.edge_table,
        "resources.kg.manifest": config.resources.kg.manifest,
        "resources.kg.drug_smiles": config.resources.kg.drug_smiles,
    }
    for path, template in relative_templates.items():
        _validate_relative_template(template.raw, path)


def _resolve_template(
    template: PathTemplate,
    environ: Mapping[str, str],
    field_path: str,
) -> Path:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = environ.get(name)
        if value is None or not value:
            raise ConfigurationError(
                f"{field_path}: required environment variable {name!r} is not set"
            )
        return value

    resolved = _ENVIRONMENT_REFERENCE.sub(replace, template.raw)
    remaining = _ENVIRONMENT_REFERENCE.search(resolved)
    if remaining:  # pragma: no cover - replacement resolves every match in one pass
        raise ConfigurationError(
            f"{field_path}: unresolved environment variable {remaining.group(1)!r}"
        )
    if not resolved:
        raise ConfigurationError(f"{field_path}: resolved path is empty")
    return Path(resolved).expanduser()


def _resolve_absolute_template(
    template: PathTemplate,
    environ: Mapping[str, str],
    field_path: str,
) -> Path:
    resolved = _resolve_template(template, environ, field_path)
    try:
        return canonical_path(resolved, must_exist=False, label=field_path)
    except Exception as exc:
        raise ConfigurationError(
            f"{field_path}: configured root/resource path must be host-native and absolute"
        ) from exc


def _resolve_child(
    root: Path,
    template: PathTemplate,
    environ: Mapping[str, str],
    field_path: str,
) -> Path:
    raw_child = str(_resolve_template(template, environ, field_path))
    _validate_relative_template(raw_child, field_path)
    parts = PurePosixPath(raw_child.replace("\\", "/")).parts
    return root.joinpath(*parts)


def _validate_relative_template(value: str, path: str) -> None:
    posix = PurePosixPath(value.replace("\\", "/"))
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        raise ConfigurationError(f"{path}: resource child path must be relative")
    if ".." in posix.parts or ".." in windows.parts:
        raise ConfigurationError(f"{path}: parent traversal is not allowed")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _positive(value: int | float, path: str) -> None:
    _assert(value > 0, f"{path}: must be greater than zero")


def _non_negative(value: int | float, path: str) -> None:
    _assert(value >= 0, f"{path}: must not be negative")


def _between(value: float, path: str) -> None:
    _assert(0.0 <= value <= 1.0, f"{path}: must be between 0 and 1")
