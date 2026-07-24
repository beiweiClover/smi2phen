"""Immutable typed models for the frozen V2 workflow configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class PathTemplate:
    """An unexpanded configured path.

    Keeping the source text is important: configuration hashes must not depend on
    host-specific environment-variable values.
    """

    raw: str


@dataclass(frozen=True, slots=True)
class WorkflowModeConfig:
    evidence_mode: str
    evidence: tuple[str, ...]
    gps_branch: str


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    id: str
    version: str
    default_mode: str
    modes: Mapping[str, WorkflowModeConfig]
    legacy_generated_results_as_default_inputs: bool
    artifact_path_root: str


@dataclass(frozen=True, slots=True)
class DiseaseConfig:
    name: str
    slug: str
    identifier: str | None
    custom_node_id: str
    species: str
    tissue: str | None
    description: str | None
    source_tag: str


@dataclass(frozen=True, slots=True)
class ResourceReferenceSemantics:
    environment_placeholder: str
    access: str
    copied_into_run_workspace: bool


@dataclass(frozen=True, slots=True)
class RunWorkspaceConfig:
    parent: PathTemplate
    child_template: str
    isolate_each_run: bool
    allow_outputs_outside_workspace: bool


@dataclass(frozen=True, slots=True)
class GPSResources:
    root: PathTemplate
    model_code: PathTemplate
    model_data: PathTemplate
    human_gene_info: PathTemplate
    gene_mapping_supported_species: tuple[str, ...]
    expression_supported_species: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NetInferResources:
    root: PathTemplate
    drug_target_network: PathTemplate
    drug_substructure_network: PathTemplate
    supplementary_workbook: PathTemplate
    supported_species: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProximityResources:
    interactome: PathTemplate
    supported_species: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KGResources:
    base_graph_root: PathTemplate
    node_table: PathTemplate
    edge_table: PathTemplate
    manifest: PathTemplate
    drug_smiles: PathTemplate
    supported_species: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResourcesConfig:
    gps: GPSResources
    netinfer: NetInferResources
    proximity: ProximityResources
    kg: KGResources


@dataclass(frozen=True, slots=True)
class BitFingerprintConfig:
    algorithm: str
    bits: int
    radius: int
    use_features: bool


@dataclass(frozen=True, slots=True)
class FeatureCountFingerprintConfig:
    algorithm: str
    radius: int
    use_features: bool


@dataclass(frozen=True, slots=True)
class GPSDrugProfilesConfig:
    cell_lines: tuple[str, ...]
    output_cell_line: str
    batch_size: int
    probability_threshold: float
    fingerprint: BitFingerprintConfig
    seed: int
    device: str
    preserve_legacy_cell_line_order: bool


@dataclass(frozen=True, slots=True)
class GPSDiseaseSignatureConfig:
    fdr_cutoff: float
    absolute_log2fc_cutoff: float
    tpm_filter_cutoff: float
    minimum_group_fraction_expressed: float
    test: str
    multiple_testing: str
    multi_comparison_combination: str


@dataclass(frozen=True, slots=True)
class GPSScoringConfig:
    random_background_samples: int
    seed: int
    lower_is_better: bool


@dataclass(frozen=True, slots=True)
class GPSConfig:
    drug_profiles: GPSDrugProfilesConfig
    disease_signature: GPSDiseaseSignatureConfig
    scoring: GPSScoringConfig


@dataclass(frozen=True, slots=True)
class NetInferParameters:
    k: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class NetInferNetworkTypes:
    nbi_types: tuple[str, ...]
    predict_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NetInferConfig:
    method: str
    top_n_predicted_targets: int
    batch_size: int
    inference_batch_size: int
    device: str
    dtype: str
    novel_compound_fingerprint: FeatureCountFingerprintConfig
    parameters: NetInferParameters
    known_drug: NetInferNetworkTypes
    novel_compound: NetInferNetworkTypes


@dataclass(frozen=True, slots=True)
class ProximityConfig:
    randomizations: int
    minimum_degree_bin_size: int
    seed: int
    job_batch_size: int
    background_component: str
    randomization: str
    lower_is_better: bool


@dataclass(frozen=True, slots=True)
class KGConstructionConfig:
    candidate_source_tag: str
    user_drug_node_prefix: str
    netinfer_dti_top_n: int
    netinfer_dti_selection: str
    pubchem_fingerprint_workers: int
    pubchem_fingerprint_chunk_size: int


@dataclass(frozen=True, slots=True)
class KGTrainingDataConfig:
    seed: int
    valid_fraction: float
    test_fraction: float
    pin_custom_disease_positive_edges_to_train: bool


@dataclass(frozen=True, slots=True)
class KGModelConfig:
    input_dimension: int
    hidden_dimension: int
    output_dimension: int
    attention: bool
    prototype_learning: bool
    prototype_count: int
    similarity_measure: str
    bert_measure: str
    aggregation_measure: str
    exponential_lambda: float
    random_walks: int
    walk_mode: str
    path_length: int


@dataclass(frozen=True, slots=True)
class KGPretrainConfig:
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    negative_samples_per_positive: int
    negative_method: str
    optimizer: str
    best_model_metric: str
    train_print_every_batches: int
    validate_every_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    save_embeddings: bool
    model: KGModelConfig


@dataclass(frozen=True, slots=True)
class KGRelationsConfig:
    forward: tuple[str, ...]
    reverse: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KGSchedulingConfig:
    maximum_concurrent_tasks_per_gpu: int
    allow_parallel_on_distinct_gpus: bool


@dataclass(frozen=True, slots=True)
class KGFinetuneConfig:
    seeds: tuple[int, ...]
    relations: KGRelationsConfig
    device: str
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    negative_samples_per_positive: int
    negative_method: str
    optimizer: str
    loss: str
    train_print_every_epochs: int
    validate_every_epochs: int
    validate_first_epoch: bool
    early_stopping_patience: int
    best_model_metric: str
    scheduler_factor: float
    scheduler_patience: int
    reset_decoder: bool
    resume_completed_seeds: bool
    save_per_seed_models: bool
    save_per_seed_embeddings: bool
    same_pretrain_checkpoint_for_every_seed: bool
    scheduling: KGSchedulingConfig


@dataclass(frozen=True, slots=True)
class KGRRAConfig:
    enabled_by_default: bool
    role: str


@dataclass(frozen=True, slots=True)
class KGAggregationConfig:
    primary_sort: str
    required_metrics: tuple[str, ...]
    rra: KGRRAConfig


@dataclass(frozen=True, slots=True)
class KGConfig:
    construction: KGConstructionConfig
    training_data: KGTrainingDataConfig
    pretrain: KGPretrainConfig
    finetune: KGFinetuneConfig
    aggregation: KGAggregationConfig


@dataclass(frozen=True, slots=True)
class RankingKGConfig:
    top_n: int
    metric: str
    direction: str


@dataclass(frozen=True, slots=True)
class RankingFilterConfig:
    operator: str
    value: float


@dataclass(frozen=True, slots=True)
class RankingEvidenceConfig:
    metric: str
    filter: RankingFilterConfig
    direction: str


@dataclass(frozen=True, slots=True)
class RankingConsensusConfig:
    method: str
    weights: str


@dataclass(frozen=True, slots=True)
class EmptyIntersectionConfig:
    status: str
    auto_relax_thresholds: bool


@dataclass(frozen=True, slots=True)
class RankingConfig:
    kg: RankingKGConfig
    proximity: RankingEvidenceConfig
    gps: RankingEvidenceConfig
    consensus: RankingConsensusConfig
    empty_intersection: EmptyIntersectionConfig


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    schema_version: str
    workflow: WorkflowDefinition
    disease: DiseaseConfig
    resource_reference_semantics: ResourceReferenceSemantics
    run_workspace: RunWorkspaceConfig
    resources: ResourcesConfig
    gps: GPSConfig
    netinfer: NetInferConfig
    proximity: ProximityConfig
    kg: KGConfig
    ranking: RankingConfig

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data with original, unexpanded path templates."""

        value = _to_primitive(self)
        if not isinstance(value, dict):  # pragma: no cover - invariant of this model
            raise TypeError("WorkflowConfig did not serialize to a mapping")
        return value


@dataclass(frozen=True, slots=True)
class ResolvedGPSResources:
    root: Path
    model_code: Path
    model_data: Path
    human_gene_info: Path


@dataclass(frozen=True, slots=True)
class ResolvedNetInferResources:
    root: Path
    drug_target_network: Path
    drug_substructure_network: Path
    supplementary_workbook: Path


@dataclass(frozen=True, slots=True)
class ResolvedProximityResources:
    interactome: Path


@dataclass(frozen=True, slots=True)
class ResolvedKGResources:
    base_graph_root: Path
    node_table: Path
    edge_table: Path
    manifest: Path
    drug_smiles: Path


@dataclass(frozen=True, slots=True)
class ResolvedResources:
    gps: ResolvedGPSResources
    netinfer: ResolvedNetInferResources
    proximity: ResolvedProximityResources
    kg: ResolvedKGResources


@dataclass(frozen=True, slots=True)
class ResolvedConfiguredPaths:
    run_workspace_parent: Path
    resources: ResolvedResources


def immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make a shallow immutable mapping for fields exposed by frozen models."""

    return MappingProxyType(dict(value))


def _to_primitive(value: Any) -> Any:
    if isinstance(value, PathTemplate):
        return value.raw
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    return value
