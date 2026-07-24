"""Registered runner boundary between orchestration and scientific code."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lipid_screening_agent.config import load_workflow_config, resolve_resource_paths
from lipid_screening_agent.runtime import RunContext

from .models import NodeRecord, RunnerOutcome

_TPM_INPUT_NAME = re.compile(r"^TPM_matrix(?:_.+)?\.tsv$")
_METADATA_INPUT_NAME = re.compile(r"^metadata(?:_.+)?\.tsv$")


@dataclass(frozen=True, slots=True)
class RunnerRequest:
    context: RunContext
    node: NodeRecord
    config_path: Path
    input_artifacts: tuple[Mapping[str, Any], ...] = ()
    code_version: str = "unknown"
    resource_hashes: Mapping[str, str] = field(default_factory=dict)
    is_cancelled: Callable[[], bool] = lambda: False
    heartbeat: Callable[[float | None], None] = lambda _progress=None: None


@dataclass(frozen=True, slots=True)
class CommandInvocation:
    argv: tuple[str, ...] | Sequence[str]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        argv = tuple(str(value) for value in self.argv)
        if not argv or any(not value for value in argv):
            raise ValueError("subprocess argv must contain non-empty values")
        object.__setattr__(self, "argv", argv)


RunnerCallable = Callable[[RunnerRequest], RunnerOutcome]
CommandFactory = Callable[[RunnerRequest], CommandInvocation]


@dataclass(frozen=True, slots=True)
class RegisteredRunner:
    node_id: str
    callable: RunnerCallable | None = None
    command_factory: CommandFactory | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if (self.callable is None) == (self.command_factory is None):
            raise ValueError("registered runner requires exactly one execution mechanism")


class RunnerRegistry:
    def __init__(self) -> None:
        self._runners: dict[str, RegisteredRunner] = {}

    def register_callable(
        self,
        node_id: str,
        runner: RunnerCallable,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        self._register(
            RegisteredRunner(node_id=node_id, callable=runner, timeout_seconds=timeout_seconds)
        )

    def register_command(
        self,
        node_id: str,
        factory: CommandFactory,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        self._register(
            RegisteredRunner(
                node_id=node_id,
                command_factory=factory,
                timeout_seconds=timeout_seconds,
            )
        )

    def _register(self, runner: RegisteredRunner) -> None:
        if runner.node_id in self._runners:
            raise ValueError(f"runner already registered: {runner.node_id}")
        self._runners[runner.node_id] = runner

    def get(self, node_id: str) -> RegisteredRunner:
        try:
            return self._runners[node_id]
        except KeyError as exc:
            raise KeyError(f"no runner registered for workflow node {node_id!r}") from exc

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._runners


RUNNER_MODULES: dict[str, str] = {
    "register_inputs": "lipid_screening_agent.runners.input_prepare.register_inputs",
    "prepare_compound_library": "lipid_screening_agent.runners.input_prepare.prepare_compound_library",
    "prepare_disease_genes": "lipid_screening_agent.runners.input_prepare.prepare_disease_genes",
    "import_drug_targets": "lipid_screening_agent.runners.input_prepare.import_drug_targets",
    "prepare_expression_inputs": "lipid_screening_agent.runners.input_prepare.prepare_expression_inputs",
    "gps_predict_drug_profiles": "lipid_screening_agent.runners.gps.predict_drug_profiles",
    "gps_build_disease_signature": "lipid_screening_agent.runners.gps.build_disease_signature",
    "gps_score_compounds": "lipid_screening_agent.runners.gps.score_compounds",
    "netinfer_prepare_inputs": "lipid_screening_agent.runners.netinfer.prepare_inputs",
    "netinfer_predict_known": "lipid_screening_agent.runners.netinfer.predict_known",
    "netinfer_predict_batch": "lipid_screening_agent.runners.netinfer.predict_batch",
    "netinfer_merge_targets": "lipid_screening_agent.runners.netinfer.merge_targets",
    "proximity_prepare_network": "lipid_screening_agent.runners.proximity.prepare_network",
    "proximity_score_compounds": "lipid_screening_agent.runners.proximity.score_compounds",
    "kg_construct_graph": "lipid_screening_agent.runners.kg.construct_graph",
    "kg_prepare_training_data": "lipid_screening_agent.runners.kg.prepare_training_data",
    "kg_pretrain": "lipid_screening_agent.runners.kg.pretrain",
    "kg_finetune_seed": "lipid_screening_agent.runners.kg.finetune_seed",
    "kg_aggregate_seeds": "lipid_screening_agent.runners.kg.aggregate_seeds",
    "rank_candidates": "lipid_screening_agent.runners.ranking.rank_candidates",
    "generate_run_report": "lipid_screening_agent.runners.ranking.generate_run_report",
}


def _artifact_path(request: RunnerRequest, artifact_type: str) -> Path:
    matches = [
        item for item in request.input_artifacts if item.get("artifact_type") == artifact_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {artifact_type} input artifact, found {len(matches)}"
        )
    value = matches[0].get("path") or matches[0].get("relative_path")
    if not value:
        raise ValueError(f"{artifact_type} input artifact has no path")
    path = Path(str(value))
    return (
        path if path.is_absolute() else request.context.resolve_run_relative(path, must_exist=True)
    )


def _artifact_manifest_path(request: RunnerRequest, artifact_type: str) -> Path:
    matches = [
        item for item in request.input_artifacts if item.get("artifact_type") == artifact_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {artifact_type} input artifact, found {len(matches)}"
        )
    item = matches[0]
    return request.context.resolve_run_relative(
        f"artifacts/manifests/{item['producer_node_id']}/{item['producer_task_id']}/"
        f"{item['artifact_id']}.json",
        must_exist=True,
    )


def _target_mapping_path(request: RunnerRequest) -> Path:
    """Accept the stable target mapping interface from either current target provider."""

    provided = _all_artifacts(request, "target_mapping")
    inferred = _all_artifacts(request, "netinfer_target_map")
    matches = [*provided, *inferred]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one target mapping artifact, found {len(matches)}")
    value = matches[0].get("path") or matches[0].get("relative_path")
    if not value:
        raise ValueError("target mapping artifact has no path")
    path = Path(str(value))
    return (
        path
        if path.is_absolute()
        else request.context.resolve_run_relative(path, must_exist=True)
    )


def _all_artifacts(request: RunnerRequest, artifact_type: str) -> list[Mapping[str, Any]]:
    return [item for item in request.input_artifacts if item.get("artifact_type") == artifact_type]


def _scientific_runner_args(request: RunnerRequest) -> list[str]:
    """Resolve typed run-local artifacts to the formal runner CLI; never accept shell text."""

    node_id = request.node.node_id
    context = request.context
    inputs = context.input_dir
    staged = {
        "compounds": inputs / "compounds.csv",
        "disease_genes": inputs / "disease_genes.tsv",
        "drug_targets": inputs / "drug_targets.json",
        "target_mapping": inputs / "target_mapping.tsv",
        "positive_drugs": inputs / "positive_drugs.tsv",
        "disease_links": inputs / "disease_links.tsv",
    }
    original = {key: inputs / "original" / path.name for key, path in staged.items()}
    if node_id == "register_inputs":
        values = [
            "--compound-library",
            str(staged["compounds"]),
            "--disease-genes",
            str(staged["disease_genes"]),
        ]
        for path in _expression_input_files(inputs):
            values += ["--expression-file", str(path)]
        for key, flag in (
            ("drug_targets", "--drug-targets"),
            ("target_mapping", "--target-mapping"),
            ("positive_drugs", "--positive-drugs"),
            ("disease_links", "--disease-links"),
        ):
            if staged[key].is_file():
                values += [flag, str(staged[key])]
        return values
    if node_id == "prepare_compound_library":
        return ["--input-file", str(original["compounds"])]
    if node_id == "prepare_disease_genes":
        return ["--input-file", str(original["disease_genes"])]
    if node_id == "import_drug_targets":
        return [
            "--drug-targets",
            str(original["drug_targets"]),
            "--target-mapping",
            str(original["target_mapping"]),
            "--compounds",
            str(_artifact_path(request, "compounds_normalized")),
        ]
    if node_id == "prepare_expression_inputs":
        return []
    if node_id == "gps_predict_drug_profiles":
        return ["--compounds", str(_artifact_path(request, "compounds_normalized"))]
    if node_id == "gps_build_disease_signature":
        return [
            "--expression-manifest",
            str(_artifact_path(request, "expression_comparisons_manifest")),
            "--drug-gps",
            str(_artifact_path(request, "gps_drug_profiles")),
        ]
    if node_id == "gps_score_compounds":
        return [
            "--drug-gps",
            str(_artifact_path(request, "gps_drug_profiles_entrez")),
            "--disease-gps",
            str(_artifact_path(request, "gps_disease_signature")),
        ]
    if node_id == "netinfer_prepare_inputs":
        return ["--compounds", str(_artifact_path(request, "compounds_normalized"))]
    if node_id == "netinfer_predict_known":
        return ["--mapping", str(_artifact_path(request, "netinfer_input_mapping"))]
    if node_id == "netinfer_predict_batch":
        batch_id = request.node.task_id
        batch = [
            item
            for item in request.input_artifacts
            if item.get("artifact_type") == "netinfer_batch_input"
            and (
                item.get("instance_key") == batch_id
                or Path(str(item.get("path") or item.get("relative_path", ""))).parent.name
                == batch_id
            )
        ]
        if len(batch) != 1:
            raise ValueError(f"expected one NetInfer batch input for {batch_id}")
        value = batch[0].get("path") or batch[0].get("relative_path")
        return [
            "--batch-id",
            batch_id,
            "--batch-manifest",
            str(_artifact_path(request, "netinfer_batch_manifest")),
            "--batch-input",
            str(value),
        ]
    if node_id == "netinfer_merge_targets":
        values = [
            "--mapping",
            str(_artifact_path(request, "netinfer_input_mapping")),
            "--target-map",
            str(_artifact_path(request, "netinfer_target_map")),
            "--batch-manifest",
            str(_artifact_path(request, "netinfer_batch_manifest")),
        ]
        known = _all_artifacts(request, "netinfer_known_predictions")
        if known:
            values += ["--known-predictions", str(known[0].get("path"))]
        for item in _all_artifacts(request, "netinfer_batch_predictions"):
            instance = item.get("instance_key") or item.get("producer_task_id")
            values += ["--batch-prediction", f"{instance}={item.get('path')}"]
        return values
    if node_id == "proximity_prepare_network":
        config = load_workflow_config(request.config_path)
        resources = resolve_resource_paths(
            config, ["resources.proximity.interactome"], environ=os.environ
        )
        return [
            "--ppi",
            str(resources["resources.proximity.interactome"]),
            "--disease-genes",
            str(_artifact_path(request, "disease_genes_normalized")),
            "--drug-targets-manifest",
            str(_artifact_manifest_path(request, "drug_targets")),
            "--target-mapping",
            str(_target_mapping_path(request)),
        ]
    if node_id == "proximity_score_compounds":
        return [
            "--prepared-manifest",
            str(_artifact_manifest_path(request, "proximity_network_manifest")),
        ]
    if node_id == "kg_construct_graph":
        config = load_workflow_config(request.config_path)
        resources = resolve_resource_paths(
            config,
            [
                "resources.kg.node_table",
                "resources.kg.edge_table",
                "resources.kg.manifest",
                "resources.kg.drug_smiles",
            ],
            environ=os.environ,
        )
        values = [
            "--compounds",
            str(_artifact_path(request, "compounds_normalized")),
            "--disease-genes",
            str(_artifact_path(request, "disease_genes_normalized")),
            "--drug-targets-manifest",
            str(_artifact_manifest_path(request, "drug_targets")),
            "--base-nodes",
            str(resources["resources.kg.node_table"]),
            "--base-edges",
            str(resources["resources.kg.edge_table"]),
            "--base-manifest",
            str(resources["resources.kg.manifest"]),
            "--base-drug-smiles",
            str(resources["resources.kg.drug_smiles"]),
            "--target-mapping",
            str(_target_mapping_path(request)),
        ]
        for key, flag in (
            ("positive_drugs", "--positive-drugs"),
            ("disease_links", "--disease-links"),
        ):
            if original[key].is_file():
                values += [flag, str(original[key])]
        return values
    if node_id == "kg_prepare_training_data":
        return [
            "--kg-nodes-manifest",
            str(_artifact_manifest_path(request, "kg_nodes")),
            "--kg-graph-manifest",
            str(_artifact_manifest_path(request, "kg_graph")),
            "--kg-construction-manifest",
            str(_artifact_manifest_path(request, "kg_construction_manifest")),
        ]
    if node_id == "kg_pretrain":
        return [
            "--training-nodes-manifest",
            str(_artifact_manifest_path(request, "kg_training_nodes")),
            "--training-edges-manifest",
            str(_artifact_manifest_path(request, "kg_training_edges")),
            "--training-manifest",
            str(_artifact_manifest_path(request, "kg_training_manifest")),
        ]
    if node_id == "kg_finetune_seed":
        return [
            "--seed",
            str(request.node.parameters["seed"]),
            "--training-nodes-manifest",
            str(_artifact_manifest_path(request, "kg_training_nodes")),
            "--training-edges-manifest",
            str(_artifact_manifest_path(request, "kg_training_edges")),
            "--training-manifest",
            str(_artifact_manifest_path(request, "kg_training_manifest")),
            "--pretrain-checkpoint-manifest",
            str(_artifact_manifest_path(request, "kg_pretrain_checkpoint")),
            "--pretrain-config-manifest",
            str(_artifact_manifest_path(request, "kg_pretrain_config")),
        ]
    if node_id == "kg_aggregate_seeds":
        values = []
        for item in _all_artifacts(request, "kg_seed_ranking"):
            values += [
                "--seed-ranking-manifest",
                str(_artifact_manifest_path_for_item(request, item)),
            ]
        for item in _all_artifacts(request, "kg_seed_summary"):
            values += [
                "--seed-summary-manifest",
                str(_artifact_manifest_path_for_item(request, item)),
            ]
        return values
    if node_id == "rank_candidates":
        values = [
            "--kg-ranking",
            str(_artifact_path(request, "kg_ensemble_ranking")),
            "--proximity-scores",
            str(_artifact_path(request, "proximity_scores")),
        ]
        gps = _all_artifacts(request, "gps_scores")
        if gps:
            values += ["--gps-scores", str(gps[0].get("path")), "--gps-status", "succeeded"]
        else:
            values += ["--gps-status", "skipped"]
        return values
    if node_id == "generate_run_report":
        values = [
            "--run-manifest",
            str(context.run_dir / "run_manifest.json"),
            "--input-manifest",
            str(context.run_dir / "inputs/input_manifest.json"),
            "--ranking-summary",
            str(_artifact_path(request, "ranking_summary")),
            "--final-candidates",
            str(_artifact_path(request, "final_candidates")),
        ]
        for path in sorted((context.run_dir / "artifacts/node_results").rglob("*.json")):
            values += ["--node-result", str(path)]
        return values
    raise ValueError(f"no scientific CLI argument mapping for node {node_id}")


def _expression_input_files(input_dir: Path) -> list[Path]:
    try:
        entries = sorted(input_dir.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.is_file()
        and (
            _TPM_INPUT_NAME.fullmatch(entry.name) is not None
            or _METADATA_INPUT_NAME.fullmatch(entry.name) is not None
        )
    ]


def _artifact_manifest_path_for_item(request: RunnerRequest, item: Mapping[str, Any]) -> Path:
    return request.context.resolve_run_relative(
        f"artifacts/manifests/{item['producer_node_id']}/{item['producer_task_id']}/{item['artifact_id']}.json",
        must_exist=True,
    )


def module_command_factory(module: str, *, python_executable: str | None = None) -> CommandFactory:
    """Create the stable CLI adapter used by Stage 10 workers.

    Artifact-to-CLI resolution is intentionally explicit: Stage 10 supplies ``runner_args`` in
    node parameters after resolving registered artifact IDs. No shell text is ever accepted.
    """

    executable = python_executable or sys.executable

    def factory(request: RunnerRequest) -> CommandInvocation:
        context = request.context
        if context.resource_dir is None:
            raise ValueError("a resource_dir is required for scientific runner commands")
        output_dir_value = request.node.parameters.get("output_dir")
        contracted_output_roots = {
            "prepare_compound_library": "inputs/prepared",
            "prepare_disease_genes": "inputs/prepared",
            "import_drug_targets": "artifacts/targets",
            "prepare_expression_inputs": "inputs/prepared",
            "gps_predict_drug_profiles": "artifacts/gps",
            "gps_build_disease_signature": "artifacts/gps",
            "gps_score_compounds": "artifacts/gps",
            "netinfer_prepare_inputs": "artifacts/netinfer",
            "netinfer_predict_known": "artifacts/netinfer",
            "netinfer_predict_batch": "artifacts/netinfer",
            "netinfer_merge_targets": "artifacts/netinfer",
            "proximity_prepare_network": "artifacts/proximity",
            "proximity_score_compounds": "artifacts/proximity",
            "kg_construct_graph": "artifacts/kg/construction",
            "rank_candidates": "artifacts/final",
            "generate_run_report": "reports",
        }
        output_dir = (
            context.resolve_run_relative(str(output_dir_value))
            if output_dir_value
            else context.resolve_run_relative(
                contracted_output_roots.get(
                    request.node.node_id,
                    f"artifacts/{request.node.node_id}/{request.node.task_id}",
                )
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        runner_input_dir = (
            context.resolve_run_relative("inputs/original", must_exist=True)
            if request.node.node_id
            in {
                "prepare_compound_library",
                "prepare_disease_genes",
                "prepare_expression_inputs",
            }
            else context.input_dir
        )
        if request.node.node_id.startswith("gps_"):
            runner_resource_dir = context.resource_dir / "gps"
        elif request.node.node_id.startswith("netinfer_"):
            runner_resource_dir = context.resource_dir / "netinfer"
        else:
            runner_resource_dir = context.resource_dir
        if not runner_resource_dir.is_dir():
            raise ValueError(f"runner resource directory does not exist: {runner_resource_dir}")
        raw_extra = request.node.parameters.get("runner_args")
        if raw_extra is None:
            raw_extra = _scientific_runner_args(request)
        if not isinstance(raw_extra, Sequence) or isinstance(raw_extra, (str, bytes)):
            raise ValueError("runner_args must be an argv sequence")
        argv = [
            executable,
            "-m",
            module,
            "--run-dir",
            str(context.run_dir),
            "--input-dir",
            str(runner_input_dir),
            "--resource-dir",
            str(runner_resource_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            str(request.config_path),
            "--task-id",
            request.node.task_id,
            "--attempt",
            str(request.node.attempt),
        ]
        for artifact in request.input_artifacts:
            artifact_id = artifact.get("artifact_id")
            if artifact_id:
                argv.extend(("--input-artifact-id", str(artifact_id)))
        argv.extend(str(value) for value in raw_extra)
        return CommandInvocation(argv=argv)

    return factory


def default_runner_registry(*, python_executable: str | None = None) -> RunnerRegistry:
    registry = RunnerRegistry()
    for node_id, module in RUNNER_MODULES.items():
        registry.register_command(
            node_id,
            module_command_factory(module, python_executable=python_executable),
        )
    return registry


__all__ = [
    "CommandInvocation",
    "RUNNER_MODULES",
    "RegisteredRunner",
    "RunnerRegistry",
    "RunnerRequest",
    "default_runner_registry",
    "module_command_factory",
]
