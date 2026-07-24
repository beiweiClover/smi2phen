"""Deterministic core/enhanced DAG planning with no LLM dependency."""

from __future__ import annotations

from collections.abc import Sequence

from lipid_screening_agent.config import (
    WorkflowConfig,
    assess_module_readiness,
    core_compatibility_errors,
)

from .components import queue_for_node
from .models import (
    DependencyRef,
    FanoutPlan,
    InputAvailability,
    InputState,
    PlannedNode,
    WorkflowPlan,
    WorkflowStatus,
)


class PlanningBlockedError(ValueError):
    def __init__(self, missing_inputs: Sequence[str]) -> None:
        self.missing_inputs = tuple(missing_inputs)
        super().__init__(
            "required workflow inputs are unavailable: " + ", ".join(self.missing_inputs)
        )


class PlanningUnsupportedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("workflow is unsupported by configured resources: " + "; ".join(reasons))


class WorkflowPlanner:
    """Build the reference DAG solely from typed inputs and frozen configuration."""

    def plan(self, *, run_id: str, input_state: InputState, config: WorkflowConfig) -> WorkflowPlan:
        missing = [
            name
            for name, state in (
                ("compounds", input_state.compounds),
                ("disease_genes", input_state.disease_genes),
            )
            if state is not InputAvailability.AVAILABLE
        ]
        if missing:
            raise PlanningBlockedError(missing)

        provided_target_states = {
            "drug_targets": input_state.drug_targets,
            "target_mapping": input_state.target_mapping,
        }
        provided_targets = all(
            value is InputAvailability.AVAILABLE for value in provided_target_states.values()
        )
        if any(
            value is InputAvailability.AVAILABLE for value in provided_target_states.values()
        ) and not provided_targets:
            raise PlanningBlockedError(
                [
                    name
                    for name, value in provided_target_states.items()
                    if value is not InputAvailability.AVAILABLE
                ]
            )

        compatibility_errors = core_compatibility_errors(config)
        if compatibility_errors:
            raise PlanningUnsupportedError(compatibility_errors)

        module_readiness = assess_module_readiness(
            config,
            disease_genes_available=True,
            expression_available=input_state.expression_pairs is InputAvailability.AVAILABLE,
        )
        enhanced = (
            input_state.expression_pairs is InputAvailability.AVAILABLE
            and module_readiness["gps"].status == "available"
        )
        configured_mode_name = "enhanced" if enhanced else "core"
        configured_mode = config.workflow.modes[configured_mode_name]
        mode = (
            f"provided_targets_{configured_mode_name}"
            if provided_targets
            else configured_mode_name
        )
        nodes: list[PlannedNode] = []

        def dependency(node_id: str, task_id: str = "main") -> DependencyRef:
            return DependencyRef(node_id=node_id, task_id=task_id)

        def add(
            node_id: str,
            stage: str,
            dependencies: Sequence[DependencyRef] = (),
            *,
            task_id: str = "main",
            status: WorkflowStatus = WorkflowStatus.PENDING,
            skip_reason: str | None = None,
            resource_class: str | None = None,
            parameters: dict[str, object] | None = None,
        ) -> None:
            nodes.append(
                PlannedNode(
                    node_id=node_id,
                    task_id=task_id,
                    stage=stage,
                    dependencies=tuple(dependencies),
                    initial_status=status,
                    skip_reason=skip_reason,
                    resource_class=resource_class or queue_for_node(node_id),
                    parameters=parameters or {},
                )
            )

        # The context already exists when the service plans, so this internal lifecycle node is
        # an auditable success rather than a second filesystem mutation.
        add("create_run_workspace", "preparation", status=WorkflowStatus.SUCCEEDED)
        add("register_inputs", "preparation", [dependency("create_run_workspace")])
        add("prepare_compound_library", "preparation", [dependency("register_inputs")])
        add("prepare_disease_genes", "preparation", [dependency("register_inputs")])

        gps_nodes = (
            "prepare_expression_inputs",
            "gps_predict_drug_profiles",
            "gps_build_disease_signature",
            "gps_score_compounds",
        )
        if enhanced:
            add("prepare_expression_inputs", "preparation", [dependency("register_inputs")])
            add("gps_predict_drug_profiles", "gps", [dependency("prepare_compound_library")])
            add(
                "gps_build_disease_signature",
                "gps",
                [dependency("prepare_expression_inputs"), dependency("gps_predict_drug_profiles")],
            )
            add(
                "gps_score_compounds",
                "gps",
                [
                    dependency("gps_predict_drug_profiles"),
                    dependency("gps_build_disease_signature"),
                ],
            )
        else:
            if module_readiness["gps"].status == "unsupported":
                reason = "GPS unsupported: " + module_readiness["gps"].reason
            else:
                reason = (
                    "expression inputs explicitly skipped by user"
                    if input_state.expression_pairs is InputAvailability.SKIPPED
                    else "expression inputs unavailable; core evidence mode selected"
                )
            for node_id in gps_nodes:
                add(
                    node_id,
                    "preparation" if node_id == "prepare_expression_inputs" else "gps",
                    status=WorkflowStatus.SKIPPED,
                    skip_reason=reason,
                )

        fanouts: tuple[FanoutPlan, ...] = ()
        if provided_targets:
            add(
                "import_drug_targets",
                "preparation",
                [dependency("register_inputs"), dependency("prepare_compound_library")],
                parameters={"target_source": "provided"},
            )
            reason = "NetInfer deferred; validated user-provided targets are used"
            for node_id in (
                "netinfer_prepare_inputs",
                "netinfer_predict_known",
                "netinfer_predict_batch",
                "netinfer_merge_targets",
            ):
                add(
                    node_id,
                    "netinfer",
                    status=WorkflowStatus.SKIPPED,
                    skip_reason=reason,
                )
            target_dependencies = [dependency("import_drug_targets")]
        else:
            add("netinfer_prepare_inputs", "netinfer", [dependency("prepare_compound_library")])
            add("netinfer_predict_known", "netinfer", [dependency("netinfer_prepare_inputs")])

            batch_ids = input_state.netinfer_batch_ids
            batch_dependencies: list[DependencyRef] = []
            if batch_ids is not None:
                for batch_id in batch_ids:
                    add(
                        "netinfer_predict_batch",
                        "netinfer",
                        [dependency("netinfer_prepare_inputs")],
                        task_id=batch_id,
                        parameters={"batch_id": batch_id},
                    )
                    batch_dependencies.append(dependency("netinfer_predict_batch", batch_id))
            else:
                # This wildcard is a persisted barrier, not an executable node. It becomes
                # satisfied only after prepare materializes the exact batch task set.
                batch_dependencies.append(dependency("netinfer_predict_batch", "*"))

            add(
                "netinfer_merge_targets",
                "netinfer",
                [
                    dependency("netinfer_prepare_inputs"),
                    dependency("netinfer_predict_known"),
                    *batch_dependencies,
                ],
            )
            target_dependencies = [
                dependency("netinfer_prepare_inputs"),
                dependency("netinfer_merge_targets"),
            ]
            fanouts = (
                FanoutPlan(
                    source_node_id="netinfer_prepare_inputs",
                    target_node_id="netinfer_predict_batch",
                    consumer_node_id="netinfer_merge_targets",
                    item_key="batch_id",
                    items=batch_ids,
                ),
            )
        add(
            "proximity_prepare_network",
            "proximity",
            [
                dependency("prepare_disease_genes"),
                *target_dependencies,
            ],
        )
        add("proximity_score_compounds", "proximity", [dependency("proximity_prepare_network")])
        add(
            "kg_construct_graph",
            "kg",
            [
                dependency("prepare_compound_library"),
                dependency("prepare_disease_genes"),
                *target_dependencies,
            ],
        )
        add("kg_prepare_training_data", "kg", [dependency("kg_construct_graph")])
        add(
            "kg_pretrain",
            "kg",
            [dependency("kg_prepare_training_data")],
        )
        seed_dependencies: list[DependencyRef] = []
        for seed in config.kg.finetune.seeds:
            task_id = f"seed-{seed}"
            add(
                "kg_finetune_seed",
                "kg",
                [dependency("kg_prepare_training_data"), dependency("kg_pretrain")],
                task_id=task_id,
                parameters={"seed": seed},
            )
            seed_dependencies.append(dependency("kg_finetune_seed", task_id))
        add("kg_aggregate_seeds", "kg", seed_dependencies)

        ranking_dependencies = [
            dependency("kg_aggregate_seeds"),
            dependency("proximity_score_compounds"),
        ]
        if enhanced:
            ranking_dependencies.append(dependency("gps_score_compounds"))
        add(
            "rank_candidates",
            "final",
            ranking_dependencies,
            parameters={"evidence_mode": configured_mode.evidence_mode},
        )
        add("generate_run_report", "final", [dependency("rank_candidates")])

        readiness = {name: item.to_dict() for name, item in module_readiness.items()}
        if provided_targets:
            readiness["netinfer"] = {
                "status": "skipped",
                "reason": "NetInfer deferred; validated user-provided targets are used",
            }
        return WorkflowPlan(
            run_id=run_id,
            workflow_id=config.workflow.id,
            workflow_version=config.workflow.version,
            mode=mode,
            evidence_mode=configured_mode.evidence_mode,
            nodes=nodes,
            fanouts=fanouts,
            module_readiness=readiness,
        )


__all__ = ["PlanningBlockedError", "PlanningUnsupportedError", "WorkflowPlanner"]
