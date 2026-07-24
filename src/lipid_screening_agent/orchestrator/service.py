"""Framework-neutral Workflow service API for Agent, Web, and queue workers."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.config import (
    WorkflowConfig,
    hash_workflow_config,
    parse_workflow_config,
)
from lipid_screening_agent.runtime import (
    RunContext,
    atomic_write_json,
    atomic_write_yaml,
    sha256_file,
)

from .cache import build_cache_key, cached_artifacts_are_valid
from .components import queue_for_node
from .eta import ETAEstimator, HistoricalETAEstimator
from .executors import LocalExecutor, QueueExecutor, QueueJob
from .models import (
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    InputAvailability,
    InputState,
    NodeRecord,
    RunnerOutcome,
    WorkflowStatus,
)
from .planner import PlanningBlockedError, PlanningUnsupportedError, WorkflowPlanner
from .registry import RunnerRegistry, RunnerRequest
from .store import WorkflowStore

_FAILURE_DEPENDENCY_STATUSES = {
    WorkflowStatus.FAILED,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.CANCELLED,
    WorkflowStatus.SKIPPED,
}


class WorkflowService:
    """Deterministic orchestration facade implementing create/plan/start/status/etc."""

    def __init__(
        self,
        *,
        store: WorkflowStore,
        registry: RunnerRegistry,
        executor: LocalExecutor | QueueExecutor,
        project_root: str | Path,
        resource_dir: str | Path | None = None,
        planner: WorkflowPlanner | None = None,
        eta_estimator: ETAEstimator | None = None,
        code_version: str = __version__,
        resource_hashes: dict[str, str] | None = None,
        recover_on_startup: bool = True,
    ) -> None:
        self.store = store
        self.registry = registry
        self.executor = executor
        self.project_root = Path(project_root).resolve()
        self.resource_dir = None if resource_dir is None else Path(resource_dir).resolve()
        self.planner = planner or WorkflowPlanner()
        self.eta_estimator = eta_estimator or HistoricalETAEstimator(store)
        self.code_version = code_version
        self.resource_hashes = dict(resource_hashes or {})
        self._contexts: dict[str, RunContext] = {}
        if recover_on_startup:
            self.recover_interrupted()

    # Public service API -------------------------------------------------
    def create(
        self,
        *,
        context: RunContext,
        input_state: InputState,
        config: WorkflowConfig,
        hardware_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        config_hash = hash_workflow_config(config)
        config_path = context.resolve_run_relative("workflow_config.yaml")
        atomic_write_yaml(config_path, config.to_dict(), allowed_root=context.run_dir)
        manifest_path = context.resolve_run_relative("run_manifest.json")
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(decoded, dict) or decoded.get("run_id") != context.run_id:
                raise ValueError("existing run manifest does not belong to this run")
            manifest.update(decoded)
        gps_compatible = config.disease.species in config.resources.gps.expression_supported_species
        enhanced = input_state.expression_pairs is InputAvailability.AVAILABLE and gps_compatible
        provided_targets = (
            input_state.drug_targets is InputAvailability.AVAILABLE
            and input_state.target_mapping is InputAvailability.AVAILABLE
        )
        decisions = dict(manifest.get("decisions", {}))
        decisions["target_source"] = (
            "provided" if provided_targets else "python_netinfer"
        )
        decisions["validation_profile"] = config.workflow.id.endswith("_validation")
        manifest.update(
            {
                "schema_version": "1.0",
                "run_id": context.run_id,
                "workflow_version": config.workflow.version,
                "config_hash": config_hash,
                "evidence_mode": "kg_proximity_gps" if enhanced else "kg_proximity",
                "disease": config.to_dict()["disease"],
                "created_at": manifest.get("created_at") or datetime.now().astimezone().isoformat(),
                "input_sources": manifest.get("input_sources", {}),
                "decisions": decisions,
            }
        )
        atomic_write_json(manifest_path, manifest, allowed_root=context.run_dir)
        self.store.create_run(
            run_id=context.run_id,
            run_dir=str(context.run_dir),
            workflow_id=config.workflow.id,
            workflow_version=config.workflow.version,
            config_hash=config_hash,
            config=config.to_dict(),
            input_state=input_state.to_dict(),
            hardware_fingerprint=hardware_fingerprint,
            input_scale=input_state.input_scale,
        )
        self._contexts[context.run_id] = context
        return self.status(context.run_id)

    def preview(
        self,
        *,
        run_id: str,
        input_state: InputState,
        config: WorkflowConfig,
    ) -> dict[str, Any]:
        """Build a deterministic plan without persisting or locking the run."""

        return self.planner.plan(
            run_id=run_id,
            input_state=input_state,
            config=config,
        ).to_dict()

    def plan(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run["plan"] is not None:
            return run["plan"]
        config = parse_workflow_config(run["config"])
        input_state = InputState.from_dict(run["input_state"])
        try:
            plan = self.planner.plan(run_id=run_id, input_state=input_state, config=config)
        except PlanningBlockedError as exc:
            self.store.set_run_status(
                run_id,
                WorkflowStatus.BLOCKED,
                event_type="planning_blocked",
                payload={"missing_inputs": list(exc.missing_inputs)},
            )
            raise
        except PlanningUnsupportedError as exc:
            self.store.set_run_status(
                run_id,
                WorkflowStatus.BLOCKED,
                event_type="planning_unsupported",
                payload={"reasons": list(exc.reasons)},
            )
            raise
        context = self._context(run_id)
        manifest_path = context.resolve_run_relative("run_manifest.json", must_exist=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["planning"] = {
            "mode": plan.mode,
            "evidence_mode": plan.evidence_mode,
            "skipped_nodes": [
                {
                    "node_id": node.node_id,
                    "task_id": node.task_id,
                    "reason": node.skip_reason,
                }
                for node in plan.nodes
                if node.initial_status is WorkflowStatus.SKIPPED
            ],
        }
        atomic_write_json(manifest_path, manifest, allowed_root=context.run_dir)
        self.store.save_plan(plan)
        self._refresh_readiness(run_id)
        return self.store.get_run(run_id)["plan"]

    def start(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if run["plan"] is None:
            self.plan(run_id)
        if run.get("cancel_requested"):
            return self.status(run_id)
        self.store.set_run_status(run_id, WorkflowStatus.RUNNING, event_type="run_started")
        self._refresh_readiness(run_id)
        if isinstance(self.executor, QueueExecutor):
            self._enqueue_ready(run_id)
            self._recompute_run_status(run_id)
            return self.status(run_id)

        while True:
            run = self.store.get_run(run_id)
            if run["cancel_requested"]:
                self._cancel_waiting_nodes(run_id)
            self._refresh_readiness(run_id)
            ready = [
                node
                for node in self.store.list_nodes(run_id)
                if node.status is WorkflowStatus.READY
            ]
            if not ready:
                break
            for node in ready:
                if self.store.get_run(run_id)["cancel_requested"]:
                    break
                if self._apply_cache_if_available(run_id, node):
                    continue
                self._execute_local(run_id, node)
        self._refresh_readiness(run_id)
        self._recompute_run_status(run_id)
        return self.status(run_id)

    def status(self, run_id: str, *, include_events: bool = False) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        nodes = self.store.list_nodes(run_id)
        eta = self.eta_estimator.estimate(
            nodes=nodes,
            hardware_fingerprint=run["hardware_fingerprint"],
            input_scale=run["input_scale"],
        )
        result = {
            "run_id": run_id,
            "status": run["status"],
            "mode": run["mode"],
            "evidence_mode": run["evidence_mode"],
            "cancel_requested": run["cancel_requested"],
            "created_at": run["created_at"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "plan": run["plan"],
            "nodes": [node.to_dict() for node in nodes],
            "eta": eta.to_dict(),
        }
        if include_events:
            result["events"] = self.store.events(run_id)
        return result

    def cancel(self, run_id: str) -> dict[str, Any]:
        current = WorkflowStatus(self.store.get_run(run_id)["status"])
        if current in {WorkflowStatus.SUCCEEDED, WorkflowStatus.CANCELLED}:
            return self.status(run_id)
        self.store.request_cancel(run_id)
        self.executor.cancel(run_id)
        self._cancel_waiting_nodes(run_id)
        self._recompute_run_status(run_id)
        return self.status(run_id)

    def retry(
        self,
        run_id: str,
        *,
        node_id: str | None = None,
        task_id: str | None = None,
        start: bool = False,
    ) -> dict[str, Any]:
        nodes = self.store.list_nodes(run_id)
        selected = [
            node
            for node in nodes
            if node.status is WorkflowStatus.FAILED
            and (node_id is None or node.node_id == node_id)
            and (task_id is None or node.task_id == task_id)
        ]
        if not selected:
            raise ValueError("no matching failed node tasks to retry")
        roots = {(node.node_id, node.task_id) for node in selected}
        affected = self.store.downstream(run_id, roots)
        records = {(node.node_id, node.task_id): node for node in nodes}
        for key in affected:
            record = records.get(key)
            if record is None:
                continue
            if key in roots:
                self.store.transition_node(
                    run_id,
                    record.node_id,
                    record.task_id,
                    WorkflowStatus.READY,
                    event_type="node_retry_scheduled",
                    payload={"previous_attempt": record.attempt},
                )
            elif record.status in TERMINAL_STATUSES and record.status is not WorkflowStatus.SKIPPED:
                self.store.transition_node(
                    run_id,
                    record.node_id,
                    record.task_id,
                    WorkflowStatus.PENDING,
                    invalidation=True,
                    event_type="downstream_invalidated",
                    payload={"retry_root": sorted(f"{a}:{b}" for a, b in roots)},
                )
        # A retry is an explicit new execution decision, so clear run-level cancellation.
        self._clear_cancel_request(run_id)
        self._refresh_readiness(run_id)
        self.store.set_run_status(run_id, WorkflowStatus.READY, event_type="run_retry_ready")
        return self.start(run_id) if start else self.status(run_id)

    def results(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        nodes = self.store.list_nodes(run_id)
        final_nodes = [node for node in nodes if node.stage == "final"]
        return {
            "run_id": run_id,
            "status": run["status"],
            "evidence_mode": run["evidence_mode"],
            "artifacts": [dict(artifact) for node in final_nodes for artifact in node.artifacts],
            "ranking": next(
                (node.to_dict() for node in final_nodes if node.node_id == "rank_candidates"),
                None,
            ),
            "report": next(
                (node.to_dict() for node in final_nodes if node.node_id == "generate_run_report"),
                None,
            ),
        }

    # Queue worker integration ------------------------------------------
    def claim_job(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        *,
        attempt: int | None = None,
        worker_id: str = "unknown-worker",
        queue: str = "cpu",
    ) -> dict[str, Any]:
        current = self.store.get_node(run_id, node_id, task_id)
        node = self.store.claim_node(
            run_id,
            node_id,
            task_id,
            attempt=current.attempt + 1 if attempt is None else attempt,
            worker_id=worker_id,
            queue=queue,
        )
        self.store.set_run_status(run_id, WorkflowStatus.RUNNING, event_type="worker_started")
        return self._runner_payload(run_id, node)

    def heartbeat(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        *,
        progress: float | None = None,
        attempt: int | None = None,
    ) -> None:
        node = self.store.get_node(run_id, node_id, task_id)
        if attempt is not None and node.attempt != attempt:
            raise ValueError("stale_attempt")
        self.store.heartbeat(run_id, node_id, task_id, progress=progress)

    def complete_job(
        self,
        run_id: str,
        node_id: str,
        task_id: str,
        outcome: RunnerOutcome,
        *,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        node = self.store.get_node(run_id, node_id, task_id)
        if attempt is not None and node.attempt != attempt:
            raise ValueError("stale_attempt")
        if node.status is WorkflowStatus.QUEUED:
            node = self.store.transition_node(
                run_id, node_id, task_id, WorkflowStatus.RUNNING, event_type="queue_job_claimed"
            )
            self.store.set_run_status(run_id, WorkflowStatus.RUNNING, event_type="worker_started")
        self._complete(run_id, node, outcome)
        self._refresh_readiness(run_id)
        if isinstance(self.executor, QueueExecutor):
            self._enqueue_ready(run_id)
        self._recompute_run_status(run_id)
        return self.status(run_id)

    def interrupt_job(
        self, run_id: str, node_id: str, task_id: str, *, attempt: int
    ) -> dict[str, Any]:
        node = self.store.get_node(run_id, node_id, task_id)
        if node.status is not WorkflowStatus.RUNNING or node.attempt != attempt:
            raise ValueError("stale_or_inactive_lease")
        outcome = RunnerOutcome(
            status=WorkflowStatus.FAILED,
            error={
                "category": "execution",
                "code": "interrupted",
                "message": "worker lease expired before completion",
                "exception_type": "WorkerLeaseExpired",
                "retryable": True,
                "details": {"attempt": attempt},
            },
        )
        return self.complete_job(run_id, node_id, task_id, outcome, attempt=attempt)

    def record_queue_rejection(
        self, run_id: str, node_id: str, task_id: str, *, reason: str, job_id: str
    ) -> None:
        with self.store.transaction() as connection:
            self.store._event(
                connection,
                run_id=run_id,
                node_id=node_id,
                task_id=task_id,
                event_type="queue_job_rejected",
                payload={"job_id": job_id, "reason": reason[:200]},
            )

    def materialize_fanout(
        self, run_id: str, target_node_id: str, items: list[str] | tuple[str, ...]
    ) -> list[str]:
        resource_class = queue_for_node(target_node_id)
        stage = "netinfer" if target_node_id == "netinfer_predict_batch" else "workflow"
        result = self.store.materialize_fanout(
            run_id,
            target_node_id,
            items,
            stage=stage,
            resource_class=resource_class,
        )
        self._refresh_readiness(run_id)
        return result

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        error = {
            "category": "execution",
            "code": "interrupted",
            "message": "worker stopped while the node was running",
            "exception_type": "InterruptedExecution",
            "retryable": True,
            "details": {},
        }
        for run in self.store.list_runs():
            recovered_this_run = False
            for node in self.store.list_nodes(run["run_id"]):
                if node.status is WorkflowStatus.RUNNING:
                    self.store.transition_node(
                        run["run_id"],
                        node.node_id,
                        node.task_id,
                        WorkflowStatus.FAILED,
                        event_type="node_recovered_as_interrupted",
                        error=error,
                    )
                    recovered.append(node.key)
                    recovered_this_run = True
            if recovered_this_run and run["status"] == WorkflowStatus.RUNNING.value:
                self.store.set_run_status(
                    run["run_id"], WorkflowStatus.FAILED, event_type="run_recovered_after_restart"
                )
        return recovered

    # Internal scheduling ------------------------------------------------
    def _context(self, run_id: str) -> RunContext:
        context = self._contexts.get(run_id)
        if context is not None:
            return context
        run = self.store.get_run(run_id)
        context = RunContext.open_existing(
            run_dir=run["run_dir"],
            project_root=self.project_root,
            resource_dir=self.resource_dir,
            create_missing_directories=True,
        )
        self._contexts[run_id] = context
        return context

    def _refresh_readiness(self, run_id: str) -> None:
        changed = True
        while changed:
            changed = False
            nodes = self.store.list_nodes(run_id)
            by_key = {(node.node_id, node.task_id): node for node in nodes}
            for node in nodes:
                if node.status is not WorkflowStatus.PENDING:
                    continue
                dependency_statuses: list[WorkflowStatus] = []
                waiting = False
                for dependency in node.dependencies:
                    if dependency.task_id != "*":
                        dependency_node = by_key.get((dependency.node_id, dependency.task_id))
                        if dependency_node is None:
                            waiting = True
                            continue
                        dependency_statuses.append(dependency_node.status)
                        continue
                    fanout = self.store.fanout_state(run_id, dependency.node_id)
                    if fanout is None or not fanout["resolved"]:
                        waiting = True
                        continue
                    for item in fanout["items"] or ():
                        dependency_node = by_key.get((dependency.node_id, item))
                        if dependency_node is None:
                            waiting = True
                            continue
                        dependency_statuses.append(dependency_node.status)
                failures = [
                    status
                    for status in dependency_statuses
                    if status in _FAILURE_DEPENDENCY_STATUSES
                ]
                if failures:
                    self.store.transition_node(
                        run_id,
                        node.node_id,
                        node.task_id,
                        WorkflowStatus.BLOCKED,
                        event_type="node_blocked_by_dependency",
                        error={
                            "category": "execution",
                            "code": "dependency_not_successful",
                            "message": "one or more required dependencies did not succeed",
                            "exception_type": "DependencyBlocked",
                            "retryable": True,
                            "details": {"statuses": [status.value for status in failures]},
                        },
                    )
                    changed = True
                elif waiting:
                    continue
                elif all(status in SUCCESS_STATUSES for status in dependency_statuses):
                    self.store.transition_node(
                        run_id,
                        node.node_id,
                        node.task_id,
                        WorkflowStatus.READY,
                        event_type="node_became_ready",
                    )
                    changed = True

    def _execute_local(self, run_id: str, node: NodeRecord) -> None:
        node = self.store.transition_node(
            run_id, node.node_id, node.task_id, WorkflowStatus.QUEUED, event_type="node_queued"
        )
        node = self.store.transition_node(
            run_id, node.node_id, node.task_id, WorkflowStatus.RUNNING, event_type="node_started"
        )
        try:
            registered = self.registry.get(node.node_id)
        except KeyError as exc:
            self._complete(
                run_id,
                node,
                RunnerOutcome(
                    status=WorkflowStatus.FAILED,
                    error={
                        "category": "configuration",
                        "code": "runner_not_registered",
                        "message": str(exc),
                        "exception_type": "RunnerRegistryError",
                        "retryable": False,
                        "details": {},
                    },
                ),
            )
            return
        payload = self._runner_request(run_id, node)
        outcome = self.executor.execute(
            registered,
            payload,
            heartbeat=lambda progress=None: self.store.heartbeat(
                run_id, node.node_id, node.task_id, progress=progress
            ),
        )
        self._complete(run_id, node, outcome)

    def _runner_request(self, run_id: str, node: NodeRecord) -> RunnerRequest:
        context = self._context(run_id)
        return RunnerRequest(
            context=context,
            node=node,
            config_path=context.resolve_run_relative("workflow_config.yaml", must_exist=True),
            input_artifacts=tuple(self._input_artifacts(run_id, node)),
            code_version=self.code_version,
            resource_hashes=self.resource_hashes,
        )

    def _runner_payload(self, run_id: str, node: NodeRecord) -> dict[str, Any]:
        request = self._runner_request(run_id, node)
        return {
            "run_id": run_id,
            "node_id": node.node_id,
            "task_id": node.task_id,
            "attempt": node.attempt,
            "run_dir": str(request.context.run_dir),
            "config_path": str(request.config_path),
            "resource_class": node.resource_class,
            "parameters": dict(node.parameters),
            "input_artifacts": [dict(item) for item in request.input_artifacts],
            "code_version": self.code_version,
            "resource_hashes": dict(self.resource_hashes),
        }

    def _complete(self, run_id: str, node: NodeRecord, outcome: RunnerOutcome) -> None:
        artifacts = self._enrich_artifacts(run_id, node, outcome.artifacts)
        cache_key = self._cache_key(run_id, node)
        persisted_metrics = dict(outcome.metrics)
        if outcome.fanout_items:
            persisted_metrics["_workflow_fanout_items"] = {
                name: list(items) for name, items in outcome.fanout_items.items()
            }
        updated = self.store.transition_node(
            run_id,
            node.node_id,
            node.task_id,
            outcome.status,
            event_type="node_completed",
            payload={"warnings": list(outcome.warnings)},
            error=outcome.error,
            artifacts=artifacts,
            metrics=persisted_metrics,
            warnings=outcome.warnings,
            cache_key=cache_key,
        )
        for target_node_id, items in outcome.fanout_items.items():
            self.materialize_fanout(run_id, target_node_id, list(items))
        if outcome.status is WorkflowStatus.SUCCEEDED and cache_key is not None:
            self.store.cache_put(
                node_id=node.node_id,
                task_id=node.task_id,
                cache_key=cache_key,
                artifacts=artifacts,
                metrics=persisted_metrics,
                source_run_id=run_id,
            )
        if outcome.status in SUCCESS_STATUSES and updated.started_at and updated.finished_at:
            duration = (
                datetime.fromisoformat(updated.finished_at)
                - datetime.fromisoformat(updated.started_at)
            ).total_seconds()
            run = self.store.get_run(run_id)
            self.store.add_history(
                node_id=node.node_id,
                task_id=node.task_id,
                duration_seconds=max(0.0, duration),
                hardware_fingerprint=run["hardware_fingerprint"],
                input_scale=run["input_scale"],
                metrics=persisted_metrics,
                status=outcome.status,
            )

    def _enrich_artifacts(
        self,
        run_id: str,
        node: NodeRecord,
        artifacts: tuple[dict[str, Any], ...] | Any,
    ) -> list[dict[str, Any]]:
        context = self._context(run_id)
        enriched: list[dict[str, Any]] = []
        for artifact in artifacts:
            value = dict(artifact)
            artifact_id = value.get("artifact_id")
            if artifact_id and "path" not in value:
                manifest = context.resolve_run_relative(
                    f"artifacts/manifests/{node.node_id}/{node.task_id}/{artifact_id}.json"
                )
                if manifest.is_file():
                    decoded = json.loads(manifest.read_text(encoding="utf-8"))
                    value.update(decoded)
                    value["path"] = str(
                        context.resolve_run_relative(decoded["relative_path"], must_exist=True)
                    )
            enriched.append(value)
        return enriched

    def _input_artifacts(self, run_id: str, node: NodeRecord) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for dependency in node.dependencies:
            if dependency.task_id == "*":
                fanout = self.store.fanout_state(run_id, dependency.node_id)
                task_ids = () if fanout is None else (fanout["items"] or ())
            else:
                task_ids = (dependency.task_id,)
            for task_id in task_ids:
                dependency_node = self.store.get_node(run_id, dependency.node_id, task_id)
                artifacts.extend(dict(item) for item in dependency_node.artifacts)
        return artifacts

    def _cache_key(self, run_id: str, node: NodeRecord) -> str | None:
        run = self.store.get_run(run_id)
        hashes = dict(run["input_state"].get("input_artifact_hashes", {}))
        for artifact in self._input_artifacts(run_id, node):
            digest = artifact.get("sha256")
            artifact_id = artifact.get("artifact_id")
            if artifact_id and digest is None:
                return None
            if artifact_id:
                hashes[str(artifact_id)] = str(digest)
        return build_cache_key(
            input_artifact_hashes=hashes,
            config_hash=run["config_hash"],
            code_version=self.code_version,
            resource_hashes=self.resource_hashes,
            parameters=node.parameters,
        )

    def _apply_cache_if_available(self, run_id: str, node: NodeRecord) -> bool:
        cache_key = self._cache_key(run_id, node)
        if cache_key is None:
            return False
        cached = self.store.cache_get(node.node_id, node.task_id, cache_key)
        if cached is None or not cached_artifacts_are_valid(cached["artifacts"]):
            return False
        materialized = self._materialize_cached_artifacts(
            run_id,
            node,
            cached["artifacts"],
            source_run_id=cached["source_run_id"],
        )
        self.store.transition_node(
            run_id,
            node.node_id,
            node.task_id,
            WorkflowStatus.CACHED,
            event_type="node_cache_hit",
            payload={"source_run_id": cached["source_run_id"]},
            artifacts=materialized,
            metrics=cached["metrics"],
            cache_key=cache_key,
        )
        for target_node_id, items in cached["metrics"].get("_workflow_fanout_items", {}).items():
            self.materialize_fanout(run_id, target_node_id, list(items))
        return True

    def _materialize_cached_artifacts(
        self,
        run_id: str,
        node: NodeRecord,
        artifacts: list[dict[str, Any]],
        *,
        source_run_id: str,
    ) -> list[dict[str, Any]]:
        """Place concrete cache hits inside the destination run workspace.

        Hard links avoid copying large immutable artifacts on a shared volume. A safe copy fallback
        handles filesystems that do not support links. Pathless fake artifacts remain metadata-only.
        """

        context = self._context(run_id)
        destination_root = context.resolve_run_relative(
            f"artifacts/cache/{node.node_id}/{node.task_id}"
        )
        destination_root.mkdir(parents=True, exist_ok=True)
        materialized: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            value = dict(artifact)
            source_value = value.get("path")
            if source_value is None:
                materialized.append(value)
                continue
            source = Path(str(source_value)).resolve()
            artifact_id = str(value.get("artifact_id", f"artifact-{index}"))
            destination = destination_root / f"{artifact_id}-{source.name}"
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    temporary = destination.with_name(destination.name + ".tmp")
                    shutil.copy2(source, temporary)
                    temporary.replace(destination)
            expected = value.get("sha256")
            if expected is not None and sha256_file(destination) != expected:
                raise ValueError(f"materialized cache artifact hash mismatch: {artifact_id}")
            value.update(
                path=str(destination),
                relative_path=context.relative_path(destination),
                cached_from_run_id=source_run_id,
            )
            materialized.append(value)
            manifest_fields = {
                "artifact_id",
                "artifact_type",
                "relative_path",
                "size_bytes",
                "sha256",
                "created_at",
                "producer_node_id",
                "producer_task_id",
                "input_artifact_ids",
                "config_hash",
                "code_version",
                "resource_hashes",
            }
            if manifest_fields <= value.keys():
                manifest_path = context.resolve_run_relative(
                    f"artifacts/manifests/{node.node_id}/{node.task_id}/{artifact_id}.json"
                )
                atomic_write_json(
                    manifest_path,
                    {field: value[field] for field in manifest_fields},
                    allowed_root=context.run_dir,
                )
        return materialized

    def _enqueue_ready(self, run_id: str) -> None:
        assert isinstance(self.executor, QueueExecutor)
        # A cache hit is terminal and can make one or more downstream nodes
        # ready immediately.  Continue refreshing until every newly exposed
        # ready node is either cached or placed on its external queue.
        while True:
            self._refresh_readiness(run_id)
            ready = [
                node
                for node in self.store.list_nodes(run_id)
                if node.status is WorkflowStatus.READY
            ]
            if not ready:
                return
            for node in ready:
                if self._apply_cache_if_available(run_id, node):
                    continue
                queued = self.store.transition_node(
                    run_id,
                    node.node_id,
                    node.task_id,
                    WorkflowStatus.QUEUED,
                    event_type="node_queued",
                )
                payload = self._runner_payload(run_id, queued)
                payload["attempt"] = queued.attempt + 1
                job = QueueJob(
                    run_id=run_id,
                    node_id=queued.node_id,
                    task_id=queued.task_id,
                    attempt=queued.attempt + 1,
                    resource_class=queued.resource_class,
                    payload=payload,
                )
                queue_job_id = self.executor.enqueue(job)
                # Queue ID is captured in the event stream without inventing another node state.
                with self.store.transaction() as connection:
                    self.store._event(
                        connection,
                        run_id=run_id,
                        node_id=queued.node_id,
                        task_id=queued.task_id,
                        event_type="external_job_enqueued",
                        payload={"queue_job_id": queue_job_id},
                    )

    def _cancel_waiting_nodes(self, run_id: str) -> None:
        for node in self.store.list_nodes(run_id):
            if node.status in {WorkflowStatus.PENDING, WorkflowStatus.READY, WorkflowStatus.QUEUED}:
                self.store.transition_node(
                    run_id,
                    node.node_id,
                    node.task_id,
                    WorkflowStatus.CANCELLED,
                    event_type="node_cancelled",
                )

    def _clear_cancel_request(self, run_id: str) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                "UPDATE runs SET cancel_requested=0, version=version+1 WHERE run_id=?", (run_id,)
            )
            self.store._event(connection, run_id=run_id, event_type="cancel_request_cleared")

    def _recompute_run_status(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        nodes = self.store.list_nodes(run_id)
        statuses = {node.status for node in nodes}
        if run["cancel_requested"] and not statuses.intersection({WorkflowStatus.RUNNING}):
            target = WorkflowStatus.CANCELLED
        # A failed branch does not make the whole run terminal while independent
        # work is already running or queued.  Keeping the run active lets those
        # workers finish and preserves their queue messages for a scoped retry.
        elif WorkflowStatus.RUNNING in statuses:
            target = WorkflowStatus.RUNNING
        elif WorkflowStatus.QUEUED in statuses:
            target = WorkflowStatus.QUEUED
        elif WorkflowStatus.FAILED in statuses:
            target = WorkflowStatus.FAILED
        elif all(status in TERMINAL_STATUSES for status in statuses):
            target = (
                WorkflowStatus.BLOCKED
                if WorkflowStatus.BLOCKED in statuses
                else WorkflowStatus.SUCCEEDED
            )
        elif WorkflowStatus.READY in statuses:
            target = WorkflowStatus.READY
        else:
            target = WorkflowStatus.BLOCKED
        self.store.set_run_status(run_id, target)


__all__ = ["WorkflowService"]
