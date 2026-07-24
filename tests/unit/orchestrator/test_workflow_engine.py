from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

import pytest

from lipid_screening_agent.config import load_workflow_config
from lipid_screening_agent.orchestrator import (
    InMemoryQueueExecutor,
    InputAvailability,
    InputState,
    InvalidStateTransition,
    LocalExecutor,
    RunnerOutcome,
    RunnerRegistry,
    WorkflowPlanner,
    WorkflowService,
    WorkflowStatus,
    WorkflowStore,
    build_cache_key,
)
from lipid_screening_agent.orchestrator.executors import CancellationToken
from lipid_screening_agent.orchestrator.registry import RUNNER_MODULES
from lipid_screening_agent.runtime import RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "workflow.yaml"


def test_subprocess_executor_drains_large_stderr_without_pipe_deadlock():
    executor = LocalExecutor(heartbeat_interval_seconds=0.01)
    payload = (
        "import json,sys; "
        "sys.stderr.write('warning\\n' * 200000); sys.stderr.flush(); "
        "print(json.dumps({'status':'succeeded','outputs':[],"
        "'metrics':{'large_stderr_drained':True},'warnings':[],"
        "'fanout_items':{'netinfer_predict_batch':['batch_0001']}}))"
    )
    outcome = executor._run_subprocess(
        (sys.executable, "-c", payload),
        cwd=None,
        env=None,
        timeout=10,
        token=CancellationToken(),
        external_cancel=lambda: False,
        heartbeat=lambda _progress=None: None,
    )

    assert outcome.status is WorkflowStatus.SUCCEEDED
    assert outcome.metrics["large_stderr_drained"] is True
    assert outcome.fanout_items == {"netinfer_predict_batch": ("batch_0001",)}


class FakeRunners:
    def __init__(self) -> None:
        self.calls: Counter[tuple[str, str]] = Counter()
        self.fail_once: set[tuple[str, str]] = set()
        self.block_once: set[tuple[str, str]] = set()
        self.dynamic_batches: tuple[str, ...] | None = None

    def run(self, request):
        key = (request.node.node_id, request.node.task_id)
        self.calls[key] += 1
        request.heartbeat(0.5)
        if key in self.block_once and self.calls[key] == 1:
            return RunnerOutcome(status="blocked", warnings=("intentional fake block",))
        if key in self.fail_once and self.calls[key] == 1:
            return RunnerOutcome(
                status="failed",
                error={
                    "category": "execution",
                    "code": "fake_failure",
                    "message": f"intentional failure for {request.node.key}",
                    "exception_type": "FakeRunnerError",
                    "retryable": True,
                    "details": {},
                },
            )
        digest = hashlib.sha256(
            f"{request.node.node_id}:{request.node.task_id}".encode()
        ).hexdigest()
        fanout_items = {}
        if request.node.node_id == "netinfer_prepare_inputs" and self.dynamic_batches is not None:
            fanout_items = {"netinfer_predict_batch": self.dynamic_batches}
        return RunnerOutcome(
            status="succeeded",
            artifacts=(
                {
                    "artifact_id": f"artifact-{digest[:20]}",
                    "artifact_type": f"fake_{request.node.node_id}",
                    "sha256": digest,
                },
            ),
            metrics={"input_count": 10},
            fanout_items=fanout_items,
        )


@pytest.fixture
def config():
    return load_workflow_config(CONFIG_PATH)


def make_context(tmp_path: Path, run_id: str) -> tuple[RunContext, Path]:
    resources = tmp_path / "resources"
    resources.mkdir(exist_ok=True)
    context = RunContext.create(
        runs_root=tmp_path / "runs",
        run_id=run_id,
        project_root=PROJECT_ROOT,
        resource_dir=resources,
    )
    return context, resources


def make_registry(fake: FakeRunners) -> RunnerRegistry:
    registry = RunnerRegistry()
    for node_id in RUNNER_MODULES:
        registry.register_callable(node_id, fake.run, timeout_seconds=5)
    return registry


def make_service(tmp_path: Path, fake: FakeRunners, resources: Path) -> WorkflowService:
    return WorkflowService(
        store=WorkflowStore(tmp_path / "workflow.sqlite3"),
        registry=make_registry(fake),
        executor=LocalExecutor(heartbeat_interval_seconds=0.01),
        project_root=PROJECT_ROOT,
        resource_dir=resources,
        code_version="test-code-v1",
        resource_hashes={"resources": "a" * 64},
    )


def input_state(*, enhanced=False, batches=("batch-0001", "batch-0002"), digest="b"):
    return InputState(
        compounds="available",
        disease_genes="available",
        expression_pairs="available" if enhanced else "skipped",
        netinfer_batch_ids=batches,
        input_artifact_hashes={"raw-inputs": digest * 64},
        input_scale={"compound_count": 10, "disease_gene_count": 3},
    )


def node_map(status: dict):
    return {(node["node_id"], node["task_id"]): node for node in status["nodes"]}


def test_planner_builds_core_skips_gps_and_fans_out_seeds(config):
    plan = WorkflowPlanner().plan(run_id="core-plan", input_state=input_state(), config=config)
    nodes = {(node.node_id, node.task_id): node for node in plan.nodes}
    assert plan.mode == "core"
    assert plan.evidence_mode == "kg_proximity"
    assert all(
        nodes[(node_id, "main")].initial_status is WorkflowStatus.SKIPPED
        for node_id in (
            "prepare_expression_inputs",
            "gps_predict_drug_profiles",
            "gps_build_disease_signature",
            "gps_score_compounds",
        )
    )
    assert {task_id for node_id, task_id in nodes if node_id == "kg_finetune_seed"} == {
        "seed-5",
        "seed-6",
        "seed-7",
        "seed-8",
        "seed-9",
    }
    rank_dependencies = {item.node_id for item in nodes[("rank_candidates", "main")].dependencies}
    assert rank_dependencies == {"kg_aggregate_seeds", "proximity_score_compounds"}


def test_planner_builds_enhanced_gps_dependencies(config):
    plan = WorkflowPlanner().plan(
        run_id="enhanced-plan", input_state=input_state(enhanced=True), config=config
    )
    nodes = {(node.node_id, node.task_id): node for node in plan.nodes}
    assert plan.evidence_mode == "kg_proximity_gps"
    assert nodes[("gps_score_compounds", "main")].initial_status is WorkflowStatus.PENDING
    assert "gps_score_compounds" in {
        item.node_id for item in nodes[("rank_candidates", "main")].dependencies
    }


def test_dynamic_netinfer_fanout_and_fanin(tmp_path, config):
    context, resources = make_context(tmp_path, "dynamic-batches")
    fake = FakeRunners()
    fake.dynamic_batches = ("batch-a", "batch-b")
    service = make_service(tmp_path, fake, resources)
    service.create(
        context=context,
        input_state=input_state(batches=None),
        config=config,
        hardware_fingerprint="test-gpu",
    )
    service.plan(context.run_id)
    status = service.start(context.run_id)
    nodes = node_map(status)
    assert status["status"] == "succeeded"
    assert nodes[("netinfer_predict_batch", "batch-a")]["status"] == "succeeded"
    assert nodes[("netinfer_predict_batch", "batch-b")]["status"] == "succeeded"
    assert fake.calls[("netinfer_merge_targets", "main")] == 1


def test_seed_failure_retries_only_failed_seed_and_invalid_downstream(tmp_path, config):
    context, resources = make_context(tmp_path, "seed-retry")
    fake = FakeRunners()
    fake.fail_once.add(("kg_finetune_seed", "seed-7"))
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    failed = service.start(context.run_id)
    failed_nodes = node_map(failed)
    assert failed["status"] == "failed"
    assert failed_nodes[("kg_finetune_seed", "seed-7")]["status"] == "failed"
    assert failed_nodes[("kg_aggregate_seeds", "main")]["status"] == "blocked"

    completed = service.retry(
        context.run_id,
        node_id="kg_finetune_seed",
        task_id="seed-7",
        start=True,
    )
    completed_nodes = node_map(completed)
    assert completed["status"] == "succeeded"
    assert completed_nodes[("kg_finetune_seed", "seed-7")]["attempt"] == 2
    for seed in (5, 6, 8, 9):
        assert fake.calls[("kg_finetune_seed", f"seed-{seed}")] == 1
    assert fake.calls[("kg_finetune_seed", "seed-7")] == 2


def test_batch_failure_retries_only_failed_batch(tmp_path, config):
    context, resources = make_context(tmp_path, "batch-retry")
    fake = FakeRunners()
    fake.fail_once.add(("netinfer_predict_batch", "batch-0002"))
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    assert service.start(context.run_id)["status"] == "failed"
    completed = service.retry(
        context.run_id,
        node_id="netinfer_predict_batch",
        task_id="batch-0002",
        start=True,
    )
    assert completed["status"] == "succeeded"
    assert fake.calls[("netinfer_predict_batch", "batch-0001")] == 1
    assert fake.calls[("netinfer_predict_batch", "batch-0002")] == 2


def test_planned_gps_failure_blocks_ranking_without_downgrade(tmp_path, config):
    context, resources = make_context(tmp_path, "gps-failure")
    fake = FakeRunners()
    fake.fail_once.add(("gps_score_compounds", "main"))
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(enhanced=True), config=config)
    service.plan(context.run_id)
    status = service.start(context.run_id)
    nodes = node_map(status)
    assert status["status"] == "failed"
    assert status["evidence_mode"] == "kg_proximity_gps"
    assert nodes[("rank_candidates", "main")]["status"] == "blocked"
    assert fake.calls[("rank_candidates", "main")] == 0


def test_planned_gps_block_propagates_without_downgrade(tmp_path, config):
    context, resources = make_context(tmp_path, "gps-blocked")
    fake = FakeRunners()
    fake.block_once.add(("gps_build_disease_signature", "main"))
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(enhanced=True), config=config)
    service.plan(context.run_id)
    status = service.start(context.run_id)
    nodes = node_map(status)
    assert status["status"] == "blocked"
    assert status["evidence_mode"] == "kg_proximity_gps"
    assert nodes[("gps_build_disease_signature", "main")]["status"] == "blocked"
    assert nodes[("gps_score_compounds", "main")]["status"] == "blocked"
    assert nodes[("rank_candidates", "main")]["status"] == "blocked"


def test_cache_hit_and_input_hash_invalidation(tmp_path, config):
    first_context, resources = make_context(tmp_path, "cache-first")
    fake = FakeRunners()
    service = make_service(tmp_path, fake, resources)
    service.create(context=first_context, input_state=input_state(), config=config)
    service.plan(first_context.run_id)
    assert service.start(first_context.run_id)["status"] == "succeeded"

    second_context, _ = make_context(tmp_path, "cache-second")
    service.create(context=second_context, input_state=input_state(), config=config)
    service.plan(second_context.run_id)
    second = service.start(second_context.run_id)
    assert node_map(second)[("rank_candidates", "main")]["status"] == "cached"

    third_context, _ = make_context(tmp_path, "cache-invalidated")
    service.create(
        context=third_context,
        input_state=input_state(digest="c"),
        config=config,
    )
    service.plan(third_context.run_id)
    third = service.start(third_context.run_id)
    assert node_map(third)[("register_inputs", "main")]["status"] == "succeeded"
    assert fake.calls[("register_inputs", "main")] == 2


def test_queue_executor_continues_scheduling_after_cache_hit(tmp_path, config):
    first_context, resources = make_context(tmp_path, "queue-cache-source")
    fake = FakeRunners()
    local_service = make_service(tmp_path, fake, resources)
    local_service.create(context=first_context, input_state=input_state(), config=config)
    local_service.plan(first_context.run_id)
    assert local_service.start(first_context.run_id)["status"] == "succeeded"

    queue = InMemoryQueueExecutor()
    queue_service = WorkflowService(
        store=local_service.store,
        registry=make_registry(fake),
        executor=queue,
        project_root=PROJECT_ROOT,
        resource_dir=resources,
        code_version="test-code-v1",
        resource_hashes={"resources": "a" * 64},
        recover_on_startup=False,
    )
    second_context, _ = make_context(tmp_path, "queue-cache-destination")
    queue_service.create(context=second_context, input_state=input_state(), config=config)
    queue_service.plan(second_context.run_id)
    queued = queue_service.start(second_context.run_id)

    assert queued["status"] == "succeeded"
    assert node_map(queued)[("register_inputs", "main")]["status"] == "cached"
    assert queue.jobs == []


def test_restart_recovers_running_as_retryable_interrupted(tmp_path, config):
    context, resources = make_context(tmp_path, "restart")
    fake = FakeRunners()
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    service.store.set_run_status(context.run_id, WorkflowStatus.RUNNING)
    service.store.transition_node(context.run_id, "register_inputs", "main", WorkflowStatus.QUEUED)
    service.store.transition_node(context.run_id, "register_inputs", "main", WorkflowStatus.RUNNING)

    restarted = make_service(tmp_path, fake, resources)
    recovered = restarted.store.get_node(context.run_id, "register_inputs")
    assert recovered.status is WorkflowStatus.FAILED
    assert recovered.error["code"] == "interrupted"
    completed = restarted.retry(
        context.run_id, node_id="register_inputs", task_id="main", start=True
    )
    assert completed["status"] == "succeeded"
    assert node_map(completed)[("register_inputs", "main")]["attempt"] == 2


def test_queue_cancel_and_illegal_transition(tmp_path, config):
    context, resources = make_context(tmp_path, "cancel")
    fake = FakeRunners()
    queue = InMemoryQueueExecutor()
    store = WorkflowStore(tmp_path / "workflow.sqlite3")
    service = WorkflowService(
        store=store,
        registry=make_registry(fake),
        executor=queue,
        project_root=PROJECT_ROOT,
        resource_dir=resources,
    )
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    queued = service.start(context.run_id)
    assert queued["status"] == "queued"
    assert queue.jobs
    cancelled = service.cancel(context.run_id)
    assert cancelled["status"] == "cancelled"
    assert all(
        node["status"] in {"succeeded", "skipped", "cancelled"} for node in cancelled["nodes"]
    )

    succeeded = store.get_node(context.run_id, "create_run_workspace")
    assert succeeded.status is WorkflowStatus.SUCCEEDED
    with pytest.raises(InvalidStateTransition):
        store.transition_node(
            context.run_id,
            "create_run_workspace",
            "main",
            WorkflowStatus.RUNNING,
        )


def test_event_log_records_every_transition(tmp_path, config):
    context, resources = make_context(tmp_path, "events")
    fake = FakeRunners()
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    service.start(context.run_id)
    events = service.status(context.run_id, include_events=True)["events"]
    assert events[0]["event_type"] == "run_created"
    assert any(event["event_type"] == "node_started" for event in events)
    assert any(event["event_type"] == "node_completed" for event in events)
    assert all(event["created_at"] for event in events)


def test_cache_key_invalidates_every_provenance_dimension():
    baseline = {
        "input_artifact_hashes": {"input": "1" * 64},
        "config_hash": "2" * 64,
        "code_version": "code-a",
        "resource_hashes": {"ppi": "3" * 64},
        "parameters": {"seed": 7},
    }
    original = build_cache_key(**baseline)
    variants = (
        {**baseline, "input_artifact_hashes": {"input": "4" * 64}},
        {**baseline, "config_hash": "5" * 64},
        {**baseline, "code_version": "code-b"},
        {**baseline, "resource_hashes": {"ppi": "6" * 64}},
        {**baseline, "parameters": {"seed": 8}},
    )
    assert all(build_cache_key(**variant) != original for variant in variants)


def test_eta_is_unknown_without_history(tmp_path, config):
    context, resources = make_context(tmp_path, "eta-unknown")
    fake = FakeRunners()
    service = make_service(tmp_path, fake, resources)
    service.create(context=context, input_state=input_state(), config=config)
    service.plan(context.run_id)
    assert service.status(context.run_id)["eta"]["status"] == "unknown"


def test_eta_uses_matching_hardware_history_and_input_scale(tmp_path, config):
    first_context, resources = make_context(tmp_path, "eta-history-first")
    fake = FakeRunners()
    service = make_service(tmp_path, fake, resources)
    service.create(
        context=first_context,
        input_state=input_state(),
        config=config,
        hardware_fingerprint="gpu-a",
    )
    service.plan(first_context.run_id)
    service.start(first_context.run_id)

    second_context, _ = make_context(tmp_path, "eta-history-second")
    larger = input_state(digest="d")
    larger = InputState(
        **{
            **larger.to_dict(),
            "input_scale": {"compound_count": 20, "disease_gene_count": 6},
        }
    )
    service.create(
        context=second_context,
        input_state=larger,
        config=config,
        hardware_fingerprint="gpu-a",
    )
    service.plan(second_context.run_id)
    eta = service.status(second_context.run_id)["eta"]
    assert eta["status"] == "estimated"
    assert "matching_hardware_history" in eta["basis"]
    assert "input_scale_adjusted" in eta["basis"]


def test_missing_required_input_blocks_planning(config):
    with pytest.raises(Exception) as error:
        WorkflowPlanner().plan(
            run_id="missing",
            input_state=InputState(
                compounds=InputAvailability.MISSING,
                disease_genes=InputAvailability.AVAILABLE,
            ),
            config=config,
        )
    assert "compounds" in str(error.value)
