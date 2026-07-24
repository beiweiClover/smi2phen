from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from lipid_screening_agent.backend.runtime import Runtime, build_runtime
from lipid_screening_agent.backend.service import ScreeningService
from lipid_screening_agent.backend.store import V3Store
from lipid_screening_agent.backend.tool_loop import (
    SYSTEM_PROMPT,
    ModelReply,
    ToolCall,
    ToolCallingAgent,
    ToolDispatcher,
)
from lipid_screening_agent.backend.uploads import UploadService
from lipid_screening_agent.config import load_workflow_config
from lipid_screening_agent.orchestrator import (
    InMemoryQueueExecutor,
    WorkflowService,
    WorkflowStore,
    default_runner_registry,
    project_source_digest,
)
from lipid_screening_agent.runtime.context import RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ScriptedModel:
    def __init__(self, replies: list[ModelReply]) -> None:
        self.replies = replies
        self.calls = 0

    def complete(self, *, messages, tools):
        assert messages[0]["role"] == "system"
        assert len(tools) == 7
        reply = self.replies[self.calls]
        self.calls += 1
        return reply


def make_runtime(tmp_path: Path, *, model=None):
    database = tmp_path / "state.sqlite3"
    runs_root = tmp_path / "runs"
    resources = tmp_path / "resources"
    runs_root.mkdir()
    resources.mkdir()
    store = V3Store(database)
    queue = InMemoryQueueExecutor()
    workflow = WorkflowService(
        store=WorkflowStore(database),
        registry=default_runner_registry(),
        executor=queue,
        project_root=PROJECT_ROOT,
        resource_dir=resources,
        code_version="v3-test",
        recover_on_startup=False,
    )
    service = ScreeningService(
        store=store,
        workflow=workflow,
        workflow_config=load_workflow_config(PROJECT_ROOT / "configs" / "workflow.yaml"),
        runs_root=runs_root,
        project_root=PROJECT_ROOT,
        resource_root=resources,
    )
    uploads = UploadService(store=store, runs_root=runs_root)
    agent = (
        None
        if model is None
        else ToolCallingAgent(model=model, store=store, dispatcher=ToolDispatcher(service))
    )
    return Runtime(service, uploads, agent, store, model is not None), queue


def add_target_inputs(
    runtime: Runtime, run_id: str, *, compound_id: str = "A", smiles: str = "CCO"
) -> None:
    runtime.uploads.save(
        run_id=run_id,
        kind="drug_targets",
        original_name="drug_targets.json",
        stream=BytesIO(
            json.dumps(
                {
                    compound_id: {
                        "smiles": smiles,
                        "targets": [
                            {
                                "gene_symbol": "TP53",
                                "uniprot_id": "P04637",
                                "evidence": "known",
                            }
                        ],
                    }
                }
            ).encode()
        ),
    )
    runtime.uploads.save(
        run_id=run_id,
        kind="target_mapping",
        original_name="target_mapping.tsv",
        stream=BytesIO(b"gene_symbol\tentrez_id\nTP53\t7157\n"),
    )


def add_required_inputs(runtime: Runtime, run_id: str) -> None:
    runtime.uploads.save(
        run_id=run_id,
        kind="compounds",
        original_name="library.csv",
        stream=BytesIO(b"ID,SMILES\nA,CCO\n"),
    )
    runtime.uploads.save(
        run_id=run_id,
        kind="disease_genes",
        original_name="genes.txt",
        stream=BytesIO(b"TP53\n"),
    )
    add_target_inputs(runtime, run_id)


def test_target_pair_is_optional_and_selects_python_netinfer(tmp_path: Path) -> None:
    runtime, _ = make_runtime(tmp_path)
    session = runtime.service.create_session("thread-netinfer")
    run_id = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    runtime.uploads.save(
        run_id=run_id,
        kind="compounds",
        original_name="library.csv",
        stream=BytesIO(b"ID,SMILES\nA,CCO\n"),
    )
    runtime.uploads.save(
        run_id=run_id,
        kind="disease_genes",
        original_name="genes.txt",
        stream=BytesIO(b"TP53\n"),
    )

    requirements = runtime.service.requirements(run_id)
    assert requirements["ready"] is True
    assert requirements["target_source"] == "python_netinfer"
    assert requirements["inputs"]["target_pair"] == "generated"
    plan = runtime.service.preview_plan(run_id)
    assert plan["mode"] == "core"
    assert all(
        node["initial_status"] == "pending"
        for node in plan["nodes"]
        if node["node_id"].startswith("netinfer_")
    )


def test_uploads_match_formal_workflow_paths_and_start_is_explicit(tmp_path: Path):
    runtime, queue = make_runtime(tmp_path)
    session = runtime.service.create_session("thread-1")
    run = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )
    run_id = run["run_id"]
    add_required_inputs(runtime, run_id)

    assert (tmp_path / "runs" / run_id / "inputs" / "compounds.csv").is_file()
    assert (tmp_path / "runs" / run_id / "inputs" / "disease_genes.tsv").is_file()
    assert runtime.service.requirements(run_id)["ready"] is True

    plan = runtime.service.preview_plan(run_id)
    assert plan["evidence_mode"] == "kg_proximity"
    assert plan["mode"] == "provided_targets_core"
    assert any(node["node_id"] == "register_inputs" for node in plan["nodes"])
    assert any(node["node_id"] == "import_drug_targets" for node in plan["nodes"])
    assert all(
        node["initial_status"] == "skipped"
        for node in plan["nodes"]
        if node["node_id"].startswith("netinfer_")
    )
    stored = runtime.store.run(run_id)
    assert stored["plan_previewed"] is True
    assert stored["workflow_created"] is False
    assert runtime.service.requirements(run_id)["inputs_locked"] is False
    assert not (tmp_path / "runs" / run_id / "workflow_config.yaml").exists()
    assert queue.jobs == []
    with pytest.raises(ValueError, match="confirmation"):
        runtime.service.start(run_id, confirmed=False)

    status = runtime.service.start(run_id, confirmed=True)
    assert status["status"] == "queued"
    assert queue.jobs[0][1].node_id == "register_inputs"
    assert runtime.store.run(run_id)["workflow_created"] is True
    assert runtime.service.requirements(run_id)["inputs_locked"] is True
    manifest = json.loads(
        (tmp_path / "runs" / run_id / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["planning"]["mode"] == "provided_targets_core"
    assert {
        item["node_id"] for item in manifest["planning"]["skipped_nodes"]
    } >= {
        "netinfer_prepare_inputs",
        "netinfer_predict_known",
        "netinfer_predict_batch",
        "netinfer_merge_targets",
    }


def test_replacement_is_explicit_and_inputs_lock_only_after_start(tmp_path: Path):
    runtime, _ = make_runtime(tmp_path)
    session = runtime.service.create_session("thread-replace")
    run_id = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    runtime.uploads.save(
        run_id=run_id,
        kind="compounds",
        original_name="first.csv",
        stream=BytesIO(b"ID,SMILES\nA,CCO\n"),
    )
    with pytest.raises(ValueError, match="replace=true"):
        runtime.uploads.save(
            run_id=run_id,
            kind="compounds",
            original_name="second.csv",
            stream=BytesIO(b"ID,SMILES\nB,CCC\n"),
        )
    runtime.uploads.save(
        run_id=run_id,
        kind="compounds",
        original_name="second.csv",
        stream=BytesIO(b"ID,SMILES\nB,CCC\n"),
        replace=True,
    )
    runtime.uploads.save(
        run_id=run_id,
        kind="disease_genes",
        original_name="genes.tsv",
        stream=BytesIO(b"TP53\n"),
    )
    add_target_inputs(runtime, run_id, compound_id="B", smiles="CCC")
    runtime.service.preview_plan(run_id)
    assert runtime.store.run(run_id)["plan_previewed"] is True
    runtime.uploads.save(
        run_id=run_id,
        kind="positive_drugs",
        original_name="positive_drugs.tsv",
        stream=BytesIO(b"input_type\tvalue\nlibrary_id\tB\n"),
    )
    assert runtime.store.run(run_id)["plan_previewed"] is False
    assert runtime.store.run(run_id)["workflow_created"] is False
    runtime.service.preview_plan(run_id)
    runtime.service.start(run_id, confirmed=True)
    with pytest.raises(ValueError, match="locked"):
        runtime.uploads.save(
            run_id=run_id,
            kind="disease_links",
            original_name="disease_links.tsv",
            stream=BytesIO(b"input_type\tvalue\nbase_disease_name\tNAFLD\n"),
        )


def test_legacy_persisted_preview_is_invalidated_by_supplementary_upload(
    tmp_path: Path,
) -> None:
    runtime, _ = make_runtime(tmp_path)
    session = runtime.service.create_session("legacy-preview-thread")
    run_id = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    add_required_inputs(runtime, run_id)
    context = RunContext.open_existing(
        run_dir=tmp_path / "runs" / run_id,
        run_id=run_id,
        project_root=PROJECT_ROOT,
        resource_dir=tmp_path / "resources",
        create_missing_directories=True,
    )
    runtime.service.workflow.create(
        context=context,
        input_state=runtime.service._input_state(run_id),
        config=runtime.service.workflow_config,
    )
    runtime.service.workflow.plan(run_id)
    runtime.store.mark_workflow_created(run_id)

    assert runtime.store.inputs_locked(run_id) is False
    runtime.uploads.save(
        run_id=run_id,
        kind="positive_drugs",
        original_name="positive_drugs.tsv",
        stream=BytesIO(b"input_type\tvalue\nlibrary_id\tA\n"),
    )

    assert runtime.store.run(run_id)["workflow_created"] is False
    assert runtime.store.run(run_id)["plan_previewed"] is False
    with pytest.raises(KeyError, match="unknown run"):
        runtime.service.workflow.store.get_run(run_id)


def test_optional_expression_pair_selects_enhanced_plan(tmp_path: Path):
    runtime, _ = make_runtime(tmp_path)
    session = runtime.service.create_session("thread-enhanced")
    run_id = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    add_required_inputs(runtime, run_id)

    runtime.uploads.save(
        run_id=run_id,
        kind="expression_tpm",
        original_name="real_tpm.tsv",
        stream=BytesIO(b"GeneID\td1\tc1\n7157\t2\t1\n"),
    )
    assert runtime.service.requirements(run_id)["next_required"] == "expression_pair"
    runtime.uploads.save(
        run_id=run_id,
        kind="expression_metadata",
        original_name="real_metadata.tsv",
        stream=BytesIO(b"sample_id\tgroup\nd1\tdisease\nc1\tcontrol\n"),
    )

    requirements = runtime.service.requirements(run_id)
    assert requirements["ready"] is True
    assert requirements["inputs"]["expression_pair"] == "available"
    assert requirements["evidence_mode"] == "kg_proximity_gps"
    assert (tmp_path / "runs" / run_id / "inputs" / "TPM_matrix_1.tsv").is_file()
    assert (tmp_path / "runs" / run_id / "inputs" / "metadata_1.tsv").is_file()

    plan = runtime.service.preview_plan(run_id)
    assert plan["mode"] == "provided_targets_enhanced"
    assert plan["evidence_mode"] == "kg_proximity_gps"
    assert all(
        node["initial_status"] == "pending"
        for node in plan["nodes"]
        if node["node_id"].startswith("gps_")
        or node["node_id"] == "prepare_expression_inputs"
    )
    assert all(
        node["initial_status"] == "skipped"
        for node in plan["nodes"]
        if node["node_id"].startswith("netinfer_")
    )


def test_expression_pair_endpoint_accepts_multiple_complete_pairs(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    runtime, _ = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))
    client.post("/sessions", json={"thread_id": "multi-expression-thread"})
    created = client.post(
        "/sessions/multi-expression-thread/runs",
        json={"disease_name": "hepatic steatosis", "disease_slug": "hepatic_steatosis"},
    )
    run_id = created.json()["run_id"]
    add_required_inputs(runtime, run_id)

    first = client.post(
        f"/runs/{run_id}/expression-pairs/1",
        files={
            "tpm": ("TPM_matrix_1.tsv", b"GeneID\td1\tc1\n7157\t2\t1\n", "text/tab-separated-values"),
            "metadata": (
                "metadata_1.tsv",
                b"sample_id\tgroup\nd1\tdisease\nc1\tcontrol\n",
                "text/tab-separated-values",
            ),
        },
    )
    second = client.post(
        f"/runs/{run_id}/expression-pairs/2",
        files={
            "tpm": ("TPM_matrix_2.tsv", b"GeneID\td2\tc2\n7157\t3\t1\n", "text/tab-separated-values"),
            "metadata": (
                "metadata_2.tsv",
                b"sample_id\tgroup\nd2\tdisease\nc2\tcontrol\n",
                "text/tab-separated-values",
            ),
        },
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert "表达 TPM / metadata 对" in first.json()["receipt"]["content"]
    assert first.json()["receipt"]["ready"] is True
    requirements = second.json()["requirements"]
    assert requirements["ready"] is True
    assert requirements["inputs"]["expression_pair"] == "available"
    assert requirements["inputs"]["expression_pair_count"] == 2
    assert {item["pair_id"] for item in requirements["expression_pairs"]} == {"1", "2"}
    assert (tmp_path / "runs" / run_id / "inputs" / "TPM_matrix_1.tsv").is_file()
    assert (tmp_path / "runs" / run_id / "inputs" / "metadata_1.tsv").is_file()
    assert (tmp_path / "runs" / run_id / "inputs" / "TPM_matrix_2.tsv").is_file()
    assert (tmp_path / "runs" / run_id / "inputs" / "metadata_2.tsv").is_file()

    plan = runtime.service.preview_plan(run_id)
    assert plan["mode"] == "provided_targets_enhanced"


def test_incomplete_expression_pair_blocks_plan_until_pair_is_complete(tmp_path: Path):
    runtime, _ = make_runtime(tmp_path)
    session = runtime.service.create_session("thread-incomplete-expression")
    run_id = runtime.service.create_run(
        thread_id=session["thread_id"],
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    add_required_inputs(runtime, run_id)

    runtime.uploads.save(
        run_id=run_id,
        kind="expression_tpm",
        pair_id="2",
        original_name="TPM_matrix_2.tsv",
        stream=BytesIO(b"GeneID\td2\tc2\n7157\t3\t1\n"),
    )

    requirements = runtime.service.requirements(run_id)
    assert requirements["ready"] is False
    assert requirements["next_required"] == "expression_pair"
    assert requirements["inputs"]["expression_pair"] == "incomplete"
    with pytest.raises(ValueError, match="expression_pair"):
        runtime.service.preview_plan(run_id)


def test_tool_calling_loop_has_no_implicit_start(tmp_path: Path):
    scripted = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "call-1",
                        "create_screening_run",
                        json.dumps(
                            {
                                "disease_name": "hepatic steatosis",
                                "disease_slug": "hepatic_steatosis",
                            }
                        ),
                    ),
                )
            ),
            ModelReply(content="任务已创建，请上传 compounds 和 disease_genes。"),
        ]
    )
    runtime, queue = make_runtime(tmp_path, model=scripted)
    runtime.service.create_session("thread-agent")
    result = runtime.agent.chat(thread_id="thread-agent", user_message="创建脂肪肝筛选任务")

    assert result["run_id"].startswith("run-")
    assert result["tool_events"] == [
        {"name": "create_screening_run", "status": "succeeded"}
    ]
    assert queue.jobs == []
    assert runtime.service.requirements(result["run_id"])["next_required"] == "compounds"


def test_fastapi_minimal_flow(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    runtime, queue = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))
    session = client.post("/sessions", json={"thread_id": "api-thread"})
    assert session.status_code == 201
    created = client.post(
        "/sessions/api-thread/runs",
        json={"disease_name": "hepatic steatosis", "disease_slug": "hepatic_steatosis"},
    )
    assert created.status_code == 201
    run_id = created.json()["run_id"]
    first = client.post(
        f"/runs/{run_id}/files/compounds",
        files={"upload": ("compounds.csv", b"ID,SMILES\nA,CCO\n", "text/csv")},
    )
    second = client.post(
        f"/runs/{run_id}/files/disease_genes",
        files={"upload": ("genes.tsv", b"TP53\n", "text/tab-separated-values")},
    )
    targets = client.post(
        f"/runs/{run_id}/files/drug_targets",
        files={
            "upload": (
                "drug_targets.json",
                json.dumps(
                    {
                        "A": {
                            "smiles": "CCO",
                            "targets": [
                                {
                                    "gene_symbol": "TP53",
                                    "uniprot_id": "P04637",
                                    "evidence": "known",
                                }
                            ],
                        }
                    }
                ).encode(),
                "application/json",
            )
        },
    )
    mapping = client.post(
        f"/runs/{run_id}/files/target_mapping",
        files={
            "upload": (
                "target_mapping.tsv",
                b"gene_symbol\tentrez_id\nTP53\t7157\n",
                "text/tab-separated-values",
            )
        },
    )
    positive = client.post(
        f"/runs/{run_id}/files/positive_drugs",
        files={
            "upload": (
                "positive_drugs.tsv",
                b"input_type\tvalue\nlibrary_id\tA\n",
                "text/tab-separated-values",
            )
        },
    )
    links = client.post(
        f"/runs/{run_id}/files/disease_links",
        files={
            "upload": (
                "disease_links.tsv",
                b"input_type\tvalue\tnode_name\nbase_disease_name\tNAFLD\tNAFLD\n",
                "text/tab-separated-values",
            )
        },
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert targets.status_code == 201
    assert positive.status_code == 201
    assert links.status_code == 201
    assert first.json()["receipt"]["next_required"] == "disease_genes"
    assert "已接收化合物库：compounds.csv" in first.json()["receipt"]["content"]
    assert second.json()["receipt"]["ready"] is True
    history = client.get("/sessions/api-thread").json()
    assert any(
        message["content"] == first.json()["receipt"]["content"]
        for message in history["messages"]
    )
    assert mapping.json()["requirements"]["ready"] is True
    assert positive.json()["requirements"]["inputs"]["positive_drugs"] == "available"
    assert links.json()["requirements"]["inputs"]["disease_links"] == "available"
    denied = client.post(f"/runs/{run_id}/start", json={"confirmed": False})
    assert denied.status_code == 409
    started = client.post(f"/runs/{run_id}/start", json={"confirmed": True})
    assert started.status_code == 200
    assert queue.jobs[0][1].node_id == "register_inputs"


@pytest.mark.parametrize(
    ("kind", "relative_path"),
    [
        ("compounds", "examples/invalid_inputs/compounds_missing_smiles.csv"),
        ("disease_genes", "examples/invalid_inputs/disease_genes_header_only.tsv"),
        ("drug_targets", "examples/invalid_inputs/drug_targets_wrong_shape.json"),
        ("target_mapping", "examples/invalid_inputs/target_mapping_missing_entrez.tsv"),
        ("positive_drugs", "examples/invalid_inputs/positive_drugs_bad_input_type.tsv"),
        ("disease_links", "examples/invalid_inputs/disease_links_missing_value.tsv"),
        ("expression_tpm", "examples/invalid_inputs/TPM_matrix_wrong_first_column.tsv"),
        ("expression_metadata", "examples/invalid_inputs/metadata_invalid_group.tsv"),
    ],
)
def test_invalid_reference_files_are_rejected_before_registration(
    tmp_path: Path,
    kind: str,
    relative_path: str,
) -> None:
    runtime, _ = make_runtime(tmp_path)
    thread_id = f"invalid-{kind}"
    runtime.service.create_session(thread_id)
    run_id = runtime.service.create_run(
        thread_id=thread_id,
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    path = PROJECT_ROOT / relative_path

    with path.open("rb") as handle, pytest.raises(ValueError, match="basic validation failed"):
        runtime.uploads.save(
            run_id=run_id,
            kind=kind,
            original_name=path.name,
            stream=handle,
        )

    assert runtime.store.inputs(run_id) == {}


def test_input_catalog_and_examples_are_available_from_web_ui(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    runtime, _ = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))
    catalog = client.get("/input-specs")

    assert catalog.status_code == 200
    body = catalog.json()
    specs = body["inputs"]
    assert len(specs) == 4
    assert [item["kind"] for item in specs] == [
        "compounds",
        "disease_genes",
        "positive_drugs",
        "disease_links",
    ]
    assert {item["kind"] for item in specs} == {
        "compounds",
        "disease_genes",
        "positive_drugs",
        "disease_links",
    }
    labels = {item["kind"]: item["label"] for item in specs}
    assert labels["positive_drugs"] == "KG 阳性药物先验"
    assert labels["disease_links"] == "KG 疾病链接先验"
    expression = body["expression_pair"]
    assert expression["tpm_kind"] == "expression_tpm"
    assert expression["metadata_kind"] == "expression_metadata"
    assert "可选多对" in expression["format"]
    for item in specs:
        assert item["source"]
        assert item["collection_note"]
        assert client.get(item["valid_example_url"]).status_code == 200
        assert client.get(item["invalid_example_url"]).status_code == 200
    assert client.get(expression["tpm_valid_example_url"]).status_code == 200
    assert client.get(expression["metadata_valid_example_url"]).status_code == 200

    page = client.get("/")
    assert 'id="inputCards"' in page.text
    assert 'id="expressionPairSection"' in page.text
    assert page.text.index('id="inputCards"') < page.text.index(
        'id="expressionPairSection"'
    )
    assert page.text.index("</aside>") < page.text.index('id="results"')
    assert 'id="moduleCards"' in page.text
    assert 'id="dagGroups"' in page.text
    assert 'id="cancelRun"' in page.text
    assert 'id="chatProgressTrack"' in page.text
    assert 'id="artifactHint"' in page.text
    assert "完整候选排名表" in page.text
    assert "候选小分子清单" in page.text
    assert 'id="historyDialog"' in page.text
    assert 'id="chatUploadGuide"' in page.text
    assert "candidateVisibleCount = 12" in page.text
    assert "界面最多加载 Top100" in page.text
    assert "二维结构" in page.text
    assert "选择并上传文件" in page.text
    assert 'input.addEventListener("change"' in page.text
    assert "合规样例" in page.text
    assert "不合规样例" not in page.text
    assert "确认启动" in page.text
    assert "NetInfer" in page.text
    assert "不是深度学习模型" in page.text
    assert "不自动放宽阈值" in page.text


def test_session_history_can_be_reopened_and_downloaded_as_zip(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    runtime, _ = make_runtime(tmp_path)
    runtime.service.create_session("history-thread")
    runtime.store.append_message(
        "history-thread", {"role": "user", "content": "为脂肪肝创建任务"}
    )
    runtime.store.append_message(
        "history-thread", {"role": "assistant", "content": "请上传化合物库"}
    )
    run_id = runtime.service.create_run(
        thread_id="history-thread",
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    client = TestClient(create_app(runtime))

    listed = client.get("/sessions")
    assert listed.status_code == 200
    item = listed.json()["sessions"][0]
    assert item["thread_id"] == "history-thread"
    assert item["run_id"] == run_id
    assert item["status"] == "collecting"
    assert item["message_count"] == 2

    restored = client.get("/sessions/history-thread")
    assert restored.status_code == 200
    assert [message["role"] for message in restored.json()["messages"]] == [
        "user",
        "assistant",
    ]
    assert restored.json()["run"]["run_id"] == run_id

    downloaded = client.get("/sessions/history-thread/export")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(downloaded.content)) as archive:
        assert {
            "session.json",
            "conversation.json",
            "conversation.md",
            "inputs.json",
            "results.json",
        }.issubset(archive.namelist())
        transcript = archive.read("conversation.md").decode("utf-8")
        assert "为脂肪肝创建任务" in transcript
        assert "请上传化合物库" in transcript


def test_result_preview_enriches_and_limits_candidate_cards_to_top100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = make_runtime(tmp_path)
    runtime.service.create_session("candidate-thread")
    run_id = runtime.service.create_run(
        thread_id="candidate-thread",
        disease_name="hepatic steatosis",
        disease_slug="hepatic_steatosis",
    )["run_id"]
    prepared = tmp_path / "runs" / run_id / "inputs" / "prepared"
    final = tmp_path / "runs" / run_id / "artifacts" / "final"
    prepared.mkdir(parents=True)
    final.mkdir(parents=True)
    normalized_rows = ["ID,SMILES,Formula,MolWt"]
    candidate_rows = [
        "final_rank\tcompound_id\tcompound_name\tevidence_count\tkg_rank_mean\tproximity_z"
    ]
    for index in range(1, 106):
        normalized_rows.append(f"C{index},CCO,C2H6O,46.07")
        candidate_rows.append(f"{index}\tC{index}\tCandidate {index}\t2\t{index}.0\t-1.0")
    (prepared / "compounds.normalized.csv").write_text(
        "\n".join(normalized_rows) + "\n", encoding="utf-8"
    )
    (final / "final_candidates.tsv").write_text(
        "\n".join(candidate_rows) + "\n", encoding="utf-8"
    )
    runtime.store.mark_workflow_created(run_id)
    monkeypatch.setattr(
        runtime.service.workflow,
        "results",
        lambda requested: {
            "run_id": requested,
            "status": "succeeded",
            "evidence_mode": "kg_proximity",
            "artifacts": [],
        },
    )

    results = runtime.service.results(run_id)

    assert results["candidate_count"] == 105
    assert results["candidate_preview_limit"] == 100
    assert len(results["candidate_preview"]) == 100
    first = results["candidate_preview"][0]
    assert first["compound_id"] == "C1"
    assert first["smiles"] == "CCO"
    assert first["properties"]["formula"] == "C2H6O"
    assert first["properties"]["molecular_weight"] == 46.07
    assert first["structure_url"].endswith("/candidates/C1/structure.svg")


def test_results_artifacts_have_download_metadata_and_download_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    runtime, _ = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))
    client.post("/sessions", json={"thread_id": "download-thread"})
    created = client.post(
        "/sessions/download-thread/runs",
        json={"disease_name": "hepatic steatosis", "disease_slug": "hepatic_steatosis"},
    )
    run_id = created.json()["run_id"]
    final_dir = tmp_path / "runs" / run_id / "artifacts" / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "final_candidates.tsv").write_text(
        "final_rank\tcompound_id\n1\tA\n",
        encoding="utf-8",
    )
    runtime.store.mark_workflow_created(run_id)

    def fake_results(requested_run_id: str) -> dict:
        return {
            "run_id": requested_run_id,
            "status": "succeeded",
            "evidence_mode": "kg_proximity",
            "artifacts": [
                {
                    "artifact_id": "a-final",
                    "artifact_type": "final_candidates",
                    "relative_path": "artifacts/final/final_candidates.tsv",
                    "sha256": "abc",
                }
            ],
            "ranking": None,
            "report": None,
        }

    monkeypatch.setattr(runtime.service.workflow, "results", fake_results)

    response = client.get(f"/runs/{run_id}/results")

    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    assert artifact["download_label"] == "完整候选排名表（final_candidates.tsv）"
    assert artifact["file_name"] == "final_candidates.tsv"
    assert artifact["download_url"] == f"/runs/{run_id}/artifacts/a-final"

    download = client.get(artifact["download_url"])
    assert download.status_code == 200
    assert download.content.replace(b"\r\n", b"\n") == b"final_rank\tcompound_id\n1\tA\n"
    assert "final_candidates-final_candidates.tsv" in download.headers["content-disposition"]


def test_results_tool_exposes_download_links_and_prompt_allows_downloads() -> None:
    class FakeService:
        def run_for_thread(self, thread_id: str) -> dict:
            return {"run_id": f"run-for-{thread_id}"}

        def results(self, run_id: str) -> dict:
            return {
                "run_id": run_id,
                "status": "succeeded",
                "evidence_mode": "kg_proximity",
                "artifacts": [
                    {
                        "artifact_id": "a-final",
                        "artifact_type": "final_candidates",
                        "relative_path": "artifacts/final/final_candidates.tsv",
                        "file_name": "final_candidates.tsv",
                        "download_label": "完整候选排名表（final_candidates.tsv）",
                        "download_url": f"/runs/{run_id}/artifacts/a-final",
                        "sha256": "abc",
                    }
                ],
            }

    result = ToolDispatcher(FakeService()).execute(
        "tool-thread",
        ToolCall("call-results", "get_results", "{}"),
    )

    artifact = result["artifacts"][0]
    assert artifact["download_label"] == "完整候选排名表（final_candidates.tsv）"
    assert artifact["download_url"] == "/runs/run-for-tool-thread/artifacts/a-final"
    assert "不得声称系统没有下载入口" in SYSTEM_PROMPT
    assert "ranking.kg.top_n=200" in SYSTEM_PROMPT
    assert "NetInfer 不作为" in SYSTEM_PROMPT
    assert "没有文献检索" in SYSTEM_PROMPT


def test_compose_defaults_to_full_ranking_and_validation_mode_is_explicit(
    tmp_path: Path,
) -> None:
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "LIPID_AGENT_VALIDATION_MODE: ${LIPID_AGENT_VALIDATION_MODE:-0}" in compose

    full_state = tmp_path / "full"
    full_runtime = build_runtime(
        environ={
            "LIPID_AGENT_PROJECT_ROOT": str(PROJECT_ROOT),
            "LIPID_AGENT_STATE_DIR": str(full_state),
            "LIPID_AGENT_DB": str(full_state / "state.sqlite3"),
            "LIPID_AGENT_RUNS_DIR": str(full_state / "runs"),
            "LIPID_AGENT_RESOURCE_ROOT": str(full_state / "resources"),
        },
        executor=InMemoryQueueExecutor(),
    )
    assert full_runtime.service.workflow_config.ranking.kg.top_n == 200

    validation_state = tmp_path / "validation"
    validation_runtime = build_runtime(
        environ={
            "LIPID_AGENT_PROJECT_ROOT": str(PROJECT_ROOT),
            "LIPID_AGENT_STATE_DIR": str(validation_state),
            "LIPID_AGENT_DB": str(validation_state / "state.sqlite3"),
            "LIPID_AGENT_RUNS_DIR": str(validation_state / "runs"),
            "LIPID_AGENT_RESOURCE_ROOT": str(validation_state / "resources"),
            "LIPID_AGENT_VALIDATION_MODE": "1",
        },
        executor=InMemoryQueueExecutor(),
    )
    assert validation_runtime.service.workflow_config.ranking.kg.top_n == 20


def test_web_ui_status_refresh_preserves_selected_expression_files() -> None:
    from lipid_screening_agent.backend.web_ui import WEB_UI

    block = WEB_UI.split("const renderInputStatuses = () => {", 1)[1].split(
        "const syncRequirements =",
        1,
    )[0]

    assert "renderExpressionPairStatuses();" in block
    assert "renderExpressionPairs();" not in block


def test_web_ui_uses_request_scoped_manual_api_key(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import lipid_screening_agent.backend.runtime as runtime_module
    from lipid_screening_agent.backend.api import create_app

    captured: list[dict] = []

    class ManualKeyModel:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def complete(self, *, messages, tools):
            return ModelReply(content="连接成功")

    monkeypatch.setattr(runtime_module, "OpenAICompatibleModel", ManualKeyModel)
    runtime, _ = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))

    page = client.get("/")
    assert page.status_code == 200
    assert "DeepSeek API Key" in page.text
    assert "localStorage" not in page.text

    client.post("/sessions", json={"thread_id": "manual-key-thread"})
    response = client.post(
        "/sessions/manual-key-thread/chat",
        headers={"X-Model-API-Key": "test-secret-key"},
        json={"message": "测试连接"},
    )

    assert response.status_code == 200
    assert response.json()["content"] == "连接成功"
    assert captured[0]["api_key"] == "test-secret-key"
    assert captured[0]["model"] == "deepseek-v4-flash"
    persisted = json.dumps(runtime.store.messages("manual-key-thread"), ensure_ascii=False)
    assert "test-secret-key" not in persisted


def test_streaming_chat_emits_true_stages_and_preserves_chat_endpoint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lipid_screening_agent.backend.api import create_app

    streamed_model = ScriptedModel(
        [
            ModelReply(
                tool_calls=(
                    ToolCall(
                        "call-stream",
                        "create_screening_run",
                        json.dumps(
                            {
                                "disease_name": "hepatic steatosis",
                                "disease_slug": "hepatic_steatosis",
                            }
                        ),
                    ),
                )
            ),
            ModelReply(content="任务已创建，请上传必需输入。"),
        ]
    )
    runtime, _ = make_runtime(tmp_path, model=streamed_model)
    client = TestClient(create_app(runtime))
    client.post("/sessions", json={"thread_id": "stream-thread"})

    response = client.post(
        "/sessions/stream-thread/chat/stream",
        json={"message": "创建筛选任务"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[0] == {"type": "stage", "stage": "agent", "status": "started"}
    assert any(
        event.get("type") == "tool"
        and event.get("name") == "create_screening_run"
        and event.get("status") == "started"
        for event in events
    )
    assert any(
        event.get("type") == "tool"
        and event.get("name") == "create_screening_run"
        and event.get("status") == "succeeded"
        for event in events
    )
    final = events[-1]
    assert final["type"] == "final"
    assert final["content"] == "任务已创建，请上传必需输入。"
    assert final["run_id"].startswith("run-")

    compatible_model = ScriptedModel([ModelReply(content="原接口仍然可用")])
    compatible_root = tmp_path / "compatible"
    compatible_root.mkdir()
    compatible_runtime, _ = make_runtime(compatible_root, model=compatible_model)
    compatible_client = TestClient(create_app(compatible_runtime))
    compatible_client.post("/sessions", json={"thread_id": "compatible-thread"})
    compatible = compatible_client.post(
        "/sessions/compatible-thread/chat",
        json={"message": "测试原接口"},
    )
    assert compatible.status_code == 200
    assert compatible.json()["content"] == "原接口仍然可用"


def test_streaming_chat_turns_model_exception_into_redacted_retry_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import lipid_screening_agent.backend.runtime as runtime_module
    from lipid_screening_agent.backend.api import create_app

    secret = "stream-test-secret"

    class FailingManualModel:
        def __init__(self, **kwargs):
            self.api_key = kwargs["api_key"]

        def complete(self, *, messages, tools):
            raise RuntimeError(f"upstream rejected {self.api_key}")

    monkeypatch.setattr(runtime_module, "OpenAICompatibleModel", FailingManualModel)
    runtime, _ = make_runtime(tmp_path)
    client = TestClient(create_app(runtime))
    client.post("/sessions", json={"thread_id": "stream-error-thread"})

    response = client.post(
        "/sessions/stream-error-thread/chat/stream",
        headers={"X-Model-API-Key": secret},
        json={"message": "测试异常"},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line]
    assert events[-1]["type"] == "error"
    assert events[-1]["retryable"] is True
    assert "[redacted]" in events[-1]["message"]
    assert secret not in response.text
    persisted = json.dumps(runtime.store.messages("stream-error-thread"), ensure_ascii=False)
    assert secret not in persisted


def test_source_digest_only_uses_files_shared_by_api_and_workers(tmp_path: Path):
    for relative in ("src", "configs", "contracts", "docker"):
        (tmp_path / relative).mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "workflow.yaml").write_text("version: 1\n", encoding="utf-8")
    (tmp_path / "contracts" / "inputs.yaml").write_text("inputs: []\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    before = project_source_digest(tmp_path)

    (tmp_path / "README.md").write_text("API-only documentation\n", encoding="utf-8")
    (tmp_path / "docker" / "Dockerfile.api").write_text("FROM python\n", encoding="utf-8")
    assert project_source_digest(tmp_path) == before

    (tmp_path / "src" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert project_source_digest(tmp_path) != before
