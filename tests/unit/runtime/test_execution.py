import json
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.runtime import RunContext
from lipid_screening_agent.runtime.atomic import atomic_write_text
from lipid_screening_agent.runtime.errors import ExecutionError, InputError
from lipid_screening_agent.runtime.execution import execute_node
from lipid_screening_agent.runtime.hashing import sha256_bytes


def _context(tmp_path: Path, run_id: str = "run-01") -> RunContext:
    project = tmp_path / "project"
    resource = tmp_path / "resources"
    project.mkdir()
    resource.mkdir()
    return RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resource,
    )


def _config_hash() -> str:
    return sha256_bytes(b"config")


def test_execute_node_commits_manifest_then_success_result(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        output = context.resolve_output("prepared.txt")
        atomic_write_text(output, "abc", allowed_root=context.run_dir)
        execution.add_output("compounds_normalized", output)
        execution.metric("valid_count", 1)

    result = execute_node(
        operation,
        context=context,
        node_id="prepare_compound_library",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["valid_count"] == 1
    assert len(result.outputs) == 1

    manifest_path = (
        context.run_dir
        / "artifacts"
        / "manifests"
        / "prepare_compound_library"
        / "main"
        / f"{result.outputs[0]}.json"
    )
    manifest = load_artifact_manifest(manifest_path, run_root=context.run_dir)
    assert manifest.artifact_type == "compounds_normalized"
    assert manifest.sha256 == sha256_bytes(b"abc")

    result_path = (
        context.run_dir / "artifacts" / "node_results" / "prepare_compound_library" / "main.json"
    )
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "succeeded"


def test_execute_node_records_classified_failure_and_logs(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        raise InputError("input is invalid", details={"column": "ID"})

    result = execute_node(
        operation,
        context=context,
        node_id="register_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert result.error.details["column"] == "ID"
    json_lines = (
        (context.log_dir / "register_inputs" / "main.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert any(json.loads(line)["event"] == "node_failed" for line in json_lines)
    assert "node execution failed" in (context.log_dir / "register_inputs" / "main.log").read_text(
        encoding="utf-8"
    )


def test_execute_node_output_validation_failure_is_structured(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        execution.add_output("compounds_normalized", context.output_dir / "missing.csv")

    result = execute_node(
        operation,
        context=context,
        node_id="prepare_compound_library",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "output_contract"
    assert not result.outputs


def test_execute_node_supports_repeated_logical_artifact_type(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        for batch_id in ("batch-0001", "batch-0002"):
            output = context.resolve_output(f"{batch_id}.tsv")
            atomic_write_text(output, batch_id, allowed_root=context.run_dir)
            execution.add_output(
                "netinfer_batch_input",
                output,
                instance_key=batch_id,
            )

    result = execute_node(
        operation,
        context=context,
        node_id="netinfer_prepare_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert len(set(result.outputs)) == 2
    for artifact_id in result.outputs:
        assert (
            context.run_dir
            / "artifacts"
            / "manifests"
            / "netinfer_prepare_inputs"
            / "main"
            / f"{artifact_id}.json"
        ).is_file()


def test_execute_node_can_persist_skipped_terminal_result(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = execute_node(
        lambda execution: execution.mark_skipped("no expression input"),
        context=context,
        node_id="prepare_expression_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SKIPPED
    assert result.warnings == ("no expression input",)
    assert result.outputs == ()


def test_execute_node_can_reraise_after_persisting_failure(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        raise InputError("bad input")

    with pytest.raises(InputError, match="bad input"):
        execute_node(
            operation,
            context=context,
            node_id="register_inputs",
            config_hash=_config_hash(),
            code_version="test-version",
            raise_on_error=True,
        )

    result_path = context.run_dir / "artifacts/node_results/register_inputs/main.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_invalid_metric_is_rejected_before_it_can_break_failure_persistence(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    result = execute_node(
        lambda execution: execution.metric("bad", Path("not-json")),
        context=context,
        node_id="register_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "output_contract"
    result_path = context.run_dir / "artifacts/node_results/register_inputs/main.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_oversized_warning_is_rejected_before_failure_persistence(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    result = execute_node(
        lambda execution: execution.warn("x" * 5000),
        context=context,
        node_id="register_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.warnings == ()
    assert result.error is not None
    assert result.error.category.value == "output_contract"
    result_path = context.run_dir / "artifacts/node_results/register_inputs/main.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_oversized_exception_payload_is_bounded_and_persisted(tmp_path: Path) -> None:
    context = _context(tmp_path)

    def operation(execution) -> None:
        raise ExecutionError(
            "x" * 5000,
            details={"payload": "y" * 20000},
        )

    result = execute_node(
        operation,
        context=context,
        node_id="register_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert len(result.error.message) == 4096
    assert result.error.details["truncated_or_invalid"] is True
    result_path = context.run_dir / "artifacts/node_results/register_inputs/main.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_logger_initialization_failure_still_persists_node_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    operation_called = False

    def fail_logger(**kwargs):
        raise OSError("log destination unavailable")

    def operation(execution) -> None:
        nonlocal operation_called
        operation_called = True

    monkeypatch.setattr(
        "lipid_screening_agent.runtime.execution.create_node_logger",
        fail_logger,
    )
    result = execute_node(
        operation,
        context=context,
        node_id="register_inputs",
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert not operation_called
    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "execution"
    result_path = context.run_dir / "artifacts/node_results/register_inputs/main.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "failed"
