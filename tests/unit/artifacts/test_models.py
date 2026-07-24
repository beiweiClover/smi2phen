import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import (
    ArtifactManifest,
    ErrorCategory,
    ErrorInfo,
    NodeResult,
    NodeStatus,
    artifact_matches_manifest,
    create_artifact_manifest,
    make_artifact_id,
    verify_artifact_manifest,
)
from lipid_screening_agent.runtime.errors import (
    InputError,
    OutputContractError,
    PathSafetyError,
)
from lipid_screening_agent.runtime.hashing import hash_config

NOW = datetime(2026, 7, 20, 12, 30, 45, 123456, tzinfo=timezone(timedelta(hours=8)))
CONFIG_HASH = hash_config({"workflow": {"mode": "core"}})


def _manifest(run_root: Path) -> ArtifactManifest:
    output = run_root / "artifacts" / "module" / "result.csv"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"abc")
    return create_artifact_manifest(
        output,
        run_root=run_root,
        artifact_type="example_result",
        producer_node_id="example_node",
        producer_task_id="main",
        input_artifact_ids=(make_artifact_id("upstream", "main", "normalized_inputs"),),
        config_hash=CONFIG_HASH,
        code_version="0.1.0",
        resource_hashes={"resource.table": CONFIG_HASH},
        created_at=NOW,
    )


def test_artifact_ids_are_deterministic_unique_and_portable() -> None:
    first = make_artifact_id("prepare", "main", "batch_input", instance_key="batch-0001")
    same = make_artifact_id("prepare", "main", "batch_input", instance_key="batch-0001")
    second = make_artifact_id("prepare", "main", "batch_input", instance_key="batch-0002")

    assert first == same
    assert first != second
    assert "/" not in first and "\\" not in first and ":" not in first
    assert first.startswith("a-")
    assert len(first) == 34


def test_manifest_hashes_file_and_round_trips_with_utc_timestamp(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    assert manifest.artifact_type == "example_result"
    assert manifest.relative_path == "artifacts/module/result.csv"
    assert manifest.size_bytes == 3
    assert manifest.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert manifest.created_at.tzinfo == timezone.utc
    assert manifest.created_at.hour == 4

    serialized = manifest.to_json()
    restored = ArtifactManifest.from_dict(json.loads(serialized))
    assert restored == manifest
    assert restored.to_dict() == manifest.to_dict()


def test_manifest_rejects_naive_time_bad_hash_and_nonportable_path(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    data = manifest.to_dict()

    data["created_at"] = "2026-07-20T12:00:00"
    with pytest.raises(OutputContractError, match="timezone"):
        ArtifactManifest.from_dict(data)

    data = manifest.to_dict()
    data["sha256"] = "ABC"
    with pytest.raises(OutputContractError, match="SHA-256"):
        ArtifactManifest.from_dict(data)

    for bad_path in ("../escape.csv", "/tmp/escape.csv", r"C:\escape.csv"):
        data = manifest.to_dict()
        data["relative_path"] = bad_path
        with pytest.raises(OutputContractError):
            ArtifactManifest.from_dict(data)


def test_manifest_detects_tampering_and_rejects_output_outside_run(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    manifest = _manifest(run_root)

    verify_artifact_manifest(manifest, run_root=run_root)
    assert artifact_matches_manifest(manifest, run_root=run_root)

    (run_root / manifest.relative_path).write_bytes(b"changed")
    with pytest.raises(OutputContractError, match="failed verification"):
        verify_artifact_manifest(manifest, run_root=run_root)
    assert not artifact_matches_manifest(manifest, run_root=run_root)

    outside = tmp_path / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(OutputContractError):
        create_artifact_manifest(
            outside,
            run_root=run_root,
            artifact_type="example_result",
            producer_node_id="example_node",
            producer_task_id="main",
            config_hash=CONFIG_HASH,
            code_version="0.1.0",
        )


def test_success_node_result_round_trip_and_strict_metrics() -> None:
    output_id = make_artifact_id("node", "main", "result")
    result = NodeResult(
        node_id="node",
        task_id="main",
        status=NodeStatus.SUCCEEDED,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2.5),
        attempt=1,
        outputs=[output_id],
        metrics={"processed": 3, "nested": [True, None]},
        warnings=["一个可恢复警告"],
        error=None,
    )

    restored = NodeResult.from_dict(json.loads(result.to_json()))

    assert restored == result
    assert result.duration_seconds == 2.5
    assert restored.status is NodeStatus.SUCCEEDED
    with pytest.raises(OutputContractError, match="JSON-compatible"):
        NodeResult(
            node_id="node",
            task_id="main",
            status="succeeded",
            started_at=NOW,
            finished_at=NOW,
            attempt=1,
            metrics={"bad": float("nan")},
        )


def test_failed_node_result_requires_and_serializes_structured_error() -> None:
    error = ErrorInfo(
        category=ErrorCategory.INPUT,
        code="invalid_columns",
        message="ID column is missing",
        exception_type="lipid_screening_agent.runtime.errors.InputError",
        retryable=False,
        details={"required": ["ID", "SMILES"]},
    )
    result = NodeResult(
        node_id="prepare_compounds",
        task_id="main",
        status="failed",
        started_at=NOW,
        finished_at=NOW + timedelta(milliseconds=10),
        attempt=2,
        error=error,
    )

    assert result.to_dict()["error"]["category"] == "input"
    assert NodeResult.from_dict(result.to_dict()) == result

    with pytest.raises(OutputContractError, match="requires structured error"):
        NodeResult(
            node_id="prepare_compounds",
            task_id="main",
            status="failed",
            started_at=NOW,
            finished_at=NOW,
            attempt=1,
        )


def test_error_info_maps_runtime_errors_and_sanitizes_details() -> None:
    input_error = InputError(
        "missing columns\nplease repair the input",
        details={"path": Path("input.csv")},
    )
    info = ErrorInfo.from_exception(input_error)

    assert info.category is ErrorCategory.INPUT
    assert info.code == "input_error"
    assert "reported_details" in info.details

    path_info = ErrorInfo.from_exception(PathSafetyError("escaped workspace"))
    assert path_info.category is ErrorCategory.OUTPUT_CONTRACT


def test_error_info_bounds_custom_exception_fields() -> None:
    long_named_error = type("X" * 700, (Exception,), {})
    error = long_named_error("m" * 5000)
    error.code = "CON." + ("c" * 5000)
    error.details = {"payload": "d" * 20000}

    info = ErrorInfo.from_exception(error)

    assert len(info.message) == 4096
    assert len(info.code) <= 180
    assert info.code.startswith("error_CON")
    assert len(info.exception_type) == 512
    assert info.details["truncated_or_invalid"] is True


def test_node_result_rejects_nonterminal_state_naive_time_and_invalid_consistency() -> None:
    with pytest.raises(OutputContractError, match="terminal"):
        NodeResult(
            node_id="node",
            task_id="main",
            status="running",
            started_at=NOW,
            finished_at=NOW,
            attempt=1,
        )

    with pytest.raises(OutputContractError, match="timezone"):
        NodeResult(
            node_id="node",
            task_id="main",
            status="succeeded",
            started_at=datetime(2026, 7, 20),
            finished_at=NOW,
            attempt=1,
        )

    error = ErrorInfo(
        category="execution",
        code="unexpected",
        message="unexpected error",
        exception_type="RuntimeError",
    )
    with pytest.raises(OutputContractError, match="cannot contain an error"):
        NodeResult(
            node_id="node",
            task_id="main",
            status="succeeded",
            started_at=NOW,
            finished_at=NOW,
            attempt=1,
            error=error,
        )
