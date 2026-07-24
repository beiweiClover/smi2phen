import csv
import json
import shutil
from importlib import import_module
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.runtime import RunContext
from lipid_screening_agent.runtime.errors import EnvironmentError
from lipid_screening_agent.runtime.hashing import sha256_bytes

runner = import_module("lipid_screening_agent.runners.input_prepare.prepare_compound_library")

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "small_inputs"
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class _FakeChem:
    @staticmethod
    def MolFromSmiles(smiles: str):
        if smiles == "not-a-smiles":
            return None
        return object()


def _context(tmp_path: Path, run_id: str = "run-01") -> RunContext:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    resources.mkdir()
    run_dir = project / "runs" / run_id
    return RunContext.create(
        runs_root=project / "runs",
        run_id=run_id,
        project_root=project,
        resource_dir=resources,
        output_dir=run_dir / "inputs" / "prepared",
    )


def _registered_fixture(context: RunContext, filename: str) -> Path:
    original = context.resolve_run_relative("inputs/original")
    original.mkdir(parents=True, exist_ok=True)
    destination = original / filename
    shutil.copy2(FIXTURES / filename, destination)
    return destination


def _config_hash() -> str:
    return sha256_bytes(b"stage-02-test-config")


def _manifest_by_type(context: RunContext, result) -> dict[str, object]:
    manifests = {}
    for artifact_id in result.outputs:
        path = (
            context.run_dir
            / "artifacts"
            / "manifests"
            / runner.NODE_ID
            / "main"
            / f"{artifact_id}.json"
        )
        manifest = load_artifact_manifest(path, run_root=context.run_dir)
        manifests[manifest.artifact_type] = manifest
    return manifests


def test_prepare_accepts_aliases_preserves_columns_and_reports_invalid_smiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = _registered_fixture(context, "compound_library.csv")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
        input_artifact_ids=("registered-input",),
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics == {
        "input_count": 3,
        "valid_count": 2,
        "invalid_count": 1,
        "skipped_count": 1,
        "additional_column_count": 2,
    }
    assert len(result.warnings) == 1

    normalized_path = context.resolve_run_relative(runner.NORMALIZED_PATH)
    with normalized_path.open("r", encoding="utf-8", newline="") as handle:
        normalized_rows = list(csv.DictReader(handle))
    assert list(normalized_rows[0]) == ["ID", "SMILES", "name", "source"]
    assert [row["ID"] for row in normalized_rows] == ["cmp-001", "cmp-003"]
    assert normalized_rows[0]["name"] == "ethanol"

    invalid_path = context.resolve_run_relative(runner.INVALID_PATH)
    with invalid_path.open("r", encoding="utf-8", newline="") as handle:
        invalid_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert invalid_rows == [
        {
            "ID": "cmp-002",
            "SMILES": "not-a-smiles",
            "reason": "rdkit_parse_failed",
        }
    ]

    manifests = _manifest_by_type(context, result)
    assert set(manifests) == {"compounds_normalized", "invalid_smiles"}
    assert manifests["compounds_normalized"].relative_path == runner.NORMALIZED_PATH
    assert manifests["invalid_smiles"].relative_path == runner.INVALID_PATH
    assert manifests["compounds_normalized"].input_artifact_ids == ("registered-input",)

    result_document = json.loads(
        (context.run_dir / "artifacts/node_results/prepare_compound_library/main.json").read_text(
            encoding="utf-8"
        )
    )
    assert result_document["metrics"]["valid_count"] == 2
    assert len(result_document["outputs"]) == 2


def test_prepare_rejects_duplicate_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    source = _registered_fixture(context, "compound_library_duplicate.csv")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert result.error.details["duplicate_ids"] == ("cmp-001",)
    assert not context.resolve_run_relative(runner.NORMALIZED_PATH).exists()


def test_prepare_rejects_empty_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = _context(tmp_path)
    source = context.resolve_run_relative("inputs/original/empty-id.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("ID,SMILES\n,CCO\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert result.error.details["empty_id_count"] == 1


def test_prepare_fails_when_no_valid_compound_but_keeps_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = context.resolve_run_relative("inputs/original/all-invalid.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "ID,SMILES\ncompound-a,not-a-smiles\ncompound-b,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.metrics["input_count"] == 2
    assert result.metrics["valid_count"] == 0
    assert result.metrics["invalid_count"] == 2
    assert context.resolve_run_relative(runner.INVALID_PATH).exists()
    assert not context.resolve_run_relative(runner.NORMALIZED_PATH).exists()


def test_prepare_reads_tsv_and_writes_empty_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = context.resolve_run_relative("inputs/original/compounds.tsv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "molecule_id\tstandardized_smiles\tlabel\ncompound-a\tCCO\tvalid\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["invalid_count"] == 0
    invalid_text = context.resolve_run_relative(runner.INVALID_PATH).read_text(encoding="utf-8")
    assert invalid_text == "ID\tSMILES\treason\n"


def test_prepare_reports_missing_rdkit_as_environment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = _registered_fixture(context, "compound_library.csv")

    def missing_rdkit():
        raise EnvironmentError("RDKit unavailable")

    monkeypatch.setattr(runner, "_load_rdkit", missing_rdkit)
    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "environment"


def test_prepare_uses_real_rdkit_when_available(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    context = _context(tmp_path)
    source = _registered_fixture(context, "compound_library.csv")

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["valid_count"] == 2
    assert result.metrics["invalid_count"] == 1


def test_prepare_rejects_output_dir_outside_contracted_prepared_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    resources.mkdir()
    context = RunContext.create(
        runs_root=project / "runs",
        run_id="wrong-output",
        project_root=project,
        resource_dir=resources,
    )
    source = _registered_fixture(context, "compound_library.csv")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "output_contract"
    assert not context.resolve_run_relative(runner.NORMALIZED_PATH).exists()


def test_prepare_rejects_input_that_was_not_registered_under_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = context.resolve_run_relative("inputs/unregistered.csv")
    source.write_text("ID,SMILES\ncompound-a,CCO\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"


def test_prepare_does_not_guess_the_first_two_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    source = context.resolve_run_relative("inputs/original/unknown-columns.csv")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("first,second\ncompound-a,CCO\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_load_rdkit", lambda: _FakeChem)

    result = runner.prepare_compound_library(
        context=context,
        input_path=source,
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"


def test_cli_prints_failed_node_result_and_returns_exit_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)
    source = _registered_fixture(context, "compound_library_duplicate.csv")

    exit_code = runner.main(
        [
            "--run-dir",
            str(context.run_dir),
            "--input-dir",
            str(context.input_dir),
            "--resource-dir",
            str(context.resource_dir),
            "--output-dir",
            str(context.output_dir),
            "--config",
            str(PROJECT_ROOT / "configs" / "workflow.yaml"),
            "--input-file",
            str(source),
        ]
    )

    assert exit_code == 1
    result_document = json.loads(capsys.readouterr().out)
    assert result_document["node_id"] == runner.NODE_ID
    assert result_document["status"] == "failed"
