import json
import shutil
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.runners.input_prepare.models import (
    InputRegistrationRequest,
    load_input_registration_manifest,
)
from lipid_screening_agent.runners.input_prepare.register_inputs import (
    main,
    register_inputs,
)
from lipid_screening_agent.runtime import InputError, RunContext, sha256_bytes

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "small_inputs"
CONFIG = Path(__file__).resolve().parents[4] / "configs" / "workflow.yaml"


def _context(tmp_path: Path) -> RunContext:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    resources.mkdir()
    runs_root = project / "runs"
    return RunContext.create(
        runs_root=runs_root,
        run_id="run-register",
        project_root=project,
        resource_dir=resources,
        input_dir=runs_root / "run-register" / "uploads",
    )


def _config_hash() -> str:
    return sha256_bytes(b"stage-02-config")


def test_register_inputs_copies_file_and_commits_provenance(tmp_path: Path) -> None:
    context = _context(tmp_path)
    upload = context.input_dir / "library.csv"
    shutil.copy2(FIXTURES / "registration_upload.txt", upload)

    result = register_inputs(
        context=context,
        inputs=[
            InputRegistrationRequest(
                input_key="compound_library",
                path=upload,
                source="explicit_fixture",
            )
        ],
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["registered_count"] == 1
    assert result.metrics["total_size_bytes"] == upload.stat().st_size
    registered = context.run_dir / "inputs/original/library.csv"
    assert registered.read_bytes() == upload.read_bytes()

    input_manifest_path = context.run_dir / "inputs/input_manifest.json"
    manifest = load_input_registration_manifest(
        input_manifest_path,
        run_root=context.run_dir,
    )
    record = manifest.records_for("compound_library")[0]
    assert record.registered_path == "inputs/original/library.csv"
    assert record.source == "explicit_fixture"
    assert record.sha256 == sha256_bytes(upload.read_bytes())
    assert "+00:00" not in record.registered_at
    assert record.registered_at.endswith("Z")

    artifact_manifest_path = (
        context.run_dir / "artifacts/manifests/register_inputs/main" / f"{result.outputs[0]}.json"
    )
    artifact_manifest = load_artifact_manifest(
        artifact_manifest_path,
        run_root=context.run_dir,
    )
    assert artifact_manifest.artifact_type == "input_registration_manifest"
    assert artifact_manifest.relative_path == "inputs/input_manifest.json"


def test_register_inputs_reports_expression_pair_roles(tmp_path: Path) -> None:
    context = _context(tmp_path)
    tpm = context.input_dir / "TPM_matrix_1.tsv"
    metadata = context.input_dir / "metadata_1.tsv"
    tpm.write_text("GeneID\ts1\ts2\n1\t1\t2\n", encoding="utf-8")
    metadata.write_text(
        "sample_id\tgroup\ns1\tcontrol\ns2\tdisease\n",
        encoding="utf-8",
    )

    result = register_inputs(
        context=context,
        inputs=[
            InputRegistrationRequest(
                input_key="expression_pairs",
                path=tpm,
                role="tpm",
                pair_id="comparison_1",
            ),
            InputRegistrationRequest(
                input_key="expression_pairs",
                path=metadata,
                role="metadata",
                pair_id="comparison_1",
            ),
        ],
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["expression_pair_count"] == 1
    assert result.metrics["incomplete_expression_pair_count"] == 0
    payload = json.loads(
        (context.run_dir / "inputs/input_manifest.json").read_text(encoding="utf-8")
    )
    assert {record["role"] for record in payload["inputs"]} == {"tpm", "metadata"}
    assert {record["pair_id"] for record in payload["inputs"]} == {"comparison_1"}


def test_registration_accepts_provided_target_inputs(tmp_path: Path) -> None:
    targets = tmp_path / "drug_targets.json"
    mapping = tmp_path / "target_mapping.tsv"
    targets.write_text("{}\n", encoding="utf-8")
    mapping.write_text("gene_symbol\tentrez_id\n", encoding="utf-8")

    requests = (
        InputRegistrationRequest(input_key="drug_targets", path=targets),
        InputRegistrationRequest(input_key="target_mapping", path=mapping),
    )

    assert {request.input_key for request in requests} == {
        "drug_targets",
        "target_mapping",
    }


def test_register_inputs_rejects_filename_collision_before_copying(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same.tsv"
    second = second_dir / "same.tsv"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    result = register_inputs(
        context=context,
        inputs=[
            InputRegistrationRequest(input_key="compound_library", path=first),
            InputRegistrationRequest(input_key="disease_genes", path=second),
        ],
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert not list((context.run_dir / "inputs/original").glob("*"))


def test_register_inputs_never_overwrites_existing_original(tmp_path: Path) -> None:
    context = _context(tmp_path)
    original_dir = context.run_dir / "inputs/original"
    original_dir.mkdir(parents=True)
    existing = original_dir / "library.csv"
    existing.write_text("preserve-me", encoding="utf-8")
    upload = context.input_dir / "library.csv"
    upload.write_text("replacement", encoding="utf-8")

    result = register_inputs(
        context=context,
        inputs=[InputRegistrationRequest(input_key="compound_library", path=upload)],
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert existing.read_text(encoding="utf-8") == "preserve-me"
    assert not (context.run_dir / "inputs/input_manifest.json").exists()


def test_register_inputs_never_overwrites_existing_manifest(tmp_path: Path) -> None:
    context = _context(tmp_path)
    manifest_path = context.run_dir / "inputs/input_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = b'{"preserve": true}\n'
    manifest_path.write_bytes(sentinel)
    upload = context.input_dir / "library.csv"
    upload.write_text("ID,SMILES\ncompound-a,CCO\n", encoding="utf-8")

    result = register_inputs(
        context=context,
        inputs=[InputRegistrationRequest(input_key="compound_library", path=upload)],
        config_hash=_config_hash(),
        code_version="test-version",
    )

    assert result.status is NodeStatus.FAILED
    assert manifest_path.read_bytes() == sentinel
    assert not (context.run_dir / "inputs/original/library.csv").exists()


def test_registration_request_rejects_untrusted_traversal_filename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("data", encoding="utf-8")

    with pytest.raises(InputError):
        InputRegistrationRequest(
            input_key="compound_library",
            path=source,
            original_name="../escape.csv",
        )


def test_register_inputs_cli_success_and_execution_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)
    upload = context.input_dir / "library.csv"
    shutil.copy2(FIXTURES / "registration_upload.txt", upload)

    exit_code = main(
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
            str(CONFIG),
            "--compound-library",
            str(upload),
            "--input-artifact-id",
            "run-manifest",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    artifact_path = (
        context.run_dir
        / "artifacts/manifests/register_inputs/main"
        / f"{payload['outputs'][0]}.json"
    )
    artifact = load_artifact_manifest(artifact_path, run_root=context.run_dir)
    assert artifact.input_artifact_ids == ("run-manifest",)


def test_register_cli_rejects_expression_filename_case_mismatch_structurally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)
    upload = context.input_dir / "tpm_matrix_1.tsv"
    upload.write_text("GeneID\ts1\ts2\n1\t1\t2\n", encoding="utf-8")

    exit_code = main(
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
            str(CONFIG),
            "--expression-file",
            str(upload),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error"]["category"] == "input"


def test_expression_pair_id_has_a_portable_length_limit(tmp_path: Path) -> None:
    source = tmp_path / "TPM_matrix.tsv"
    source.write_text("data", encoding="utf-8")

    with pytest.raises(InputError, match="128"):
        InputRegistrationRequest(
            input_key="expression_pairs",
            path=source,
            role="tpm",
            pair_id="x" * 129,
        )
