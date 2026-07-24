import json
import shutil
from pathlib import Path

import pytest

from lipid_screening_agent.artifacts import NodeStatus, load_artifact_manifest
from lipid_screening_agent.runners.input_prepare.prepare_expression_inputs import (
    OUTPUT_RELATIVE_PATH,
    build_parser,
    main,
    prepare_expression_inputs,
)
from lipid_screening_agent.runtime import RunContext, sha256_bytes, sha256_file

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "small_inputs"


def _context(tmp_path: Path) -> RunContext:
    project = tmp_path / "project"
    resource = tmp_path / "resources"
    project.mkdir()
    resource.mkdir()
    return RunContext.create(
        runs_root=project / "runs",
        run_id="expression-run",
        project_root=project,
        resource_dir=resource,
        output_dir=project / "runs/expression-run/inputs/prepared",
    )


def _original_input_dir(context: RunContext) -> Path:
    original = context.input_dir / "original"
    original.mkdir()
    return original


def _copy_pair(directory: Path) -> None:
    for name in ("TPM_matrix_1.tsv", "metadata_1.tsv"):
        shutil.copy2(FIXTURES / name, directory / name)


def _run(context: RunContext, input_dir: Path, *, skip: bool = False):
    return prepare_expression_inputs(
        context=context,
        input_dir=input_dir,
        no_expression_data=skip,
        config_hash=sha256_bytes(b"test-config"),
        code_version="test-version",
        input_artifact_ids=("registered-expression-inputs",),
    )


def test_prepares_manifest_metrics_hashes_and_artifact(tmp_path: Path) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    _copy_pair(original)

    result = _run(context, original)

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics == {
        "input_file_count": 2,
        "orphan_file_count": 0,
        "comparison_count": 1,
        "sample_count": 2,
        "control_sample_count": 1,
        "disease_sample_count": 1,
        "tpm_extra_sample_count": 1,
    }
    assert len(result.outputs) == 1

    output_path = context.resolve_run_relative(OUTPUT_RELATIVE_PATH, must_exist=True)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["comparisons"] == [
        {
            "comparison_id": "comparison_1",
            "tpm_path": "inputs/original/TPM_matrix_1.tsv",
            "metadata_path": "inputs/original/metadata_1.tsv",
            "sample_counts": {"total": 2, "control": 1, "disease": 1},
            "sha256": {
                "tpm": sha256_file(original / "TPM_matrix_1.tsv"),
                "metadata": sha256_file(original / "metadata_1.tsv"),
            },
        }
    ]

    manifest_path = (
        context.run_dir
        / "artifacts"
        / "manifests"
        / "prepare_expression_inputs"
        / "main"
        / f"{result.outputs[0]}.json"
    )
    artifact = load_artifact_manifest(manifest_path, run_root=context.run_dir)
    assert artifact.artifact_type == "expression_comparisons_manifest"
    assert artifact.sha256 == sha256_file(output_path)
    assert artifact.input_artifact_ids == ("registered-expression-inputs",)


@pytest.mark.parametrize("orphan_name", ["TPM_matrix_1.tsv", "metadata_1.tsv"])
def test_rejects_orphan_expression_file(tmp_path: Path, orphan_name: str) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    shutil.copy2(FIXTURES / orphan_name, original / orphan_name)

    result = _run(context, original)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.category.value == "input"
    assert result.metrics["orphan_file_count"] == 1
    assert not context.resolve_run_relative(OUTPUT_RELATIVE_PATH).exists()


def test_explicit_no_expression_data_returns_skipped_without_artifact(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)

    result = _run(context, original, skip=True)

    assert result.status is NodeStatus.SKIPPED
    assert result.metrics == {
        "input_file_count": 0,
        "comparison_count": 0,
        "sample_count": 0,
        "control_sample_count": 0,
        "disease_sample_count": 0,
        "tpm_extra_sample_count": 0,
        "orphan_file_count": 0,
    }
    assert result.outputs == ()
    assert not context.resolve_run_relative(OUTPUT_RELATIVE_PATH).exists()


def test_metadata_samples_must_be_a_subset_not_equal_to_tpm_samples(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    _copy_pair(original)

    result = _run(context, original)

    assert result.status is NodeStatus.SUCCEEDED
    assert result.metrics["sample_count"] == 2
    assert result.metrics["tpm_extra_sample_count"] == 1


def test_rejects_comparison_id_collision_between_base_and_suffix_one(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    _copy_pair(original)
    shutil.copy2(FIXTURES / "TPM_matrix_1.tsv", original / "TPM_matrix.tsv")
    shutil.copy2(FIXTURES / "metadata_1.tsv", original / "metadata.tsv")

    result = _run(context, original)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert "duplicate comparison IDs" in result.error.message


def test_rejects_nonportable_comparison_suffix(tmp_path: Path) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    shutil.copy2(FIXTURES / "TPM_matrix_1.tsv", original / "TPM_matrix_bad..tsv")
    shutil.copy2(FIXTURES / "metadata_1.tsv", original / "metadata_bad..tsv")

    result = _run(context, original)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert "portable comparison ID" in result.error.message


def test_rejects_metadata_sample_missing_from_tpm_header(tmp_path: Path) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    shutil.copy2(FIXTURES / "TPM_matrix_1.tsv", original / "TPM_matrix_1.tsv")
    (original / "metadata_1.tsv").write_text(
        "sample_id\tgroup\ncontrol_1\tcontrol\nmissing_1\tdisease\n",
        encoding="utf-8",
    )

    result = _run(context, original)

    assert result.status is NodeStatus.FAILED
    assert result.error is not None
    assert result.error.details["missing_sample_ids"] == ("missing_1",)


def test_cli_parser_reuses_common_arguments_and_supports_explicit_skip(
    tmp_path: Path,
) -> None:
    namespace = build_parser().parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--input-dir",
            str(tmp_path / "run" / "inputs" / "original"),
            "--resource-dir",
            str(tmp_path / "resources"),
            "--output-dir",
            str(tmp_path / "run" / "inputs" / "prepared"),
            "--config",
            str(tmp_path / "workflow.yaml"),
            "--no-expression-data",
        ]
    )

    assert namespace.no_expression_data is True


def _cli_arguments(context: RunContext, original: Path) -> list[str]:
    return [
        "--run-dir",
        str(context.run_dir),
        "--input-dir",
        str(original),
        "--resource-dir",
        str(context.resource_dir),
        "--output-dir",
        str(context.input_dir / "prepared"),
        "--config",
        str(Path(__file__).resolve().parents[4] / "configs" / "workflow.yaml"),
    ]


def test_cli_main_prints_successful_node_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)
    _copy_pair(original)

    exit_code = main(_cli_arguments(context, original))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["node_id"] == "prepare_expression_inputs"
    assert payload["status"] == "succeeded"


def test_cli_main_returns_one_for_structured_input_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context(tmp_path)
    original = _original_input_dir(context)

    exit_code = main(_cli_arguments(context, original))

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error"]["category"] == "input"
