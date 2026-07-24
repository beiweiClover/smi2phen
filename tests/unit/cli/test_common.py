from pathlib import Path

import pytest

from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    load_common_runner_environment,
    parse_common_runner_arguments,
)
from lipid_screening_agent.runtime import RunContext
from lipid_screening_agent.runtime.errors import ConfigurationError


def test_common_runner_arguments_parse_all_explicit_paths(tmp_path: Path) -> None:
    arguments = parse_common_runner_arguments(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--input-dir",
            str(tmp_path / "run" / "inputs"),
            "--resource-dir",
            str(tmp_path / "resources"),
            "--output-dir",
            str(tmp_path / "run" / "artifacts"),
            "--config",
            str(tmp_path / "workflow.yaml"),
        ]
    )

    assert arguments.run_dir == tmp_path / "run"
    assert arguments.output_dir == tmp_path / "run" / "artifacts"


def test_common_runner_arguments_require_every_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_common_runner_arguments(["--run-dir", str(tmp_path / "run")])


def test_common_runner_arguments_reject_relative_paths() -> None:
    with pytest.raises(ConfigurationError, match="absolute"):
        parse_common_runner_arguments(
            [
                "--run-dir",
                "runs/example",
                "--input-dir",
                "runs/example/inputs",
                "--resource-dir",
                "resources",
                "--output-dir",
                "runs/example/artifacts",
                "--config",
                "configs/workflow.yaml",
            ]
        )


def test_load_common_runner_environment_reuses_typed_config_and_safe_context(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    resources = tmp_path / "resources"
    project.mkdir()
    resources.mkdir()
    created = RunContext.create(
        runs_root=project / "runs",
        run_id="run-01",
        project_root=project,
        resource_dir=resources,
    )
    config_path = Path(__file__).resolve().parents[3] / "configs" / "workflow.yaml"
    arguments = CommonRunnerArguments(
        run_dir=created.run_dir,
        input_dir=created.input_dir,
        resource_dir=resources.resolve(),
        output_dir=(created.input_dir / "prepared").resolve(),
        config=config_path.resolve(),
    )

    environment = load_common_runner_environment(arguments, project_root=project)

    assert environment.context.run_id == "run-01"
    assert environment.config.workflow.id == "lipid_screening_v2"
    assert len(environment.config_hash) == 64
