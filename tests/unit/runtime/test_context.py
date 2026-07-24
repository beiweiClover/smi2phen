from __future__ import annotations

from pathlib import Path

import pytest

from lipid_screening_agent.runtime.context import RunContext
from lipid_screening_agent.runtime.errors import (
    ConfigurationError,
    ExecutionError,
    PathSafetyError,
)


def _boundaries(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    source = project / "src"
    resources = tmp_path / "resources"
    source.mkdir(parents=True)
    resources.mkdir()
    return project, source, resources


def test_create_builds_isolated_project_local_workspace(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)

    context = RunContext.create(
        runs_root=project / "runs",
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )

    assert context.run_dir == (project / "runs" / "run-001").resolve()
    assert context.input_dir == context.run_dir / "inputs"
    assert context.artifact_dir == context.run_dir / "artifacts"
    assert context.log_dir == context.run_dir / "logs"
    assert context.output_dir == context.artifact_dir
    assert context.resource_dir == resources.resolve()
    assert all(path.is_dir() for path in (context.input_dir, context.artifact_dir, context.log_dir))
    assert context.resolve_artifact("kg/result.csv") == (context.artifact_dir / "kg" / "result.csv")


@pytest.mark.parametrize("run_id", ["../escape", "bad/id", r"bad\id", "CON", " space"])
def test_create_rejects_nonportable_run_id(tmp_path: Path, run_id: str) -> None:
    project, source, resources = _boundaries(tmp_path)

    with pytest.raises(ConfigurationError):
        RunContext.create(
            runs_root=project / "runs",
            run_id=run_id,
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )


def test_create_does_not_reuse_existing_run_by_default(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)
    arguments = {
        "runs_root": project / "runs",
        "run_id": "run-001",
        "project_root": project,
        "source_roots": (source,),
        "resource_dir": resources,
    }
    RunContext.create(**arguments)

    with pytest.raises(ExecutionError, match="already exists"):
        RunContext.create(**arguments)


def test_create_rejects_disk_and_project_roots(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)

    with pytest.raises(PathSafetyError, match="project root"):
        RunContext.create(
            runs_root=project,
            run_id="run-001",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )
    with pytest.raises(PathSafetyError, match="filesystem root"):
        RunContext.create(
            runs_root=Path(tmp_path.anchor),
            run_id="run-001",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )


def test_create_rejects_source_and_resource_tree_overlap(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)

    with pytest.raises(PathSafetyError, match="source root"):
        RunContext.create(
            runs_root=source / "runs",
            run_id="run-001",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )
    with pytest.raises(PathSafetyError, match="resource root"):
        RunContext.create(
            runs_root=resources / "runs",
            run_id="run-001",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )


def test_create_rejects_explicit_output_outside_run(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)

    with pytest.raises(PathSafetyError):
        RunContext.create(
            runs_root=project / "runs",
            run_id="run-001",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
            output_dir=tmp_path / "outside",
        )


def test_open_existing_round_trip_and_run_id_check(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)
    created = RunContext.create(
        runs_root=project / "runs",
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )

    opened = RunContext.open_existing(
        run_dir=created.run_dir,
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )
    assert opened.to_dict() == created.to_dict()

    with pytest.raises(ConfigurationError, match="does not match"):
        RunContext.open_existing(
            run_dir=created.run_dir,
            run_id="another-run",
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )


def test_for_runner_accepts_explicit_dirs_and_creates_writable_subdirs(
    tmp_path: Path,
) -> None:
    project, source, resources = _boundaries(tmp_path)
    created = RunContext.create(
        runs_root=project / "runs",
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )
    output = created.run_dir / "inputs" / "prepared"

    runner_context = RunContext.for_runner(
        run_dir=created.run_dir,
        input_dir=created.input_dir,
        resource_dir=resources,
        output_dir=output,
        project_root=project,
        source_roots=(source,),
    )

    assert runner_context.output_dir == output.resolve()
    assert output.is_dir()


def test_for_runner_rejects_output_that_overlaps_immutable_original_inputs(
    tmp_path: Path,
) -> None:
    project, source, resources = _boundaries(tmp_path)
    created = RunContext.create(
        runs_root=project / "runs",
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )
    original = created.input_dir / "original"
    original.mkdir()

    with pytest.raises(PathSafetyError, match="immutable original"):
        RunContext.for_runner(
            run_dir=created.run_dir,
            input_dir=created.input_dir,
            resource_dir=resources,
            output_dir=original,
            project_root=project,
            source_roots=(source,),
        )


def test_open_existing_does_not_create_missing_directories_by_default(
    tmp_path: Path,
) -> None:
    project, source, resources = _boundaries(tmp_path)
    run = project / "runs" / "manual-run"
    (run / "inputs").mkdir(parents=True)

    with pytest.raises(ConfigurationError, match="artifact dir does not exist"):
        RunContext.open_existing(
            run_dir=run,
            project_root=project,
            source_roots=(source,),
            resource_dir=resources,
        )


def test_for_runner_rejects_symlink_output_escape(tmp_path: Path) -> None:
    project, source, resources = _boundaries(tmp_path)
    created = RunContext.create(
        runs_root=project / "runs",
        run_id="run-001",
        project_root=project,
        source_roots=(source,),
        resource_dir=resources,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    output_link = created.run_dir / "linked-output"
    try:
        output_link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available for this test user")

    with pytest.raises(PathSafetyError):
        RunContext.for_runner(
            run_dir=created.run_dir,
            input_dir=created.input_dir,
            resource_dir=resources,
            output_dir=output_link,
            project_root=project,
            source_roots=(source,),
        )
