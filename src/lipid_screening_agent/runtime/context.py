"""Isolated run-workspace context shared by CLI runners and the future workflow engine."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

from .errors import ConfigurationError, ExecutionError, PathSafetyError, ResourceError
from .paths import (
    PathLike,
    canonical_path,
    ensure_within,
    is_filesystem_root,
    paths_overlap,
    resolve_run_relative,
    to_run_relative_posix,
    validate_portable_segment,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_RUN_ID_LENGTH = 128


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise ConfigurationError("run_id must be a string")
    try:
        validate_portable_segment(run_id, label="run_id")
    except PathSafetyError as exc:
        raise ConfigurationError(str(exc)) from exc
    if len(run_id) > _MAX_RUN_ID_LENGTH or not _RUN_ID.fullmatch(run_id):
        raise ConfigurationError(
            "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ and be at most 128 characters"
        )
    return run_id


def _existing_directory(
    value: PathLike,
    *,
    label: str,
    error_type: type[ConfigurationError] | type[ResourceError],
) -> Path:
    try:
        path = canonical_path(value, must_exist=True, label=label)
    except PathSafetyError as exc:
        raise error_type(str(exc)) from exc
    if not path.is_dir():
        raise error_type(f"{label} is not a directory: {path}")
    return path


def _canonical_source_roots(
    project_root: Path,
    source_roots: Sequence[PathLike] | None,
) -> tuple[Path, ...]:
    candidates: list[PathLike] = []
    project_source = project_root / "src"
    if project_source.exists():
        candidates.append(project_source)

    # This is derived from the installed module location, not from the current working directory.
    package_source = Path(__file__).resolve().parents[2]
    candidates.append(package_source)
    candidates.extend(source_roots or ())

    canonical: list[Path] = []
    for candidate in candidates:
        source = _existing_directory(
            candidate,
            label="source root",
            error_type=ConfigurationError,
        )
        if source not in canonical:
            canonical.append(source)
    return tuple(canonical)


def _canonical_resource_roots(
    resource_dir: PathLike | None,
    resource_roots: Sequence[PathLike] | None,
) -> tuple[Path, ...]:
    candidates: list[PathLike] = list(resource_roots or ())
    if resource_dir is not None:
        candidates.append(resource_dir)

    canonical: list[Path] = []
    for candidate in candidates:
        resource = _existing_directory(
            candidate,
            label="resource root",
            error_type=ResourceError,
        )
        if resource not in canonical:
            canonical.append(resource)
    return tuple(canonical)


def _assert_safe_run_location(
    *,
    run_dir: Path,
    project_root: Path,
    source_roots: Sequence[Path],
    resource_roots: Sequence[Path],
) -> None:
    if is_filesystem_root(run_dir):
        raise PathSafetyError("run directory cannot be a filesystem root")
    if run_dir == project_root:
        raise PathSafetyError("run directory cannot be the project root")
    for source_root in source_roots:
        if paths_overlap(run_dir, source_root):
            raise PathSafetyError(
                f"run directory overlaps protected source root: {run_dir} and {source_root}"
            )
    for resource_root in resource_roots:
        if paths_overlap(run_dir, resource_root):
            raise PathSafetyError(
                f"run directory overlaps protected resource root: {run_dir} and {resource_root}"
            )


def _safe_child_directory(path: PathLike, *, run_dir: Path, label: str) -> Path:
    candidate = ensure_within(path, run_dir)
    # This also rejects an existing symlink component and verifies a stable portable spelling.
    to_run_relative_posix(Path(path), run_dir)
    if candidate.exists() and not candidate.is_dir():
        raise PathSafetyError(f"{label} is not a directory: {candidate}")
    return candidate


def _mkdir_checked(path: Path, *, run_dir: Path) -> Path:
    ensure_within(path, run_dir)
    path.mkdir(parents=True, exist_ok=True)
    return ensure_within(path, run_dir)


def _reject_immutable_input_overlap(path: Path, *, run_dir: Path, label: str) -> None:
    original_inputs = canonical_path(
        run_dir / "inputs" / "original",
        label="immutable original input root",
    )
    if paths_overlap(path, original_inputs):
        raise PathSafetyError(f"{label} overlaps immutable original inputs: {path}")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Canonical paths and identity for one isolated workflow run."""

    run_id: str
    runs_root: Path
    run_dir: Path
    input_dir: Path
    resource_dir: Path | None
    artifact_dir: Path
    log_dir: Path
    output_dir: Path
    project_root: Path
    source_roots: tuple[Path, ...]
    resource_roots: tuple[Path, ...]

    @classmethod
    def create(
        cls,
        *,
        runs_root: PathLike,
        run_id: str,
        project_root: PathLike,
        source_roots: Sequence[PathLike] | None = None,
        resource_roots: Sequence[PathLike] | None = None,
        resource_dir: PathLike | None = None,
        input_dir: PathLike | None = None,
        artifact_dir: PathLike | None = None,
        log_dir: PathLike | None = None,
        output_dir: PathLike | None = None,
        exist_ok: bool = False,
    ) -> RunContext:
        """Create a new isolated run workspace and its lightweight directory layout."""

        valid_run_id = _validate_run_id(run_id)
        project = _existing_directory(
            project_root,
            label="project root",
            error_type=ConfigurationError,
        )
        sources = _canonical_source_roots(project, source_roots)
        resources = _canonical_resource_roots(resource_dir, resource_roots)

        root = canonical_path(runs_root, label="runs root")
        if is_filesystem_root(root):
            raise PathSafetyError("runs root cannot be a filesystem root")
        if root == project:
            raise PathSafetyError("runs root cannot be the project root")

        proposed_run = canonical_path(root / valid_run_id, label="run directory")
        _assert_safe_run_location(
            run_dir=proposed_run,
            project_root=project,
            source_roots=sources,
            resource_roots=resources,
        )

        root.mkdir(parents=True, exist_ok=True)
        root = canonical_path(root, must_exist=True, label="runs root")
        raw_run_dir = root / valid_run_id
        if raw_run_dir.is_symlink():
            raise PathSafetyError(f"run directory cannot be a symlink: {raw_run_dir}")
        if raw_run_dir.exists() and not exist_ok:
            raise ExecutionError(
                f"run directory already exists: {raw_run_dir}",
                retryable=False,
            )
        try:
            raw_run_dir.mkdir(exist_ok=exist_ok)
        except OSError as exc:
            raise ExecutionError(f"cannot create run directory: {raw_run_dir}") from exc

        run = ensure_within(raw_run_dir, root)
        _assert_safe_run_location(
            run_dir=run,
            project_root=project,
            source_roots=sources,
            resource_roots=resources,
        )

        inputs = _safe_child_directory(input_dir or run / "inputs", run_dir=run, label="input dir")
        artifacts = _safe_child_directory(
            artifact_dir or run / "artifacts",
            run_dir=run,
            label="artifact dir",
        )
        logs = _safe_child_directory(log_dir or run / "logs", run_dir=run, label="log dir")
        outputs = _safe_child_directory(
            output_dir or artifacts,
            run_dir=run,
            label="output dir",
        )
        for label, directory in (
            ("artifact dir", artifacts),
            ("log dir", logs),
            ("output dir", outputs),
        ):
            _reject_immutable_input_overlap(directory, run_dir=run, label=label)
        for directory in dict.fromkeys((inputs, artifacts, logs, outputs)):
            _mkdir_checked(directory, run_dir=run)

        selected_resource = resources[0] if resource_dir is None and resources else resource_dir
        canonical_resource = (
            None
            if selected_resource is None
            else _existing_directory(
                selected_resource,
                label="resource dir",
                error_type=ResourceError,
            )
        )
        return cls(
            run_id=valid_run_id,
            runs_root=root,
            run_dir=run,
            input_dir=inputs,
            resource_dir=canonical_resource,
            artifact_dir=artifacts,
            log_dir=logs,
            output_dir=outputs,
            project_root=project,
            source_roots=sources,
            resource_roots=resources,
        )

    @classmethod
    def open_existing(
        cls,
        *,
        run_dir: PathLike,
        project_root: PathLike,
        run_id: str | None = None,
        input_dir: PathLike | None = None,
        resource_dir: PathLike | None = None,
        artifact_dir: PathLike | None = None,
        log_dir: PathLike | None = None,
        output_dir: PathLike | None = None,
        source_roots: Sequence[PathLike] | None = None,
        resource_roots: Sequence[PathLike] | None = None,
        create_missing_directories: bool = False,
    ) -> RunContext:
        """Validate and reopen a workspace without relying on the process working directory."""

        raw_run = Path(run_dir)
        run = _existing_directory(
            run_dir,
            label="run directory",
            error_type=ConfigurationError,
        )
        if raw_run.is_symlink():
            raise PathSafetyError(f"run directory cannot be a symlink: {raw_run}")
        valid_run_id = _validate_run_id(run.name if run_id is None else run_id)
        if run.name != valid_run_id:
            raise ConfigurationError(
                f"run_id {valid_run_id!r} does not match run directory name {run.name!r}"
            )
        project = _existing_directory(
            project_root,
            label="project root",
            error_type=ConfigurationError,
        )
        sources = _canonical_source_roots(project, source_roots)
        resources = _canonical_resource_roots(resource_dir, resource_roots)
        _assert_safe_run_location(
            run_dir=run,
            project_root=project,
            source_roots=sources,
            resource_roots=resources,
        )

        root = canonical_path(run.parent, must_exist=True, label="runs root")
        if is_filesystem_root(root):
            raise PathSafetyError("runs root cannot be a filesystem root")
        if root == project:
            raise PathSafetyError("runs root cannot be the project root")

        inputs = _safe_child_directory(input_dir or run / "inputs", run_dir=run, label="input dir")
        artifacts = _safe_child_directory(
            artifact_dir or run / "artifacts",
            run_dir=run,
            label="artifact dir",
        )
        logs = _safe_child_directory(log_dir or run / "logs", run_dir=run, label="log dir")
        outputs = _safe_child_directory(
            output_dir or artifacts,
            run_dir=run,
            label="output dir",
        )
        for label, directory in (
            ("artifact dir", artifacts),
            ("log dir", logs),
            ("output dir", outputs),
        ):
            _reject_immutable_input_overlap(directory, run_dir=run, label=label)

        if create_missing_directories:
            for directory in dict.fromkeys((artifacts, logs, outputs)):
                _mkdir_checked(directory, run_dir=run)
        for label, directory in (
            ("input dir", inputs),
            ("artifact dir", artifacts),
            ("log dir", logs),
            ("output dir", outputs),
        ):
            if not directory.is_dir():
                raise ConfigurationError(f"{label} does not exist: {directory}")

        canonical_resource = (
            None
            if resource_dir is None
            else _existing_directory(
                resource_dir,
                label="resource dir",
                error_type=ResourceError,
            )
        )
        if canonical_resource is None and resources:
            canonical_resource = resources[0]
        return cls(
            run_id=valid_run_id,
            runs_root=root,
            run_dir=run,
            input_dir=inputs,
            resource_dir=canonical_resource,
            artifact_dir=artifacts,
            log_dir=logs,
            output_dir=outputs,
            project_root=project,
            source_roots=sources,
            resource_roots=resources,
        )

    @classmethod
    def for_runner(
        cls,
        *,
        run_dir: PathLike,
        input_dir: PathLike,
        resource_dir: PathLike,
        output_dir: PathLike,
        project_root: PathLike,
        run_id: str | None = None,
        artifact_dir: PathLike | None = None,
        log_dir: PathLike | None = None,
        source_roots: Sequence[PathLike] | None = None,
        resource_roots: Sequence[PathLike] | None = None,
    ) -> RunContext:
        """Open an existing run for a CLI runner, creating only writable runtime directories."""

        return cls.open_existing(
            run_dir=run_dir,
            project_root=project_root,
            run_id=run_id,
            input_dir=input_dir,
            resource_dir=resource_dir,
            artifact_dir=artifact_dir,
            log_dir=log_dir,
            output_dir=output_dir,
            source_roots=source_roots,
            resource_roots=resource_roots,
            create_missing_directories=True,
        )

    def resolve_run_relative(
        self,
        value: str | PurePath,
        *,
        must_exist: bool = False,
    ) -> Path:
        return resolve_run_relative(self.run_dir, value, must_exist=must_exist)

    def resolve_input(self, value: str | PurePath, *, must_exist: bool = False) -> Path:
        return resolve_run_relative(self.input_dir, value, must_exist=must_exist)

    def resolve_artifact(self, value: str | PurePath, *, must_exist: bool = False) -> Path:
        return resolve_run_relative(self.artifact_dir, value, must_exist=must_exist)

    def resolve_log(self, value: str | PurePath, *, must_exist: bool = False) -> Path:
        return resolve_run_relative(self.log_dir, value, must_exist=must_exist)

    def resolve_output(self, value: str | PurePath, *, must_exist: bool = False) -> Path:
        return resolve_run_relative(self.output_dir, value, must_exist=must_exist)

    def relative_path(self, path: PathLike) -> str:
        """Return the stable manifest spelling for a path inside this run."""

        return to_run_relative_posix(path, self.run_dir)

    def to_dict(self) -> dict[str, object]:
        """Return stable primitive values suitable for logs and diagnostic JSON."""

        return {
            "run_id": self.run_id,
            "runs_root": str(self.runs_root),
            "run_dir": str(self.run_dir),
            "input_dir": str(self.input_dir),
            "resource_dir": (None if self.resource_dir is None else str(self.resource_dir)),
            "artifact_dir": str(self.artifact_dir),
            "log_dir": str(self.log_dir),
            "output_dir": str(self.output_dir),
            "project_root": str(self.project_root),
            "source_roots": [str(path) for path in self.source_roots],
            "resource_roots": [str(path) for path in self.resource_roots],
        }


__all__ = ["RunContext"]
