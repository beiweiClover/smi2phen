"""Reusable command-line arguments shared by every registered runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lipid_screening_agent.config import (
    WorkflowConfig,
    hash_workflow_config,
    load_workflow_config,
)
from lipid_screening_agent.runtime.context import RunContext
from lipid_screening_agent.runtime.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class CommonRunnerArguments:
    """The five explicit paths required by every scientific runner."""

    run_dir: Path
    input_dir: Path
    resource_dir: Path
    output_dir: Path
    config: Path

    def __post_init__(self) -> None:
        non_absolute: list[str] = []
        for name in ("run_dir", "input_dir", "resource_dir", "output_dir", "config"):
            value = Path(getattr(self, name))
            object.__setattr__(self, name, value)
            if not value.is_absolute():
                non_absolute.append(name)
        if non_absolute:
            joined = ", ".join(f"--{name.replace('_', '-')}" for name in non_absolute)
            raise ConfigurationError(f"runner paths must be absolute: {joined}")

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> CommonRunnerArguments:
        values = {
            field: Path(getattr(namespace, field))
            for field in (
                "run_dir",
                "input_dir",
                "resource_dir",
                "output_dir",
                "config",
            )
        }
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CommonRunnerEnvironment:
    """Validated common paths plus the typed, canonically hashed workflow config."""

    arguments: CommonRunnerArguments
    context: RunContext
    config: WorkflowConfig
    config_hash: str


def add_common_runner_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the frozen common path interface to an existing parser."""

    parser.add_argument("--run-dir", required=True, type=Path, help="Existing run workspace")
    parser.add_argument("--input-dir", required=True, type=Path, help="Runner input boundary")
    parser.add_argument(
        "--resource-dir", required=True, type=Path, help="Read-only resource boundary"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Runner output boundary")
    parser.add_argument("--config", required=True, type=Path, help="Workflow YAML configuration")
    return parser


def common_runner_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    """Create a parser containing only the shared runner arguments."""

    return add_common_runner_arguments(argparse.ArgumentParser(description=description))


def parse_common_runner_arguments(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> CommonRunnerArguments:
    """Parse and type-check the common runner path arguments."""

    namespace = common_runner_parser(description=description).parse_args(argv)
    return CommonRunnerArguments.from_namespace(namespace)


def load_common_runner_environment(
    arguments: CommonRunnerArguments,
    *,
    project_root: str | os.PathLike[str],
    run_id: str | None = None,
    source_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> CommonRunnerEnvironment:
    """Open a safe runner context and load config without resolving unrelated resources."""

    config = load_workflow_config(arguments.config)
    context = RunContext.for_runner(
        run_dir=arguments.run_dir,
        run_id=run_id,
        input_dir=arguments.input_dir,
        resource_dir=arguments.resource_dir,
        output_dir=arguments.output_dir,
        project_root=project_root,
        source_roots=source_roots,
    )
    return CommonRunnerEnvironment(
        arguments=arguments,
        context=context,
        config=config,
        config_hash=hash_workflow_config(config),
    )
