"""Small Stage 02 helpers shared by the input-preparation CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from lipid_screening_agent.runtime import OutputContractError, RunContext, ensure_within


def add_execution_identity_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add optional execution identity/provenance arguments after the five common paths."""

    parser.add_argument("--task-id", default="main")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--input-artifact-id",
        action="append",
        default=[],
        help="Upstream artifact instance ID; repeat for multiple inputs",
    )
    return parser


def execution_identity(
    namespace: argparse.Namespace,
) -> tuple[str, int, Sequence[str]]:
    """Return the normalized execution identity fields from an argparse namespace."""

    return (
        namespace.task_id,
        namespace.attempt,
        tuple(namespace.input_artifact_id),
    )


def resolve_prepared_output(context: RunContext, contract_path: str) -> Path:
    """Resolve a fixed prepared-input contract path within the CLI output boundary."""

    expected = context.resolve_run_relative(contract_path)
    prepared_root = context.resolve_run_relative("inputs/prepared")
    if context.output_dir != prepared_root:
        raise OutputContractError(
            "input-preparation output_dir must be the contracted inputs/prepared directory",
            details={
                "configured_output_dir": str(context.output_dir),
                "expected_output_dir": str(prepared_root),
            },
        )
    return ensure_within(expected, context.output_dir)


__all__ = [
    "add_execution_identity_arguments",
    "execution_identity",
    "resolve_prepared_output",
]
