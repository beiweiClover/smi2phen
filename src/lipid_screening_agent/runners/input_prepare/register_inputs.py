"""Register uploaded files into one run's immutable original-input area."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import CommonRunnerArguments, common_runner_parser
from lipid_screening_agent.cli.common import load_common_runner_environment
from lipid_screening_agent.runtime import (
    FileDigest,
    InputError,
    RunContext,
    atomic_write_json,
    canonical_path,
    ensure_within,
    isoformat_utc,
    utc_now,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ._common import add_execution_identity_arguments, execution_identity
from .models import (
    InputRegistrationManifest,
    InputRegistrationRequest,
    RegisteredInputRecord,
)

NODE_ID = "register_inputs"
MANIFEST_RELATIVE_PATH = "inputs/input_manifest.json"
ORIGINAL_INPUTS_RELATIVE_PATH = "inputs/original"
COPY_CHUNK_SIZE = 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[4]

_TPM_NAME = re.compile(r"^TPM_matrix(?:_(?P<suffix>.+))?\.tsv$")
_METADATA_NAME = re.compile(r"^metadata(?:_(?P<suffix>.+))?\.tsv$")


def _copy_file_exclusive(source: Path, destination: Path) -> FileDigest:
    """Stream one file to a newly created destination without ever overwriting."""

    before = source.stat()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor: int | None = None
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        descriptor = os.open(destination, flags, 0o644)
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target_handle:
            descriptor = None
            while chunk := source_handle.read(COPY_CHUNK_SIZE):
                target_handle.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        after = source.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or size_bytes != after.st_size
        ):
            raise InputError(
                f"source file changed while it was registered: {source}",
                retryable=False,
            )
    except FileExistsError as exc:
        raise InputError(
            f"registered filename already exists and cannot be overwritten: {destination.name}",
            retryable=False,
        ) from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise
    return FileDigest(size_bytes=size_bytes, sha256=digest.hexdigest())


def _canonical_source(path: Path) -> Path:
    try:
        source = canonical_path(path, must_exist=True, label="uploaded input")
    except Exception as exc:
        raise InputError(f"uploaded input is not an existing absolute file: {path}") from exc
    if not source.is_file():
        raise InputError(f"uploaded input is not a regular file: {source}")
    return source


def _preflight_requests(
    requests: Sequence[InputRegistrationRequest],
    *,
    original_dir: Path,
) -> list[tuple[InputRegistrationRequest, Path, Path, str]]:
    if not requests:
        raise InputError("at least one input file must be registered", retryable=False)

    seen_names: set[str] = set()
    seen_logical: set[tuple[str, str | None, str]] = set()
    prepared: list[tuple[InputRegistrationRequest, Path, Path, str]] = []
    for request in requests:
        if not isinstance(request, InputRegistrationRequest):
            raise InputError("register_inputs received an invalid request object")
        source = _canonical_source(Path(request.path))
        original_name = request.original_name or source.name
        try:
            validated = InputRegistrationRequest(
                input_key=request.input_key,
                path=source,
                source=request.source,
                role=request.role,
                pair_id=request.pair_id,
                original_name=original_name,
            )
        except InputError:
            raise

        name_key = original_name.casefold()
        if name_key in seen_names:
            raise InputError(
                f"registered filenames would collide: {original_name}",
                retryable=False,
            )
        seen_names.add(name_key)

        logical_key = (validated.input_key, validated.pair_id, validated.role)
        if logical_key in seen_logical:
            raise InputError(
                f"duplicate logical input registration: {logical_key}",
                retryable=False,
            )
        seen_logical.add(logical_key)

        destination = original_dir / original_name
        ensure_within(destination, original_dir)
        if destination.exists():
            raise InputError(
                f"registered filename already exists and cannot be overwritten: {original_name}",
                retryable=False,
            )
        prepared.append((validated, source, destination, original_name))
    return prepared


def register_inputs(
    *,
    context: RunContext,
    inputs: Sequence[InputRegistrationRequest],
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Copy uploaded files, write their provenance manifest, and commit one node result."""

    requests = tuple(inputs)

    def operation(execution: NodeExecution) -> None:
        execution.update_metrics(
            {
                "input_count": len(requests),
                "registered_count": 0,
                "expression_pair_count": 0,
                "incomplete_expression_pair_count": 0,
                "total_size_bytes": 0,
            }
        )
        original_dir = context.resolve_run_relative(ORIGINAL_INPUTS_RELATIVE_PATH)
        original_dir.mkdir(parents=True, exist_ok=True)
        original_dir = context.resolve_run_relative(ORIGINAL_INPUTS_RELATIVE_PATH, must_exist=True)
        manifest_path = context.resolve_run_relative(MANIFEST_RELATIVE_PATH)

        try:
            reservation = os.open(
                manifest_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError as exc:
            raise InputError(
                "input registration manifest already exists and cannot be overwritten",
                retryable=False,
            ) from exc
        else:
            os.close(reservation)

        created: list[Path] = []
        try:
            prepared = _preflight_requests(requests, original_dir=original_dir)
            records: list[RegisteredInputRecord] = []
            for request, source, destination, original_name in prepared:
                digest = _copy_file_exclusive(source, destination)
                created.append(destination)
                records.append(
                    RegisteredInputRecord(
                        input_key=request.input_key,
                        role=request.role,
                        pair_id=request.pair_id,
                        original_name=original_name,
                        registered_path=context.relative_path(destination),
                        size_bytes=digest.size_bytes,
                        sha256=digest.sha256,
                        registered_at=isoformat_utc(utc_now()),
                        source=request.source,
                    )
                )

            manifest = InputRegistrationManifest(schema_version="1.0", inputs=records)
            atomic_write_json(
                manifest_path,
                manifest.to_dict(),
                allowed_root=context.run_dir,
            )

            pair_roles: dict[str, set[str]] = defaultdict(set)
            for record in records:
                if record.pair_id is not None:
                    pair_roles[record.pair_id].add(record.role)
            incomplete_pairs = sorted(
                pair_id for pair_id, roles in pair_roles.items() if roles != {"tpm", "metadata"}
            )
            if incomplete_pairs:
                sample = ", ".join(incomplete_pairs[:12])
                omitted = len(incomplete_pairs) - min(len(incomplete_pairs), 12)
                suffix = "" if omitted == 0 else f" (+{omitted} more)"
                execution.warn(
                    f"expression registration contains incomplete pairs: {sample}{suffix}"
                )

            execution.update_metrics(
                {
                    "registered_count": len(records),
                    "expression_pair_count": len(pair_roles),
                    "incomplete_expression_pair_count": len(incomplete_pairs),
                    "total_size_bytes": sum(record.size_bytes for record in records),
                }
            )
            execution.add_output("input_registration_manifest", manifest_path)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            for created_path in reversed(created):
                created_path.unlink(missing_ok=True)
            raise

    return execute_node(
        operation,
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def _expression_request(path: Path, *, source: str) -> InputRegistrationRequest:
    match = _TPM_NAME.fullmatch(path.name)
    role = "tpm"
    if match is None:
        match = _METADATA_NAME.fullmatch(path.name)
        role = "metadata"
    if match is None:
        raise InputError(
            "expression filename must be TPM_matrix[_{suffix}].tsv or metadata[_{suffix}].tsv"
        )
    suffix = match.group("suffix")
    pair_id = f"comparison_{suffix or '1'}"
    return InputRegistrationRequest(
        input_key="expression_pairs",
        path=path,
        source=source,
        role=role,
        pair_id=pair_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = common_runner_parser(description="Register uploaded workflow inputs")
    parser.add_argument("--compound-library", type=Path)
    parser.add_argument("--disease-genes", type=Path)
    parser.add_argument("--drug-targets", type=Path)
    parser.add_argument("--target-mapping", type=Path)
    parser.add_argument("--expression-file", action="append", type=Path, default=[])
    parser.add_argument("--positive-drugs", type=Path)
    parser.add_argument("--disease-links", type=Path)
    parser.add_argument(
        "--source",
        choices=(
            "user_upload",
            "user_approved_research",
            "agent_research",
            "explicit_fixture",
        ),
        default="user_upload",
    )
    add_execution_identity_arguments(parser)
    return parser


def _requests_from_namespace(
    namespace: argparse.Namespace,
) -> tuple[InputRegistrationRequest, ...]:
    requests: list[InputRegistrationRequest] = []
    for attribute, input_key in (
        ("compound_library", "compound_library"),
        ("disease_genes", "disease_genes"),
        ("drug_targets", "drug_targets"),
        ("target_mapping", "target_mapping"),
        ("positive_drugs", "positive_drugs"),
        ("disease_links", "disease_links"),
    ):
        path = getattr(namespace, attribute)
        if path is not None:
            requests.append(
                InputRegistrationRequest(
                    input_key=input_key,
                    path=path,
                    source=namespace.source,
                )
            )
    requests.extend(
        _expression_request(path, source=namespace.source) for path in namespace.expression_file
    )
    return tuple(requests)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    environment = load_common_runner_environment(arguments, project_root=PROJECT_ROOT)
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    try:
        requests = _requests_from_namespace(namespace)
    except InputError as request_error:

        def fail_invalid_request(
            execution: NodeExecution,
            error: InputError = request_error,
        ) -> None:
            execution.update_metrics(
                {
                    "input_count": 0,
                    "registered_count": 0,
                    "expression_pair_count": 0,
                    "incomplete_expression_pair_count": 0,
                    "total_size_bytes": 0,
                }
            )
            raise error

        result = execute_node(
            fail_invalid_request,
            context=environment.context,
            node_id=NODE_ID,
            task_id=task_id,
            attempt=attempt,
            config_hash=environment.config_hash,
            code_version=__version__,
            input_artifact_ids=input_artifact_ids,
        )
    else:
        result = register_inputs(
            context=environment.context,
            inputs=requests,
            config_hash=environment.config_hash,
            code_version=__version__,
            task_id=task_id,
            attempt=attempt,
            input_artifact_ids=input_artifact_ids,
        )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.SKIPPED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InputRegistrationRequest",
    "build_parser",
    "main",
    "register_inputs",
]
