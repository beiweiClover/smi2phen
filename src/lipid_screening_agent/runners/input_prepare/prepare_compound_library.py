"""Normalize a registered compound library for downstream scientific runners."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.runtime import RunContext, atomic_write_text, ensure_within
from lipid_screening_agent.runtime.errors import (
    EnvironmentError,
    InputError,
    PathSafetyError,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ._common import (
    add_execution_identity_arguments,
    execution_identity,
    resolve_prepared_output,
)

NODE_ID = "prepare_compound_library"
NORMALIZED_PATH = "inputs/prepared/compounds.normalized.csv"
INVALID_PATH = "inputs/prepared/invalid_smiles.tsv"

_ID_ALIASES = frozenset(
    {
        "id",
        "compoundid",
        "drugid",
        "targetmolid",
        "moleculeid",
    }
)
_SMILES_ALIASES = frozenset(
    {
        "smiles",
        "smile",
        "canonicalsmiles",
        "smilesstandardized",
        "standardizedsmiles",
    }
)


def _canonical_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _resolve_registered_input(context: RunContext, input_path: str | Path) -> Path:
    candidate = Path(input_path)
    try:
        if candidate.is_absolute():
            resolved = ensure_within(candidate, context.input_dir)
        else:
            resolved = context.resolve_input(candidate.as_posix(), must_exist=True)
        resolved = ensure_within(
            resolved,
            context.resolve_run_relative("inputs/original", must_exist=True),
        )
    except PathSafetyError as exc:
        raise InputError(
            "compound library must be a registered file inside inputs/original",
            details={"input_path": str(input_path)},
        ) from exc

    if not resolved.exists():
        raise InputError(
            "compound library does not exist",
            details={"input_path": str(input_path)},
        )
    if not resolved.is_file():
        raise InputError(
            "compound library must be a regular file",
            details={"input_path": str(input_path)},
        )
    return resolved


def _candidate_delimiters(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".csv":
        return ",", "\t"
    return "\t", ","


def _detect_delimiter(path: Path, sample: str) -> str:
    for delimiter in _candidate_delimiters(path):
        try:
            header = next(
                csv.reader(io.StringIO(sample), delimiter=delimiter, strict=True),
                [],
            )
        except csv.Error:
            continue
        if len(header) > 1:
            return delimiter
    raise InputError(
        "compound library must be a comma- or tab-delimited table with a header",
        details={"filename": path.name},
    )


def _read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(8192)
            if not sample.strip():
                raise InputError(
                    "compound library is empty",
                    details={"filename": path.name},
                )
            delimiter = _detect_delimiter(path, sample)
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            raw_header = next(reader, None)
            if raw_header is None:
                raise InputError(
                    "compound library is empty",
                    details={"filename": path.name},
                )
            headers = [column.strip() for column in raw_header]
            if any(not column for column in headers):
                raise InputError(
                    "compound library contains an empty column name",
                    details={"columns": headers},
                )
            if len(set(headers)) != len(headers):
                raise InputError(
                    "compound library contains duplicate column names",
                    details={"columns": headers},
                )

            rows: list[list[str]] = []
            for logical_row, row in enumerate(reader, start=2):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(headers):
                    raise InputError(
                        "compound library row has a different number of fields than the header",
                        details={
                            "row_number": logical_row,
                            "expected_fields": len(headers),
                            "actual_fields": len(row),
                        },
                    )
                rows.append(row)
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "compound library could not be read as UTF-8 CSV or TSV",
            details={"filename": path.name, "error_type": type(exc).__name__},
        ) from exc

    if not rows:
        raise InputError(
            "compound library contains no data records",
            details={"filename": path.name},
        )
    return headers, rows


def _required_column_index(headers: Sequence[str], aliases: frozenset[str], label: str) -> int:
    matches = [
        index for index, header in enumerate(headers) if _canonical_column_name(header) in aliases
    ]
    if not matches:
        raise InputError(
            f"compound library is missing a recognized {label} column",
            details={"required_column": label, "columns": list(headers)},
        )
    if len(matches) > 1:
        raise InputError(
            f"compound library has ambiguous {label} columns",
            details={
                "required_column": label,
                "matching_columns": [headers[index] for index in matches],
            },
        )
    return matches[0]


def _load_rdkit() -> Any:
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception as exc:
        raise EnvironmentError(
            "RDKit is required to validate compound SMILES",
            details={"dependency": "rdkit"},
        ) from exc
    return Chem


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, delimiter: str) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def _prepare_operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    input_path: str | Path,
) -> None:
    execution.update_metrics(
        {
            "input_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "skipped_count": 0,
            "additional_column_count": 0,
        }
    )
    normalized_output = resolve_prepared_output(context, NORMALIZED_PATH)
    invalid_output = resolve_prepared_output(context, INVALID_PATH)
    source = _resolve_registered_input(context, input_path)
    headers, rows = _read_table(source)
    execution.metric("input_count", len(rows))
    id_index = _required_column_index(headers, _ID_ALIASES, "ID")
    smiles_index = _required_column_index(headers, _SMILES_ALIASES, "SMILES")

    empty_id_rows = [
        row_number for row_number, row in enumerate(rows, start=2) if not row[id_index].strip()
    ]
    if empty_id_rows:
        execution.metric("empty_id_count", len(empty_id_rows))
        raise InputError(
            "compound IDs must be non-empty",
            details={
                "empty_id_count": len(empty_id_rows),
                "row_numbers": empty_id_rows[:50],
            },
        )

    normalized_ids = [row[id_index].strip() for row in rows]
    id_counts = Counter(normalized_ids)
    duplicate_ids = sorted(identifier for identifier, count in id_counts.items() if count > 1)
    if duplicate_ids:
        execution.metric("duplicate_id_count", len(duplicate_ids))
        raise InputError(
            "compound IDs must be unique",
            details={
                "duplicate_id_count": len(duplicate_ids),
                "duplicate_ids": duplicate_ids[:50],
            },
        )

    chem = _load_rdkit()
    additional_indices = [
        index for index in range(len(headers)) if index not in {id_index, smiles_index}
    ]
    output_headers = ["ID", "SMILES", *(headers[index] for index in additional_indices)]
    valid_rows: list[list[str]] = []
    invalid_rows: list[list[str]] = []

    for row in rows:
        identifier = row[id_index].strip()
        smiles = row[smiles_index].strip()
        if not smiles:
            invalid_rows.append([identifier, smiles, "empty_smiles"])
            continue
        try:
            molecule = chem.MolFromSmiles(smiles)
        except Exception:
            molecule = None
            reason = "rdkit_parse_error"
        else:
            reason = "rdkit_parse_failed"
        if molecule is None:
            invalid_rows.append([identifier, smiles, reason])
            continue
        valid_rows.append([identifier, smiles, *(row[index] for index in additional_indices)])

    invalid_text = _render_table(["ID", "SMILES", "reason"], invalid_rows, delimiter="\t")
    atomic_write_text(invalid_output, invalid_text, allowed_root=context.output_dir)

    execution.update_metrics(
        {
            "valid_count": len(valid_rows),
            "invalid_count": len(invalid_rows),
            "skipped_count": len(invalid_rows),
            "additional_column_count": len(additional_indices),
        }
    )

    if not valid_rows:
        raise InputError(
            "compound library contains no valid SMILES",
            details={"input_count": len(rows), "invalid_count": len(invalid_rows)},
        )

    normalized_text = _render_table(output_headers, valid_rows, delimiter=",")
    atomic_write_text(normalized_output, normalized_text, allowed_root=context.output_dir)

    execution.add_output("compounds_normalized", normalized_output)
    execution.add_output("invalid_smiles", invalid_output)
    if invalid_rows:
        execution.warn(
            f"Skipped {len(invalid_rows)} compound record(s) with empty or invalid SMILES."
        )


def prepare_compound_library(
    *,
    context: RunContext,
    input_path: str | Path,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Normalize one registered compound library and emit both contracted artifacts."""

    return execute_node(
        lambda execution: _prepare_operation(
            execution,
            context=context,
            input_path=input_path,
        ),
        context=context,
        node_id=NODE_ID,
        task_id=task_id,
        attempt=attempt,
        config_hash=config_hash,
        code_version=code_version,
        input_artifact_ids=input_artifact_ids,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize a registered compound ID/SMILES library."
    )
    add_common_runner_arguments(parser)
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Registered compound CSV or TSV inside --input-dir",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    common_arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(
        common_arguments,
        project_root=project_root,
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = prepare_compound_library(
        context=environment.context,
        input_path=namespace.input_file,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    sys.stdout.write(result.to_json() + "\n")
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the callable main
    raise SystemExit(main())


__all__ = ["build_parser", "main", "prepare_compound_library"]
