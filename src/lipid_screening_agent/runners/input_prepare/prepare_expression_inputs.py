"""Discover and minimally validate registered TPM/metadata comparison pairs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lipid_screening_agent import __version__
from lipid_screening_agent.artifacts import NodeResult, NodeStatus
from lipid_screening_agent.cli import (
    CommonRunnerArguments,
    add_common_runner_arguments,
    load_common_runner_environment,
)
from lipid_screening_agent.runtime import (
    InputError,
    PathSafetyError,
    RunContext,
    atomic_write_json,
    ensure_within,
    file_digest,
    validate_portable_segment,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node

from ._common import (
    add_execution_identity_arguments,
    execution_identity,
    resolve_prepared_output,
)

NODE_ID = "prepare_expression_inputs"
OUTPUT_RELATIVE_PATH = "inputs/prepared/expression_comparisons.json"

_TPM_NAME = re.compile(r"^TPM_matrix(?:_(?P<suffix>.+))?\.tsv$")
_METADATA_NAME = re.compile(r"^metadata(?:_(?P<suffix>.+))?\.tsv$")


@dataclass(frozen=True, slots=True)
class _ExpressionPair:
    suffix: str
    tpm_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class _MetadataSummary:
    sample_ids: tuple[str, ...]
    control_count: int
    disease_count: int


def _zero_metrics() -> dict[str, int]:
    return {
        "input_file_count": 0,
        "comparison_count": 0,
        "sample_count": 0,
        "control_sample_count": 0,
        "disease_sample_count": 0,
        "tpm_extra_sample_count": 0,
        "orphan_file_count": 0,
    }


def _validated_input_directory(context: RunContext, input_dir: str | Path) -> Path:
    try:
        directory = ensure_within(input_dir, context.input_dir, allow_equal=True)
        original_root = context.resolve_run_relative(
            "inputs/original",
            must_exist=True,
        )
    except (OSError, PathSafetyError) as exc:
        raise InputError(
            "expression input directory must be the registered inputs/original directory",
            details={"input_dir": str(input_dir)},
        ) from exc
    if not directory.is_dir():
        raise InputError(
            "expression input directory does not exist or is not a directory",
            details={"input_dir": str(directory)},
        )
    if directory != original_root:
        raise InputError(
            "expression input directory must be the registered inputs/original directory",
            details={
                "input_dir": str(directory),
                "expected_input_dir": str(original_root),
            },
        )
    return directory


def _suffix_from_match(match: re.Match[str]) -> str:
    return match.group("suffix") or ""


def _pair_sort_key(pair: _ExpressionPair) -> tuple[int, int, str]:
    if pair.suffix == "":
        return (0, 0, "")
    if pair.suffix.isdigit():
        return (1, int(pair.suffix), pair.suffix)
    return (2, 0, pair.suffix)


def _discover_pairs(
    directory: Path,
) -> tuple[list[_ExpressionPair], list[str], list[str]]:
    tpm_files: dict[str, Path] = {}
    metadata_files: dict[str, Path] = {}
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise InputError(
            "cannot enumerate expression input directory",
            details={"input_dir": str(directory)},
        ) from exc

    for entry in entries:
        tpm_match = _TPM_NAME.fullmatch(entry.name)
        metadata_match = _METADATA_NAME.fullmatch(entry.name)
        if tpm_match is None and metadata_match is None:
            continue
        if not entry.is_file():
            raise InputError(
                "an expression input path matching the naming contract is not a file",
                details={"path": str(entry)},
            )
        if tpm_match is not None:
            tpm_files[_suffix_from_match(tpm_match)] = entry
        else:
            assert metadata_match is not None
            metadata_files[_suffix_from_match(metadata_match)] = entry

    orphan_tpm = sorted(
        path.name for suffix, path in tpm_files.items() if suffix not in metadata_files
    )
    orphan_metadata = sorted(
        path.name for suffix, path in metadata_files.items() if suffix not in tpm_files
    )
    shared_suffixes = set(tpm_files) & set(metadata_files)
    pairs = sorted(
        (
            _ExpressionPair(suffix, tpm_files[suffix], metadata_files[suffix])
            for suffix in shared_suffixes
        ),
        key=_pair_sort_key,
    )
    return pairs, orphan_tpm, orphan_metadata


def _comparison_ids(pairs: Sequence[_ExpressionPair]) -> dict[str, str]:
    comparison_ids: dict[str, str] = {}
    owners: dict[str, str] = {}
    for pair in pairs:
        comparison_id = f"comparison_{pair.suffix}" if pair.suffix else "comparison_1"
        try:
            validate_portable_segment(comparison_id, label="comparison_id")
        except PathSafetyError as exc:
            raise InputError(
                "expression filename suffix cannot form a portable comparison ID",
                details={"suffix": pair.suffix, "comparison_id": comparison_id},
            ) from exc
        if len(comparison_id) > 128:
            raise InputError(
                "expression comparison ID cannot exceed 128 characters",
                details={"suffix": pair.suffix, "comparison_id": comparison_id},
            )
        if comparison_id in owners:
            raise InputError(
                "expression filenames produce duplicate comparison IDs",
                details={
                    "comparison_id": comparison_id,
                    "first_tpm": owners[comparison_id],
                    "second_tpm": pair.tpm_path.name,
                },
            )
        owners[comparison_id] = pair.tpm_path.name
        comparison_ids[pair.suffix] = comparison_id
    return comparison_ids


def _read_metadata(path: Path) -> _MetadataSummary:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise InputError(
                    "metadata file is empty",
                    details={"path": str(path)},
                ) from exc

            header = [value.strip() for value in header]
            if len(header) != len(set(header)):
                duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
                raise InputError(
                    "metadata file contains duplicate columns",
                    details={"path": str(path), "columns": duplicates[:10]},
                )
            missing = sorted({"sample_id", "group"} - set(header))
            if missing:
                raise InputError(
                    "metadata file is missing required columns",
                    details={"path": str(path), "missing_columns": missing},
                )

            sample_index = header.index("sample_id")
            group_index = header.index("group")
            sample_ids: list[str] = []
            groups: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != len(header):
                    raise InputError(
                        "metadata row does not match the header width",
                        details={
                            "path": str(path),
                            "row_number": row_number,
                            "expected_columns": len(header),
                            "actual_columns": len(row),
                        },
                    )
                sample_id = row[sample_index].strip()
                group = row[group_index].strip().lower()
                if not sample_id or not group:
                    raise InputError(
                        "metadata sample_id and group must be non-empty",
                        details={"path": str(path), "row_number": row_number},
                    )
                if group not in {"control", "disease"}:
                    raise InputError(
                        "metadata group values must be control or disease",
                        details={
                            "path": str(path),
                            "row_number": row_number,
                            "group": group,
                        },
                    )
                sample_ids.append(sample_id)
                groups.append(group)
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "metadata file could not be read as UTF-8 TSV",
            details={"path": str(path)},
        ) from exc

    duplicate_samples = sorted(sample for sample, count in Counter(sample_ids).items() if count > 1)
    if duplicate_samples:
        raise InputError(
            "metadata sample_id values must be unique",
            details={"path": str(path), "duplicate_sample_ids": duplicate_samples[:10]},
        )

    group_counts = Counter(groups)
    if set(group_counts) != {"control", "disease"}:
        raise InputError(
            "metadata must contain both control and disease groups",
            details={"path": str(path), "groups": sorted(group_counts)},
        )
    return _MetadataSummary(
        sample_ids=tuple(sample_ids),
        control_count=group_counts["control"],
        disease_count=group_counts["disease"],
    )


def _read_tpm_sample_ids(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise InputError(
                    "TPM file is empty",
                    details={"path": str(path)},
                ) from exc
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "TPM file header could not be read as UTF-8 TSV",
            details={"path": str(path)},
        ) from exc

    if len(header) < 3:
        raise InputError(
            "TPM file requires GeneID and at least two sample columns",
            details={"path": str(path), "column_count": len(header)},
        )
    if header[0] != "GeneID":
        raise InputError(
            "TPM first column must be GeneID",
            details={"path": str(path), "first_column": header[0]},
        )
    sample_ids = tuple(header[1:])
    if any(not sample_id.strip() for sample_id in sample_ids):
        raise InputError(
            "TPM sample column names must be non-empty",
            details={"path": str(path)},
        )
    duplicate_samples = sorted(sample for sample, count in Counter(sample_ids).items() if count > 1)
    if duplicate_samples:
        raise InputError(
            "TPM sample column names must be unique",
            details={"path": str(path), "duplicate_sample_ids": duplicate_samples[:10]},
        )
    return sample_ids


def _digest(path: Path) -> str:
    try:
        return file_digest(path).sha256
    except (OSError, RuntimeError) as exc:
        raise InputError(
            "registered expression input could not be hashed safely",
            details={"path": str(path)},
        ) from exc


def _prepare_operation(
    execution: NodeExecution,
    *,
    context: RunContext,
    input_dir: str | Path,
    no_expression_data: bool,
) -> None:
    execution.update_metrics(_zero_metrics())
    output_path = resolve_prepared_output(context, OUTPUT_RELATIVE_PATH)
    if not isinstance(no_expression_data, bool):
        raise InputError("no_expression_data must be a boolean")
    if no_expression_data:
        execution.mark_skipped("expression data explicitly declared unavailable")
        return

    directory = _validated_input_directory(context, input_dir)
    pairs, orphan_tpm, orphan_metadata = _discover_pairs(directory)
    input_file_count = len(pairs) * 2 + len(orphan_tpm) + len(orphan_metadata)
    execution.metric("input_file_count", input_file_count)
    execution.metric("orphan_file_count", len(orphan_tpm) + len(orphan_metadata))
    if orphan_tpm or orphan_metadata:
        raise InputError(
            "every TPM file must have a metadata file with the same suffix",
            details={
                "orphan_tpm_files": orphan_tpm,
                "orphan_metadata_files": orphan_metadata,
            },
        )
    if not pairs:
        raise InputError(
            "no TPM/metadata comparison pairs were found",
            details={
                "input_dir": str(directory),
                "expected": [
                    "TPM_matrix.tsv + metadata.tsv",
                    "TPM_matrix_<suffix>.tsv + metadata_<suffix>.tsv",
                ],
            },
        )

    comparison_ids = _comparison_ids(pairs)
    comparisons: list[dict[str, object]] = []
    total_samples = 0
    total_control = 0
    total_disease = 0
    total_extra_tpm_samples = 0
    for pair in pairs:
        metadata = _read_metadata(pair.metadata_path)
        tpm_samples = _read_tpm_sample_ids(pair.tpm_path)
        missing_samples = sorted(set(metadata.sample_ids) - set(tpm_samples))
        if missing_samples:
            raise InputError(
                "metadata contains samples that are absent from the paired TPM header",
                details={
                    "comparison_id": comparison_ids[pair.suffix],
                    "metadata_file": pair.metadata_path.name,
                    "tpm_file": pair.tpm_path.name,
                    "missing_sample_ids": missing_samples[:10],
                },
            )

        extra_tpm_samples = len(set(tpm_samples) - set(metadata.sample_ids))
        sample_count = len(metadata.sample_ids)
        total_samples += sample_count
        total_control += metadata.control_count
        total_disease += metadata.disease_count
        total_extra_tpm_samples += extra_tpm_samples
        comparisons.append(
            {
                "comparison_id": comparison_ids[pair.suffix],
                "tpm_path": context.relative_path(pair.tpm_path),
                "metadata_path": context.relative_path(pair.metadata_path),
                "sample_counts": {
                    "total": sample_count,
                    "control": metadata.control_count,
                    "disease": metadata.disease_count,
                },
                "sha256": {
                    "tpm": _digest(pair.tpm_path),
                    "metadata": _digest(pair.metadata_path),
                },
            }
        )

    atomic_write_json(
        output_path,
        {"schema_version": "1.0", "comparisons": comparisons},
        allowed_root=context.output_dir,
    )
    execution.add_output("expression_comparisons_manifest", output_path)
    execution.update_metrics(
        {
            "comparison_count": len(comparisons),
            "sample_count": total_samples,
            "control_sample_count": total_control,
            "disease_sample_count": total_disease,
            "tpm_extra_sample_count": total_extra_tpm_samples,
        }
    )
    execution.logger.info(
        "expression_inputs_prepared",
        "expression comparison inputs were paired and minimally validated",
        comparison_count=len(comparisons),
        sample_count=total_samples,
    )


def prepare_expression_inputs(
    *,
    context: RunContext,
    input_dir: str | Path,
    no_expression_data: bool,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Prepare a stable comparison manifest without reading TPM matrix bodies."""

    return execute_node(
        lambda execution: _prepare_operation(
            execution,
            context=context,
            input_dir=input_dir,
            no_expression_data=no_expression_data,
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
    """Build the standalone runner CLI parser."""

    parser = argparse.ArgumentParser(
        description="Pair and minimally validate registered TPM/metadata inputs."
    )
    add_common_runner_arguments(parser)
    parser.add_argument(
        "--no-expression-data",
        action="store_true",
        help="Explicitly record that no expression data is available and return skipped.",
    )
    add_execution_identity_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the expression-input preparer and print its terminal NodeResult JSON."""

    namespace = build_parser().parse_args(argv)
    arguments = CommonRunnerArguments.from_namespace(namespace)
    project_root = Path(__file__).resolve().parents[4]
    environment = load_common_runner_environment(
        arguments,
        project_root=project_root,
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    result = prepare_expression_inputs(
        context=environment.context,
        input_dir=environment.arguments.input_dir,
        no_expression_data=namespace.no_expression_data,
        config_hash=environment.config_hash,
        code_version=__version__,
        task_id=task_id,
        attempt=attempt,
        input_artifact_ids=input_artifact_ids,
    )
    print(result.to_json())
    return 0 if result.status in {NodeStatus.SUCCEEDED, NodeStatus.SKIPPED} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "prepare_expression_inputs"]
