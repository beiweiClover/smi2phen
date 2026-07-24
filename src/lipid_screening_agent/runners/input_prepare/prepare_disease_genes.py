"""Normalize disease-gene inputs to official human Symbol and Entrez identifiers."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from collections import defaultdict
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
from lipid_screening_agent.config import resolve_resource_path
from lipid_screening_agent.runtime.atomic import atomic_write_text
from lipid_screening_agent.runtime.context import RunContext
from lipid_screening_agent.runtime.errors import (
    ConfigurationError,
    InputError,
    PathSafetyError,
    ResourceError,
)
from lipid_screening_agent.runtime.execution import NodeExecution, execute_node
from lipid_screening_agent.runtime.hashing import sha256_file
from lipid_screening_agent.runtime.paths import (
    canonical_path,
    ensure_within,
    resolve_run_relative,
)

from ._common import (
    add_execution_identity_arguments,
    execution_identity,
    resolve_prepared_output,
)

NODE_ID = "prepare_disease_genes"
NORMALIZED_PATH = "inputs/prepared/disease_genes.normalized.tsv"
UNMAPPED_PATH = "inputs/prepared/unmapped_genes.tsv"
GENE_INFO_RESOURCE_KEY = "resources.gps.human_gene_info"

_CANONICAL_COLUMN = re.compile(r"[^a-z0-9]+")
_ENTREZ_ID = re.compile(r"^[0-9]+$")
_SYMBOL_COLUMN_ALIASES = frozenset(
    {
        "symbol",
        "genesymbol",
        "gene",
        "genename",
        "hgncsymbol",
    }
)
_ENTREZ_COLUMN_ALIASES = frozenset(
    {
        "entrezid",
        "geneid",
        "ncbiid",
        "ncbigeneid",
    }
)


@dataclass(frozen=True, slots=True)
class _GeneMaps:
    official_by_entrez: dict[str, str]
    official_ids_by_key: dict[str, frozenset[str]]
    synonym_ids_by_key: dict[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class _InputRecord:
    symbol: str
    entrez_id: str


@dataclass(frozen=True, slots=True)
class _MappingOutcome:
    symbol: str | None
    entrez_id: str | None
    source: str | None
    reason: str | None


def _canonical_column_name(value: object) -> str:
    return _CANONICAL_COLUMN.sub("", str(value).strip().lower())


def _symbol_key(value: str) -> str:
    return value.strip().casefold()


def _resolve_input_file(context: RunContext, value: str | Path) -> Path:
    path = Path(value)
    try:
        if path.is_absolute():
            candidate = canonical_path(path, must_exist=True, label="disease gene input")
            candidate = ensure_within(candidate, context.input_dir)
        else:
            candidate = context.resolve_input(path.as_posix(), must_exist=True)
        candidate = ensure_within(
            candidate,
            context.resolve_run_relative("inputs/original", must_exist=True),
        )
    except (OSError, PathSafetyError) as exc:
        raise InputError(
            f"Disease gene input is missing or outside inputs/original: {value}",
            details={"input_path": str(value)},
        ) from exc
    if not candidate.is_file():
        raise InputError(
            f"Disease gene input is not a regular file: {candidate}",
            details={"input_path": str(candidate)},
        )
    return candidate


def _resolve_gene_info_file(context: RunContext, value: str | Path) -> Path:
    if context.resource_dir is None:
        raise ResourceError("A resource directory is required for gene identifier mapping")
    path = Path(value)
    try:
        if path.is_absolute():
            candidate = canonical_path(path, must_exist=True, label="human gene_info")
            candidate = ensure_within(candidate, context.resource_dir)
        else:
            candidate = resolve_run_relative(
                context.resource_dir,
                path.as_posix(),
                must_exist=True,
            )
    except (OSError, PathSafetyError) as exc:
        raise ResourceError(
            f"Human gene_info is missing or outside the resource boundary: {value}",
            details={"gene_info_path": str(value)},
        ) from exc
    if not candidate.is_file():
        raise ResourceError(
            f"Human gene_info is not a regular file: {candidate}",
            details={"gene_info_path": str(candidate)},
        )
    return candidate


def _open_gene_info(path: Path):
    if path.suffix.casefold() == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return path.open(mode="r", encoding="utf-8-sig", newline="")


def _load_gene_maps(path: Path) -> _GeneMaps:
    official_by_entrez: dict[str, str] = {}
    official_candidates: dict[str, set[str]] = defaultdict(set)
    synonym_candidates: dict[str, set[str]] = defaultdict(set)
    try:
        with _open_gene_info(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = set(reader.fieldnames or ())
            required = {"GeneID", "Symbol", "Synonyms"}
            if not required.issubset(fieldnames):
                missing = sorted(required - fieldnames)
                raise ResourceError(
                    "Human gene_info is missing required columns",
                    details={"missing_columns": missing, "path": str(path)},
                )

            for row in reader:
                gene_id = (row.get("GeneID") or "").strip()
                symbol = (row.get("Symbol") or "").strip()
                if not gene_id or not symbol:
                    continue
                if not _ENTREZ_ID.fullmatch(gene_id):
                    raise ResourceError(
                        "Human gene_info contains a non-numeric GeneID",
                        details={"gene_id": gene_id, "path": str(path)},
                    )

                official_by_entrez.setdefault(gene_id, symbol)
                official_candidates[_symbol_key(symbol)].add(gene_id)
                for synonym in (row.get("Synonyms") or "").split("|"):
                    synonym = synonym.strip()
                    if synonym and synonym != "-":
                        synonym_candidates[_symbol_key(synonym)].add(gene_id)
    except ResourceError:
        raise
    except (OSError, UnicodeError, csv.Error, EOFError) as exc:
        raise ResourceError(
            f"Cannot read human gene_info resource: {path}",
            details={"path": str(path)},
        ) from exc

    if not official_by_entrez:
        raise ResourceError(
            "Human gene_info contains no usable GeneID/Symbol records",
            details={"path": str(path)},
        )
    return _GeneMaps(
        official_by_entrez=official_by_entrez,
        official_ids_by_key={key: frozenset(values) for key, values in official_candidates.items()},
        synonym_ids_by_key={key: frozenset(values) for key, values in synonym_candidates.items()},
    )


def _nonempty_input_lines(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [line.rstrip("\r\n") for line in handle if line.strip()]
    except (OSError, UnicodeError) as exc:
        raise InputError(
            f"Cannot read disease gene input: {path}",
            details={"input_path": str(path)},
        ) from exc


def _read_rows(lines: Sequence[str], delimiter: str | None) -> list[list[str]]:
    if delimiter is None:
        return [[line.strip()] for line in lines]
    try:
        return [
            [value.strip() for value in row]
            for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
        ]
    except csv.Error as exc:
        raise InputError("Cannot parse disease gene input as a delimited table") from exc


def _input_records(path: Path) -> list[_InputRecord]:
    lines = _nonempty_input_lines(path)
    if not lines:
        raise InputError(
            "Disease gene input is empty",
            details={"input_path": str(path)},
        )

    first = lines[0]
    delimiter = "\t" if "\t" in first else "," if "," in first else None
    rows = _read_rows(lines, delimiter)
    if not rows:
        raise InputError("Disease gene input contains no records")

    first_canonical = [_canonical_column_name(value) for value in rows[0]]
    header_present = bool(set(first_canonical) & (_SYMBOL_COLUMN_ALIASES | _ENTREZ_COLUMN_ALIASES))
    data_rows = rows[1:] if header_present else rows

    if header_present:
        expected_width = len(rows[0])
        for row_number, row in enumerate(data_rows, start=2):
            if len(row) != expected_width:
                raise InputError(
                    "Disease gene input row does not match the header width",
                    details={
                        "row_number": row_number,
                        "expected_columns": expected_width,
                        "actual_columns": len(row),
                    },
                )

    symbol_index: int | None = None
    entrez_index: int | None = None
    if header_present:
        symbol_indices = [
            index for index, value in enumerate(first_canonical) if value in _SYMBOL_COLUMN_ALIASES
        ]
        entrez_indices = [
            index for index, value in enumerate(first_canonical) if value in _ENTREZ_COLUMN_ALIASES
        ]
        if len(symbol_indices) > 1 or len(entrez_indices) > 1:
            raise InputError("Disease gene input has duplicate identifier columns")
        symbol_index = symbol_indices[0] if symbol_indices else None
        entrez_index = entrez_indices[0] if entrez_indices else None
    else:
        width = max(len(row) for row in data_rows)
        if width == 1:
            values = [row[0].strip() for row in data_rows if row and row[0].strip()]
            if values and all(_ENTREZ_ID.fullmatch(value) for value in values):
                entrez_index = 0
            else:
                symbol_index = 0
        elif width == 2 and all(len(row) <= 2 for row in data_rows):
            symbol_index, entrez_index = 0, 1
        else:
            raise InputError(
                "Headerless disease gene input must contain one or two columns",
                details={"column_count": width},
            )

    if symbol_index is None and entrez_index is None:
        raise InputError("Disease gene input must contain a Symbol or Entrez identifier column")

    records: list[_InputRecord] = []
    for row in data_rows:
        symbol = (
            row[symbol_index].strip()
            if symbol_index is not None and symbol_index < len(row)
            else ""
        )
        entrez_id = (
            row[entrez_index].strip()
            if entrez_index is not None and entrez_index < len(row)
            else ""
        )
        if symbol or entrez_id:
            records.append(_InputRecord(symbol=symbol, entrez_id=entrez_id))

    if not records:
        raise InputError("Disease gene input contains no non-empty records")
    return records


def _map_symbol(symbol: str, maps: _GeneMaps) -> tuple[str | None, str | None]:
    key = _symbol_key(symbol)
    official_ids = maps.official_ids_by_key.get(key, frozenset())
    if official_ids:
        if len(official_ids) != 1:
            return None, "ambiguous_symbol"
        return next(iter(official_ids)), "symbol"

    synonym_ids = maps.synonym_ids_by_key.get(key, frozenset())
    if not synonym_ids:
        return None, "unknown_symbol"
    if len(synonym_ids) != 1:
        return None, "ambiguous_synonym"
    return next(iter(synonym_ids)), "synonym"


def _map_record(record: _InputRecord, maps: _GeneMaps) -> _MappingOutcome:
    if record.entrez_id:
        if not _ENTREZ_ID.fullmatch(record.entrez_id):
            return _MappingOutcome(None, None, None, "invalid_entrez_id")
        official = maps.official_by_entrez.get(record.entrez_id)
        if official is None:
            return _MappingOutcome(None, None, None, "unknown_entrez_id")
        if record.symbol:
            mapped_id, source_or_reason = _map_symbol(record.symbol, maps)
            if mapped_id is None:
                return _MappingOutcome(None, None, None, source_or_reason)
            if mapped_id != record.entrez_id:
                return _MappingOutcome(None, None, None, "symbol_entrez_mismatch")
            return _MappingOutcome(official, record.entrez_id, source_or_reason, None)
        return _MappingOutcome(official, record.entrez_id, "entrez", None)

    if not record.symbol:
        return _MappingOutcome(None, None, None, "empty_input_value")
    mapped_id, source_or_reason = _map_symbol(record.symbol, maps)
    if mapped_id is None:
        return _MappingOutcome(None, None, None, source_or_reason)
    return _MappingOutcome(
        maps.official_by_entrez[mapped_id],
        mapped_id,
        source_or_reason,
        None,
    )


def _input_value(record: _InputRecord) -> str:
    if record.symbol and record.entrez_id:
        return f"symbol={record.symbol};entrez_id={record.entrez_id}"
    return record.symbol or record.entrez_id


def _render_tsv(fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def prepare_disease_genes(
    *,
    context: RunContext,
    input_path: str | Path,
    gene_info_path: str | Path,
    config_hash: str,
    code_version: str,
    task_id: str = "main",
    attempt: int = 1,
    input_artifact_ids: Sequence[str] = (),
) -> NodeResult:
    """Normalize one registered disease-gene file and commit both output manifests."""

    def operation(execution: NodeExecution) -> None:
        execution.update_metrics(
            {
                "input_count": 0,
                "valid_count": 0,
                "unmapped_count": 0,
                "duplicate_count": 0,
                "skipped_count": 0,
                "mapped_by_symbol_count": 0,
                "mapped_by_synonym_count": 0,
                "mapped_by_entrez_count": 0,
            }
        )
        normalized_path = resolve_prepared_output(context, NORMALIZED_PATH)
        unmapped_path = resolve_prepared_output(context, UNMAPPED_PATH)
        input_file = _resolve_input_file(context, input_path)
        gene_info_file = _resolve_gene_info_file(context, gene_info_path)
        try:
            gene_info_hash = sha256_file(gene_info_file)
        except (OSError, RuntimeError) as exc:
            raise ResourceError(
                f"Cannot hash human gene_info resource: {gene_info_file}",
                details={"path": str(gene_info_file)},
            ) from exc
        execution.resource_hashes[GENE_INFO_RESOURCE_KEY] = gene_info_hash

        maps = _load_gene_maps(gene_info_file)
        records = _input_records(input_file)
        normalized: list[dict[str, str]] = []
        unmapped: list[dict[str, str]] = []
        seen_entrez: set[str] = set()
        duplicate_count = 0
        source_counts = {"symbol": 0, "synonym": 0, "entrez": 0}

        for record in records:
            outcome = _map_record(record, maps)
            if outcome.reason is not None:
                unmapped.append({"input_value": _input_value(record), "reason": outcome.reason})
                continue
            assert outcome.symbol is not None
            assert outcome.entrez_id is not None
            assert outcome.source is not None
            if outcome.entrez_id in seen_entrez:
                duplicate_count += 1
                continue
            seen_entrez.add(outcome.entrez_id)
            normalized.append({"symbol": outcome.symbol, "entrez_id": outcome.entrez_id})
            source_counts[outcome.source] += 1

        execution.update_metrics(
            {
                "input_count": len(records),
                "valid_count": len(normalized),
                "unmapped_count": len(unmapped),
                "duplicate_count": duplicate_count,
                "skipped_count": len(unmapped) + duplicate_count,
                "mapped_by_symbol_count": source_counts["symbol"],
                "mapped_by_synonym_count": source_counts["synonym"],
                "mapped_by_entrez_count": source_counts["entrez"],
            }
        )
        atomic_write_text(
            unmapped_path,
            _render_tsv(("input_value", "reason"), unmapped),
            allowed_root=context.output_dir,
        )
        if unmapped:
            execution.warn(f"Skipped {len(unmapped)} unmapped disease gene record(s)")
        if duplicate_count:
            execution.warn(f"Removed {duplicate_count} duplicate Entrez disease gene record(s)")

        if not normalized:
            raise InputError(
                "Disease gene normalization produced no valid genes",
                details={
                    "input_count": len(records),
                    "unmapped_count": len(unmapped),
                    "duplicate_count": duplicate_count,
                },
            )

        atomic_write_text(
            normalized_path,
            _render_tsv(("symbol", "entrez_id"), normalized),
            allowed_root=context.output_dir,
        )
        execution.add_output("disease_genes_normalized", normalized_path)
        execution.add_output("unmapped_genes", unmapped_path)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize disease genes to official human Symbol and Entrez IDs"
    )
    add_common_runner_arguments(parser)
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Registered disease-gene file, absolute or relative to --input-dir",
    )
    add_execution_identity_arguments(parser)
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    common = CommonRunnerArguments.from_namespace(namespace)
    environment = load_common_runner_environment(
        common,
        project_root=_project_root(),
    )
    task_id, attempt, input_artifact_ids = execution_identity(namespace)
    try:
        configured_gene_info = resolve_resource_path(
            environment.config,
            GENE_INFO_RESOURCE_KEY,
        )
    except ConfigurationError as configuration_error:

        def fail_resource_configuration(
            execution: NodeExecution,
            error: ConfigurationError = configuration_error,
        ) -> None:
            execution.update_metrics(
                {
                    "input_count": 0,
                    "valid_count": 0,
                    "unmapped_count": 0,
                    "duplicate_count": 0,
                    "skipped_count": 0,
                    "mapped_by_symbol_count": 0,
                    "mapped_by_synonym_count": 0,
                    "mapped_by_entrez_count": 0,
                }
            )
            raise error

        result = execute_node(
            fail_resource_configuration,
            context=environment.context,
            node_id=NODE_ID,
            task_id=task_id,
            attempt=attempt,
            config_hash=environment.config_hash,
            code_version=__version__,
            input_artifact_ids=input_artifact_ids,
        )
    else:
        result = prepare_disease_genes(
            context=environment.context,
            input_path=namespace.input_file,
            gene_info_path=configured_gene_info,
            config_hash=environment.config_hash,
            code_version=__version__,
            task_id=task_id,
            attempt=attempt,
            input_artifact_ids=input_artifact_ids,
        )
    print(result.to_json())
    return 0 if result.status is NodeStatus.SUCCEEDED else 1


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())


__all__ = ["build_parser", "main", "prepare_disease_genes"]
