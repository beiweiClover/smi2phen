"""Lightweight upload-time checks before the scientific DAG performs full validation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_CANONICAL_COLUMN = re.compile(r"[^a-z0-9]+")
_ID_ALIASES = frozenset({"id", "compoundid", "drugid", "targetmolid", "moleculeid"})
_SMILES_ALIASES = frozenset(
    {"smiles", "smile", "canonicalsmiles", "smilesstandardized", "standardizedsmiles"}
)
_GENE_HEADERS = frozenset(
    {"symbol", "genesymbol", "gene", "genename", "hgncsymbol", "entrezid", "geneid"}
)


def _canonical(value: object) -> str:
    return _CANONICAL_COLUMN.sub("", str(value).strip().lower())


def _open_text(path: Path):
    try:
        return path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError("file cannot be opened for basic validation") from exc


def _table_preview(
    path: Path,
    *,
    delimiter: str,
    row_limit: int = 100,
) -> tuple[list[str], list[list[str]]]:
    try:
        with _open_text(path) as handle:
            reader = csv.reader(handle, delimiter=delimiter, strict=True)
            header = next(reader, None)
            rows: list[list[str]] = []
            for row in reader:
                if row and any(value.strip() for value in row):
                    rows.append(row)
                if len(rows) >= row_limit:
                    break
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("file must be readable UTF-8 delimited text") from exc
    if header is None or not any(value.strip() for value in header):
        raise ValueError("file has no readable header or content")
    return [value.strip() for value in header], rows


def _validate_compounds(path: Path, original_name: str) -> dict[str, Any]:
    delimiter = "," if original_name.casefold().endswith(".csv") else "\t"
    header, rows = _table_preview(path, delimiter=delimiter)
    canonical = [_canonical(value) for value in header]
    id_columns = [index for index, value in enumerate(canonical) if value in _ID_ALIASES]
    smiles_columns = [
        index for index, value in enumerate(canonical) if value in _SMILES_ALIASES
    ]
    if len(id_columns) != 1 or len(smiles_columns) != 1:
        raise ValueError("compound table requires one recognized ID column and one SMILES column")
    if not rows:
        raise ValueError("compound table contains no data rows")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("compound table preview contains rows with the wrong number of fields")
    usable = sum(
        1
        for row in rows
        if row[id_columns[0]].strip() and row[smiles_columns[0]].strip()
    )
    if usable == 0:
        raise ValueError("compound table preview contains no row with both ID and SMILES")
    return {
        "status": "basic_passed",
        "message": "recognized compound ID/SMILES table",
        "details": {"preview_rows": len(rows), "usable_preview_rows": usable},
    }


def _validate_disease_genes(path: Path) -> dict[str, Any]:
    try:
        with _open_text(path) as handle:
            lines = [line.strip() for line in handle if line.strip()][:101]
    except UnicodeError as exc:
        raise ValueError("disease gene file must be readable UTF-8 text") from exc
    if not lines:
        raise ValueError("disease gene file contains no non-empty records")
    first_fields = [value.strip() for value in re.split(r"[\t,]", lines[0])]
    first_is_header = any(_canonical(value) in _GENE_HEADERS for value in first_fields)
    data_count = len(lines) - 1 if first_is_header else len(lines)
    if data_count < 1:
        raise ValueError("disease gene file contains a header but no gene records")
    return {
        "status": "basic_passed",
        "message": "recognized disease-gene text or table",
        "details": {"preview_records": data_count, "header_detected": first_is_header},
    }


def _validate_drug_targets(path: Path) -> dict[str, Any]:
    try:
        with _open_text(path) as handle:
            value = json.load(handle)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("drug target file must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("drug target JSON must be a non-empty object keyed by compound ID")
    inspected = 0
    for compound_id, record in value.items():
        if not isinstance(compound_id, str) or not compound_id.strip():
            raise ValueError("drug target JSON contains an empty compound ID")
        if not isinstance(record, dict) or not isinstance(record.get("targets"), list):
            raise ValueError("each drug target record must contain a targets array")
        inspected += 1
        if inspected >= 20:
            break
    return {
        "status": "basic_passed",
        "message": "recognized compound-keyed drug target JSON",
        "details": {"compound_count": len(value), "inspected_records": inspected},
    }


def _validate_target_mapping(path: Path) -> dict[str, Any]:
    header, rows = _table_preview(path, delimiter="\t")
    canonical = {_canonical(value) for value in header}
    if not {"genesymbol", "entrezid"}.issubset(canonical):
        raise ValueError("target mapping requires gene_symbol and entrez_id columns")
    if not rows:
        raise ValueError("target mapping contains no data rows")
    return {
        "status": "basic_passed",
        "message": "recognized target mapping TSV",
        "details": {"preview_rows": len(rows)},
    }


def _validate_optional_kg_table(
    path: Path, *, label: str, allowed_input_types: set[str]
) -> dict[str, Any]:
    header, rows = _table_preview(path, delimiter="\t")
    observed = set(header)
    missing = [column for column in ("input_type", "value") if column not in observed]
    if missing:
        raise ValueError(
            f"{label} requires input_type and value columns; missing: {', '.join(missing)}"
        )
    input_type_index = header.index("input_type")
    value_index = header.index("value")
    usable = 0
    seen_types: set[str] = set()
    for row in rows:
        if len(row) <= max(input_type_index, value_index):
            raise ValueError(f"{label} preview contains rows with missing required fields")
        input_type = row[input_type_index].strip()
        value = row[value_index].strip()
        if not input_type and not value:
            continue
        if not input_type or not value:
            raise ValueError(f"{label} preview contains an incomplete record")
        if input_type not in allowed_input_types:
            raise ValueError(f"{label} contains unsupported input_type: {input_type}")
        seen_types.add(input_type)
        usable += 1
    if usable == 0:
        raise ValueError(f"{label} contains no usable rows")
    return {
        "status": "basic_passed",
        "message": f"recognized {label} TSV",
        "details": {"preview_rows": usable, "input_types": sorted(seen_types)},
    }


def _validate_positive_drugs(path: Path) -> dict[str, Any]:
    return _validate_optional_kg_table(
        path,
        label="positive_drugs.tsv",
        allowed_input_types={"library_id", "base_drug_name", "base_drug_id"},
    )


def _validate_disease_links(path: Path) -> dict[str, Any]:
    return _validate_optional_kg_table(
        path,
        label="disease_links.tsv",
        allowed_input_types={"base_disease_id", "base_disease_name"},
    )


def _validate_expression_tpm(path: Path) -> dict[str, Any]:
    header, _ = _table_preview(path, delimiter="\t", row_limit=1)
    if len(header) < 3:
        raise ValueError("TPM table requires GeneID and at least two sample columns")
    if header[0] != "GeneID":
        raise ValueError("TPM first column must be GeneID")
    sample_ids = header[1:]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("TPM sample column names must be non-empty and unique")
    return {
        "status": "basic_passed",
        "message": "recognized TPM header",
        "details": {"sample_column_count": len(sample_ids)},
    }


def _validate_expression_metadata(path: Path) -> dict[str, Any]:
    try:
        with _open_text(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = [value.strip() for value in (reader.fieldnames or ())]
            if not {"sample_id", "group"}.issubset(fieldnames):
                raise ValueError("metadata requires sample_id and group columns")
            groups: Counter[str] = Counter()
            samples: set[str] = set()
            row_count = 0
            for row in reader:
                if not row or not any((value or "").strip() for value in row.values()):
                    continue
                sample_id = (row.get("sample_id") or "").strip()
                group = (row.get("group") or "").strip().lower()
                if not sample_id or group not in {"control", "disease"}:
                    raise ValueError(
                        "metadata rows require a sample_id and group=control or disease"
                    )
                if sample_id in samples:
                    raise ValueError("metadata sample_id values must be unique")
                samples.add(sample_id)
                groups[group] += 1
                row_count += 1
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("metadata must be readable UTF-8 TSV") from exc
    if set(groups) != {"control", "disease"}:
        raise ValueError("metadata must contain both control and disease groups")
    return {
        "status": "basic_passed",
        "message": "recognized expression metadata TSV",
        "details": {"sample_count": row_count, "groups": dict(groups)},
    }


def validate_upload(*, kind: str, path: Path, original_name: str) -> dict[str, Any]:
    """Run intentionally shallow checks and return a small auditable summary."""

    validators = {
        "compounds": lambda: _validate_compounds(path, original_name),
        "disease_genes": lambda: _validate_disease_genes(path),
        "drug_targets": lambda: _validate_drug_targets(path),
        "target_mapping": lambda: _validate_target_mapping(path),
        "positive_drugs": lambda: _validate_positive_drugs(path),
        "disease_links": lambda: _validate_disease_links(path),
        "expression_tpm": lambda: _validate_expression_tpm(path),
        "expression_metadata": lambda: _validate_expression_metadata(path),
    }
    try:
        validation = validators[kind]()
    except KeyError as exc:
        raise ValueError(f"unsupported upload kind: {kind}") from exc
    return {
        **validation,
        "scope": "upload_time_structure_only",
        "deep_validation": "performed by deterministic workflow nodes after confirmation",
    }


__all__ = ["validate_upload"]
