"""Validated Stage 04 mapping and batch-manifest readers."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime import InputError

REQUIRED_MAPPING_COLUMNS = {
    "ID",
    "SMILES",
    "match_key",
    "official_drug_id",
    "netinfer_input_type",
    "netinfer_input_id",
    "batch_id",
}
REQUIRED_BATCH_FIELDS = {
    "batch_id",
    "task_id",
    "compound_count",
    "compound_ids",
    "input_path",
    "input_sha256",
    "prediction_path",
}


def read_mapping(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise InputError("NetInfer mapping has no header")
            missing = sorted(REQUIRED_MAPPING_COLUMNS - set(reader.fieldnames))
            if missing:
                raise InputError(
                    "NetInfer mapping is missing required columns",
                    details={"missing_columns": missing},
                )
            rows = [
                {str(key): str(value or "").strip() for key, value in row.items()} for row in reader
            ]
    except InputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "NetInfer mapping could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    if not rows:
        raise InputError("NetInfer mapping contains no compounds")
    identifiers = [row["ID"] for row in rows]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise InputError("NetInfer mapping IDs must be non-empty and unique")
    for row in rows:
        if row["netinfer_input_type"] not in {"DRUG", "COMPOUND"}:
            raise InputError(
                "NetInfer mapping contains an unsupported input type",
                details={"ID": row["ID"], "value": row["netinfer_input_type"]},
            )
        if not row["netinfer_input_id"]:
            raise InputError(
                "NetInfer mapping contains an empty input ID",
                details={"ID": row["ID"]},
            )
        if row["netinfer_input_type"] == "DRUG":
            if (
                not row["match_key"]
                or row["official_drug_id"] != row["netinfer_input_id"]
                or row["batch_id"]
            ):
                raise InputError(
                    "known NetInfer mapping fields are inconsistent",
                    details={"ID": row["ID"]},
                )
        elif (
            row["official_drug_id"]
            or row["netinfer_input_id"] != row["ID"]
            or bool(row["match_key"]) != bool(row["batch_id"])
        ):
            raise InputError(
                "novel NetInfer mapping fields are inconsistent",
                details={"ID": row["ID"]},
            )
    return rows


def read_batch_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(
            "NetInfer batch manifest could not be read",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "1.0":
        raise InputError("NetInfer batch manifest has an unsupported schema")
    batches = payload.get("batches")
    if not isinstance(batches, list):
        raise InputError("NetInfer batch manifest batches must be a list")
    if payload.get("batch_count") != len(batches):
        raise InputError("NetInfer batch manifest batch_count is inconsistent")
    seen_batches: set[str] = set()
    seen_compounds: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping) or not REQUIRED_BATCH_FIELDS.issubset(batch):
            raise InputError(
                "NetInfer batch manifest item is missing required fields",
                details={"batch_index": index},
            )
        batch_id = batch["batch_id"]
        compound_ids = batch["compound_ids"]
        if not isinstance(batch_id, str) or not batch_id or batch_id in seen_batches:
            raise InputError("NetInfer batch IDs must be non-empty and unique")
        if not isinstance(compound_ids, list) or not all(
            isinstance(value, str) and value for value in compound_ids
        ):
            raise InputError(
                "NetInfer batch compound_ids must be non-empty strings",
                details={"batch_id": batch_id},
            )
        if batch["compound_count"] != len(compound_ids) or not compound_ids:
            raise InputError(
                "NetInfer batch compound_count is inconsistent",
                details={"batch_id": batch_id},
            )
        overlap = seen_compounds.intersection(compound_ids)
        if overlap:
            raise InputError(
                "NetInfer compounds cannot occur in multiple batches",
                details={"compound_ids": sorted(overlap)[:50]},
            )
        seen_batches.add(batch_id)
        seen_compounds.update(compound_ids)
        normalized.append(dict(batch))
    if payload.get("novel_compound_count") != len(seen_compounds):
        raise InputError("NetInfer batch manifest novel_compound_count is inconsistent")
    result = dict(payload)
    result["batches"] = normalized
    return result


def batch_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(batch["batch_id"]): dict(batch) for batch in manifest["batches"]}


__all__ = [
    "REQUIRED_BATCH_FIELDS",
    "REQUIRED_MAPPING_COLUMNS",
    "batch_index",
    "read_batch_manifest",
    "read_mapping",
]
