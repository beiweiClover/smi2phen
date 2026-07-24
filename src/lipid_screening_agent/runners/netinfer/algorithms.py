"""Pure NetInfer raw parsing and known-first target merge logic."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime import InputError

RAW_COLUMNS = (
    "source_type",
    "source_id",
    "target_type",
    "uniprot_id",
    "score",
    "rank",
)


@dataclass(frozen=True, slots=True)
class RawPrediction:
    source_type: str
    source_id: str
    target_type: str
    uniprot_id: str
    score: str
    rank: str
    ordinal: int

    @property
    def numeric_score(self) -> float | None:
        try:
            value = float(self.score)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @property
    def numeric_rank(self) -> int | None:
        try:
            number = float(self.rank)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)

    @property
    def is_known(self) -> bool:
        return self.rank.strip().casefold() in {"-", "known"}


def parse_raw_prediction_rows(
    rows: Sequence[Sequence[str]],
    *,
    source_label: str,
) -> tuple[RawPrediction, ...]:
    """Validate the legacy headerless six-column raw output grammar."""

    predictions: list[RawPrediction] = []
    for ordinal, raw in enumerate(rows, start=1):
        if not raw or all(not str(value).strip() for value in raw):
            continue
        if len(raw) != len(RAW_COLUMNS):
            raise InputError(
                "NetInfer raw output row must contain six tab-separated fields",
                details={"source": source_label, "row": ordinal, "field_count": len(raw)},
            )
        values = tuple(str(value).strip() for value in raw)
        if any(not value for value in values[:4]):
            raise InputError(
                "NetInfer raw output contains an empty identity field",
                details={"source": source_label, "row": ordinal},
            )
        predictions.append(RawPrediction(*values, ordinal=ordinal))
    return tuple(predictions)


def read_raw_predictions(path: Path) -> tuple[RawPrediction, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t", strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(
            "NetInfer raw output could not be read",
            details={"path": str(path), "error_type": type(exc).__name__},
        ) from exc
    return parse_raw_prediction_rows(rows, source_label=str(path))


def validate_raw_predictions(
    predictions: Sequence[RawPrediction],
    *,
    expected_source_type: str,
    allowed_source_ids: set[str] | None = None,
) -> None:
    """Reject malformed or cross-batch raw output without changing its contents."""

    unexpected_types = sorted(
        {row.source_type for row in predictions if row.source_type != expected_source_type}
    )
    if unexpected_types:
        raise InputError(
            "NetInfer raw output contains an unexpected source type",
            details={
                "expected_source_type": expected_source_type,
                "unexpected_source_types": unexpected_types,
            },
        )
    unexpected_targets = sorted(
        {row.target_type for row in predictions if row.target_type != "TARGET"}
    )
    if unexpected_targets:
        raise InputError(
            "NetInfer raw output contains an unexpected target type",
            details={"unexpected_target_types": unexpected_targets},
        )
    if allowed_source_ids is not None:
        unexpected_ids = sorted(
            {row.source_id for row in predictions if row.source_id not in allowed_source_ids}
        )
        if unexpected_ids:
            raise InputError(
                "NetInfer batch output contains compounds from another batch",
                details={"unexpected_source_ids": unexpected_ids[:50]},
            )


def _known_sort_key(row: RawPrediction) -> tuple[bool, float, int]:
    score = row.numeric_score
    return (score is None, -(score or 0.0), row.ordinal)


def _predicted_sort_key(row: RawPrediction) -> tuple[int, int]:
    return (row.numeric_rank or 2**31 - 1, row.ordinal)


def ordered_targets_for_source(
    rows: Sequence[RawPrediction],
    uniprot_to_symbol: Mapping[str, str],
    *,
    top_n_predicted: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return all Known targets first, followed by prediction ranks 1..top-N."""

    known = sorted((row for row in rows if row.is_known), key=_known_sort_key)
    predicted = sorted(
        (
            row
            for row in rows
            if row.numeric_rank is not None and 1 <= int(row.numeric_rank) <= top_n_predicted
        ),
        key=_predicted_sort_key,
    )
    targets: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    known_count = predicted_count = unmapped_count = 0
    for row, evidence in [
        *((item, "known") for item in known),
        *((item, "predicted") for item in predicted),
    ]:
        symbol = str(uniprot_to_symbol.get(row.uniprot_id, row.uniprot_id)).strip()
        if not symbol:
            continue
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        if row.uniprot_id not in uniprot_to_symbol:
            unmapped_count += 1
        item: dict[str, Any] = {
            "gene_symbol": symbol,
            "uniprot_id": row.uniprot_id,
            "evidence": evidence,
        }
        if row.numeric_score is not None:
            item["score"] = row.numeric_score
        if evidence == "predicted":
            item["prediction_rank"] = row.numeric_rank
            predicted_count += 1
        else:
            known_count += 1
        targets.append(item)
    return targets, {
        "known_target_count": known_count,
        "predicted_target_count": predicted_count,
        "unmapped_uniprot_count": unmapped_count,
    }


def merge_compound_targets(
    mapping_rows: Sequence[Mapping[str, str]],
    predictions: Sequence[RawPrediction],
    uniprot_to_symbol: Mapping[str, str],
    *,
    top_n_predicted: int,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]], dict[str, int]]:
    """Map raw source IDs back to every user ID and build the unified JSON payload."""

    grouped: dict[tuple[str, str], list[RawPrediction]] = {}
    for prediction in predictions:
        grouped.setdefault((prediction.source_type, prediction.source_id), []).append(prediction)

    result: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    compounds_with_targets = known_targets = predicted_targets = unmapped = 0
    for row in mapping_rows:
        compound_id = row["ID"]
        key = (row["netinfer_input_type"], row["netinfer_input_id"])
        source_rows = grouped.get(key, [])
        targets, counts = ordered_targets_for_source(
            source_rows,
            uniprot_to_symbol,
            top_n_predicted=top_n_predicted,
        )
        if not targets:
            reason = (
                "standardization_failed"
                if not row.get("match_key", "").strip()
                else "no_prediction_rows"
            )
            missing.append((compound_id, reason))
        else:
            compounds_with_targets += 1
        result[compound_id] = {"smiles": row["SMILES"], "targets": targets}
        known_targets += counts["known_target_count"]
        predicted_targets += counts["predicted_target_count"]
        unmapped += counts["unmapped_uniprot_count"]

    return (
        result,
        missing,
        {
            "input_compound_count": len(mapping_rows),
            "compounds_with_targets": compounds_with_targets,
            "missing_prediction_count": len(missing),
            "known_target_count": known_targets,
            "predicted_target_count": predicted_targets,
            "unmapped_uniprot_count": unmapped,
        },
    )


__all__ = [
    "RAW_COLUMNS",
    "RawPrediction",
    "merge_compound_targets",
    "ordered_targets_for_source",
    "parse_raw_prediction_rows",
    "read_raw_predictions",
    "validate_raw_predictions",
]
