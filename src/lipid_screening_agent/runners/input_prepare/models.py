"""Typed input-registration records shared by preparation runners and later stages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lipid_screening_agent.runtime.errors import InputError, OutputContractError
from lipid_screening_agent.runtime.paths import (
    parse_run_relative_path,
    resolve_run_relative,
    validate_portable_segment,
)
from lipid_screening_agent.runtime.time import parse_iso8601

INPUT_KEYS = frozenset(
    {
        "compound_library",
        "disease_genes",
        "drug_targets",
        "target_mapping",
        "expression_pairs",
        "positive_drugs",
        "disease_links",
    }
)
EXPRESSION_ROLES = frozenset({"tpm", "metadata"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_PAIR_ID_LENGTH = 128


def _input_error(message: str) -> InputError:
    return InputError(message, retryable=False)


def _portable_name(value: str, *, label: str) -> str:
    try:
        return validate_portable_segment(value, label=label)
    except Exception as exc:
        raise _input_error(str(exc)) from exc


def _nonempty_text(value: Any, *, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _input_error(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum or "\x00" in normalized:
        raise _input_error(f"{label} is not a valid bounded text value")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class InputRegistrationRequest:
    """One uploaded file to copy into the immutable original-input area."""

    input_key: str
    path: str | Path
    source: str = "user_upload"
    role: str = "primary"
    pair_id: str | None = None
    original_name: str | None = None

    def __post_init__(self) -> None:
        if self.input_key not in INPUT_KEYS:
            raise _input_error(f"unsupported input_key: {self.input_key!r}")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "source", _nonempty_text(self.source, label="source"))

        if self.input_key == "expression_pairs":
            if self.role not in EXPRESSION_ROLES:
                raise _input_error("expression input role must be 'tpm' or 'metadata'")
            if self.pair_id is None:
                raise _input_error("expression input requires pair_id")
            object.__setattr__(
                self,
                "pair_id",
                _portable_name(self.pair_id, label="pair_id"),
            )
            if len(self.pair_id) > MAX_PAIR_ID_LENGTH:
                raise _input_error(f"pair_id cannot exceed {MAX_PAIR_ID_LENGTH} characters")
        else:
            if self.role != "primary":
                raise _input_error("non-expression input role must be 'primary'")
            if self.pair_id is not None:
                raise _input_error("non-expression input cannot define pair_id")

        if self.original_name is not None:
            object.__setattr__(
                self,
                "original_name",
                _portable_name(self.original_name, label="original_name"),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RegisteredInputRecord:
    """One immutable file entry from ``inputs/input_manifest.json``."""

    input_key: str
    role: str
    pair_id: str | None
    original_name: str
    registered_path: str
    size_bytes: int
    sha256: str
    registered_at: str
    source: str

    def __post_init__(self) -> None:
        request = InputRegistrationRequest(
            input_key=self.input_key,
            path=Path(self.original_name),
            source=self.source,
            role=self.role,
            pair_id=self.pair_id,
            original_name=self.original_name,
        )
        object.__setattr__(self, "input_key", request.input_key)
        object.__setattr__(self, "role", request.role)
        object.__setattr__(self, "pair_id", request.pair_id)
        object.__setattr__(self, "original_name", request.original_name)
        object.__setattr__(self, "source", request.source)

        relative = parse_run_relative_path(self.registered_path)
        if len(relative.parts) != 3 or relative.parts[:2] != ("inputs", "original"):
            raise OutputContractError(
                "registered_path must name one file directly below inputs/original"
            )
        if relative.name != self.original_name:
            raise OutputContractError("registered_path filename must match original_name")
        object.__setattr__(self, "registered_path", relative.as_posix())

        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise OutputContractError("registered input size_bytes must be non-negative")
        if not isinstance(self.sha256, str) or SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise OutputContractError("registered input sha256 must be a lowercase SHA-256")
        try:
            parse_iso8601(self.registered_at)
        except ValueError as exc:
            raise OutputContractError(
                "registered_at must be a timezone-aware ISO 8601 timestamp"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_key": self.input_key,
            "role": self.role,
            "pair_id": self.pair_id,
            "original_name": self.original_name,
            "registered_path": self.registered_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "registered_at": self.registered_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RegisteredInputRecord:
        expected = {
            "input_key",
            "role",
            "pair_id",
            "original_name",
            "registered_path",
            "size_bytes",
            "sha256",
            "registered_at",
            "source",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if missing or extra:
            raise OutputContractError("registered input record fields do not match the contract")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True, kw_only=True)
class InputRegistrationManifest:
    """Strict model for the Stage 02 input-registration manifest."""

    schema_version: str
    inputs: tuple[RegisteredInputRecord, ...] | Sequence[RegisteredInputRecord]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise OutputContractError("unsupported input manifest schema_version")
        if not isinstance(self.inputs, Sequence) or isinstance(self.inputs, (str, bytes)):
            raise OutputContractError("input manifest inputs must be a sequence")
        records = tuple(self.inputs)
        if not all(isinstance(record, RegisteredInputRecord) for record in records):
            raise OutputContractError("input manifest contains an invalid record")
        identities = [(record.input_key, record.pair_id, record.role) for record in records]
        if len(identities) != len(set(identities)):
            raise OutputContractError("input manifest contains duplicate logical records")
        filenames = [record.original_name.casefold() for record in records]
        if len(filenames) != len(set(filenames)):
            raise OutputContractError("input manifest contains colliding filenames")
        object.__setattr__(self, "inputs", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": [record.to_dict() for record in self.inputs],
        }

    def records_for(
        self,
        input_key: str,
        *,
        role: str | None = None,
    ) -> tuple[RegisteredInputRecord, ...]:
        if input_key not in INPUT_KEYS:
            raise InputError(f"unsupported input_key: {input_key!r}")
        return tuple(
            record
            for record in self.inputs
            if record.input_key == input_key and (role is None or record.role == role)
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputRegistrationManifest:
        if set(value) != {"schema_version", "inputs"}:
            raise OutputContractError("input manifest fields do not match the contract")
        raw_records = value["inputs"]
        if not isinstance(raw_records, list):
            raise OutputContractError("input manifest inputs must be a JSON array")
        records = []
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise OutputContractError("input manifest record must be a JSON object")
            records.append(RegisteredInputRecord.from_dict(raw_record))
        return cls(schema_version=value["schema_version"], inputs=records)


def load_input_registration_manifest(
    path: str | Path,
    *,
    run_root: str | Path,
) -> InputRegistrationManifest:
    """Read, contain, and strictly validate an input-registration manifest."""

    manifest_path = Path(path)
    try:
        if not manifest_path.is_absolute():
            manifest_path = resolve_run_relative(
                run_root,
                manifest_path.as_posix(),
                must_exist=True,
            )
        else:
            relative = manifest_path.resolve(strict=True).relative_to(
                Path(run_root).resolve(strict=True)
            )
            manifest_path = resolve_run_relative(
                run_root,
                relative.as_posix(),
                must_exist=True,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputContractError(
            "input registration manifest is outside the run workspace or missing"
        ) from exc
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OutputContractError(
            f"could not read input registration manifest: {manifest_path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise OutputContractError("input registration manifest must be a JSON object")
    return InputRegistrationManifest.from_dict(value)


__all__ = [
    "EXPRESSION_ROLES",
    "INPUT_KEYS",
    "MAX_PAIR_ID_LENGTH",
    "InputRegistrationManifest",
    "InputRegistrationRequest",
    "RegisteredInputRecord",
    "load_input_registration_manifest",
]
