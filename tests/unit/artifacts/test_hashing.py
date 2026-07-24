from pathlib import Path

import pytest

from lipid_screening_agent.config import hash_workflow_config, load_workflow_config
from lipid_screening_agent.runtime.hashing import (
    file_digest,
    hash_config,
    hash_json,
    normalize_json_value,
    sha256_bytes,
    sha256_file,
    stable_json_dumps,
)

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_file_digest_reports_exact_size_and_sha256(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"abc")

    digest = file_digest(source, chunk_size=2)

    assert digest.size_bytes == 3
    assert digest.sha256 == ABC_SHA256
    assert sha256_file(source) == ABC_SHA256
    assert sha256_bytes(b"abc") == ABC_SHA256


def test_file_digest_rejects_non_file_and_invalid_chunk_size(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        file_digest(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        file_digest(tmp_path / "missing", chunk_size=0)
    with pytest.raises(FileNotFoundError):
        file_digest(tmp_path / "missing")


def test_stable_json_and_config_hash_ignore_mapping_insertion_order() -> None:
    left = {"z": [3, 2, 1], "a": {"enabled": True, "value": 0.5}}
    right = {"a": {"value": 0.5, "enabled": True}, "z": (3, 2, 1)}

    assert stable_json_dumps(left) == stable_json_dumps(right)
    assert hash_config(left) == hash_config(right)
    assert hash_config(left) != hash_config({**left, "new": None})


def test_generic_config_hash_accepts_the_typed_workflow_model() -> None:
    config = load_workflow_config(PROJECT_ROOT / "configs" / "workflow.yaml")

    assert hash_config(config) == hash_workflow_config(config)


def test_hash_json_uses_utf8_without_ascii_escaping() -> None:
    value = {"disease": "脂肪肝"}

    serialized = stable_json_dumps(value)

    assert "脂肪肝" in serialized
    assert hash_json(value) == sha256_bytes(serialized.encode("utf-8"))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_not_json_compatible(invalid: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        normalize_json_value({"metric": invalid})


def test_non_string_mapping_keys_and_arbitrary_objects_are_rejected() -> None:
    with pytest.raises(TypeError, match="mapping key"):
        normalize_json_value({1: "not portable"})
    with pytest.raises(TypeError, match="not JSON-compatible"):
        normalize_json_value({"path": Path("artifact.csv")})
