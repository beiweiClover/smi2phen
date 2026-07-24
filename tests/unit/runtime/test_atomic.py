import json
import os
from pathlib import Path

import pytest

from lipid_screening_agent.runtime.atomic import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from lipid_screening_agent.runtime.errors import PathSafetyError


def test_atomic_writers_replace_content_and_emit_stable_json(tmp_path: Path) -> None:
    target = tmp_path / "metadata" / "value.json"
    atomic_write_text(target, "old", allowed_root=tmp_path)
    atomic_write_json(target, {"z": 1, "a": "值"}, allowed_root=tmp_path)

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": "值", "z": 1}
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert target.read_text(encoding="utf-8").index('"a"') < target.read_text(
        encoding="utf-8"
    ).index('"z"')


def test_atomic_write_rejects_target_outside_allowed_root(tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError, match="outside|within|root"):
        atomic_write_bytes(tmp_path.parent / "outside.bin", b"x", allowed_root=tmp_path)


@pytest.mark.parametrize("relative", ["../outside.txt", r"windows\style.txt", "C:drive.txt"])
def test_atomic_write_never_misjoins_ambiguous_relative_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    with pytest.raises(PathSafetyError):
        atomic_write_text(relative, "x", allowed_root=tmp_path)


def test_atomic_replace_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("stable", encoding="utf-8")

    def fail_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_text(target, "new", allowed_root=tmp_path)

    assert target.read_text(encoding="utf-8") == "stable"
    assert not list(tmp_path.glob(".result.txt.*.tmp"))


def test_atomic_json_rejects_non_standard_floats(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_write_json(tmp_path / "bad.json", {"value": float("nan")}, allowed_root=tmp_path)
