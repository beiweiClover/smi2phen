from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from lipid_screening_agent.runtime.errors import PathSafetyError
from lipid_screening_agent.runtime.paths import (
    ensure_within,
    parse_run_relative_path,
    resolve_run_relative,
    to_run_relative_posix,
    validate_portable_segment,
)


def test_parse_run_relative_path_accepts_canonical_posix_path() -> None:
    parsed = parse_run_relative_path("artifacts/kg/seed-5/ranking.csv")

    assert parsed == PurePosixPath("artifacts/kg/seed-5/ranking.csv")


@pytest.mark.parametrize(
    "value",
    [
        "/etc/passwd",
        "//server/share/file",
        r"C:\outside\file",
        "C:/outside/file",
        r"C:outside\file",
        r"\Windows\file",
        r"\\server\share\file",
        r"\\?\C:\outside\file",
    ],
)
def test_parse_run_relative_path_rejects_rooted_paths_from_either_flavour(
    value: str,
) -> None:
    with pytest.raises(PathSafetyError):
        parse_run_relative_path(value)


@pytest.mark.parametrize(
    "value",
    ["", ".", "../file", "a/../file", "a/./file", "a//file", "a/", r"a\file"],
)
def test_parse_run_relative_path_rejects_noncanonical_or_ambiguous_paths(
    value: str,
) -> None:
    with pytest.raises(PathSafetyError):
        parse_run_relative_path(value)


@pytest.mark.parametrize(
    "value",
    ["CON", "CON .txt", "aux.txt", "COM1.log", "LPT9", "trailing.", "space "],
)
def test_portable_segment_rejects_windows_incompatible_names(value: str) -> None:
    with pytest.raises(PathSafetyError):
        validate_portable_segment(value)


def test_ensure_within_uses_path_components_not_string_prefixes(tmp_path: Path) -> None:
    run_root = (tmp_path / "run").resolve()
    sibling = (tmp_path / "run2" / "artifact.json").resolve()
    run_root.mkdir()

    with pytest.raises(PathSafetyError):
        ensure_within(sibling, run_root)


def test_ensure_within_rejects_root_by_default_and_supports_compatibility_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()

    with pytest.raises(PathSafetyError):
        ensure_within(root, root)
    assert ensure_within(root, root, allow_root=True) == root


def test_resolve_and_serialize_unicode_run_path(tmp_path: Path) -> None:
    run_root = tmp_path / "运行目录"
    run_root.mkdir()
    candidate = resolve_run_relative(run_root, "artifacts/final/result.tsv")

    assert candidate == (run_root / "artifacts" / "final" / "result.tsv").resolve()
    assert to_run_relative_posix(candidate, run_root) == "artifacts/final/result.tsv"


def test_resolve_run_relative_requires_existing_file_when_requested(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()

    with pytest.raises(PathSafetyError, match="does not exist"):
        resolve_run_relative(run_root, "artifacts/missing.json", must_exist=True)


def test_symlink_parent_cannot_escape_run_workspace(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outside = tmp_path / "outside"
    run_root.mkdir()
    outside.mkdir()
    link = run_root / "artifacts"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not available for this test user")

    with pytest.raises(PathSafetyError):
        resolve_run_relative(run_root, "artifacts/result.json")


def test_symlink_file_is_not_serialized_as_an_artifact(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    outside = tmp_path / "outside.txt"
    run_root.mkdir()
    outside.write_text("outside", encoding="utf-8")
    link = run_root / "result.txt"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not available for this test user")

    with pytest.raises(PathSafetyError):
        to_run_relative_posix(link, run_root)
