"""Cross-platform path validation for isolated workflow runs.

There are intentionally two path domains:

* run-relative paths stored in contracts and manifests are canonical POSIX strings;
* paths used against the current filesystem are absolute native :class:`~pathlib.Path` objects.

Keeping those domains separate prevents a Windows path from becoming a literal filename on Linux
and prevents a POSIX absolute path from being joined to a Windows run directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TypeAlias

from .errors import PathSafetyError

PathLike: TypeAlias = str | os.PathLike[str]

_WINDOWS_INVALID_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f]")


def _path_text(value: PathLike | PurePath, *, label: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise PathSafetyError(f"{label} must be a string or path-like value") from exc
    if isinstance(raw, bytes):
        raise PathSafetyError(f"{label} must be text, not bytes")
    if "\x00" in raw:
        raise PathSafetyError(f"{label} contains a NUL character")
    return raw


def validate_portable_segment(value: str, *, label: str = "path segment") -> str:
    """Validate one segment that must work on both Windows and POSIX filesystems."""

    if not isinstance(value, str):
        raise PathSafetyError(f"{label} must be a string")
    if not value or value in {".", ".."}:
        raise PathSafetyError(f"{label} cannot be empty, '.' or '..'")
    if "/" in value or "\\" in value:
        raise PathSafetyError(f"{label} must be one path segment")
    if _CONTROL_CHARACTER.search(value):
        raise PathSafetyError(f"{label} contains a control character")
    invalid = sorted(set(value) & _WINDOWS_INVALID_CHARACTERS)
    if invalid:
        raise PathSafetyError(
            f"{label} contains characters that are invalid on Windows: {''.join(invalid)}"
        )
    if value.endswith((" ", ".")):
        raise PathSafetyError(f"{label} cannot end with a space or period")

    basename = value.split(".", 1)[0].rstrip(" .").upper()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        raise PathSafetyError(f"{label} uses the reserved Windows name {basename!r}")
    return value


def parse_run_relative_path(value: str | PurePath) -> PurePosixPath:
    """Parse a canonical, portable path relative to a run workspace.

    Absolute paths under either path flavour, Windows drive-relative paths such as ``C:temp``,
    UNC/device paths, backslash-separated paths, and traversal are rejected before joining.
    """

    raw = _path_text(value, label="run-relative path")
    if not raw:
        raise PathSafetyError("run-relative path cannot be empty")
    if "\\" in raw:
        raise PathSafetyError("run-relative paths must use POSIX '/' separators")

    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)
    if posix_path.is_absolute() or windows_path.drive or windows_path.root:
        raise PathSafetyError(
            "run-relative path cannot be absolute, rooted, drive-qualified, or UNC"
        )

    raw_segments = raw.split("/")
    if any(segment in {"", ".", ".."} for segment in raw_segments):
        raise PathSafetyError(
            "run-relative path must be canonical and cannot contain empty, '.' or '..' segments"
        )
    for segment in raw_segments:
        validate_portable_segment(segment)
    return posix_path


def _native_absolute_path(value: PathLike, *, label: str) -> Path:
    """Coerce an absolute path for this host without accepting a foreign path flavour."""

    raw = _path_text(value, label=label)
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)

    if os.name == "nt":
        if posix_path.is_absolute() and not windows_path.is_absolute():
            raise PathSafetyError(
                f"{label} is an ambiguous POSIX/root-relative path on Windows: {raw!r}"
            )
    elif not posix_path.is_absolute() and (windows_path.drive or windows_path.root):
        raise PathSafetyError(f"{label} is a Windows path on a POSIX host: {raw!r}")
    elif posix_path.is_absolute() and windows_path.drive:
        # A leading ``//server/share`` is lexically both POSIX-absolute and a Windows UNC path.
        # Rejecting it avoids silently changing its meaning between development and containers.
        raise PathSafetyError(f"{label} is an ambiguous UNC/POSIX path: {raw!r}")

    native = Path(raw)
    if not native.is_absolute():
        raise PathSafetyError(f"{label} must be an absolute path for the current host")
    return native


def canonical_path(
    value: PathLike,
    *,
    must_exist: bool = False,
    label: str = "path",
) -> Path:
    """Return an absolute canonical native path, resolving existing links and junctions."""

    native = _native_absolute_path(value, label=label)
    try:
        return native.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        existence = "existing " if must_exist else ""
        raise PathSafetyError(f"cannot resolve {existence}{label}: {native}") from exc


def is_filesystem_root(value: PathLike) -> bool:
    """Return whether *value* is the current host's drive, share, or POSIX root."""

    canonical = canonical_path(value, label="filesystem path")
    anchor = Path(canonical.anchor)
    return bool(canonical.anchor) and canonical == anchor


def ensure_within(
    path: PathLike,
    root: PathLike,
    *,
    allow_equal: bool = False,
    allow_root: bool | None = None,
) -> Path:
    """Return canonical *path* if it is contained by canonical *root*.

    The comparison follows existing symlink and junction components.  It deliberately uses
    ``relative_to`` rather than textual prefix checks, so sibling names such as ``run`` and
    ``run2`` cannot be confused.
    """

    if allow_root is not None:
        if allow_equal and not allow_root:
            raise TypeError("allow_equal and allow_root specify conflicting containment policies")
        allow_equal = allow_root

    canonical_root = canonical_path(root, label="containment root")
    canonical_candidate = canonical_path(path, label="candidate path")
    try:
        relative = canonical_candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise PathSafetyError(
            f"path escapes the allowed root: {canonical_candidate} is outside {canonical_root}"
        ) from exc
    if relative == Path(".") and not allow_equal:
        raise PathSafetyError(f"path must be below, not equal to, {canonical_root}")
    return canonical_candidate


def paths_overlap(first: PathLike, second: PathLike) -> bool:
    """Return whether two canonical filesystem trees contain one another."""

    first_path = canonical_path(first, label="first path")
    second_path = canonical_path(second, label="second path")
    if first_path == second_path:
        return True
    return first_path in second_path.parents or second_path in first_path.parents


def _reject_existing_symlink_components(root: Path, candidate: Path) -> None:
    """Reject existing symlinks below *root* before an output is opened or replaced."""

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PathSafetyError(f"path is not lexically below run root: {candidate}") from exc

    cursor = root
    for segment in relative.parts:
        cursor = cursor / segment
        try:
            is_link = cursor.is_symlink()
        except OSError as exc:
            raise PathSafetyError(f"cannot inspect path component: {cursor}") from exc
        if is_link:
            raise PathSafetyError(
                f"symlink path components are not allowed in run outputs: {cursor}"
            )


def resolve_run_relative(
    root: PathLike,
    value: str | PurePath,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a portable run-relative path below an existing native root."""

    canonical_root = canonical_path(root, must_exist=True, label="run root")
    relative = parse_run_relative_path(value)
    lexical_candidate = canonical_root.joinpath(*relative.parts)
    _reject_existing_symlink_components(canonical_root, lexical_candidate)
    canonical_candidate = ensure_within(lexical_candidate, canonical_root)
    if must_exist and not canonical_candidate.exists():
        raise PathSafetyError(f"required run path does not exist: {canonical_candidate}")
    return canonical_candidate


def to_run_relative_posix(path: PathLike, run_root: PathLike) -> str:
    """Serialize an absolute run path as a stable POSIX-relative manifest path."""

    canonical_root = canonical_path(run_root, must_exist=True, label="run root")
    lexical_candidate = _native_absolute_path(path, label="run path")
    _reject_existing_symlink_components(canonical_root, lexical_candidate)
    canonical_candidate = ensure_within(lexical_candidate, canonical_root)
    relative = canonical_candidate.relative_to(canonical_root)
    portable = PurePosixPath(*relative.parts).as_posix()
    return parse_run_relative_path(portable).as_posix()


__all__ = [
    "PathLike",
    "canonical_path",
    "ensure_within",
    "is_filesystem_root",
    "parse_run_relative_path",
    "paths_overlap",
    "resolve_run_relative",
    "to_run_relative_posix",
    "validate_portable_segment",
]
