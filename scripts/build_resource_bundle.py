"""Build the deterministic full scientific-resource bundle for a GitHub release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    resources = payload.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("manifest resources must be a non-empty list")
    return payload


def safe_resource_path(resource_root: Path, item: dict[str, Any]) -> tuple[PurePosixPath, Path]:
    raw = str(item["expected_relative_path"])
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in relative.parts[0]
    ):
        raise ValueError(f"{item['resource_id']} has an unsafe resource path: {raw}")
    root = resource_root.resolve(strict=True)
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{item['resource_id']} escapes the resource root") from exc
    if not candidate.is_file():
        raise ValueError(f"{item['resource_id']} is not a regular file: {candidate}")
    return relative, candidate


def verify_resource(path: Path, item: dict[str, Any]) -> None:
    actual_size = path.stat().st_size
    expected_size = int(item["size"])
    if actual_size != expected_size:
        raise ValueError(
            f"{item['resource_id']} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    expected_hash = str(item["sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise ValueError(
            f"{item['resource_id']} SHA-256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )


def add_deterministic_file(
    archive: tarfile.TarFile,
    source: Path,
    archive_path: PurePosixPath,
    handle: BinaryIO,
) -> None:
    info = tarfile.TarInfo(archive_path.as_posix())
    info.size = source.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, handle)


def build_bundle(
    manifest: dict[str, Any],
    resource_root: Path,
    destination: Path,
) -> dict[str, Any]:
    resources = [item for item in manifest["resources"] if bool(item.get("required", False))]
    if not resources:
        raise ValueError("manifest has no required resources")

    prepared: list[tuple[PurePosixPath, Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()
    for item in resources:
        relative, source = safe_resource_path(resource_root, item)
        normalized = relative.as_posix()
        if normalized in seen_paths:
            raise ValueError(f"duplicate bundle path: {normalized}")
        seen_paths.add(normalized)
        verify_resource(source, item)
        prepared.append((relative, source, item))
    prepared.sort(key=lambda entry: entry[0].as_posix())

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for relative, source, _item in prepared:
                        with source.open("rb") as handle:
                            add_deterministic_file(archive, source, relative, handle)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "filename": destination.name,
        "format": "tar.gz",
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "resource_count": len(prepared),
        "uncompressed_size": sum(source.stat().st_size for _, source, _ in prepared),
        "resource_ids": [str(item["resource_id"]) for _, _, item in prepared],
    }


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "resources" / "manifest.json",
    )
    parser.add_argument(
        "--resource-root",
        type=Path,
        default=Path(os.environ.get("SMI2PHEN_RESOURCE_DIR", ".local-resources")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / ".release-staging",
    )
    parser.add_argument("--version", default="v0.1.0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filename = f"smi2phen-resources-{args.version}.tar.gz"
    destination = args.output_dir / filename
    try:
        manifest = load_manifest(args.manifest)
        metadata = build_bundle(manifest, args.resource_root, destination)
        checksum_path = args.output_dir / "SHA256SUMS"
        checksum_path.write_text(
            f"{metadata['sha256']}  {metadata['filename']}\n",
            encoding="utf-8",
            newline="\n",
        )
        metadata_path = args.output_dir / "resource-bundle-metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        raise SystemExit(f"resource bundle build failed: {exc}") from None
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
