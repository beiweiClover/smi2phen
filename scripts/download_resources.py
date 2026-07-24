"""Download authorized scientific resources and verify the audited snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

MODES = ("core", "enhanced")
DOWNLOADABLE_SCHEMES = {"http", "https", "file"}
BUNDLE_FIELDS = {
    "bundle_id",
    "filename",
    "format",
    "size",
    "sha256",
    "download_url",
    "resource_ids",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ValueError("manifest resources must be a list")
    for index, item in enumerate(resources):
        if not isinstance(item, dict):
            raise ValueError(f"resource at index {index} is not an object")
        for field in (
            "resource_id",
            "expected_relative_path",
            "size",
            "sha256",
            "download_url",
            "required",
            "required_for",
        ):
            if field not in item:
                raise ValueError(f"resource at index {index} is missing field: {field}")
    bundle = payload.get("resource_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("manifest resource_bundle must be an object")
    missing_bundle_fields = sorted(BUNDLE_FIELDS - bundle.keys())
    if missing_bundle_fields:
        raise ValueError(
            "manifest resource_bundle is missing fields: "
            + ", ".join(missing_bundle_fields)
        )
    if bundle["format"] != "tar.gz":
        raise ValueError("manifest resource_bundle format must be tar.gz")
    resource_ids = [str(item["resource_id"]) for item in resources]
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError("manifest contains duplicate resource_ids")
    bundle_resource_ids = list(map(str, bundle["resource_ids"]))
    if sorted(bundle_resource_ids) != sorted(resource_ids):
        raise ValueError("resource_bundle resource_ids must match all manifest resources")
    if len(bundle_resource_ids) != len(set(bundle_resource_ids)):
        raise ValueError("resource_bundle contains duplicate resource_ids")
    return payload


def selected_resources(manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    resources = list(manifest["resources"])
    if mode == "all":
        return resources
    return [item for item in resources if mode in item["required_for"]]


def destination_for(resource_root: Path, item: dict[str, Any]) -> Path:
    root = resource_root.resolve(strict=False)
    relative = Path(str(item["expected_relative_path"]))
    if relative.is_absolute():
        raise ValueError(f"{item['resource_id']} has an absolute destination path")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{item['resource_id']} destination escapes the resource root"
        ) from exc
    return candidate


def verify_file(path: Path, item: dict[str, Any]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    expected_size = int(item["size"])
    if actual_size != expected_size:
        return False, f"size mismatch: expected {expected_size}, got {actual_size}"
    actual_hash = sha256_file(path)
    expected_hash = str(item["sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        return False, f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
    return True, "verified"


def downloadable_url(item: dict[str, Any]) -> str | None:
    value = item.get("download_url")
    if not isinstance(value, str) or not value.strip():
        return None
    url = value.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in DOWNLOADABLE_SCHEMES:
        return None
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{item['resource_id']} download URL must not contain credentials")
    return url


def manual_message(item: dict[str, Any], destination: Path) -> str:
    manual = item.get("manual_download")
    if isinstance(manual, dict):
        page = manual.get("official_page")
        reason = manual.get("reason")
    else:
        page = None
        reason = None
    details = [
        f"MANUAL {item['resource_id']}: place an authorized copy at {destination}",
    ]
    if reason:
        details.append(f"reason={reason}")
    if page:
        details.append(f"official_page={page}")
    details.append(f"expected_sha256={item['sha256']}")
    return "; ".join(details)


def download_to_temporary(url: str, destination: Path, timeout: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "smi2phen-resource-downloader/0.1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def bundle_download_url(bundle: dict[str, Any]) -> str:
    value = bundle.get("download_url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("resource_bundle download_url must be a non-empty URL")
    url = value.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in DOWNLOADABLE_SCHEMES:
        raise ValueError(f"resource_bundle URL scheme is not supported: {parsed.scheme}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("resource_bundle download URL must not contain credentials")
    return url


def verify_bundle_archive(path: Path, bundle: dict[str, Any]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    expected_size = int(bundle["size"])
    if actual_size != expected_size:
        return False, f"size mismatch: expected {expected_size}, got {actual_size}"
    actual_hash = sha256_file(path)
    expected_hash = str(bundle["sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        return False, f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
    return True, "verified"


def bundle_member_name(item: dict[str, Any]) -> str:
    raw = str(item["expected_relative_path"])
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in relative.parts[0]
    ):
        raise ValueError(f"{item['resource_id']} has an unsafe bundle path: {raw}")
    return relative.as_posix()


def extract_verified_bundle(
    archive_path: Path,
    manifest: dict[str, Any],
    staging_root: Path,
) -> None:
    expected = {
        bundle_member_name(item): item
        for item in manifest["resources"]
        if str(item["resource_id"]) in manifest["resource_bundle"]["resource_ids"]
    }
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("resource bundle contains duplicate member names")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ValueError(
                f"resource bundle member set mismatch; missing={missing}, extra={extra}"
            )
        for member in members:
            item = expected[member.name]
            if not member.isfile():
                raise ValueError(f"resource bundle member is not a regular file: {member.name}")
            expected_size = int(item["size"])
            if member.size != expected_size:
                raise ValueError(
                    f"resource bundle member size mismatch for {member.name}: "
                    f"expected {expected_size}, got {member.size}"
                )
            destination = destination_for(staging_root, item)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read resource bundle member: {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    for item in manifest["resources"]:
        destination = destination_for(staging_root, item)
        valid, detail = verify_file(destination, item)
        if not valid:
            raise ValueError(f"extracted {item['resource_id']} failed verification: {detail}")


def install_staged_resources(
    manifest: dict[str, Any],
    staging_root: Path,
    resource_root: Path,
) -> None:
    for item in manifest["resources"]:
        source = destination_for(staging_root, item)
        destination = destination_for(resource_root, item)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def verified_resource_count(manifest: dict[str, Any], resource_root: Path) -> int:
    return sum(
        verify_file(destination_for(resource_root, item), item)[0]
        for item in manifest["resources"]
    )


def process_bundle(
    manifest: dict[str, Any],
    resource_root: Path,
    *,
    timeout: float,
    dry_run: bool,
    bundle_url_override: str | None = None,
) -> tuple[str, str]:
    bundle = dict(manifest["resource_bundle"])
    if bundle_url_override is not None:
        bundle["download_url"] = bundle_url_override
    total = len(manifest["resources"])
    verified_before = verified_resource_count(manifest, resource_root)
    if verified_before == total:
        return (
            "verified",
            f"SKIP {bundle['bundle_id']}: all {total}/{total} resources already verified",
        )

    url = bundle_download_url(bundle)
    if dry_run:
        return (
            "available",
            f"AVAILABLE {bundle['bundle_id']}: {url} "
            f"({verified_before}/{total} resources already verified)",
        )

    root = resource_root.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive: Path | None = None
    try:
        temporary_archive = download_to_temporary(
            url,
            root.parent / str(bundle["filename"]),
            timeout,
        )
        valid, detail = verify_bundle_archive(temporary_archive, bundle)
        if not valid:
            return (
                "error",
                f"ERROR {bundle['bundle_id']}: downloaded bundle {detail}; source={url}",
            )
        with tempfile.TemporaryDirectory(
            prefix=".smi2phen-resource-extract-",
            dir=root.parent,
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            extract_verified_bundle(temporary_archive, manifest, staging_root)
            install_staged_resources(manifest, staging_root, root)
        verified_after = verified_resource_count(manifest, root)
        if verified_after != total:
            return (
                "error",
                f"ERROR {bundle['bundle_id']}: installation left only "
                f"{verified_after}/{total} verified resources",
            )
        return (
            "downloaded",
            f"DOWNLOADED {bundle['bundle_id']}: installed and verified "
            f"{verified_after}/{total} resources under {root}",
        )
    except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as exc:
        return "error", f"ERROR {bundle['bundle_id']}: bundle installation failed: {exc}"
    finally:
        if temporary_archive is not None:
            temporary_archive.unlink(missing_ok=True)


def process_resource(
    item: dict[str, Any],
    resource_root: Path,
    *,
    timeout: float,
    dry_run: bool,
) -> tuple[str, str]:
    destination = destination_for(resource_root, item)
    verified, detail = verify_file(destination, item)
    if verified:
        return "verified", f"SKIP {item['resource_id']}: already verified at {destination}"

    url = downloadable_url(item)
    if url is None:
        return "manual", manual_message(item, destination)
    if dry_run:
        return (
            "available",
            f"AVAILABLE {item['resource_id']}: {url} -> {destination} ({detail})",
        )

    temporary: Path | None = None
    try:
        temporary = download_to_temporary(url, destination, timeout)
        valid, downloaded_detail = verify_file(temporary, item)
        if not valid:
            return (
                "error",
                f"ERROR {item['resource_id']}: downloaded file {downloaded_detail}; "
                f"source={url}",
            )
        temporary.replace(destination)
        temporary = None
        return (
            "downloaded",
            f"DOWNLOADED {item['resource_id']}: verified and saved to {destination}",
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return "error", f"ERROR {item['resource_id']}: download failed: {exc}"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        "--mode",
        choices=("enhanced", "all"),
        default="enhanced",
        help="Compatibility option; the downloader always installs the complete resource bundle.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--bundle-url",
        help="Override the manifest bundle URL for local release testing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(args.manifest)
        status, message = process_bundle(
            manifest,
            args.resource_root,
            timeout=args.timeout,
            dry_run=args.dry_run,
            bundle_url_override=args.bundle_url,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"resource download configuration error: {exc}") from None

    if args.as_json:
        print(
            json.dumps(
                {
                    "mode": "enhanced",
                    "bundle_id": manifest["resource_bundle"]["bundle_id"],
                    "status": status,
                    "message": message,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(message)
    return 0 if status in {"verified", "downloaded", "available"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
