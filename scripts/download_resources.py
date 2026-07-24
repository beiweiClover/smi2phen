"""Download authorized scientific resources and verify the audited snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MODES = ("core", "enhanced")
DOWNLOADABLE_SCHEMES = {"http", "https", "file"}


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
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(args.manifest)
        resources = selected_resources(manifest, args.mode)
        results = []
        for item in resources:
            status, message = process_resource(
                item,
                args.resource_root,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
            results.append(
                {
                    "resource_id": str(item["resource_id"]),
                    "required": bool(item["required"]),
                    "status": status,
                    "message": message,
                }
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"resource download configuration error: {exc}") from None

    if args.as_json:
        print(json.dumps({"mode": args.mode, "results": results}, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(result["message"])

    incomplete = [
        result
        for result in results
        if result["required"] and result["status"] not in {"verified", "downloaded"}
    ]
    if incomplete:
        if not args.as_json:
            print(
                "Required resources remain unavailable or unverified. "
                "Follow the MANUAL/ERROR messages, then rerun this command.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
