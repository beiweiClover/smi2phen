"""Check external scientific resources and report Core/Enhanced readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "resource_id",
    "filename",
    "expected_relative_path",
    "size",
    "sha256",
    "source",
    "version",
    "license",
    "redistribution_status",
    "download_url",
    "required",
    "module",
}
MODES = ("core", "enhanced")


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
    seen: set[str] = set()
    for index, item in enumerate(resources):
        if not isinstance(item, dict):
            raise ValueError(f"resource at index {index} is not an object")
        missing = sorted(REQUIRED_FIELDS - item.keys())
        if missing:
            raise ValueError(
                f"resource at index {index} is missing fields: {', '.join(missing)}"
            )
        resource_id = str(item["resource_id"])
        if resource_id in seen:
            raise ValueError(f"duplicate resource_id: {resource_id}")
        seen.add(resource_id)
        required_for = item.get("required_for")
        if not isinstance(required_for, list) or not required_for:
            raise ValueError(f"{resource_id} requires a non-empty required_for list")
        unknown_modes = sorted(set(map(str, required_for)) - set(MODES))
        if unknown_modes:
            raise ValueError(f"{resource_id} has unknown modes: {', '.join(unknown_modes)}")
    return payload


def check_resources(manifest: dict[str, Any], resource_root: Path) -> list[dict[str, Any]]:
    root = resource_root.resolve(strict=False)
    results: list[dict[str, Any]] = []
    for item in manifest["resources"]:
        relative = Path(str(item["expected_relative_path"]))
        candidate = (root / relative).resolve(strict=False)
        result = {
            "resource_id": str(item["resource_id"]),
            "expected_relative_path": relative.as_posix(),
            "required_for": list(map(str, item["required_for"])),
            "module": str(item["module"]),
            "status": "missing",
        }
        try:
            candidate.relative_to(root)
        except ValueError:
            result["status"] = "invalid_path"
            results.append(result)
            continue
        if not candidate.is_file():
            results.append(result)
            continue
        actual_size = candidate.stat().st_size
        expected_size = int(item["size"])
        result["actual_size"] = actual_size
        result["expected_size"] = expected_size
        if actual_size != expected_size:
            result["status"] = "size_mismatch"
            results.append(result)
            continue
        actual_hash = sha256_file(candidate)
        result["actual_sha256"] = actual_hash
        if actual_hash.lower() != str(item["sha256"]).lower():
            result["status"] = "sha256_mismatch"
        else:
            result["status"] = "ok"
        results.append(result)
    return results


def readiness(results: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        mode: all(
            result["status"] == "ok"
            for result in results
            if mode in result["required_for"]
        )
        for mode in MODES
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
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = load_manifest(args.manifest)
        results = check_resources(manifest, args.resource_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"resource check configuration error: {exc}") from None

    mode_readiness = readiness(results)
    selected_modes = MODES if args.mode == "all" else (args.mode,)
    selected_results = [
        result
        for result in results
        if any(mode in result["required_for"] for mode in selected_modes)
    ]
    payload = {
        "resource_root": str(args.resource_root),
        "selected_mode": args.mode,
        "readiness": mode_readiness,
        "checked": len(selected_results),
        "problems": [
            result for result in selected_results if result["status"] != "ok"
        ],
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in payload["problems"]:
            print(
                f"{result['status'].upper()} {result['resource_id']}: "
                f"{result['expected_relative_path']}"
            )
        for mode in MODES:
            label = "READY" if mode_readiness[mode] else "NOT READY"
            print(f"{mode}: {label}")
        if payload["problems"]:
            print(
                "Place authorized files at the listed relative paths and verify the "
                "manifest metadata before use."
            )
    return 0 if all(mode_readiness[mode] for mode in selected_modes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
