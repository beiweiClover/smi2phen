"""Submit the bundled real-data sample to the unified scientific workflow."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FILES = {
    "compounds": "compounds.csv",
    "disease_genes": "disease_genes.tsv",
}
PROVIDED_TARGET_FILES = {
    "drug_targets": "drug_targets.json",
    "target_mapping": "target_mapping.tsv",
}
TERMINAL = {"succeeded", "failed", "blocked", "cancelled"}


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{method} {path} returned a non-object response")
    return decoded


def _json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return _request(
        base_url,
        method,
        path,
        body=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )


def _upload(base_url: str, run_id: str, kind: str, path: Path) -> None:
    boundary = f"lipid-agent-{uuid.uuid4().hex}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="upload"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode()
    body = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    _request(
        base_url,
        "POST",
        f"/runs/{run_id}/files/{kind}",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )


def _status_summary(snapshot: dict[str, Any]) -> str:
    counts = Counter(str(node.get("status")) for node in snapshot.get("nodes", []))
    rendered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    active = [
        f"{node.get('node_id')}/{node.get('task_id')}:{node.get('status')}"
        for node in snapshot.get("nodes", [])
        if node.get("status") in {"ready", "queued", "running"}
    ]
    suffix = f"; active={','.join(active)}" if active else ""
    return f"run={snapshot.get('status')}; {rendered}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the unified workflow with Python NetInfer by default"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples"
        / "provided_targets_validation",
    )
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--expression-tpm", type=Path)
    parser.add_argument("--expression-metadata", type=Path)
    parser.add_argument(
        "--provided-targets",
        action="store_true",
        help="Upload the bundled target pair and skip Python NetInfer",
    )
    args = parser.parse_args()

    if (args.expression_tpm is None) != (args.expression_metadata is None):
        raise RuntimeError("--expression-tpm and --expression-metadata must be supplied together")
    health = _request(args.base_url, "GET", "/healthz")
    if health.get("status") != "ok":
        raise RuntimeError(f"API is not healthy: {health}")

    expected_files = dict(FILES)
    if args.provided_targets:
        expected_files.update(PROVIDED_TARGET_FILES)
    missing = [
        name for name in expected_files.values() if not (args.sample_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"sample files are missing: {', '.join(missing)}")

    thread_id = f"unified-validation-{uuid.uuid4().hex[:12]}"
    _json_request(args.base_url, "POST", "/sessions", {"thread_id": thread_id})
    run = _json_request(
        args.base_url,
        "POST",
        f"/sessions/{thread_id}/runs",
        {"disease_name": "hepatic steatosis", "disease_slug": "hepatic_steatosis"},
    )
    run_id = str(run["run_id"])
    print(f"created {run_id}", flush=True)

    for kind, filename in FILES.items():
        _upload(args.base_url, run_id, kind, args.sample_dir / filename)
        print(f"uploaded {kind}: {filename}", flush=True)
    if args.provided_targets:
        for kind, filename in PROVIDED_TARGET_FILES.items():
            _upload(args.base_url, run_id, kind, args.sample_dir / filename)
            print(f"uploaded {kind}: {filename}", flush=True)
    if args.expression_tpm is not None and args.expression_metadata is not None:
        _upload(args.base_url, run_id, "expression_tpm", args.expression_tpm)
        _upload(
            args.base_url,
            run_id,
            "expression_metadata",
            args.expression_metadata,
        )
        print("uploaded expression pair: TPM + metadata", flush=True)

    plan = _request(args.base_url, "GET", f"/runs/{run_id}/plan")
    expected_mode = "enhanced" if args.expression_tpm is not None else "core"
    if args.provided_targets:
        expected_mode = f"provided_targets_{expected_mode}"
    if plan.get("mode") != expected_mode:
        raise RuntimeError(f"unexpected plan mode: {plan.get('mode')}")
    netinfer = [
        node
        for node in plan.get("nodes", [])
        if str(node.get("node_id", "")).startswith("netinfer_")
    ]
    expected_netinfer_status = "skipped" if args.provided_targets else "pending"
    if not netinfer or any(
        node.get("initial_status") != expected_netinfer_status for node in netinfer
    ):
        raise RuntimeError(f"NetInfer nodes were not all {expected_netinfer_status!r}")
    gps = [
        node
        for node in plan.get("nodes", [])
        if str(node.get("node_id", "")).startswith("gps_")
        or node.get("node_id") == "prepare_expression_inputs"
    ]
    if args.expression_tpm is not None and (
        not gps or any(node.get("initial_status") == "skipped" for node in gps)
    ):
        raise RuntimeError("the enhanced validation plan unexpectedly skipped GPS")
    target_source = "provided targets" if args.provided_targets else "Python NetInfer"
    print(f"plan accepted: {expected_mode}; target source={target_source}", flush=True)

    _json_request(
        args.base_url,
        "POST",
        f"/runs/{run_id}/start",
        {"confirmed": True},
    )
    deadline = time.monotonic() + args.timeout_seconds
    previous = ""
    while True:
        snapshot = _request(args.base_url, "GET", f"/runs/{run_id}")
        summary = _status_summary(snapshot)
        if summary != previous:
            print(summary, flush=True)
            previous = summary
        status = str(snapshot.get("status"))
        if status in TERMINAL:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"validation timed out after {args.timeout_seconds} seconds")
        time.sleep(args.poll_seconds)

    result = _request(args.base_url, "GET", f"/runs/{run_id}/results")
    output = {
        "run_id": run_id,
        "status": status,
        "evidence_mode": result.get("evidence_mode"),
        "artifact_types": sorted(
            str(artifact.get("artifact_type"))
            for artifact in result.get("artifacts", [])
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0 if status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
