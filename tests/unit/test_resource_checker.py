from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = PROJECT_ROOT / "scripts" / "check_resources.py"
    spec = importlib.util.spec_from_file_location("check_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(payload: bytes) -> dict[str, object]:
    return {
        "resource_id": "fixture",
        "filename": "fixture.bin",
        "expected_relative_path": "module/fixture.bin",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": "test",
        "version": "test",
        "license": "test",
        "redistribution_status": "test_only",
        "download_url": "not_applicable",
        "required": True,
        "required_for": ["core", "enhanced"],
        "module": "test",
    }


def test_resource_checker_reports_ready_for_matching_file(tmp_path: Path) -> None:
    module = _module()
    payload = b"audited fixture\n"
    target = tmp_path / "module" / "fixture.bin"
    target.parent.mkdir()
    target.write_bytes(payload)

    results = module.check_resources({"resources": [_record(payload)]}, tmp_path)

    assert results[0]["status"] == "ok"
    assert module.readiness(results) == {"core": True, "enhanced": True}


def test_resource_checker_reports_missing_and_hash_mismatch(tmp_path: Path) -> None:
    module = _module()
    expected = b"expected\n"
    record = _record(expected)

    missing = module.check_resources({"resources": [record]}, tmp_path)
    assert missing[0]["status"] == "missing"
    assert module.readiness(missing) == {"core": False, "enhanced": False}

    target = tmp_path / "module" / "fixture.bin"
    target.parent.mkdir()
    target.write_bytes(b"different")
    record["size"] = len(b"different")
    mismatch = module.check_resources({"resources": [record]}, tmp_path)
    assert mismatch[0]["status"] == "sha256_mismatch"
