from __future__ import annotations

import hashlib
import importlib.util
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = PROJECT_ROOT / "scripts" / "download_resources.py"
    spec = importlib.util.spec_from_file_location("download_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(payload: bytes, download_url: str | None) -> dict[str, object]:
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
        "download_url": download_url,
        "required": True,
        "required_for": ["core", "enhanced"],
        "module": "test",
    }


def _bundle_manifest(
    record: dict[str, object],
    archive: Path,
    *,
    sha256: str | None = None,
) -> dict[str, object]:
    return {
        "resources": [record],
        "resource_bundle": {
            "bundle_id": "fixture-bundle",
            "filename": archive.name,
            "format": "tar.gz",
            "size": archive.stat().st_size,
            "sha256": sha256 or hashlib.sha256(archive.read_bytes()).hexdigest(),
            "download_url": archive.as_uri(),
            "resource_ids": [record["resource_id"]],
        },
    }


def _write_bundle(archive: Path, member_name: str, payload: bytes) -> None:
    source = archive.parent / "bundle-source.bin"
    source.write_bytes(payload)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=member_name)


def test_downloads_and_then_skips_verified_file(tmp_path: Path) -> None:
    module = _module()
    payload = b"audited fixture\n"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    resource_root = tmp_path / "resources"
    record = _record(payload, source.as_uri())

    status, _ = module.process_resource(
        record, resource_root, timeout=5.0, dry_run=False
    )
    assert status == "downloaded"
    assert (resource_root / "module" / "fixture.bin").read_bytes() == payload

    status, _ = module.process_resource(
        record, resource_root, timeout=5.0, dry_run=False
    )
    assert status == "verified"


def test_rejects_hash_mismatch_without_replacing_destination(tmp_path: Path) -> None:
    module = _module()
    expected = b"expected\n"
    source = tmp_path / "source.bin"
    source.write_bytes(b"different")
    resource_root = tmp_path / "resources"
    target = resource_root / "module" / "fixture.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    record = _record(expected, source.as_uri())

    status, message = module.process_resource(
        record, resource_root, timeout=5.0, dry_run=False
    )

    assert status == "error"
    assert "mismatch" in message
    assert target.read_bytes() == b"existing"


def test_manual_resource_reports_path_and_source_page(tmp_path: Path) -> None:
    module = _module()
    record = _record(b"manual\n", None)
    record["manual_download"] = {
        "official_page": "https://example.invalid/resource",
        "reason": "license needs review",
    }

    status, message = module.process_resource(
        record, tmp_path, timeout=5.0, dry_run=False
    )

    assert status == "manual"
    assert "license needs review" in message
    assert "https://example.invalid/resource" in message


def test_destination_cannot_escape_resource_root(tmp_path: Path) -> None:
    module = _module()
    record = _record(b"unsafe\n", None)
    record["expected_relative_path"] = "../outside.bin"

    try:
        module.destination_for(tmp_path, record)
    except ValueError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("unsafe destination was accepted")


def test_full_bundle_downloads_installs_and_then_skips(tmp_path: Path) -> None:
    module = _module()
    payload = b"complete audited fixture\n"
    archive = tmp_path / "resources.tar.gz"
    _write_bundle(archive, "module/fixture.bin", payload)
    record = _record(payload, None)
    manifest = _bundle_manifest(record, archive)
    resource_root = tmp_path / "installed"

    status, message = module.process_bundle(
        manifest,
        resource_root,
        timeout=5.0,
        dry_run=False,
    )

    assert status == "downloaded"
    assert "1/1" in message
    assert (resource_root / "module" / "fixture.bin").read_bytes() == payload

    status, message = module.process_bundle(
        manifest,
        resource_root,
        timeout=5.0,
        dry_run=False,
    )
    assert status == "verified"
    assert "1/1" in message


def test_full_bundle_rejects_archive_hash_mismatch(tmp_path: Path) -> None:
    module = _module()
    payload = b"expected fixture\n"
    archive = tmp_path / "resources.tar.gz"
    _write_bundle(archive, "module/fixture.bin", payload)
    record = _record(payload, None)
    manifest = _bundle_manifest(record, archive, sha256="0" * 64)
    resource_root = tmp_path / "installed"

    status, message = module.process_bundle(
        manifest,
        resource_root,
        timeout=5.0,
        dry_run=False,
    )

    assert status == "error"
    assert "SHA-256 mismatch" in message
    assert not (resource_root / "module" / "fixture.bin").exists()


def test_full_bundle_rejects_unexpected_member_without_path_escape(tmp_path: Path) -> None:
    module = _module()
    payload = b"expected fixture\n"
    archive = tmp_path / "resources.tar.gz"
    _write_bundle(archive, "../escape.bin", payload)
    record = _record(payload, None)
    manifest = _bundle_manifest(record, archive)
    resource_root = tmp_path / "installed"

    status, message = module.process_bundle(
        manifest,
        resource_root,
        timeout=5.0,
        dry_run=False,
    )

    assert status == "error"
    assert "member set mismatch" in message
    assert not (tmp_path / "escape.bin").exists()
