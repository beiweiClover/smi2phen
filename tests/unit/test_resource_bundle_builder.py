from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = PROJECT_ROOT / "scripts" / "build_resource_bundle.py"
    spec = importlib.util.spec_from_file_location("build_resource_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_builder_is_deterministic_and_selective(tmp_path: Path) -> None:
    module = _module()
    payload = b"audited scientific fixture\n"
    resource_root = tmp_path / "resources"
    target = resource_root / "module" / "fixture.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    (resource_root / "unused.bin").write_bytes(b"must not be bundled")
    manifest = {
        "resources": [
            {
                "resource_id": "fixture",
                "expected_relative_path": "module/fixture.bin",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "required": True,
            }
        ]
    }
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_metadata = module.build_bundle(manifest, resource_root, first)
    second_metadata = module.build_bundle(manifest, resource_root, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["sha256"] == second_metadata["sha256"]
    assert first_metadata["resource_ids"] == ["fixture"]
