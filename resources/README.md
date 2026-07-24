# External resource metadata

`manifest.json` records every file in the complete scientific-resource snapshot, including its
runtime-relative path, byte size, SHA-256, source, version, license note, and consuming module.
Resource payloads are distributed as a single versioned GitHub Release asset rather than being
committed to Git history or baked into the Docker image.

Install and verify the complete bundle:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

The downloader retrieves `smi2phen-resources-v0.1.0.tar.gz`, verifies the bundle checksum, safely
extracts only the 18 declared files beneath `.local-resources/`, and then verifies every extracted
file independently. Existing verified resources are skipped.

The release archive is built deterministically from an authorized local snapshot with:

```bash
python scripts/build_resource_bundle.py --resource-root /path/to/resources
```

See `docs/RESOURCES.md` for contents, provenance boundaries, and release instructions.
