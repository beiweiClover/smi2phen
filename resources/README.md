# External resource metadata

No scientific resource payload is stored in this directory. `manifest.json` records the filenames,
relative paths, sizes, and SHA-256 values of the local snapshot audited during packaging.

Place authorized resource files in a separate ignored directory such as `.local-resources/`, then
run:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --resource-root .local-resources
```

Nine GPS files have pinned, byte-verified official download URLs. The NCBI, NetInfer, PPI, and
transformed KG entries remain manual-only because an exact revision, reproducible derivation, or
redistribution right is unresolved. See `docs/RESOURCES.md`. Matching a checksum does not grant
redistribution rights.
