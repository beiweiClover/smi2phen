# Scientific resources

Scientific resource payloads are distributed as one complete, checksummed GitHub Release asset.
They are not committed to Git history or baked into the Docker image. The expected layout below is
relative to the directory mounted at `/resources` (the Compose default is `.local-resources/`):

```text
.local-resources/
├── gps/
│   ├── Homo_sapiens.gene_info.gz
│   └── GPS4Drugs/
├── netinfer/
│   ├── DT.tsv
│   ├── DS.tsv
│   └── d1sc05613a1_suppl.xlsx
├── ppi/
│   └── HumanInteractome.tsv
└── kg-base/
    ├── node.csv
    ├── edges.csv
    ├── manifest.json
    └── base_drug_smiles.tsv
```

The complete file list, target paths, sizes, SHA-256 values, modules, readiness modes, sources, and
redistribution decisions are in `resources/manifest.json`.

The complete bundle contains exactly the 18 paths declared by `resources/manifest.json`. The unused
local `kg.csv` is not a runtime dependency and is deliberately excluded.

## Complete bundle download and verification

Run:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

The downloader:

- writes only below `.local-resources/` unless `--resource-root` is supplied;
- skips the download when all 18 existing files match size and SHA-256;
- downloads the full archive to a temporary file;
- checks the archive size and SHA-256 before opening it;
- rejects duplicate, missing, extra, non-regular, wrong-size, or path-escaping archive members;
- extracts into a temporary staging directory rather than over an active resource tree;
- verifies all 18 extracted files independently before installing them;
- reports a clear non-zero error for download, archive, extraction, size, or hash failures;
- requires no private token after the GitHub Release asset is public.

`--dry-run` reports the configured archive without downloading it, `--json` emits machine-readable
status, and `--bundle-url` is available for local release testing. The public command always
installs the complete Enhanced resource snapshot; it does not offer a reduced Core-only download.

## Release classification

No project-trained checkpoint is required by the current code. KG checkpoints are generated during
a run and are run artifacts, not release inputs.

The following third-party GPS files were downloaded from the official
`Bin-Chen-Lab/GPS` repository at commit
`c11668aaa08a68ec3e2e9d93d79ca4dd1956ba98`, and every file matched the audited manifest size and
SHA-256:

- `GPS4Drugs/code/model.py`;
- four configured `model.pkl` checkpoints;
- four configured `go_fingerprints_2k_*.csv` feature tables.

The upstream repository identifies the GPS code snapshot as Apache-2.0. The remaining source,
version, derivation, and license notes are retained per file in the manifest. On 2026-07-24, the
project owner explicitly confirmed that the selected full-bundle snapshot may be redistributed.
That owner decision is recorded as `owner_confirmed_release_bundle`; it does not erase upstream
attribution or turn the project MIT license into a license for third-party data.

The deterministic release archive can be rebuilt from an authorized resource root:

```bash
python scripts/build_resource_bundle.py \
  --resource-root /path/to/audited/resources \
  --version v0.1.0
```

The output directory `.release-staging/` is ignored by Git and Docker. The builder verifies every
source file against the manifest before packaging it, normalizes archive metadata, includes only
declared required files, and emits the archive, `SHA256SUMS`, and
`resource-bundle-metadata.json`.

## Readiness meaning

```bash
python scripts/check_resources.py --mode core
python scripts/check_resources.py --mode enhanced --json
```

Core readiness still describes the scientific evidence boundary, but distribution uses only the
complete Enhanced bundle. Enhanced readiness requires mapping, NetInfer, PPI, KG, and all configured
GPS files. A provided-target run can skip NetInfer at planning time, but the conservative readiness
report verifies the complete shipped snapshot.

Size/hash readiness proves only byte identity to the audited snapshot. It does not establish
scientific validity, currency, license compliance, clinical validity, or permission to
redistribute.
