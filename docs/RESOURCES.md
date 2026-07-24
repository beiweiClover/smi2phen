# Scientific resources

Large resources are not committed to the source repository or baked into the Docker image. The
expected layout below is relative to the directory mounted at `/resources` (the Compose default is
`.local-resources/`):

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

## Automatic download and verification

Run:

```bash
python scripts/download_resources.py
python scripts/check_resources.py
```

The downloader:

- writes only below `.local-resources/` unless `--resource-root` is supplied;
- skips an existing file only after size and SHA-256 both match;
- downloads to a temporary file and replaces the destination only after verification;
- reports a clear error for download, size, or hash failures;
- prints an official page, expected path, and review reason for manual-only resources;
- does not contain or require a private token for public downloads.

Use `--mode core`, `--mode enhanced`, `--dry-run`, or `--json` when needed. The command exits
non-zero while any selected required resource is missing or unverified; that is expected when the
manual-only resources have not yet been supplied.

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

The upstream repository identifies its license as Apache-2.0. The manifest therefore uses pinned
upstream raw links for these exact files. Rehosting the same bytes on Google Drive is optional; if
done, preserve upstream attribution/license information and replace a manifest URL only after the
public Drive download has been tested against the recorded hash.

The following resources are not approved for rehosting:

- NCBI `Homo_sapiens.gene_info.gz`: the official current file is mutable and was not confirmed as
  the exact audited snapshot;
- NetInfer `DT.tsv`, `DS.tsv`, and workbook: the article is open access, but the exact file revision
  and all incorporated data-source redistribution rights are not established;
- `HumanInteractome.tsv`: the assembled InWeb_IM/IntAct/PINA version and combined terms are not
  reproducibly documented;
- the four KG base files: these are locally transformed PrimeKG-derived files, not byte-identical
  upstream PrimeKG downloads, and their transformation provenance and component-source licenses
  need review.

For those entries, `download_url` is `null`, `redistribution_status` is
`manual_only_needs_review`, and the downloader provides a manual prompt. Do not upload them merely
because a local snapshot or hash exists.

## Readiness meaning

```bash
python scripts/check_resources.py --mode core
python scripts/check_resources.py --mode enhanced --json
```

Core readiness requires mapping, NetInfer, PPI, and KG resources. Enhanced readiness requires the
Core set plus GPS model definitions, checkpoints, and configured cell-line feature tables. A
provided-target run can skip NetInfer at planning time, but the conservative Core readiness report
checks the default NetInfer path.

Size/hash readiness proves only byte identity to the audited snapshot. It does not establish
scientific validity, currency, license compliance, clinical validity, or permission to
redistribute.
