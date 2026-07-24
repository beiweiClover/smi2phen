# Validation record

This file records commands executed in the clean `smi2phen` directory on 2026-07-24. “Not run”
never means “passed.”

## Level 1 — completed

- Unit/API tests: **225 passed, 10 skipped, 1 warning** with Python 3.13.5 and pytest 8.4.2.
  Five optional scientific test modules were collection-skipped because the host Python environment
  lacked optional statsmodels/RDKit/openpyxl dependencies; individual skips also covered missing
  RDKit/DGL and unavailable Windows symlink privileges. Container import assertions separately
  covered the locked scientific environment.
- Python compilation: **passed** for `src`, `scripts`, and `tests`.
- Ruff 0.15.22: **passed**.
- `docker compose config --quiet` with Compose 5.3.0: **passed**.
- Git candidate audit: 191 files, approximately 1.5 MB total, largest file 79,585 bytes; **no ordinary
  file over 100 MB**.
- Forbidden state/filename scan: **no candidate database, runtime state, log, cache, private-key
  file, real `.env`, or credential file**.
- Content risk scan: ten files contain credential-related terminology because they define,
  document, redact, or test the optional model API-key path. **No credential literal, recognized
  token format, private-key block, Windows user-profile path, local project absolute path, or email
  address was detected.**
- Ignore-rule probes: `.env`, runtime SQLite, external resources, private resource payloads, and
  `deepseekapi.txt` were all ignored as intended.
- Static example: **81 lines** (1 header + 80 records), SHA-256
  `098f96829d5de39ee5a30a0615c10139c0b437cd4ecb0466564473db8e58485d`.
- Resource checker and downloader tests: included in the passing test total.

## Release-preparation revalidation

- Resource manifest: **18 records parsed**, comprising 9 pinned automatic GPS downloads and 9
  manual-only entries.
- Pinned GPS audit: the 9 configured files were compared with official
  `Bin-Chen-Lab/GPS` commit `c11668aaa08a68ec3e2e9d93d79ca4dd1956ba98`; **all sizes and
  SHA-256 values matched**.
- Downloader network test: in a new empty ignored directory, **9/9 GPS files downloaded and
  verified**. The command returned non-zero because the 9 manual-only required resources were
  intentionally absent; each produced an acquisition/review prompt.
- Current local image build: **passed** as
  `ghcr.io/beiweiclover/smi2phen:v0.1.0`, approximately 4.24 GB. The MIT `LICENSE` is present in
  the image. No image was pushed.
- Isolated Compose start: **Redis, API, and Worker were healthy**; the Web root returned HTTP 200,
  `/healthz` returned `status=ok`, Worker heartbeat keys were present, and the state database
  initially contained zero sessions.
- Image content audit: **no `.env`, local resource directory, historical runtime/database path, or
  scientific resource payload** was present.
- The isolated service check used an empty resource mount. A successful scientific task was
  therefore **not run and not claimed**. Containers, the named Redis volume, runtime directories,
  downloaded audit files, and staging files were removed afterward.
- A new local clone with no hard-linked working-tree files checked out the release-preparation
  commit with a clean status. Compose parsing passed, no forbidden state/credential file was
  present, and the static example remained 81 lines with SHA-256
  `098f96829d5de39ee5a30a0615c10139c0b437cd4ecb0466564473db8e58485d`.
  This is a local source-integrity check, not a clone-from-GitHub result.
- The public target `https://github.com/beiweiClover/smi2phen`, MIT source license, and
  `ghcr.io/beiweiclover/smi2phen:v0.1.0` image name are recorded. A normal GitHub source push and a
  GHCR image push were attempted; both were **rejected by authentication** because the workstation
  had no valid GitHub/GHCR login. No force push was used and no remote source commit or image digest
  was created. Google Drive mirror upload and clone-from-remote clean-room validation remain
  **not run**.

## Level 2 — completed with a reduced smoke configuration

- External resource check: **18/18 files matched size and SHA-256**; Core and Enhanced readiness
  both reported true for the audited local snapshot.
- Docker image build: **passed** for local `smi2phen:latest`; final image size was approximately
  4.24 GB. Container build assertions imported FastAPI, Torch 2.1.2, DGL 2.4.0, NumPy, pandas,
  NetworkX, SciPy, statsmodels, scikit-learn, scikit-fingerprints, and RDKit.
- Compose startup: **passed** using host port 18000 because port 8000 was already occupied by the
  untouched source project's service.
- Redis, API, and unified Worker: **all healthy**.
- `GET /healthz`: **passed**, returning `status=ok` and `execution=external_workers`.
- Clean startup: **passed** with zero persisted sessions before smoke submission.
- Worker heartbeat: **present** in the isolated `smi2phen` Redis namespace.
- Queue consumption: **confirmed** by nodes transitioning from queued to running/succeeded on the
  external Worker.
- Minimal smoke: **succeeded** with `examples/minimal_inputs`, the provided-target boundary, and the
  source runtime's validation mode. The plan was `provided_targets_core`; 14 nodes succeeded and 8
  NetInfer/GPS-related nodes were skipped as planned. Final artifact types were
  `final_candidates`, `ranking_summary`, `run_report_json`, and `run_report_markdown`.

The smoke used reduced validation parameters: one KG seed (`5`), reduced epochs, and reduced
randomization. It is a service/scientific-runner smoke test, not a formal five-seed reproduction and
not the source of the packaged static demo.

## Level 3 — not executed

The audited resource bytes were available, and the machine exposed an NVIDIA RTX 4060 Ti with
16,380 MiB VRAM, approximately 25.2 GB Docker memory, and approximately 172 GB free disk. However,
the project does not specify a reviewed minimum resource envelope for the formal five-seed Enhanced
run. Hardware sufficiency for that full run therefore remains `needs_review`; no formal Core or
Enhanced scientific reproduction was attempted.

The static demo result is excluded from scientific rerun claims.

## Complete resource-bundle preparation — locally verified

This section records the subsequent full-bundle delivery change. It supersedes the earlier
GPS-only/manual acquisition method but does not rewrite the historical test record above.

- Complete example inputs: **10 source files copied byte-for-byte** into
  `examples/full_inputs/`; the two largest files are 38.81 MiB each.
- Complete-input structure: **22,966 compounds** with unique/non-empty IDs and non-empty SMILES;
  **117 disease-gene rows** with non-empty symbol/Entrez fields. Both expression metadata files
  contain control and disease groups, and all 61/95 selected samples occur in their paired
  216-column TPM headers, matching the implemented subset contract.
- Git candidate audit: **205 files**, approximately **81.61 MiB** total, largest ordinary file
  **38.81 MiB**, and **no ordinary file at or above 100 MiB**.
- Candidate filename scan: **no** database, runtime state, upload, log, cache, private-key file,
  real `.env`, or local-resource payload.
- Candidate content scan: **no** recognized token shape, private-key block, Windows user-profile
  path, local project absolute path, or email address detected. Scans report locations/types only,
  never candidate secret values.
- Full resource archive: deterministically built from the 18 manifest-declared files; unused local
  `kg.csv` excluded.
- Archive metadata: **44,452,328 bytes**, SHA-256
  `646ae28bc4bd62f2d67abf7b193017a5dbe2b70648f4dedffaee9d2d8d85996a`;
  extracted payload **318,937,454 bytes**.
- Determinism check: rebuilding the archive produced the same size and SHA-256.
- Empty-root installation test: the downloader consumed the exact local archive through a
  `file://` URL, validated the archive, safely staged extraction, and verified **18/18 files**.
  The checker reported both Core and Enhanced **READY**.
- Downloader safety tests cover archive hash mismatch and unexpected/path-escaping members.
- Full test suite: **229 passed, 10 skipped, 1 warning** on the host Python environment. The skips
  retain the optional-dependency and Windows-symlink boundaries recorded above.
- Python compilation: **passed** for `src`, `scripts`, and `tests`.
- Ruff: **passed**.
- `docker compose config --quiet`: **passed**.
- Static demo remains **81 lines** with SHA-256
  `098f96829d5de39ee5a30a0615c10139c0b437cd4ecb0466564473db8e58485d`.

The archive is currently present only in ignored local `.release-staging/`. Upload to GitHub
Releases, anonymous download from GitHub, a new GHCR push, and clean-room clone-from-remote
validation are **not yet run** and must not be reported as passed.
