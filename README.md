# smi2phen

**smi2phen: An Agent-assisted Multi-perspective Drug Screening Workflow**

**smi2phen：以科研工作流为核心、智能体辅助交互的多角度候选药物筛选系统**

smi2phen is a research software workflow for prioritizing candidate compounds from multiple
computational evidence streams. The Agent helps users understand requirements, supply inputs,
review a plan, explicitly confirm execution, inspect status, and navigate outputs. Scientific
computation is performed by a fixed, auditable Workflow/DAG rather than by the language model.

Candidate ordering is a research priority only. It does not establish efficacy, toxicity, safety,
mechanism, or clinical effectiveness, and it is not clinical or medication advice. Every output
requires further computational and experimental validation.

## Design

The FastAPI service handles Web/API requests and persists sessions and run state in a local SQLite
database. Ready Workflow nodes are placed on Redis queues. A separate Worker claims and executes
the registered scientific runner, records status, and writes artifacts inside an isolated run
workspace.

The Workflow controls input validation, dependencies, parameters, configured random seeds,
resource references and hashes, execution state, caching, and artifact manifests. The implemented
scientific branches are:

- NetInfer drug–target inference, or a validated provided-target boundary;
- degree-matched network proximity on a configured PPI interactome;
- optional GPS expression-reversal evidence;
- knowledge-graph construction, pretraining, and seeded fine-tuning;
- multi-evidence consensus ranking.

The checked-in formal KG configuration uses seeds `5, 6, 7, 8, 9`. Consensus ranking is an evidence
aggregation step, not an independent scientific model.

## Core and Enhanced modes

`core` combines KG and network-proximity evidence. It requires a compound library and disease-gene
input, plus the configured mapping, NetInfer, PPI, and KG resources. Uploading validated drug
targets and a target mapping selects the supported provided-target boundary and skips NetInfer.

`enhanced` adds GPS expression-reversal evidence when a valid TPM/metadata pair is supplied.
Without expression data, the planner skips GPS and uses the Core evidence set. The configured
species is human; successful execution for arbitrary diseases is not guaranteed.

## Inputs and outputs

Required inputs:

- compound CSV with unique `ID` and non-empty `SMILES`;
- disease genes in one of the supported symbol and/or Entrez ID table forms.

Optional inputs:

- paired TPM and metadata TSV files for Enhanced mode;
- provided `drug_targets.json` and `target_mapping.tsv`;
- positive-drug and disease-link priors for KG construction.

The Workflow registers immutable input copies with size and SHA-256, then writes prepared inputs,
per-node logs/manifests, scientific artifacts, `final_candidates.tsv`, `ranking_summary.json`, and
run reports below a run-specific workspace. Detailed schemas are in [`contracts/`](contracts/).

## Docker start

Prerequisites for the supplied Compose profile are Docker Compose, an NVIDIA-compatible Docker GPU
runtime, adequate disk/memory, and the external resources described below.

The intended public-release flow is:

```bash
git clone https://github.com/beiweiClover/smi2phen.git
cd smi2phen

cp .env.example .env
# Add your own model API key to .env only if model-assisted chat is needed.

python scripts/download_resources.py
python scripts/check_resources.py

docker compose pull
docker compose up -d
```

Then open `http://127.0.0.1:8000/`. The same address serves the API and Web application.

Compose is pinned to `ghcr.io/beiweiclover/smi2phen:v0.1.0`. Until that package has been uploaded
and made public, use the verified local build path:

```bash
cp .env.example .env
python scripts/download_resources.py
python scripts/check_resources.py
docker compose config
docker compose build
docker compose up -d
curl http://127.0.0.1:8000/healthz
```

The model API key is optional for direct Web/API workflow controls and is required only for model-assisted
chat. Keep credentials in an untracked `.env` or enter them per request in the Web interface.
Never commit credentials.

Runtime sessions, uploads, SQLite state, and artifacts are written under `.runtime/` and are
excluded from version control. Stop services with:

```bash
docker compose down
```

Do not add `-v` unless you intentionally want to remove the named Redis volume.

## Web and API use

The Web application creates sessions and runs through the API; it does not read a bundled historical
database. A clean start creates an empty SQLite database. Execution requires a plan preview and an
explicit `confirmed: true` start request. See [docs/API_AND_WEB.md](docs/API_AND_WEB.md).

A command-line validation client is retained at `scripts/validate_unified.py`. It submits real
Workflow tasks and therefore is not a cheap unit test:

```bash
python scripts/validate_unified.py --provided-targets
```

Run it only after the required resources and compute environment are ready.

## Scientific resources

Scientific resources are distributed as one complete, versioned GitHub Release asset instead of
being committed to Git history or baked into the Docker image. The default command downloads the
full Enhanced snapshot containing mapping, NetInfer, PPI, KG, and GPS resources:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

The downloader verifies the archive SHA-256, safely extracts only manifest-declared files beneath
`.local-resources/`, and verifies all 18 files individually. Correct existing files are skipped.
The v0.1.0 archive is approximately 42.4 MiB to download and approximately 304 MiB after extraction.
It does not contain the unused local `kg.csv`.

The resource URL and exact archive/file hashes are in
[`resources/manifest.json`](resources/manifest.json). The release asset must first be uploaded by
the repository owner; an unavailable asset is reported as an error, never as a successful setup.
See [docs/RESOURCES.md](docs/RESOURCES.md).

## Static example result

[`examples/demo_result/final_candidates_no_toxicity.tsv`](examples/demo_result/final_candidates_no_toxicity.tsv)
is a byte-for-byte static output-format example. It was not generated by a Docker rerun during this
packaging work and is not wired into the Web UI. The accompanying
[`README`](examples/demo_result/README.md) records its scope and checksum.

Minimal synthetic/validation inputs are under [`examples/minimal_inputs/`](examples/minimal_inputs/).
The complete retained example input set is under
[`examples/full_inputs/`](examples/full_inputs/). These inputs support reproduction and adaptation,
but their inclusion does not establish that the static result was regenerated in this release.

## Reproduction boundary

This repository contains code, configuration, contracts, tests, minimal and complete example
inputs, and resource metadata. It does not contain historical databases, chat records, uploads, run
workspaces, caches, machine logs, private provenance, scientific resource payloads, trained run
artifacts, or a claim that the static demo was reproduced. The scientific resource payload is a
separately checksummed GitHub Release asset. Exact scientific reruns additionally depend on that
verified snapshot, compatible hardware/software, and user-supplied inputs.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Methodological limitations

- Drug–target inference, network proximity, expression reversal, and KG learning are computational
  evidence sources with distinct assumptions and failure modes.
- Input coverage, identifier mapping, resource version, disease-gene selection, and graph
  construction can materially change rankings.
- Core and Enhanced rankings are not interchangeable because their evidence sets differ.
- Consensus ranking aggregates available evidence and does not calibrate clinical probability.
- `no_toxicity` in the static filename means toxicity scoring was not included in that ranking; it
  does not mean toxicity filtering or safety assessment was completed.

## Local verification

```bash
python -m pytest
python -m compileall -q src scripts tests
python -m ruff check .
docker compose config
```

Recorded packaging-time validation is in [docs/VALIDATION.md](docs/VALIDATION.md). A successful unit
test or container health check is not a successful full scientific reproduction.

## Citation

Repository: <https://github.com/beiweiClover/smi2phen>. The release tag and archive DOI remain
`needs_review`. Until a tagged release exists, cite the project title, exact commit, repository URL,
and access date. Do not invent or infer a DOI.

## License and resource rights

The smi2phen source code is released under the MIT License; see [`LICENSE`](LICENSE). This license
does not override the separate licenses or redistribution restrictions of third-party scientific
resources. Consult the per-resource fields in [`resources/manifest.json`](resources/manifest.json).
The account-owner steps for the GitHub source push, complete GitHub Release resource asset, GHCR
image, and clean-room verification are documented in
[docs/PUBLISHING.md](docs/PUBLISHING.md).
