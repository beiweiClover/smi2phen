# Reproducibility

## Included

- executable Python source;
- frozen workflow and component configuration;
- input, disease, node, and artifact contracts;
- the Compose definition and unified Dockerfile;
- the unified dependency lock file;
- unit/API tests and lightweight fixtures;
- minimal example inputs and one static output-format example;
- a machine-readable resource manifest, downloader, and checker.

## Intentionally excluded

- historical SQLite databases, sessions, messages, uploads, and run workspaces;
- Redis state and local Docker volumes;
- caches, bytecode, machine logs, and formal-run console logs;
- machine-specific provenance containing personal absolute paths;
- large GPS, NetInfer, PPI, and KG resources;
- run-trained checkpoints and intermediate/final artifacts;
- legacy comparison reports and large copied run reports.

## Reproduction layers

1. **Code validation**: unit/API tests, Python compilation, Ruff, and Compose parsing.
2. **Service validation**: build/start Redis, API, and Worker; verify health, empty state, queue
   consumption, and a minimal submitted task.
3. **Scientific reproduction**: verify every required resource, compatible GPU/runtime, adequate
   memory/disk, and any safely supplied external-service configuration before running the complete
   Workflow.

Passing a lower layer does not imply passing a higher layer.

## State and provenance

Each run receives its own workspace. The Workflow records input sizes/hashes, configuration,
decisions, status transitions, artifacts, and resource hashes. API and Workflow records share one
SQLite database. A clean `.runtime/state` directory starts with no historical user tasks.

Resource metadata in `resources/manifest.json` records audited filenames, relative paths, sizes,
SHA-256 values, sources, versions, licenses, and redistribution decisions. Nine GPS files use
pinned official URLs after byte-level verification against a named upstream commit. The NCBI,
NetInfer, PPI, and transformed KG entries remain manual-only and `needs_review` where exact
revision, derivation, or redistribution rights are unresolved. The manifest is not an acquisition
license or a complete data citation.

## Determinism boundary

Configured random seeds constrain stochastic stages, including KG seeds `5` through `9`, but exact
reruns also depend on resource bytes, dependency/runtime versions, hardware behavior, user inputs,
and the complete configuration. The system records these inputs where implemented; it does not
guarantee bitwise identity across arbitrary platforms.

## Static demo boundary

The file under `examples/demo_result/` is a static example copied byte-for-byte from an existing
result. It was not regenerated during submission preparation and is not evidence that the current
container completed a scientific run.
