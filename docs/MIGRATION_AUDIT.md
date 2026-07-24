# Migration audit

## Source boundary

The submission was derived from `Agentv3` by a whitelist copy. No source file was moved, deleted, or
overwritten. This document intentionally does not record local absolute paths.

## Read-only findings

- Entrypoints: FastAPI application and a separate Redis-backed Workflow Worker.
- Services: Redis, API, and unified Worker with shared state, runs, and read-only resources.
- Persistence: Web/API session records and Workflow state share a runtime SQLite file.
- Quality tools: pytest, Python compilation, Docker Compose configuration, and configured Ruff.
- Formal KG seeds: `5, 6, 7, 8, 9`.
- Source runtime material: 3 SQLite files, 169 cache files, 210 log files, approximately 1548
  runtime/upload/artifact files, and 44 locally copied resource files.
- One report contained a machine-specific project path.

Counts describe the source audit only. Runtime state was not copied. A subsequent owner-authorized
release extension added the selected complete example input set and packages the audited scientific
resources only as an ignored, separately checksummed GitHub Release asset.

## Included by whitelist

- Python source files, excluding bytecode and generated egg metadata;
- current workflow/component/disease configuration and contracts;
- the unified Dockerfile, Compose definition, and unified dependency lock;
- tests not tied to excluded formal/legacy reports, plus their lightweight fixtures;
- input examples required by the API, a consolidated minimal-input directory, and the selected
  complete example input set;
- the reusable unified validation client;
- rewritten method/reproducibility/API/resource documentation;
- one checksum-verified static result example.

## Excluded

- `.data/`, `.docker-data/`, tool caches, bytecode, SQLite files, Redis state;
- chat history, uploads, historical run-workspace inputs, caches, artifacts, logs, and checkpoints;
- local large research resources from Git history and the Docker image;
- generated egg metadata;
- historical `Report/` content and formal/legacy comparison scripts/tests;
- a provenance example containing machine-specific information;
- an orphan smoke override that referred to a script absent from the source tree.

## Minimal packaging changes

- project, Compose, image, API title, Web title, and exported archive branding changed to
  `smi2phen`;
- container work directory changed to `/opt/smi2phen`;
- Dockerfile dependency on an excluded historical report was removed;
- runtime mounts changed to clean ignored directories;
- component registry image name was aligned with Compose;
- public-facing documentation, resource metadata/checks, deterministic resource-bundle delivery,
  and ignore rules were added.

Scientific algorithms and the FastAPI–Redis–Worker–Workflow execution architecture were not
restructured.
