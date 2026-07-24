# smi2phen v0.1.0

Initial reproducible release of **smi2phen: An Agent-assisted Multi-perspective Drug Screening
Workflow**.

## Included

- FastAPI Web/API, Redis queue, and unified Worker deployment;
- fixed, auditable Workflow/DAG implementation;
- NetInfer, network proximity, optional GPS, KG learning, and consensus-ranking modules;
- complete example input set plus lightweight validation inputs;
- static output-format example;
- deterministic complete scientific-resource bundle;
- resource download, archive verification, safe extraction, and per-file checksum validation;
- unit/API tests and reproducibility documentation.

## Resource asset

Download `smi2phen-resources-v0.1.0.tar.gz` through:

```bash
python scripts/download_resources.py
python scripts/check_resources.py --mode enhanced
```

Expected archive metadata:

```text
size: 44452328 bytes
sha256: 646ae28bc4bd62f2d67abf7b193017a5dbe2b70648f4dedffaee9d2d8d85996a
files after extraction: 18
uncompressed payload: 318937454 bytes
```

The unused local `kg.csv` is not included.

## Scientific boundary

Candidate ordering represents research priority only. It does not establish efficacy, toxicity,
safety, mechanism, or clinical effectiveness and is not medical advice. The static example result
was not regenerated as part of packaging. All outputs require further computational or
experimental validation.
