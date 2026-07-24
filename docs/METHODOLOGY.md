# Methodology

## Responsibility boundary

The Agent supports requirement interpretation, input guidance, plan explanation, explicit execution
confirmation, status lookup, and result navigation. It does not select or alter scientific
algorithms at run time. The registered Workflow/DAG executes scientific modules from frozen
configuration and records artifacts and state transitions.

The Workflow is designed to:

- validate and register immutable user inputs;
- resolve dependencies and Core/Enhanced branching;
- record parameters, configured seeds, source-code digest, and resource hashes;
- execute nodes through Redis-backed external workers;
- confine generated files to a run workspace;
- record per-node results, logs, manifests, and final artifacts.

## Evidence flow

1. Compound and disease-gene inputs are normalized. Invalid or unmapped records are reported.
2. NetInfer produces drug–target evidence for matched and novel compounds. Alternatively, a
   validated provided-target pair can cross the same target-provider boundary.
3. Network proximity scores compound targets relative to disease genes on the configured PPI
   interactome using degree-matched randomization.
4. In Enhanced mode, GPS predicts compound perturbation profiles, builds a disease expression
   signature from paired TPM/metadata inputs, and computes an expression-reversal score.
5. KG construction combines the configured base graph with run compounds, targets, disease genes,
   and optional priors. One pretraining stage is followed by seeded fine-tuning and seed aggregation.
6. Final ranking intersects the configured evidence thresholds and averages within-intersection
   rank percentiles with equal weights over the available evidence streams.

The default KG fine-tuning seeds in `configs/workflow.yaml` are `5, 6, 7, 8, 9`. The default
proximity seed is `452456`; other module seeds and all parameters are recorded in the same
configuration. The resource checker verifies the resource snapshot but does not validate its
scientific suitability or licensing.

## Core mode

Core uses KG and proximity evidence. The default target source is NetInfer. When both provided
target files are registered, the planner uses the supported provided-target branch and skips
NetInfer. GPS nodes are skipped.

## Enhanced mode

Enhanced adds GPS evidence and requires a valid TPM/metadata pair. If expression data is absent,
the planner records GPS as skipped and the effective evidence mode becomes the Core evidence set.

## Consensus interpretation

Consensus ranking is an aggregation procedure, not a standalone scientific model. A higher
priority is evidence-relative within the submitted candidate set and configured thresholds. It does
not estimate treatment effect, toxicity, safety, or clinical benefit.

## Evidence basis

This description is limited to:

- `configs/workflow.yaml`;
- `contracts/*.yaml`;
- registered runners under `src/lipid_screening_agent/runners/`;
- orchestration code under `src/lipid_screening_agent/orchestrator/`.

No new experimental result or reproduction conclusion is asserted here.
