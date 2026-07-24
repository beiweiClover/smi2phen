# API and Web application

## Runtime relationship

The API creates sessions, validates/uploads inputs, previews plans, requires explicit execution
confirmation, reports state, and exposes final artifacts. It does not execute scientific nodes in
the API process. Ready nodes are enqueued in Redis and claimed by `unified-worker`; API and Worker
share the same SQLite state file and run workspace mounts.

The Web application is served by `GET /` and calls the same API. It lists persisted sessions through
`GET /sessions`; therefore it requires a writable runtime database, but it does not require or ship
with a historical database. A clean start initializes empty state.

Model-assisted chat is optional. A key can be supplied through `DEEPSEEK_API_KEY` or the
`X-Model-API-Key` request header. Request-scoped header keys are used to construct a request-scoped
model client and are not intended to be persisted. Direct run endpoints do not require model chat.

## Main endpoints

- `GET /healthz`: API health and execution boundary.
- `GET /input-specs`: accepted upload specifications and example links.
- `GET /examples/{example_id}`: lightweight input examples used by the Web UI.
- `POST /sessions`, `GET /sessions`, `GET /sessions/{thread_id}`: session lifecycle.
- `POST /sessions/{thread_id}/runs`: create one run for a session.
- `POST /runs/{run_id}/files/{kind}`: register a single input.
- `POST /runs/{run_id}/expression-pairs/{pair_id}`: register paired TPM/metadata.
- `GET /runs/{run_id}/plan`: validate readiness and preview the plan.
- `POST /runs/{run_id}/start`: start only with `{"confirmed": true}`.
- `GET /runs/{run_id}`: status and node snapshot.
- `GET /runs/{run_id}/results`: final result metadata.
- `GET /runs/{run_id}/artifacts/{artifact_id}`: artifact download.
- `POST /runs/{run_id}/cancel`: request cancellation.
- `POST /sessions/{thread_id}/chat` and `/chat/stream`: optional Agent interaction.

FastAPI exposes its generated OpenAPI document at `/openapi.json`.

## Minimal direct API sequence

1. Create a session.
2. Create its run with a disease name and lower-snake-case slug.
3. Upload compounds and disease genes, plus any optional target or expression inputs.
4. Preview the plan and resolve reported missing inputs/resources.
5. Start with explicit confirmation.
6. Poll run state until terminal, then inspect results/artifacts.

`scripts/validate_unified.py` automates this sequence and confirms plan branching, but it starts real
Workflow work and should only be used in a prepared service environment.

## Static demo

The existing Web application has no independent low-cost static-result route. The packaged static
demo was therefore not connected to the UI; doing so would have required adding new data/database
or task behavior outside this submission's scope.
