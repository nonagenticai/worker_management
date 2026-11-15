# improvement_workers

Plan 5 of the workflow improvement loop: seed five WM worker rows -- one per
`card_type` -- with placeholder `system_prompt`s and structured `output_schema`s.
Plan 6 (prompt improvement loop) rewrites the prompts in-place via
`PUT /workers/{name}` -- per memory `no-prompt-registry-wm-is-the-store`,
WM is the prompt store.

These workers belong to the `__system_workflow__` internal tenant
(UUID `00000000-0000-4000-8000-000000000001`, mirrored in
`fr/backend/src/middleware/internal-tenant-only.ts`); they are NOT visible
to normal tenants in fr.

## Workers

| `worker_name`                                              | Card role                                                                  |
|-------------------------------------------------------------|----------------------------------------------------------------------------|
| `improvement_workflow_statistics_analysis`                  | Aggregate execution telemetry; route to next card.                         |
| `improvement_workflow_performance_analysis`                 | Identify bottleneck nodes; route to next card.                             |
| `improvement_workflow_redesign`                             | Emit RFC-6902 JSON Patch against a workflow + predicted deltas.            |
| `improvement_workflow_pattern_catalogue_improvement`        | Propose patterns to add / deprecate.                                       |
| `improvement_workflow_development`                          | Translate approved redesign into implementation notes.                     |

Each worker carries `temperature=0.2`, `model_provider="inference_router"`,
`model="local-llm"` (the one pgx vLLM alias -- never a concrete model name), and a JSON Schema 2020-12 `output_schema` enforcing the
`{ proposal, predicted_deltas, rationale, confidence }` envelope.

## Run

In-cluster (relies on WM's RFC1918 trusted-network bypass):

```bash
uv run python scripts/improvement_workers/register.py --dry-run
uv run python scripts/improvement_workers/register.py
uv run python scripts/improvement_workers/register.py --force   # PUT overwrites
```

Off-cluster (port-forward + license token):

```bash
WORKER_MANAGEMENT_URL=http://localhost:8887 \
WORKER_MANAGEMENT_LICENSE_TOKEN=... \
uv run python scripts/improvement_workers/register.py
```

Per memory `upstream-license-token-pattern` the script sends
`X-Tenant-Id: Bearer <token>` only -- never `Authorization`.

## Plan 6 hand-off

Plan 6 will improve each prompt by calling
`PUT /workers/improvement_workflow_<card_type>` with a new `system_prompt`
(and optionally an updated `output_schema`). No re-deploy required: WM
reflects the change on the next call.
