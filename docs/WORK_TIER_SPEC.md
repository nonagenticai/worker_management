# BYO-LLM `work_tier` — contract

Status: live (Phase 2). The router + `llm_config_resolver.py` are the executable
contract; this doc is the human-readable summary.

## What it is

A **work tier** is a per-task work-size hint — `small` | `medium` | `high` —
that selects which of a tenant's BYO LLM provider/model/key rows is used for a
given worker run. Tiers map to model *classes* (flash / reasoning / frontier)
in the settings UI, but the stored config is always a plain provider+model+key.

Phase 1 gave each tenant **one** BYO row (`role='__default__'`). Phase 2 lets a
tenant configure **per-tier** rows; an absent/unknown tier resolves exactly as
Phase 1 did, so the change is fully backward compatible.

## Data model

Table `tenant_llm_config`, PK `(tenant_id, role)` — unchanged, **no migration**.
A tier is just a `role` value:

| tier     | role            |
|----------|-----------------|
| (absent) | `__default__`   |
| small    | `tier:small`    |
| medium   | `tier:medium`   |
| high     | `tier:high`     |

## Resolution order (`resolve_llm_config`)

1. **Genuine per-worker override** (explicit `api_key_override` or a non-system
   `model_provider`) — left untouched.
2. **Tenant tier row** `tenant_llm_config[tier:<work_tier>]` — when a known tier
   is requested and the row is usable.
3. **Tenant default row** `tenant_llm_config[__default__]` — the fallback when a
   tier is requested but unconfigured, or when no tier is requested.
4. **System default** — no usable row → seeded routing, or `BYORequiredError`
   for a `byo_required` tenant.

Decrypt failure is fail-closed (treated as "no usable key"). The routing
decision is logged: `byo_llm_route tenant=… tier=… role=… provider=… model=…`.

## How a tier is emitted

`work_tier` rides on the worker `config` dict. It can be set:

- **Per message** — `MessageRequest.work_tier` (`POST /workers/{name}/message`).
  Wins over the worker default. This is the primary axis (task size varies per
  call).
- **Per worker (default)** — `WorkerCreateRequest.work_tier`, persisted in the
  worker's stored config; used when a message omits `work_tier`.

Worker instances are cached by `tenant_id:worker_name:work_tier`, so different
tiers get distinct instances (each bakes its own provider/model/key) and a tier
switch never reuses the wrong model.

## Configuring tier rows (`/tenant-config/llm`)

All CRUD routes take an optional `?tier=small|medium|high`; omitting it addresses
the `__default__` row (Phase-1 behavior, keeps the existing fr/kag UI working).

- `GET    /tenant-config/llm[?tier=]`     → masked view (key never returned; `hasKey`).
- `PUT    /tenant-config/llm[?tier=]`     → upsert (apiKey tri-state: absent=preserve, ""=clear, value=set).
- `DELETE /tenant-config/llm[?tier=]`     → remove that row.
- `GET    /tenant-config/llm/all`         → every configured row + its `role`/`tier`.
- `GET    /tenant-config/llm/catalog`     → providers + `model_classes` hints
  (`{flash, reasoning, frontier}` recommended model per provider; free-text
  providers return `{}`).

## Consumers

Any caller that POSTs `/workers/{name}/message` may add `work_tier`. Wired
(capability only, default omit ⇒ unchanged): kag, coding-ai,
requirements_management (Python clients) and the `n8n-nodes-workers` node
(`Work Tier` option on *Send Worker Message*). portal-bff does not invoke
workers and is unaffected.

## Out of scope (later session)

fr / kag-frontend multi-tier editor UI; kag GPU embedding/reranker model
selection; genuine per-agent-role rows (the `role` column already generalizes).
