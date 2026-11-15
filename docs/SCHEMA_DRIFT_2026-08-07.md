# Schema drift: `postgres_schema.sql` vs live production

**Measured 2026-08-07** against the running production database `worker_management`
(Postgres 18.4). All queries were **read-only**; nothing was written.

This document records what was measured. It **does not** fix anything, and no
DDL was changed. It exists so the next person does not have to re-derive it,
and so the sequencing below is a decision someone takes deliberately rather
than a cleanup someone does by accident.

> **Bottom line:** `postgres_schema.sql` is an *aspirational* document. It is
> the DDL a fresh install gets, and CI applies it to the ephemeral
> `postgres:18` sidecar — so the `pg` test suite runs against the *declared*
> schema, not against a replica of production. Production was bootstrapped by
> something else and has diverged structurally, not just by a column or two.

---

## 1. Table inventory: 15 live, 12 declared

| Table | Declared in `postgres_schema.sql` | Live in prod | Live rows |
|---|---|---|---|
| `workers` | ✅ | ✅ | 668 |
| `graphs` | ✅ | ✅ | 42 |
| `graph_execution_sessions` | ✅ | ✅ | 830 |
| `graph_states` | ✅ | ✅ | — |
| `worker_states` | ✅ | ✅ | — |
| `tools` | ✅ | ✅ | — |
| `skills` | ✅ | ✅ | — |
| `workload` | ✅ | ✅ | — |
| `execution_events` | ✅ | ✅ | — |
| `node_execution_logs` | ✅ | ✅ | — |
| `workflow_routing_pointer` | ✅ | ✅ | — |
| `schema_migrations` | ✅ | ✅ | 15 |
| `tenant_llm_config` | ❌ **not declared** | ✅ (migration 021) | 3 |
| `tenant_modality_config` | ❌ **not declared** | ✅ (migration 022) | 1 |
| `agents` | ❌ **not declared** | ✅ **orphan** | **0** |

Two of the three undeclared tables are explainable: `tenant_llm_config` and
`tenant_modality_config` are created by migrations **021** and **022** and were
never back-ported into the base schema. A fresh install from
`postgres_schema.sql` alone does not have them; it only gets them once the
migration chain runs.

`agents` is a genuine orphan:

* No `CREATE TABLE ... agents` exists anywhere in this repo — not in
  `postgres_schema.sql`, not in `migrations/`, not in `src/`.
* It has 0 rows.
* Its columns are `id, agent_id, agent_name, config, created_at, updated_at,
  **creator_id**` — i.e. it still carries the `creator_id` column that
  migration **019** ("drop creator_id") removed service-wide. 019 only touches
  the tables it names, so the orphan was never swept.

## 2. Structural drift: 12 of 15 tables have **no primary key**

This is the largest finding and it is not limited to one table.

`SELECT ... FROM pg_constraint WHERE contype='p'` over the `public` schema
returns exactly **three** primary keys in the entire production database:

```
tenant_llm_config_pkey
tenant_modality_config_pkey
workflow_routing_pointer_pkey
```

Those are precisely the three tables created by migrations (**021**, **022**,
**014**). Every table from the original bootstrap — `workers`, `graphs`,
`graph_execution_sessions`, `tools`, `skills`, `worker_states`, `graph_states`,
`workload`, `execution_events`, `node_execution_logs`, `schema_migrations`,
`agents` — has **no primary key at all**, despite `postgres_schema.sql`
declaring one for each.

There are also **zero foreign-key constraints** in the live `public` schema.
`postgres_schema.sql:153` declares

```sql
CONSTRAINT fk_graph_exec FOREIGN KEY (graph_id) REFERENCES graphs(id) ON DELETE CASCADE
```

which **does not exist in production**.

## 3. Type drift is systematic, not incidental

Timestamp columns, counted per table (live):

| Table | `timestamp` (naive) | `timestamptz` |
|---|---|---|
| `workers` | 2 | 0 |
| `graphs` | 2 | 0 |
| `graph_execution_sessions` | 3 | 0 |
| `graph_states` | 2 | 0 |
| `tools` | 2 | 0 |
| `skills` | 2 | 0 |
| `worker_states` | 1 | 0 |
| `workload` | 1 | 0 |
| `execution_events` | 1 | 0 |
| `node_execution_logs` | 3 | 0 |
| `tenant_llm_config` | 0 | **2** |
| `workflow_routing_pointer` | 0 | **2** |

`postgres_schema.sql` declares `TIMESTAMPTZ` everywhere. **Every**
bootstrap-era table is naive `timestamp without time zone`; only the
migration-created tables got `timestamptz`. Same split as the primary keys —
strong evidence the bootstrap-era tables were not created from this file.

### `graph_execution_sessions` in detail

| Column | Declared | Live |
|---|---|---|
| `id` | `BIGSERIAL PRIMARY KEY` | `integer`, **no PK constraint** |
| `session_id` | `UUID UNIQUE NOT NULL` | `text` (UNIQUE index present, from migration 020) |
| `graph_id` | `UUID NOT NULL` | `text` |
| `execution_mode` | `VARCHAR(50) NOT NULL` | `text` |
| `status` | `VARCHAR(50) NOT NULL` | `text` |
| `created_at` / `updated_at` | `TIMESTAMPTZ NOT NULL` | `timestamp without time zone NOT NULL` |
| `completed_at` | `TIMESTAMPTZ` | `timestamp without time zone` |
| `correlation_id` | `UUID` | `uuid` ✅ (migration 012 landed correctly) |
| `CONSTRAINT fk_graph_exec` | declared | **absent** |

The tenant-axis columns (`tenant_id`, `tenant_name`, `subject_id`,
`visibility`) *are* `character varying` live and match, including the
`visibility` CHECK constraint — so migrations 010/011 applied correctly.

## 4. The `schema_migrations` ledger cannot be trusted as an "applied" record

Live ledger (15 rows):

```
1.0.4   Add operational_status column to workers table
006     Convert TEXT columns to JSONB for psycopg compatibility
007     Convert worker_states.worker_state from TEXT to JSONB
0.8.0   Add description and variables columns to graphs table
0.9.0   Add inference telemetry columns to graph_execution_sessions
010     Per-user-within-tenant isolation (tenant_id, subject_id, visibility)
013     Ensure schema_migrations.version UNIQUE + backfill bookkeeping
014     Workflow routing pointer (blue/green canary) for (card_type, list)
015     Composite (tenant_id, created_at DESC) index on graph_execution_sessions
017     Backfill: delete worker rows with reserved-literal worker_name
0.18.0  Add quality_score column to graph_execution_sessions
020     Ensure single-column UNIQUE index on graph_execution_sessions.session_id
021     Per-tenant BYO LLM provider config (tenant_llm_config)
022     Per-tenant BYO modality (tts/asr/image/embedding) provider config
023     Mark stored BYO key as owner-provided (managed BYO) on both per-tenant key tables
```

Two independent problems:

**(a) Inconsistent version keys.** Four migration files record a
semver-shaped key instead of their filename ordinal:

| File | Records as |
|---|---|
| `004_add_operational_status.sql` | `1.0.4` |
| `008_add_graphs_description_variables.sql` | `0.8.0` |
| `009_add_inference_telemetry_columns.sql` | `0.9.0` |
| `018_add_quality_score_to_sessions.sql` | `0.18.0` |

So a lookup for `'004'`, `'008'`, `'009'` or `'018'` returns nothing even
though those migrations *did* apply (their columns are present live). These
rows are **not junk** — `migrations/013_fix_schema_migrations_unique.sql:47-55`
deliberately backfills them under exactly these keys, matching what the source
files themselves write. The keys are simply not comparable to the file
ordinals.

**(b) 8 of 23 migration files never write a ledger row at all:**
`002`, `002a`, `003`, `005`, `011`, `012`, `016`, `019`. That includes
**019**, the `creator_id` drop the current tenant-native code depends on. Its
effect is observable in the schema, but the ledger is silent about it.

The header of `013_fix_schema_migrations_unique.sql` records the historical
root cause: `schema_migrations.version` had no UNIQUE constraint, so every
`INSERT ... ON CONFLICT (version) DO NOTHING` errored, and the runner
(`psql -f $f || echo`) shrugged the error off — the DDL applied, the
bookkeeping did not.

**Consequence:** "is migration N applied?" cannot be answered from this table.
Only inspecting the actual schema answers it.

---

## 5. Recommended sequencing (do these in order, not in parallel)

Nothing below has been done. Each step is a deliberate decision.

1. **Regenerate the declared schema from reality.**
   `pg_dump -s` the production database into a new candidate file and diff it
   against `postgres_schema.sql`. This turns "what is the drift?" from an
   investigation into a diff. Do **not** hand-edit the current file toward
   production — the two have diverged in enough places that a hand-merge will
   miss some.

2. **Reconcile the type drift deliberately, table by table.**
   `text` vs `UUID`, naive `timestamp` vs `TIMESTAMPTZ`, and the missing
   primary keys are three separate decisions with different blast radii:
   * Adding the missing PKs to 12 tables requires proving each candidate key is
     actually unique in live data first (830 sessions, 668 workers, 42 graphs).
   * `timestamp` → `TIMESTAMPTZ` is a rewriting `ALTER` that reinterprets
     existing values against a session timezone. It needs an explicit
     `USING ... AT TIME ZONE` and a decision about what the stored naive values
     currently mean.
   * `text` → `UUID` on `session_id` / `graph_id` will fail on any row that is
     not a parseable UUID. Check before, not during.
   * Adding `fk_graph_exec` will fail on any orphaned `graph_id`. Check first.

3. **Drop or declare `agents`.** It has 0 rows and no creator in this repo, so
   dropping is the cheap answer — but confirm no other service writes to it
   before dropping. If something does, it must be declared here and given an
   owner.

4. **Decide what `schema_migrations` is for.** Either make every migration
   record itself under its filename ordinal (and backfill the 8 silent ones +
   re-key the 4 semver ones), or stop pretending the table is authoritative and
   drive migrations purely off idempotent `IF NOT EXISTS` guards. The current
   half-state is the worst of both.

5. **Only then**, consider `WORKER_MGMT_MIGRATIONS_STRICT`.
   Migrations currently run non-strict in the FastAPI lifespan
   (`src/db/db_config.py:375-380`, `:405-419`): a failure logs a warning and
   the service proceeds on a partially-migrated schema. Flipping that to strict
   *before* steps 1–4 would turn every one of the drifts above into a
   crash-loop on the next pod restart. Strict mode is the right end state; it
   is not the right first move.

---

## How to reproduce

```bash
psql -h postgres.example.com -p 5432 -U postgres -d worker_management

-- table inventory
select table_name from information_schema.tables
 where table_schema='public' and table_type='BASE TABLE' order by 1;

-- primary keys (returns 3)
select conname from pg_constraint
 where contype='p' and connamespace='public'::regnamespace order by 1;

-- foreign keys (returns 0)
select conname from pg_constraint
 where contype='f' and connamespace='public'::regnamespace order by 1;

-- timestamp drift
select table_name,
       count(*) filter (where data_type='timestamp without time zone') as naive,
       count(*) filter (where data_type='timestamp with time zone')    as tz
  from information_schema.columns
 where table_schema='public' group by 1 order by 1;

-- the ledger
select * from schema_migrations order by 1;
```

Read-only. Production is the primary — do not write to it.

---

## Addendum 2026-08-13: `worker_id` may belong in the "prod is right" class

**This addendum does not change any DDL.** It records evidence measured *after*
the 2026-08-10 diff, because that evidence bears on a conclusion the diff drew,
and the conclusion is what gates the remediation sequence.

`postgres_schema.sql` currently states that the only class where prod is right
and the file is the stale artifact is `jsonb -> text[]` (`tools.tags`,
`skills.domains`), and that rewriting anything else to match prod "would CODIFY
THE DAMAGE". That framing treats every other divergence as prod having decayed
away from an intended design.

For `worker_id` the evidence now points the other way, and it comes from the
#55 investigation (wm #171, merged 2026-08-12) rather than from a schema diff.

### What was measured (production, read-only, 2026-08-13)

| Fact | Value |
|---|---|
| `workers.worker_id` live type | `text` (file declares `UUID`) |
| `worker_states.worker_id` live type | `text` (file declares `UUID`) |
| `worker_states` rows | 857,075 |
| ...whose `worker_id` is **not** UUID-shaped | **16,224** |
| ...with **no matching** `workers` row | **16,505** |
| Foreign keys anywhere in the database | **0** |

### Why this is not simply decay

The application *deliberately* writes non-UUID values into `worker_id`. The
registry fallback path keys state by worker **name** when it has no id, which is
the mechanism behind the split-history bug #55 fixed. So:

* a `UUID` column could not physically store 16,224 of the rows production has
  written through the normal code path; and
* `fk_worker` could not be added even in principle — 16,505 rows have no parent
  row to point at, so `ADD CONSTRAINT` aborts, and once added it would reject
  writes from live workers using the name-keyed fallback.

That is the same shape as `tools.tags` / `skills.domains`: the file describes
something the running system has never been able to do.

### The decision this needs (owner call, not a drive-by edit)

Two coherent end-states, and they are genuinely different projects:

1. **Ratify `text`.** Correct the declared type for the `worker_id` columns and
   drop `fk_worker`. Cheap, and it stops CI enforcing a rule prod does not have
   — CI bootstraps from `postgres_schema.sql` via `scripts/init_db.py`, so today
   the `pg` suite validates against UUID columns and 7 foreign keys that do not
   exist in production. Wrong-direction error: it passes CI and fails in prod.
2. **Make the application stop writing names into `worker_id`.** Keep the UUID
   intent, backfill or discard the 16,505 unparented rows, then constrain. This
   is the planka-private #39 shape — measure, repair, then constrain — and it
   needs an outage window and a data decision about the history stream.

Until that is decided, `postgres_schema.sql` is left as-is on purpose. The one
thing that should NOT happen is a silent edit in either direction: this file's
own header forbids rewriting to match prod, and that instruction was written
before the evidence above existed.

### Reproduce

```sql
-- non-UUID worker ids in the history stream
select count(*) from worker_states
 where worker_id !~ '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$';

-- history rows with no parent in workers
select count(*) from worker_states s
  left join workers w on s.worker_id::text = w.worker_id::text
 where w.worker_id is null;
```
