# Worker Management

Multi-tenant API for creating, configuring, and orchestrating LLM-powered workers and DAG workflow graphs. Port 8887.

## What Worker Management Does

When a tenant sends a request to create a worker, the service registers it in PostgreSQL with a name, agent type, model provider, system prompt, and optional output schema. Workers can be in-process PydanticAI agents backed by any of six model providers (OpenAI, DeepSeek, Groq, Cohere, MLX, inference-router), or they can be container-based agents like Claude Code and Codex that run as K8s Jobs or inside remote code-graph-helper containers. Once created, tenants send messages to workers and receive structured responses, with each interaction recorded as a state transition in the database.

Beyond individual workers, the service manages directed acyclic graphs that wire workers together into multi-step workflows. Each graph contains typed nodes (worker, function, or condition) connected by edges with optional condition expressions. When a tenant triggers execution, the graph execution engine topologically sorts the nodes, runs them sequentially through PydanticAI, and persists intermediate state at each step. Execution can happen end-to-end in a single call or step-by-step for interactive debugging, with each session tracked by ID so tenants can poll for status.

All data is tenant-isolated via license token authentication. The authentication middleware introspects bearer tokens against the licenses service and caches results for five minutes. Trusted-network requests from RFC 1918 ranges are served without a token, and they do **not** merely get the default tenant: a caller inside `TRUSTED_CIDRS` may **name any tenant** in a bare `X-Tenant-Id` header, which is accepted on syntax alone (any canonical non-nil UUID) and becomes the row-level scope for the request — `DEFAULT_TENANT_ID` applies only when the header is absent (`src/api/auth_middleware.py:649-712`). Every such request is counted (`worker_management_unauthenticated_tenant_assertion_total`) and logged, and `WORKER_MANAGEMENT_REQUIRE_AUTHENTICATED_TENANT` turns it into a 401 — deliberately **off** by default, because the live callers hold no token yet. See the per-route analysis in `src/api/tenant_config/_router.py`. Every authenticated request whose `X-Project-Id` is absent or not a canonical non-nil UUID is counted on `worker_management_project_id_missing_total{caller,reason}` (`caller` = the verified `X-Caller-Authorization` azp, else the name `WORKER_MANAGEMENT_PROJECT_ID_CALLER_LICENCES` (`<license_id>=<name>,…`) gives the introspected licence, else `licence:<license_id>`, else `trusted_network`; never `X-App-Id`), and `REQUIRE_PROJECT_ID_HEADER` (alias `WORKER_MANAGEMENT_REQUIRE_PROJECT_ID_HEADER`) turns it into a 400 `PROJECT_ID_REQUIRED` — also **off** by default, so the counter can show which callers still omit it before anything is refused. Workers support structured output enforcement through JSON Schema validation and custom Python validator functions, so downstream consumers can rely on predictable response shapes.

The service is stateful, backed by PostgreSQL via a psycopg 3 (`psycopg_pool.AsyncConnectionPool`) connection pool with configurable min/max sizes and SSL support. Worker instances are cached in memory with MCP lifecycle management and per-worker locking to avoid redundant agent creation. OpenTelemetry tracing is optional, with OTLP export plus SQLAlchemy and httpx instrumentation for full observability when enabled.

## Features

- **Five agent runtimes** — supports pydantic_ai (in-process), claude_code and codex (K8s Jobs), remote_claude and remote_codex (code-graph-helper containers) so tenants can choose the right execution model for their workload
- **Six model providers** — routes pydantic_ai agents through OpenAI, DeepSeek, Groq, Cohere, MLX, or the inference router so tenants are not locked into a single LLM vendor
- **DAG workflow graphs** — connects workers into directed acyclic graphs with worker, function, and condition node types so multi-step pipelines execute in topological order
- **Stepwise execution** — runs graphs one node at a time with session tracking so tenants can inspect intermediate state and debug complex workflows
- **Structured output validation** — enforces JSON Schema on worker responses and supports custom Python validator functions so downstream consumers get predictable data shapes
- **Worker state history** — records every state transition (created, processing, completed, error) with timestamps and metadata so tenants have a full audit trail
- **Bulk state operations** — accepts batched state submissions (up to 100 per request) so high-throughput pipelines avoid per-state HTTP overhead
- **License token authentication** — introspects tokens against the licenses service with 5-minute caching so every request is tenant-isolated without per-call latency
- **Trusted network bypass** — automatically assigns the default tenant for requests from private CIDRs so in-cluster services skip token exchange
- **OpenTelemetry tracing** — optional OTLP export with FastAPI, SQLAlchemy, and httpx instrumentation so operators get end-to-end request visibility when Grafana is available

## Authentication

Auth enabled by default (`ENABLE_AUTH=true`). The `AuthenticationMiddleware` extracts a bearer token from `Authorization` or `X-Tenant-Id` headers and introspects it against the licenses service. Results are cached for 5 minutes.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | Introspection via `/license/introspect` on the licenses service |
| 2 | Trusted network | Bypass for private CIDRs (127.0.0.1/32, 192.168.0.0/16, 172.16.0.0/12, 10.0.0.0/8) |

Health endpoints (`/`, `/health`, `/healthz`, `/ready`, `/readyz`, `/docs`, `/openapi.json`, `/redoc`, `/metrics`) are always exempt.

## Quick Start

```bash
uv sync
uv run uvicorn src.api.worker_management_api:app --host 0.0.0.0 --port 8887
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Workers `/workers`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/workers` | Create a worker (any agent type) |
| POST | `/workers/pydantic-ai` | Create a PydanticAI agent |
| POST | `/workers/claude-code` | Create a Claude Code agent |
| POST | `/workers/codex` | Create a Codex agent |
| POST | `/workers/remote-claude` | Create a remote Claude agent |
| POST | `/workers/remote-codex` | Create a remote Codex agent |
| GET | `/workers` | List all worker names for tenant |
| GET | `/workers/types` | List available agent types |
| GET | `/providers` | List all model providers worker_management can dispatch to |
| GET | `/providers/{name}/models` | Live-query the upstream backend for the provider's current model list |
| POST | `/chat/completions` | Stateless chat completions (OpenAI-style); `stream=true` returns SSE |
| GET | `/workers/states` | List all worker states across workers |
| DELETE | `/workers/states` | Delete all worker states for tenant |
| GET | `/workers/{worker_name}` | Get worker config and status |
| PUT | `/workers/{worker_name}` | Update worker configuration |
| DELETE | `/workers/{worker_name}` | Delete a worker |
| POST | `/workers/{worker_name}/message` | Send a message to an agent worker |
| GET | `/workers/{worker_name}/states` | Get worker state history (paginated) |
| GET | `/workers/{worker_name}/state/latest` | Get latest worker state |
| DELETE | `/workers/{worker_name}/states` | Delete state history for a worker |
| DELETE | `/workers/{worker_name}/instance` | Evict cached worker instance |
| DELETE | `/workers/` | Delete all workers for tenant |

### Worker States `/workers_states`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/workers_states` | List worker states (paginated) |
| POST | `/workers_states` | Bulk save worker states (up to 100) |

### Graphs `/graphs`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/graphs/` | Create a graph |
| GET | `/graphs/` | List graphs (paginated, optional details) |
| DELETE | `/graphs/` | Delete all graphs and states for tenant |
| DELETE | `/graphs/states/all` | Delete all graph states (graphs kept) |
| GET | `/graphs/{graph_id}` | Get graph by UUID |
| PATCH | `/graphs/{graph_id}` | Update graph (incremental) |
| DELETE | `/graphs/{graph_id}` | Delete graph and its states |
| GET | `/graphs/{graph_id}/structure` | Get simplified structure for visualization |
| GET | `/graphs/{graph_id}/states` | Get graph execution state history |
| DELETE | `/graphs/{graph_id}/states` | Delete state history for a graph |
| GET | `/graphs/{graph_id}/validate` | Validate graph structure and templates |

### Graph Execution `/execution`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/execution/graphs/{graph_id}/execute` | Execute a graph end-to-end |
| POST | `/execution/graphs/{graph_id}/execute/stepwise` | Execute a graph step-by-step |
| GET | `/execution/sessions/{session_id}/status` | Get execution session status |
| GET | `/execution/health` | Execution engine health check |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks DB) |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8887` | Server port |
| `DATABASE_URL` | -- | PostgreSQL DSN, e.g. `postgresql://user:pass@host:5432/worker_management`. Only host/port/database/user/password are read from it; the pool is always psycopg 3. A legacy `postgresql+asyncpg://` DSN still parses (SQLAlchemy `make_url`) but the dialect suffix is inert — prefer the plain `postgresql://` form. |
| `POSTGRES_POOL_MIN` | `10` | Connection pool minimum size |
| `POSTGRES_POOL_MAX` | `20` | Connection pool maximum size |
| `POSTGRES_COMMAND_TIMEOUT` | `30` | SQL command timeout (seconds) |
| `POSTGRES_CONNECT_TIMEOUT` | `10` | Connection timeout (seconds) |
| `POSTGRES_SSLMODE` | `require` | SSL mode (disable, allow, prefer, require) |
| `ENABLE_AUTH` | `true` | Toggle authentication middleware |
| `DEFAULT_TENANT_ID` | `n8n-dev` | Tenant ID for trusted-network requests |
| `LICENSE_INTROSPECT_URL` | `http://licenses.licenses.svc.cluster.local:8544/license/introspect` | License validation endpoint |
| `TRUSTED_CIDRS` | `127.0.0.1/32,...,10.0.0.0/8` | CIDRs that skip token auth |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8921,...` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `DEEPSEEK_API_KEY` | -- | DeepSeek API key |
| `GROQ_API_KEY` | -- | Groq API key |
| `OPENAI_API_BASE` | -- | Base URL for MLX/OpenAI-compatible endpoints |
| `INFERENCE_ROUTER_URL` | -- | Inference router base URL |
| `SECRETS_URL` | -- | Secrets service URL |
| `CODING_SERVICE_URL` | -- | Code-graph-helper URL (for remote agents) |
| `OTLP_TRACES_ENDPOINT` | -- | OpenTelemetry OTLP endpoint |
| `OTEL_SDK_DISABLED` | `false` | Disable OpenTelemetry entirely |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL for rate limiting |
| `ENABLE_RATE_LIMITING` | `true` | Toggle rate limiting middleware |

## Architecture

```
src/
├── api/
│   ├── worker_management_api.py   # FastAPI app, lifespan, health endpoints, middleware
│   ├── app_config.py              # Alternate secure app factory with rate limiting
│   ├── router.py                  # Main API router aggregating sub-routers
│   ├── workers.py                 # Workers CRUD + message endpoints
│   ├── worker_states.py           # Bulk worker state endpoints
│   ├── graph_management_api.py    # Graph CRUD + validation endpoints
│   ├── graph_execution_api.py     # Graph execution + session status endpoints
│   ├── auth_middleware.py         # License token authentication middleware
│   ├── cors_middleware.py         # CORS configuration
│   ├── security_middleware.py     # Security headers + request validation
│   ├── security_validation.py     # Query parameter sanitization
│   ├── rate_limiting.py           # Redis-backed rate limiting middleware
│   ├── error_handlers.py         # Global exception handlers
│   ├── common_responses.py        # Shared response schemas
│   ├── run_api.py                 # Uvicorn runner
│   └── db_init.py                 # Database initialization helper
├── auth/
│   ├── dependencies.py            # FastAPI auth dependencies
│   └── tenant.py                  # Tenant context extraction
├── db/
│   ├── db_config.py               # psycopg 3 pool init, SSL, asyncpg-style compat wrapper
│   ├── worker_registry_db.py      # Worker CRUD queries
│   ├── workers_state_db.py        # Worker state queries
│   ├── graph_execution_db.py      # Graph execution state queries
│   └── graph_state_db.py          # Graph state queries
├── models/
│   └── agent_registry.py          # Agent registry data model
├── services/
│   ├── worker_state_manager.py    # Worker state business logic
│   ├── mlx_model_registry.py      # MLX model discovery and caching
│   ├── inference_router_registry.py # Inference router model discovery
│   └── agent_host_registry.py     # Agent host tracking
├── worker/
│   ├── worker_factory.py          # Agent instance creation by type
│   ├── worker_manager.py          # Worker instance cache + MCP lifecycle
│   ├── worker_template.py         # Worker configuration templates
│   ├── mcp_utils.py               # MCP server management utilities
│   ├── docker_mcp_manager.py      # Docker-based MCP server management
│   ├── service_authenticator.py   # Service-to-service auth helper
│   ├── agents/
│   │   ├── base_agent.py          # Abstract base agent class
│   │   ├── agent_registry.py      # Pluggable agent type registry
│   │   ├── pydantic_ai_agent.py   # PydanticAI agent implementation
│   │   ├── pydantic_ai_agent_wrapper.py # PydanticAI wrapper with tools
│   │   ├── claude_code_agent.py   # Claude Code K8s Job agent
│   │   ├── codex_agent.py         # Codex K8s Job agent
│   │   ├── remote_claude_agent.py # Remote Claude via code-graph-helper
│   │   ├── remote_codex_agent.py  # Remote Codex via code-graph-helper
│   │   ├── remote_container_executor.py # Shared remote container executor
│   │   ├── schemas.py            # Agent request/response schemas
│   │   ├── tool_conversion.py     # MCP tool-to-PydanticAI conversion
│   │   └── tool_template.py       # Tool configuration templates
│   └── graphs/
│       ├── graph_execution_service.py   # Topological sort + execution engine
│       ├── pydantic_graph_execution_service.py # Pydantic-graph based execution
│       ├── pydantic_graph_state.py      # Pydantic-graph state management
│       ├── graph_factory.py             # Graph instance creation
│       ├── graph_models.py              # Graph data models
│       ├── graph_monitoring.py          # Execution monitoring
│       ├── graph_operations.py          # Graph CRUD operations
│       ├── execution_engines.py         # Execution engine abstraction
│       ├── node_factory.py              # Node instance creation
│       ├── persistence/
│       │   └── database_persistence.py  # Graph state DB persistence
│       └── nodes/
│           ├── worker_node.py           # Worker execution node
│           ├── function_node.py         # Function execution node
│           ├── condition_node.py        # Conditional branching node
│           └── workflow_executor_node.py # Workflow executor node
├── telemetry/
│   └── agent_metrics.py           # Agent execution metrics
├── type_definitions/
│   └── constraint_types.py        # Type constraint definitions
├── utils/
│   └── skill_mapper.py            # Skill mapping utilities
├── k8s_job_executor.py            # Kubernetes Job creation and monitoring
└── logging_config.py              # Structured logging configuration
```

## Agent Types

| Type | Runtime | Description |
|------|---------|-------------|
| `pydantic_ai` | In-process | PydanticAI agent with configurable model provider and tool support |
| `claude_code` | K8s Job container | Claude Code CLI agent running in an isolated container |
| `codex` | K8s Job container | OpenAI Codex CLI agent running in an isolated container |
| `remote_claude` | Remote container | Claude agent executed in an existing code-graph-helper container |
| `remote_codex` | Remote container | Codex agent executed in an existing code-graph-helper container |

## Model Providers

For `pydantic_ai` agents, six model providers are available:

| Provider | Env Var | Description |
|----------|---------|-------------|
| `openai` | `OPENAI_API_KEY` | OpenAI models (GPT-4, etc.) |
| `deepseek` | `DEEPSEEK_API_KEY` | DeepSeek models |
| `groq` | `GROQ_API_KEY` | Groq models |
| `cohere` | -- | Cohere models |
| `mlx` | `OPENAI_API_BASE` | Local MLX-compatible inference via OpenAI API format |
| `inference_router` | `INFERENCE_ROUTER_URL` | Routes to the best available backend (self-hosted `local-llm` vLLM, DeepSeek, Claude) |

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Coverage threshold: 79.5% (enforced in pyproject.toml).

## CI/CD

GitHub Actions workflow (`.github/workflows/ci-public.yml`): `ruff + unit tests + the no-private-refs guard --> build amd64 --> optional smoke test --> publish`.

On `main` the image is published to `ghcr.io/<owner>/<repo>` with `sha-<commit>`, `main-<run>` and `latest` tags. Publishing is opt-in: set the repository variable `PUBLISH_IMAGE=true`. Supply a smoke command through the `SMOKE_CMD` repository variable. Deploying the published image is left to you.

## Docker

```bash
docker build -t worker-management .
docker run -p 8887:8887 worker-management
docker run -p 8887:8887 -e LOG_LEVEL=DEBUG -e DATABASE_URL=postgresql://user:pass@host/db -e ENABLE_AUTH=false worker-management
```

## Dependencies

- **licenses service** (port 8544) — validates license tokens via introspection
- **inference-router** (port 8150) — routes pydantic_ai agents to best available model backend
- **code-graph-helper** — hosts remote Claude and Codex agent containers
- **PostgreSQL 18** — persistent storage via a psycopg 3 async connection pool (through PgBouncer in K8s)
- **Redis** — distributed rate limiting backend
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2** — request/response validation and settings
- **PydanticAI** — agent framework for in-process LLM agents
- **pydantic-graph** — graph execution engine
- **psycopg 3** (`psycopg[binary,pool]`) — the async PostgreSQL driver. This is the only DB driver; asyncpg is **not** a dependency. `src/db/db_config.py` wraps the psycopg `AsyncConnection` in a thin asyncpg-style facade (`fetch` / `fetchrow` / `fetchval` / `execute`, `$1` positional placeholders) so the repository layer did not have to be rewritten during the migration — the `asyncpg` names you see in `src/` refer to that compatibility shim, not to the driver.
- **httpx** — async HTTP client
- **SQLAlchemy** — database URL parsing and instrumentation
- **OpenTelemetry** — distributed tracing with OTLP export
- **kubernetes** — K8s Job creation for container-based agents
- **MCP SDK** — Model Context Protocol tool management
- **structlog** — structured logging
- **Cohere SDK** — Cohere model provider
- **Groq SDK** — Groq model provider
- **OpenAI SDK** — OpenAI and MLX model provider

## Known Issues

- **Python version mismatch in README vs pyproject.toml** — pyproject.toml requires `>=3.14` but CI runs Python 3.14. No actual issue, just noting the old README said 3.12.
- **Schemathesis contract tests disabled** — Commented out in CI workflow ("TEMPORARILY DISABLED - schemathesis blocking deploys"). Contract test infrastructure exists in `tests/contract/` but is not running.
- **Rate limiting disabled in main app** — `RateLimitMiddleware` is commented out in `worker_management_api.py` ("COMPLETELY DISABLED for schemathesis testing") though `app_config.py` still configures it.

## Licensing

The code in this repository is Apache-2.0 and needs no licence key to build, test or run.

Licence-token validation is **opt-in by configuration**: `LICENSE_INTROSPECT_URL` defaults to
empty, and while it is unset no licence request is ever made. It is one of the settings the
production validator requires — a deployment that starts with `ENVIRONMENT=production` and no
introspection URL fails at startup rather than serving unlicensed traffic. Pointed at a licences
service, tokens are introspected per request and the result is cached.

The licensing stack is open source too — [`licenses`](https://github.com/nonagenticai/licenses)
(entitlements, plans and tokens) and [`license-authz`](https://github.com/nonagenticai/license-authz)
(the Envoy `ext_authz` enforcement point). Neither is required to run this code.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
