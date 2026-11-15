# -----------------------------
# Stage 0: Base (shared)
# -----------------------------
ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Multi-arch compatible uv binary
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/


# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM base AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prime cache with metadata + lock. uv.lock is REQUIRED (no glob): a missing
# lock must fail the build, and the image never re-resolves. An in-image
# `uv lock` is how pyjwt 2.15.1 shipped while the lock said 2.14.0 and broke
# auth estate-wide (gitops#4248, license-authz#63).
COPY README.md pyproject.toml uv.lock ./

# Install dependencies exactly as locked into a local .venv. The project itself
# is never installed: the runtime imports it via PYTHONPATH=/app/src.
RUN uv sync --frozen --no-dev --no-install-project

# App source after deps for better cache reuse
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY seed/ /app/seed/
COPY migrations/ /app/migrations/

# Build-time guard: fail the build if any installed version != uv.lock
# (pyjwt asserted explicitly). Same script as media-discovery#64 / license-authz#63.
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/lock-pins.txt && \
    /app/.venv/bin/python /app/scripts/assert_lock_versions.py uv.lock /tmp/lock-pins.txt && \
    uv sync --frozen --no-dev --no-install-project --check


# -----------------------------
# Stage 2: Production
# -----------------------------
FROM base AS production
ENV PATH="/app/.venv/bin:$PATH" \
PYTHONPATH=/app/src \
RUNNING_IN_DOCKER=true

# Runtime deps: curl + docker CLI (use distro docker.io to avoid external repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
curl \
ca-certificates \
docker.io \
&& rm -rf /var/lib/apt/lists/*

# No internal package index or CA is configured: every dependency resolves from
# PyPI and the image trusts the standard CA bundle installed above.

# Create appuser BEFORE the copies so each COPY can set ownership inline via
# --chown. A trailing `chown -R appuser:appuser /app` instead rewrites every
# inode in the tree AND mints a second, duplicate image layer that is freshly
# non-dedupable on every build -- so it costs twice: once in build time, again
# in push bytes. Measured here 2026-09-05 on a cache-cold build: that single
# RUN was 1115.3s of a 1214s "Build and push" step (92%), while pushing layers
# was 9.7s. The same pattern blew the 30 min cap on coding-security (reported
# as `cancelled`, not `failure`); the fix below copies that repo's proven shape.
# /app itself is chowned NON-recursively so the directory keeps the ownership
# the old recursive chown gave it.
#
# `usermod -aG docker` needs the `docker` group, which the docker.io install
# above creates -- so this RUN must stay below that apt layer and above the
# COPYs. Do not reorder it further.
RUN useradd --create-home --shell /bin/bash appuser && \
    usermod -aG docker appuser && \
    mkdir -p /app && chown appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/
# The seed YAML is DATA for scripts/seed_w2v_workers.py, whose default --seed path is
# ../seed/w2v_workers.yaml. Without it the script ships into the image unusable, and
# applying the seed means copying the YAML into a running pod by hand — which is why
# #113/#114/#115 all merged without ever reaching the live DB.
COPY --chown=appuser:appuser seed/ /app/seed/
COPY --chown=appuser:appuser migrations/ /app/migrations/

# Non-root execution (no UID specified - K8s runAsUser overrides at runtime).
# appuser is created above the COPYs; every COPY sets ownership inline, so there
# is deliberately NO recursive chown here. Re-adding one re-creates the 1115s
# layer this change removed.
USER appuser

EXPOSE 8887
ENV PORT=8887

# The DEDICATED Prometheus scrape port (src/settings.py: metrics_port). Separate
# from the API port on purpose: the scrape relax in gitops (port-level
# PeerAuthentication PERMISSIVE + the allow-prometheus-scrape NetworkPolicy) is
# PORT-scoped, so relaxing 8887 would expose the worker-creating API to the whole
# pod CIDR. See src/telemetry/metrics_server.py.
EXPOSE 9887

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.worker_management_api:app", "--host", "0.0.0.0", "--port", "8887"]
