#!/usr/bin/env python3
"""Seed fr-agent-default into worker_management for every fr tenant.

Mirrors scripts/seed_w2v_workers.py: a single PUT is upsert (WM
auto-creates if the worker doesn't exist).

Plan 3 (fr/docs/superpowers/plans/2026-04-28-plan-3-agent-home.md:283).

Usage:
    WM_URL=http://worker-management:8887 \\
    DOWNSTREAM_LICENSE_TOKEN=<fr-tenant-license-token> \\
        python scripts/seed_fr_agent.py [--dry-run]

Tenant scoping:
    Tenant identity is carried by the license token in
    ``X-Tenant-Id: Bearer <token>`` (the canonical upstream-services
    pattern; see seed_w2v_workers.py and improvement_workers/register.py).
    To seed multiple tenants, run this script once per tenant with that
    tenant's license token in DOWNSTREAM_LICENSE_TOKEN. There is no
    server-side "list all tenants" call; tenant fan-out is the caller's
    responsibility (e.g. tenant-provisioning hook, GitOps Job per
    tenant, or CI matrix).

Environment variables:
    WM_URL                     Worker-management base URL.
                               (default: http://worker-management:8887)
    DOWNSTREAM_LICENSE_TOKEN   License token. Required (unless --dry-run).

Exit codes:
    0  worker seeded successfully
    1  worker failed (HTTP non-2xx, network error)
    2  configuration error (missing token)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx

DEFAULT_WM_URL = "http://worker-management:8887"

# Mirrors WORKER_CONFIG_REVISION in fr/backend/src/agent/wm-bootstrap.ts.
# Bump in lockstep with the TS constant so a code review of seeder changes
# naturally surfaces "did you also bump the TS revision?".
FR_AGENT_CONFIG_REVISION = "2026-05-07-plan3-r3"

# Mirrors fr/backend/src/agent/tool-output-schema.ts — keep in sync.
FR_AGENT_TOOL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "final_answer"},
                "text": {"type": "string", "minLength": 1},
            },
            "required": ["kind", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "tool_calls"},
                "calls": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "pattern": r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$",
                            },
                            "input": {},
                        },
                        "required": ["action", "input"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["kind", "calls"],
            "additionalProperties": False,
        },
    ],
}

# Mirrors the SYSTEM_PROMPT template literal in
# fr/backend/src/agent/wm-bootstrap.ts — keep in sync.
FR_AGENT_SYSTEM_PROMPT = """You are the fr conversation agent — the single user-facing
intelligence layer for an agent-first corporate platform.

Output contract (STRICT — your reply MUST match exactly one of these shapes):
  - Final answer:  { "kind": "final_answer", "text": "<answer>" }
  - Tool calls:    { "kind": "tool_calls",
                     "calls": [ { "action": "<slice>.<verb_object>",
                                  "input": { ... } }, ... ] }

`kind` is ONLY ever the literal string "final_answer" or "tool_calls".
Action names like "kaizen.log_candidate" are NEVER used as `kind`; they
go inside `calls[].action`. Tool action names must come from the tool
catalog you receive on input — never invent a tool name.

Workflow vs exception path:
  - Prefer the "deterministic workflow" path when an action exists for
    the user's intent. The system is intelligence; rely on it.
  - When no codified action covers the user's intent, emit a tool_calls
    response with one call to action "kaizen.log_candidate" and an input
    object such as { "intent": "<one-line summary of what the user asked
    for>" }. After the call result comes back, answer the user honestly
    about what is and isn't supported today.

Style:
  - Concise. Show the user what you did, not how you decided.
  - When you call tools, the user sees the tool-call cards inline; you
    don't need to narrate intermediate steps in prose.
  - When the tool catalog does not contain something the user asked
    for, say so directly."""

# Canonical fr-agent-default config. Byte-equivalent (modulo Python/TS
# spelling) to fr/backend/src/agent/wm-bootstrap.ts buildConfig() so the
# lazy-bootstrap path and this seeder produce the same WorkerCreateRequest.
FR_AGENT_DEFAULT_CONFIG: dict[str, Any] = {
    "worker_name": "fr-agent-default",
    "agent_type": "pydantic_ai",
    "agent_role": "fr Conversation Agent",
    "model_provider": "inference_router",
    # The one served-model-name of the pgx vLLM model; never a concrete name or tier-*.
    "model": "local-llm",
    "temperature": 0.2,
    "system_prompt": FR_AGENT_SYSTEM_PROMPT,
    "description": "fr conversation agent — Plan 3",
    "output_schema": FR_AGENT_TOOL_OUTPUT_SCHEMA,
    "config_revision": FR_AGENT_CONFIG_REVISION,
}


def put_worker(
    client: httpx.Client, base_url: str, worker: dict[str, Any], token: str
) -> tuple[bool, str]:
    """PUT a single worker (upsert). Returns (ok, status_string).

    WM's PUT /workers/{name} is an upsert: it auto-creates the worker if
    it doesn't exist for this tenant, otherwise refreshes its config.
    See worker_management/src/api/workers/crud_endpoints.py:498
    ("Auto-create worker if it doesn't exist") — single PUT covers both
    the "PUTs if exists, POSTs otherwise" cases per Plan 3.
    """
    name = worker["worker_name"]
    url = f"{base_url.rstrip('/')}/workers/{name}"
    headers = {
        "X-Tenant-Id": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = client.put(url, json=worker, headers=headers, timeout=30.0)
    except httpx.HTTPError as exc:
        return False, f"network error: {exc}"
    if 200 <= resp.status_code < 300:
        return True, str(resp.status_code)
    body = resp.text[:500]
    return False, f"{resp.status_code} {body}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed fr-agent-default for the tenant identified by DOWNSTREAM_LICENSE_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the worker config that would be PUT, but do not call WM.",
    )
    args = parser.parse_args()

    base_url = os.getenv("WM_URL", DEFAULT_WM_URL)
    token = os.getenv("DOWNSTREAM_LICENSE_TOKEN", "")

    worker = FR_AGENT_DEFAULT_CONFIG
    print(f"Worker: {worker['worker_name']}  (revision {FR_AGENT_CONFIG_REVISION})")
    print(f"  agent_type     = {worker['agent_type']}")
    print(f"  agent_role     = {worker['agent_role']}")
    print(f"  model_provider = {worker['model_provider']}")
    print(f"  model          = {worker['model']}")
    print(f"  temperature    = {worker['temperature']}")

    if args.dry_run:
        print(f"DRY-RUN would PUT {worker['worker_name']} to {base_url}")
        return 0

    if not token:
        print("ERROR: DOWNSTREAM_LICENSE_TOKEN is required (set env var)", file=sys.stderr)
        return 2

    print(f"Seeding to {base_url}")
    with httpx.Client(verify=False) as client:  # noqa: S501
        ok, status = put_worker(client, base_url, worker, token)
        marker = "OK" if ok else "FAIL"
        print(f"PUT {worker['worker_name']}: {marker} ({status})")
        if not ok:
            return 1

        # Read it back so the operator sees the *server-recorded* revision,
        # not just what we sent. Catches a class of "the field hit a stale
        # API that silently dropped it" failure modes.
        try:
            url = f"{base_url.rstrip('/')}/workers/{worker['worker_name']}"
            headers = {"X-Tenant-Id": f"Bearer {token}"}
            resp = client.get(url, headers=headers, timeout=30.0)
            if 200 <= resp.status_code < 300:
                server_rev = resp.json().get("config_revision")
                if server_rev == FR_AGENT_CONFIG_REVISION:
                    print(f"  server config_revision = {server_rev} (matches)")
                else:
                    print(
                        f"  WARNING: server config_revision = {server_rev!r}, "
                        f"expected {FR_AGENT_CONFIG_REVISION!r} — WM may be on an old image"
                    )
            else:
                print(f"  WARNING: GET-back returned {resp.status_code}; cannot verify revision")
        except httpx.HTTPError as exc:
            print(f"  WARNING: GET-back failed ({exc}); cannot verify revision")

    print(f"OK: {worker['worker_name']} seeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
