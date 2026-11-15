#!/usr/bin/env python3
"""Seed w2v's 5 PUT'd workers from the canonical YAML.

PUT is upsert in worker_management's API, so this script is idempotent.
Re-running it will overwrite the live workers with the YAML's contents.

Workers covered:
- brand-analysis
- analysis-site-summary
- course-curriculum-generator
- slide-bullet-extractor
- edit-plan-generator

Usage:
    WM_URL=http://worker-management.workers.svc.cluster.local:8887 \\
    DOWNSTREAM_LICENSE_TOKEN=<token> \\
        python scripts/seed_w2v_workers.py [--dry-run] [--seed PATH]

Environment variables:
    WM_URL                     Worker-management base URL.
                               (default: http://worker-management:8887)
    DOWNSTREAM_LICENSE_TOKEN   License token. Required.

Exit codes:
    0  all workers PUT successfully
    1  any worker failed (HTTP non-2xx, network error, validation error)
    2  configuration error (missing token, missing/invalid YAML)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

DEFAULT_WM_URL = "http://worker-management:8887"
DEFAULT_SEED_PATH = Path(__file__).resolve().parent.parent / "seed" / "w2v_workers.yaml"


def load_seed(path: Path) -> list[dict[str, Any]]:
    """Load the seed YAML and return the list of worker definitions."""
    if not path.exists():
        print(f"ERROR: seed file not found: {path}", file=sys.stderr)
        sys.exit(2)
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or "workers" not in spec:
        print(f"ERROR: seed file {path} has no top-level 'workers' key", file=sys.stderr)
        sys.exit(2)
    workers = spec["workers"]
    if not isinstance(workers, list) or not workers:
        print(f"ERROR: seed file {path} 'workers' must be a non-empty list", file=sys.stderr)
        sys.exit(2)
    return workers


def put_worker(
    client: httpx.Client, base_url: str, worker: dict[str, Any], token: str
) -> tuple[bool, str]:
    """PUT a single worker. Returns (ok, status_string)."""
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
    parser = argparse.ArgumentParser(description="Seed w2v workers from canonical YAML.")
    parser.add_argument(
        "--seed",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help=f"Path to seed YAML (default: {DEFAULT_SEED_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate the YAML but do not PUT to the server",
    )
    args = parser.parse_args()

    base_url = os.getenv("WM_URL", DEFAULT_WM_URL)
    token = os.getenv("DOWNSTREAM_LICENSE_TOKEN", "")

    workers = load_seed(args.seed)
    print(f"Loaded {len(workers)} workers from {args.seed}")

    if args.dry_run:
        for w in workers:
            print(f"DRY-RUN would PUT {w['worker_name']}")
        return 0

    if not token:
        print("ERROR: DOWNSTREAM_LICENSE_TOKEN is required (set env var)", file=sys.stderr)
        return 2

    print(f"Seeding to {base_url}")
    failures: list[str] = []
    with httpx.Client(verify=False) as client:  # noqa: S501
        for w in workers:
            name = w["worker_name"]
            ok, status = put_worker(client, base_url, w, token)
            marker = "OK" if ok else "FAIL"
            print(f"PUT {name}: {marker} ({status})")
            if not ok:
                failures.append(name)

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {len(workers)} workers — {', '.join(failures)}")
        return 1
    print(f"OK: all {len(workers)} workers seeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
