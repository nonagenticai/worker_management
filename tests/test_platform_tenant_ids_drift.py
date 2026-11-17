"""worker_management#339 — WM's ``PLATFORM_TENANT_IDS`` must equal fr's.

fr and WM each hold the platform tenant set (coordinator ruling: do NOT
unify). WM#336 depends on them agreeing: fr says a card is platform-owned, then
WM re-checks the tenant against its OWN set, so a drift 422s legitimate
releases or admits a stale id.

⛔ WHAT CHANGED 2026-11-16, AND WHY IT IS STRONGER. Both sides used to carry the
set as a source literal, and this test compared the two literals. That put real
production tenant uuids in a PUBLIC repository, so both sides moved to
configuration: WM reads ``PLATFORM_TENANT_IDS`` from its Deployment env, fr reads
its own from its runtime config. The comparison therefore changed from "two
literals agree" to "the two DEPLOYED values agree", which is the comparison that
actually governs behaviour.

Because the ids may no longer appear in this repository, the pinned copy of fr's
set is a FINGERPRINT (``sha256`` over the sorted, newline-joined uuids) recorded
with the fr commit it was taken from.

Three layers, no runtime dependency on fr:

1. ALWAYS RUNS — the fixture is well formed and its hash is of the documented
   shape.
2. RUNS WHEN A REAL SET IS CONFIGURED — WM's configured set hashes to the pinned
   fingerprint. The unit suite runs a SYNTHETIC set (``tests/_platform_tenant.py``)
   and skips this with that reason; a deployment or a CI job that exports the
   estate value gets the real check.
3. OPTIONAL — when a checkout of fr is available (``FR_REPO_DIR``, or a sibling
   ``fr`` directory next to this repo), parse fr's TypeScript definition and
   compare its hash with the same fingerprint. Skips, with the reason, otherwise.

⛔ Never compare against gitops ``COMMERCIAL_PLATFORM_TENANT_IDS``: it is a
DIFFERENT set (it adds the ``00000000-…-0001`` system row).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from src.api.tenant_constants import platform_tenant_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fr_platform_tenant_ids.json"
UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# Only the SYNTHETIC namespace ``tests/_platform_tenant.py`` fabricates. A set
# drawn entirely from it is the unit suite's, not an estate's.
SYNTHETIC_PREFIXES = (
    "aaaaaaaa-0000-4000-8000-",
    "bbbbbbbb-0000-4000-8000-",
    "cccccccc-0000-4000-8000-",
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def tenant_set_hash(ids: frozenset[str] | set[str]) -> str:
    """The pinned fingerprint: sha256 over the sorted, newline-joined uuids."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _is_synthetic(ids: frozenset[str]) -> bool:
    return bool(ids) and all(i.startswith(SYNTHETIC_PREFIXES) for i in ids)


# --- layer 1: the fixture (always runs) -------------------------------------


def test_fixture_is_pinned_to_an_fr_commit_and_a_hash() -> None:
    data = _fixture()
    assert data["fr_repo"] == "nonagenticai/fr"
    assert re.fullmatch(r"[0-9a-f]{40}", str(data["fr_ref"])), "fr_ref must be a full commit SHA"
    assert data["fr_source"] == "backend/src/middleware/internal-tenant-only.ts"
    assert data["fr_symbol"] == "PLATFORM_TENANT_IDS"
    assert data["hash_algorithm"] == "sha256"
    assert re.fullmatch(r"[0-9a-f]{64}", str(data["tenant_set_sha256"]))


def test_the_fixture_carries_no_tenant_uuids() -> None:
    """The whole point of the fingerprint: a public repo holds no real ids."""
    assert not re.search(UUID_RE, FIXTURE.read_text(encoding="utf-8")), (
        "the drift fixture must hold a hash, never tenant uuids"
    )


# --- layer 2: the configured set (runs when an estate value is present) -----


def test_configured_platform_set_matches_frs_pinned_fingerprint() -> None:
    configured = platform_tenant_ids()
    if not configured:
        pytest.skip("PLATFORM_TENANT_IDS is unset (fail-closed default) — nothing to compare")
    if _is_synthetic(configured):
        pytest.skip(
            "the unit suite runs the synthetic tenant set from tests/_platform_tenant.py; "
            "the estate value is checked by a deployment or CI that exports PLATFORM_TENANT_IDS"
        )
    assert tenant_set_hash(configured) == _fixture()["tenant_set_sha256"], (
        "WM's configured PLATFORM_TENANT_IDS no longer matches fr's set at "
        f"{_fixture()['fr_ref']} — re-pin the hash and the Deployment env together"
    )


# --- layer 3: fr's live source, when a checkout is available ----------------


def parse_fr_platform_tenant_ids(ts_source: str) -> frozenset[str]:
    """Resolve fr's ``PLATFORM_TENANT_IDS`` from ``internal-tenant-only.ts`` text.

    Handles the shapes fr uses: ``export const X = '<uuid>';`` constants (also split over two lines), an
    ``[A, B] as const`` array of constant names, and a ``new Set<string>([...X,
    Y, '<uuid>'])`` whose entries are spreads, names or literals. Anything it
    cannot resolve is an error, never silently dropped.
    """
    consts = {
        m.group(1): m.group(2)
        for m in re.finditer(
            rf"(?:export\s+)?const\s+(\w+)\s*(?::\s*string\s*)?=\s*['\"]({UUID_RE})['\"]",
            ts_source,
        )
    }
    arrays: dict[str, list[str]] = {}
    for m in re.finditer(r"(?:export\s+)?const\s+(\w+)\s*=\s*\[([^\]]*)\]", ts_source):
        arrays[m.group(1)] = [t.strip() for t in m.group(2).split(",") if t.strip()]

    set_match = re.search(
        r"PLATFORM_TENANT_IDS\s*:[^=]*=\s*new\s+Set(?:<[^>]*>)?\(\s*\[([^\]]*)\]", ts_source
    )
    if set_match is None:
        raise ValueError("PLATFORM_TENANT_IDS = new Set([...]) not found in fr source")

    def resolve(token: str, depth: int = 0) -> list[str]:
        if depth > 3:
            raise ValueError(f"unresolvable nesting at {token!r}")
        token = token.strip()
        if token.startswith("..."):
            name = token[3:].strip()
            if name not in arrays:
                raise ValueError(f"spread of unknown array {name!r}")
            return [v for t in arrays[name] for v in resolve(t, depth + 1)]
        literal = re.fullmatch(rf"['\"]({UUID_RE})['\"]", token)
        if literal:
            return [literal.group(1)]
        if token in consts:
            return [consts[token]]
        raise ValueError(f"cannot resolve {token!r} in fr PLATFORM_TENANT_IDS")

    entries = [t for t in set_match.group(1).split(",") if t.strip()]
    return frozenset(v for t in entries for v in resolve(t))


_SAMPLE_TS = """
export const A_ID =
  '00000000-0000-4000-8000-000000000001';
export const B_ID = '00000000-0000-4000-8000-000000000002';
export const INTERNAL_TENANT_IDS = [
  A_ID,
  B_ID,
] as const;
export const C_ID = '00000000-0000-4000-8000-000000000003';
export const PLATFORM_TENANT_IDS: ReadonlySet<string> = new Set<string>([
  ...INTERNAL_TENANT_IDS,
  C_ID,
  '00000000-0000-4000-8000-000000000004',
]);
"""


def test_parser_resolves_the_shapes_fr_uses() -> None:
    # Keeps the parser honest when layer 3 skips (no fr checkout).
    assert parse_fr_platform_tenant_ids(_SAMPLE_TS) == frozenset(
        {
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
            "00000000-0000-4000-8000-000000000003",
            "00000000-0000-4000-8000-000000000004",
        }
    )
    with pytest.raises(ValueError, match="cannot resolve"):
        parse_fr_platform_tenant_ids(_SAMPLE_TS.replace("  C_ID,\n", "  UNKNOWN_ID,\n"))


def _fr_source_file() -> Path | None:
    candidates = []
    if env_dir := os.environ.get("FR_REPO_DIR"):
        candidates.append(Path(env_dir))
    candidates.append(REPO_ROOT.parent / "fr")
    rel = str(_fixture()["fr_source"])
    for base in candidates:
        path = base / rel
        if path.is_file():
            return path
    return None


def test_fr_checkout_matches_the_pinned_fingerprint() -> None:
    path = _fr_source_file()
    if path is None:
        pytest.skip(
            "no fr checkout (set FR_REPO_DIR to an fr clone) — the pinned-fixture "
            "comparison above still ran"
        )
    source = path.read_text(encoding="utf-8")
    try:
        live = parse_fr_platform_tenant_ids(source)
    except ValueError as exc:
        if "not found in fr source" in str(exc):
            pytest.skip(
                "the fr checkout carries no PLATFORM_TENANT_IDS literal — its ids come "
                "from FR_*_TENANT_ID at runtime since 2026-11-16, so there is no source "
                "set to compare against the fingerprint"
            )
        raise
    assert tenant_set_hash(live) == _fixture()["tenant_set_sha256"], (
        f"fr's set at {path} differs from the pinned fingerprint "
        f"({_fixture()['fr_ref']}) — re-pin the hash and WM's Deployment env together"
    )
