"""worker_management#339 — WM ``PLATFORM_TENANT_IDS`` must equal fr's.

fr and WM each hard-code the platform tenant set (coordinator ruling: do NOT
unify). WM#336 depends on them agreeing: fr says a card is platform-owned, then
WM re-checks the tenant against its OWN set, so a drift 422s legitimate
releases or admits a stale id.

Two layers, no runtime dependency on fr:

1. ALWAYS RUNS — WM's frozenset == the pinned copy of fr's set in
   ``tests/fixtures/fr_platform_tenant_ids.json`` (which records the fr commit
   it was copied from).
2. OPTIONAL — when a checkout of fr is available (``FR_REPO_DIR``, or a sibling
   ``fr`` directory next to this repo), parse fr's TypeScript definition and
   compare it with BOTH the fixture and WM. Skips, with the reason, otherwise;
   layer 1 is the coverage that never skips.

⛔ Never compare against gitops ``COMMERCIAL_PLATFORM_TENANT_IDS``: it is a
DIFFERENT set (it adds ``00000000-…-0001``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from src.api.tenant_constants import PLATFORM_TENANT_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "fr_platform_tenant_ids.json"
UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# In gitops COMMERCIAL_PLATFORM_TENANT_IDS only — must never be in this set.
SYSTEM_ZERO_TENANT = "00000000-0000-0000-0000-000000000001"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fixture_ids() -> frozenset[str]:
    ids = _fixture()["tenant_ids"]
    assert isinstance(ids, list)
    return frozenset(str(i) for i in ids)


# --- layer 1: the pinned copy (always runs) ---------------------------------


def test_fixture_is_pinned_to_an_fr_commit() -> None:
    data = _fixture()
    assert data["fr_repo"] == "nonagenticai/fr"
    assert re.fullmatch(r"[0-9a-f]{40}", str(data["fr_ref"])), "fr_ref must be a full commit SHA"
    assert data["fr_source"] == "backend/src/middleware/internal-tenant-only.ts"
    assert data["fr_symbol"] == "PLATFORM_TENANT_IDS"
    ids = data["tenant_ids"]
    assert isinstance(ids, list)
    assert len(ids) == len(set(ids)), "duplicate id in the fixture"
    for tenant_id in ids:
        assert re.fullmatch(UUID_RE, tenant_id), f"not a canonical lowercase uuid: {tenant_id!r}"


def test_wm_platform_tenant_ids_equal_frs_pinned_set() -> None:
    fr_ids = _fixture_ids()
    only_wm = sorted(PLATFORM_TENANT_IDS - fr_ids)
    only_fr = sorted(fr_ids - PLATFORM_TENANT_IDS)
    assert not only_wm and not only_fr, (
        "WM PLATFORM_TENANT_IDS drifted from fr's (fixture "
        f"{FIXTURE.name} @ fr {_fixture()['fr_ref']}): only in WM {only_wm}, only in fr {only_fr}"
    )


def test_not_the_gitops_commercial_set() -> None:
    assert SYSTEM_ZERO_TENANT not in PLATFORM_TENANT_IDS
    assert SYSTEM_ZERO_TENANT not in _fixture_ids()


# --- layer 2: fr's live source, when a checkout is available ----------------


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
  '0195f48b-d000-7000-8000-000000000001';
export const B_ID = '0195f48b-d000-7000-8000-000000000002';
export const INTERNAL_TENANT_IDS = [
  A_ID,
  B_ID,
] as const;
export const C_ID = '019de842-c92a-7e37-8fdd-eb0d8c5396f9';
export const PLATFORM_TENANT_IDS: ReadonlySet<string> = new Set<string>([
  ...INTERNAL_TENANT_IDS,
  C_ID,
  '0195f48b-d000-7000-8000-000000000003',
]);
"""


def test_parser_resolves_the_shapes_fr_uses() -> None:
    # Keeps the parser honest when layer 2 skips (no fr checkout).
    assert parse_fr_platform_tenant_ids(_SAMPLE_TS) == frozenset(
        {
            "0195f48b-d000-7000-8000-000000000001",
            "0195f48b-d000-7000-8000-000000000002",
            "019de842-c92a-7e37-8fdd-eb0d8c5396f9",
            "0195f48b-d000-7000-8000-000000000003",
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


def test_fr_checkout_matches_wm_and_the_fixture() -> None:
    path = _fr_source_file()
    if path is None:
        pytest.skip(
            "no fr checkout (set FR_REPO_DIR to an fr clone) — the pinned-fixture "
            "comparison above still ran"
        )
    live = parse_fr_platform_tenant_ids(path.read_text(encoding="utf-8"))
    assert live == _fixture_ids(), (
        f"fr's set at {path} differs from the pinned fixture "
        f"(re-copy it with the new fr ref): only in fr {sorted(live - _fixture_ids())}, "
        f"only in fixture {sorted(_fixture_ids() - live)}"
    )
    assert live == PLATFORM_TENANT_IDS
