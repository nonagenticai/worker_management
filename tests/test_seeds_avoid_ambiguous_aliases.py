"""Guard: no seeded worker may pin an ambiguous inference-router alias.

`deepseek-v4-flash` / `deepseek-v4-pro` are aliased by BOTH enabled inference-router
providers, and pgx-local maps them to `local-llm`. So `_auto_pin_by_alias` treats them
as ambiguous, the request falls through `default_chain`, and pgx answers with Gemma —
silently, with no error anywhere. Measured 2026-07-19: requesting `deepseek-v4-pro`
incremented pgx vLLM's request counter by exactly 1.

Every seeded worker that named one of those was therefore running on Gemma while its
config claimed DeepSeek: fr's conversation agent and all 12 provisioning workers.
They briefly pinned `tier-flash`; with deepseek disabled (2026-09-17) that alias is
owned by pgx-local alone, so they now name the model they actually get: `local-llm`.

This test reads the seed sources as text on purpose — importing them would execute
module-level side effects, and the defect is a literal string in a config dict.
"""

from __future__ import annotations

import re
from pathlib import Path

# Names that do NOT auto-pin. Sending them is legal (that is the availability-failover
# path), but a *pinned worker config* naming one is always a bug: it says DeepSeek and
# means Gemma.
AMBIGUOUS = ("deepseek-v4-flash", "deepseek-v4-pro")

SEED_FILES = ("seed_fr_agent.py", "seed_provisioning_workers.py")


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts"


def test_no_seed_pins_an_ambiguous_alias() -> None:
    offenders: list[str] = []
    for name in SEED_FILES:
        path = _scripts_dir() / name
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            # only the actual pin, not the prose explaining why we avoid it
            if re.match(r'\s*"model"\s*:', line) and any(a in line for a in AMBIGUOUS):
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, (
        "seeded worker pins an ambiguous inference-router alias — it will silently run "
        "on pgx while claiming DeepSeek. Use local-llm:\n  " + "\n  ".join(offenders)
    )


def test_seeds_actually_pin_local_llm() -> None:
    """The complement: assert the intended pin is present, so a future edit that
    drops the pin entirely (falling back to the router default) is also caught."""
    for name in SEED_FILES:
        text = (_scripts_dir() / name).read_text()
        assert re.search(r'"model"\s*:\s*"local-llm"', text), f"{name} no longer pins local-llm"
