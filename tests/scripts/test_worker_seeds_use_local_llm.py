"""One-model rule: every worker definition / seed / default in this repo names ``local-llm``.

George's rule: pgx serves ONE vLLM model, defined once in the ``vllm serve`` line
of gitops ``scripts/host-configs/pgx/systemd/vllm.service`` and served under the
alias ``local-llm``. Every consumer names ``local-llm``. A concrete model name
(qwen*, nemotron*, gemma*, llama*, pgx-*) or a ``default`` placeholder is a
second definition of the model that silently rots on the next swap. The 10
``improvement_workflow_*`` rows ran with ``model=default`` because the seeds in
``scripts/improvement_workers/workers/`` said so.

There is NO sanctioned exception: not a ``tier-*`` alias (live, every tier alias is
owned by pgx-local alone -- deepseek and xai are ``enabled: false`` in gitops
``clusters/strato/services/inference-router/providers-configmap.yaml`` -- so
``tier-flash`` was pgx under a name that claimed a paid vendor), and not ``default``.

Coverage, and only this:

* the seed/definition trees (``scripts/``, ``seed/``): ``model`` literals by regex;
* the service code (``src/``) by AST: every string literal that is a ``model``
  default, kwarg, dict value, assignment or ``model or "x"`` / ``x if model else "y"``
  fallback -- a literal under any OTHER key is not seen by this scan;
* the tenant model catalogs, structurally, because their aliases sit under
  ``models`` / ``model_classes``, which the AST scan does not read: the platform
  (non-external) LLM preset and the platform ``vlm`` preset offer only ``local-llm``,
  and no preset in either catalog offers a ``default`` / ``tier-*`` placeholder;
* the SQL migrations: ``model`` column defaults, ``model = '...'`` assignments,
  ``INSERT`` column/value pairs, and ``model`` values written as JSON --
  ``"model": "..."`` literals, ``jsonb_set(.., '{model}', '"..."')`` and
  ``json[b]_build_object('model', '...')``.

Not covered: a model name assembled at runtime, read from env/config, or held
under a key other than ``model`` outside the catalogs above.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from scripts.improvement_workers.register import AGENT_CONFIGS
from scripts.improvement_workers.workers import (
    asset_authoring,
    asset_loop,
    procedure_loop,
    prompt_loop,
    workflow_loop,
)
from scripts.seed_fr_agent import FR_AGENT_DEFAULT_CONFIG
from scripts.seed_provisioning_workers import PROVISIONING_WORKERS

LOCAL_LLM = "local-llm"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
W2V_SEED = REPO_ROOT / "seed" / "w2v_workers.yaml"

# `model="x"`, `"model": "x"`, `model: x` (YAML) -- literal assignments only.
MODEL_LITERAL = re.compile(
    r"""(?:\bmodel\s*=\s*|["']model["']\s*:\s*|^\s*model:\s*)["']?([A-Za-z0-9_./:-]+)["']?""",
    re.MULTILINE,
)
# `config.model`-style attribute references, not model names (a real name such as
# `qwen3.5-27b` carries a digit or dash around the dot, so it is still caught).
ATTRIBUTE_REF = re.compile(r"^[A-Za-z_]+\.[A-Za-z_]+$")


def test_every_loop_module_contributes_workers() -> None:
    # Vacuity guard: an empty registry would make the next test pass trivially.
    for module in (workflow_loop, prompt_loop, asset_authoring, procedure_loop, asset_loop):
        assert module.AGENT_CONFIGS, f"{module.__name__}.AGENT_CONFIGS is empty"
    assert len(workflow_loop.AGENT_CONFIGS) == 5


@pytest.mark.parametrize("name", sorted(AGENT_CONFIGS))
def test_improvement_worker_model_is_local_llm(name: str) -> None:
    assert AGENT_CONFIGS[name].model == LOCAL_LLM, (
        f"{name}: model={AGENT_CONFIGS[name].model!r}; the one-model rule requires "
        f"{LOCAL_LLM!r} (never 'default', never a concrete model name)"
    )


def test_w2v_seed_models_are_local_llm() -> None:
    workers = yaml.safe_load(W2V_SEED.read_text())["workers"]
    assert workers, "w2v seed has no workers"
    bad = {w["worker_name"]: w.get("model") for w in workers if w.get("model") != LOCAL_LLM}
    assert not bad, f"w2v seed workers not on {LOCAL_LLM!r}: {bad}"


def test_provisioning_and_fr_agent_seeds_are_local_llm() -> None:
    configs = [*PROVISIONING_WORKERS, FR_AGENT_DEFAULT_CONFIG]
    assert len(configs) > 1
    bad = {c["worker_name"]: c["model"] for c in configs if c["model"] != LOCAL_LLM}
    assert not bad, f"seed workers not on {LOCAL_LLM!r}: {bad}"


def _literal_models() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for root in ("scripts", "seed"):
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if path.suffix not in {".py", ".yaml", ".yml", ".json"} or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            for match in MODEL_LITERAL.finditer(path.read_text(encoding="utf-8")):
                value = match.group(1)
                # Skip type annotations / attribute references (`model: str`, `config.model`).
                if value in {"str", "None", "Optional"} or ATTRIBUTE_REF.match(value):
                    continue
                found.append((rel, value))
    return found


def test_no_seed_or_definition_literal_names_another_model() -> None:
    """Source scan so a NEW seed file cannot bypass the structural tests above."""
    literals = _literal_models()
    # Positive control: the scan must actually see the improvement + w2v seeds.
    assert sum(1 for _, v in literals if v == LOCAL_LLM) >= 40, literals
    bad = [(rel, value) for rel, value in literals if value != LOCAL_LLM]
    assert not bad, f"model literals violating the one-model rule: {bad}"


# --------------------------------------------------------------------------- src/
# (relative path, literal) pairs that are NOT a platform-model default, each with
# its reason. Every other `model` string literal under src/ must be `local-llm`.
SRC_ALLOWED: dict[tuple[str, str], str] = {
    # OpenAPI request examples for the claude_code / codex CLI-agent endpoints: the
    # value is handed to the vendor CLI (`claude --model sonnet`, `codex -m ...`),
    # which never talks to inference-router or pgx.
    ("src/api/workers/convenience_endpoints.py", "sonnet"): "claude CLI model flag",
    ("src/api/workers/convenience_endpoints.py", "gpt-5-codex"): "codex CLI model flag",
    # Log label `model or 'default'` in the CLI-agent factories -- "the CLI's own
    # default", printed, never sent as a model.
    ("src/worker/agents/claude_code_agent.py", "default"): "log label, CLI agent",
    ("src/worker/agents/codex_agent.py", "default"): "log label, CLI agent",
}


def _is_model_ref(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "model") or (
        isinstance(node, ast.Attribute) and node.attr == "model"
    )


def _str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _model_literals_in_python(source: str) -> list[tuple[int, str]]:
    """Every string literal that is a `model` default / kwarg / dict value /
    assignment / fallback. AST-based, so docstrings and comments never match."""
    out: list[tuple[int, str]] = []

    def add(node: ast.AST | None) -> None:
        value = _str(node)
        if value is not None:
            out.append((getattr(node, "lineno", 0), value))

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            with_defaults = positional[len(positional) - len(args.defaults) :]
            for arg, default in zip(with_defaults, args.defaults, strict=True):
                if arg.arg == "model":
                    add(default)
            for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if arg.arg == "model":
                    add(kw_default)
        elif isinstance(node, ast.keyword) and node.arg == "model":
            add(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if _str(key) == "model":
                    add(value)
        elif (isinstance(node, ast.Assign) and any(_is_model_ref(t) for t in node.targets)) or (
            isinstance(node, ast.AnnAssign) and _is_model_ref(node.target)
        ):
            add(node.value)
        elif (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and _is_model_ref(node.values[0])
        ):
            add(node.values[-1])
        elif isinstance(node, ast.IfExp) and (_is_model_ref(node.test) or _is_model_ref(node.body)):
            add(node.orelse)
    return out


def _src_model_literals() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        found.extend((rel, v) for _, v in _model_literals_in_python(path.read_text("utf-8")))
    return found


PYTHON_SHAPES = '''
def f(model: str = "qwen3.5-27b"): ...
def g(*, model="mlx"): ...
call(model="default")
cfg = {"model": "tier-flash"}
self.model = "gemma-4"
kwargs["m"] = config.model or "nemotron"
m = request.model if request.model else "llama-3"
def h():
    """model="docstring-only" """  # model="comment-only"
'''


def test_python_scanner_catches_every_default_shape() -> None:
    # Positive control: each shape the scanner claims to cover must fire on a
    # concrete name; the docstring and the comment must NOT.
    values = sorted(v for _, v in _model_literals_in_python(PYTHON_SHAPES))
    assert values == sorted(
        ["qwen3.5-27b", "mlx", "default", "tier-flash", "gemma-4", "nemotron", "llama-3"]
    )


def test_no_src_model_default_names_another_model() -> None:
    literals = _src_model_literals()
    # Vacuity guard: the scan must see src/'s known local-llm literal.
    assert ("src/api/workers/models.py", LOCAL_LLM) in literals, literals
    bad = [(rel, v) for rel, v in literals if v != LOCAL_LLM and (rel, v) not in SRC_ALLOWED]
    assert not bad, f"src/ model defaults violating the one-model rule: {bad}"
    stale = set(SRC_ALLOWED) - set(literals)
    assert not stale, f"SRC_ALLOWED entries no longer present (delete them): {stale}"


# --------------------------------------------------------------------- migrations
MIGRATION_DIRS = ("migrations", "alembic")
MIGRATION_ALLOWED: dict[tuple[str, str], str] = {
    # A DELETE predicate that PURGES the paid-tier rows -- a value matched, never
    # written. Historical migration; see its header.
    ("migrations/025_purge_local_preset_paid_tier_rows.sql", "tier-medium"): "purge predicate",
}
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_MODEL = (
    re.compile(r"\bmodel\s*=\s*'([^']*)'", re.IGNORECASE),
    re.compile(r"\bmodel\s+[A-Za-z]+[^,;]*?\bDEFAULT\s+'([^']*)'", re.IGNORECASE),
    # JSON text literal: '{"model": "x"}'
    re.compile(r'"model"\s*:\s*"([^"]*)"'),
    # jsonb_set(config, '{model}', '"x"') / '{agent,model}' / to_jsonb('x'::text)
    re.compile(
        r"\bjsonb?_set\s*\([^,]+,\s*'\{(?:[^}']*,)?\s*model\s*\}'\s*,\s*"
        r"(?:'\"([^\"']*)\"'|to_jsonb?\s*\(\s*'([^']*)')",
        re.IGNORECASE,
    ),
    # json_build_object('model', 'x') / jsonb_build_object(..., 'model', 'x')
    re.compile(r"'model'\s*,\s*'([^']*)'", re.IGNORECASE),
)
_SQL_INSERT = re.compile(
    r"INSERT\s+INTO\s+[\w.]+\s*\(([^)]*)\)\s*VALUES\s*\((.*?)\)\s*[,;]",
    re.IGNORECASE | re.DOTALL,
)
_SQL_VALUE = re.compile(r"'(?:[^']|'')*'|[^,]+")


def _model_literals_in_sql(sql: str) -> list[str]:
    body = _SQL_COMMENT.sub("", sql)
    found = [
        next(g for g in m.groups() if g is not None)
        for pattern in _SQL_MODEL
        for m in pattern.finditer(body)
        if any(g is not None for g in m.groups())
    ]
    for insert in _SQL_INSERT.finditer(body):
        columns = [c.strip().strip('"').lower() for c in insert.group(1).split(",")]
        values = [v.strip() for v in _SQL_VALUE.findall(insert.group(2))]
        if "model" in columns and len(values) == len(columns):
            value = values[columns.index("model")]
            if value.startswith("'"):
                found.append(value.strip("'"))
    # An empty DEFAULT '' is "no model", not a second model name.
    return [v for v in found if v]


def _migration_model_literals() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for root in MIGRATION_DIRS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted([*base.rglob("*.sql"), *base.rglob("*.py")]):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text("utf-8")
            if path.suffix == ".py":
                found.extend((rel, v) for _, v in _model_literals_in_python(text))
            else:
                found.extend((rel, v) for v in _model_literals_in_sql(text))
    return found


def test_sql_scanner_catches_every_default_shape() -> None:
    sql = """
    -- model = 'comment-only'
    ALTER TABLE workers ALTER COLUMN model SET DEFAULT 'x';
    CREATE TABLE t (model TEXT NOT NULL DEFAULT 'qwen3.5', other TEXT DEFAULT '');
    UPDATE workers SET model = 'tier-flash' WHERE id = 1;
    INSERT INTO workers (worker_name, model) VALUES ('w, one', 'gemma-4');
    UPDATE workers SET config = '{"model": "nemotron"}';
    UPDATE workers SET config = jsonb_set(config, '{model}', '"qwen"');
    UPDATE workers SET config = jsonb_set(config,'{agent_config,model}','"llama-3"', true);
    UPDATE workers SET config = jsonb_set(config, '{model}', to_jsonb('pgx-a'::text));
    UPDATE workers SET config = jsonb_build_object('temperature', 0.2, 'model', 'default');
    UPDATE workers SET config = json_build_object('model', 'tier-medium');
    UPDATE workers SET config = jsonb_set(config, '{temperature}', '0.2');
    """
    assert sorted(_model_literals_in_sql(sql)) == sorted(
        [
            "x",
            "qwen3.5",
            "tier-flash",
            "gemma-4",
            "nemotron",
            "qwen",
            "llama-3",
            "pgx-a",
            "default",
            "tier-medium",
        ]
    )


def test_no_migration_seeds_another_model() -> None:
    literals = _migration_model_literals()
    # Vacuity guard: the one known literal must be seen, so the scan is live.
    assert set(MIGRATION_ALLOWED) <= set(literals), literals
    bad = [(rel, v) for rel, v in literals if v != LOCAL_LLM and (rel, v) not in MIGRATION_ALLOWED]
    assert not bad, f"migration model values violating the one-model rule: {bad}"


# ----------------------------------------------------------------------- catalogs
# The tenant catalogs hold model aliases under `models` / `model_classes`, which the
# src/ AST scan above does not read (it only follows a `model` key). They are what
# the fr /settings/llm-provider dropdown offers, so they are checked structurally.
_PLACEHOLDER = re.compile(r"^(default|tier-[a-z0-9-]+)$")


def _catalog_offers(preset: dict) -> list[str]:
    return [*preset.get("models", []), *preset.get("model_classes", {}).values()]


def _catalog_violations(label: str, presets: list[dict]) -> list[tuple[str, str, str]]:
    bad: list[tuple[str, str, str]] = []
    for preset in presets:
        for value in _catalog_offers(preset):
            placeholder = bool(_PLACEHOLDER.match(value))
            second_platform_model = not preset["is_external"] and value != LOCAL_LLM
            if placeholder or second_platform_model:
                bad.append((label, preset["key"], value))
    return bad


def test_catalog_scanner_catches_placeholders_and_second_models() -> None:
    # Positive control: the pre-2026-09-24 `local` preset shape must go red on
    # every alias, including a class value that is not in `models`.
    old_local = {
        "key": "local",
        "is_external": False,
        "models": [LOCAL_LLM, "default", "tier-flash", "qwen3.5"],
        "model_classes": {"flash": LOCAL_LLM, "reasoning": "tier-medium"},
    }
    byo = {"key": "byo", "is_external": True, "models": ["gpt-4o", "tier-x"], "model_classes": {}}
    assert sorted(v for *_, v in _catalog_violations("t", [old_local, byo])) == sorted(
        ["default", "tier-flash", "qwen3.5", "tier-medium", "tier-x"]
    )


def test_tenant_catalogs_offer_only_local_llm_on_the_platform() -> None:
    from src.worker.agents.modality_catalog import modality_catalog
    from src.worker.agents.provider_catalog import provider_catalog

    llm = provider_catalog()
    vlm = modality_catalog("vlm")
    # Vacuity guard: each catalog must contain a platform preset offering local-llm.
    for presets in (llm, vlm):
        platform = [p for p in presets if not p["is_external"]]
        assert platform and all(LOCAL_LLM in _catalog_offers(p) for p in platform), presets
    bad = _catalog_violations("llm", llm) + _catalog_violations("vlm", vlm)
    assert not bad, f"catalog model aliases violating the one-model rule: {bad}"
