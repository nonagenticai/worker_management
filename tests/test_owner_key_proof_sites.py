"""fr#1385 — membership of the commercial-platform set is NOT proof.

Owner ruling 2026-10-03: only PLATFORM tenants (gitops
``COMMERCIAL_PLATFORM_TENANT_IDS``) may use the owner's commercial LLM keys, and
the tenant must be PROVEN by the request (licence introspection, or a verified
end-user token), never asserted. A validator found paths that granted the
owner's keys on membership alone; each now goes through the ONE predicate,
``src.auth.tenant_proof.owner_commercial_keys_refusal``:

* the ``inference_provider`` graph node's env-key read (the LIVE owner-key path);
* the worker-build routing (``enforce_customer_routing``): IR pin + non-local IR
  models (``deepseek-…``, ``tier-*``, ``default``);
* the X-Provider pin (chat + message) — tests/test_ir_passthrough_sampling_correlation.py;
* ``ir_bundle_tokens`` no longer checks membership (owner ruling 2026-10-06:
  platform tenants are minted for like any other PROVEN tenant) —
  tests/test_ir_bundle_mint_proven_tenant.py.

Every test here runs the REAL predicate against the REAL request contextvars
(``platform_tenant_context``) — only the provenance is chosen.

The keep-out at the bottom fails on any NEW reader of an owner env key, and any
NEW platform-set membership check, outside an allowlist whose every entry is
justified.
"""

from __future__ import annotations

import ast
import os
import pathlib
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from src.settings import get_settings, reset_settings_cache
from tests._platform_tenant import CUSTOMER_TENANT_A, PLATFORM_TENANT, platform_tenant_context

SYSTEM_TENANT = "00000000-0000-0000-0000-000000000001"  # in the conftest gitops set
OWNER_KEY = "xai-OWNER-not-real"  # pragma: allowlist secret


@pytest.fixture
def unset_platform_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """WM ships BEFORE gitops: ``COMMERCIAL_PLATFORM_TENANT_IDS`` unset ⇒ empty."""
    monkeypatch.delenv("COMMERCIAL_PLATFORM_TENANT_IDS", raising=False)
    monkeypatch.delenv("WORKER_MANAGEMENT_COMMERCIAL_PLATFORM_TENANT_IDS", raising=False)
    reset_settings_cache()
    from src.api.tenant_constants import commercial_platform_tenant_ids

    assert commercial_platform_tenant_ids() == frozenset()  # no fallback list


# ============================================================ inference_provider node


class TestInferenceProviderNodeNeedsProof:
    """MUTATION: replace the refusal in ``_provider_env_key`` with a membership
    check (or ``refusal = None``) ⇒ the asserted/default tests go red."""

    cfg: ClassVar[dict[str, str]] = {"name": "p", "api_key_env": "XAI_API_KEY"}

    @staticmethod
    def _call(tenant: str | None) -> tuple[Any, list[str]]:
        from src.worker.graphs.nodes import inference_provider_node as n

        reads: list[str] = []
        real_get = os.environ.get

        def spy(key: str, default: Any = None) -> Any:
            if key == "XAI_API_KEY":
                reads.append(key)
            return real_get(key, default)

        with patch.object(n.os.environ, "get", side_effect=spy):
            try:
                return n._provider_env_key(TestInferenceProviderNodeNeedsProof.cfg, tenant), reads
            except n.CustomerEnvKeyRefusedError as exc:
                return exc, reads

    @pytest.mark.parametrize("provenance", ["asserted", "default", "bypass", "unknown"])
    def test_unproven_platform_tenant_is_refused_and_env_never_read(self, monkeypatch, provenance):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance=provenance):
            got, reads = self._call(PLATFORM_TENANT)
        assert isinstance(got, ValueError), got
        assert reads == []
        assert OWNER_KEY not in str(got)

    def test_header_only_delegation_is_refused(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance="delegated"):
            got, reads = self._call(PLATFORM_TENANT)
        assert isinstance(got, ValueError)
        assert reads == []

    def test_proven_platform_tenant_reads_the_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
            got, reads = self._call(PLATFORM_TENANT)
        assert got == OWNER_KEY
        assert reads == ["XAI_API_KEY"]

    def test_proven_platform_request_tenant_with_no_state_tenant_reads_the_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
            got, _ = self._call(None)
        assert got == OWNER_KEY

    def test_proven_customer_is_refused(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(CUSTOMER_TENANT_A, provenance="license"):
            got, reads = self._call(CUSTOMER_TENANT_A)
        assert isinstance(got, ValueError)
        assert reads == []

    def test_platform_state_tenant_under_another_proven_request_is_refused(self, monkeypatch):
        """A licence proves the REQUEST's tenant only — never one from graph state."""
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(CUSTOMER_TENANT_A, provenance="license"):
            got, reads = self._call(PLATFORM_TENANT)
        assert isinstance(got, ValueError)
        assert reads == []

    def test_customer_state_tenant_under_a_proven_platform_request_is_refused(self, monkeypatch):
        """The graph state's tenant (the run is FOR it) takes precedence over the
        request's: a proven PLATFORM caller running a CUSTOMER's graph must not
        spend the owner's key on it.

        MUTATION: swap the precedence in ``_provider_env_key`` to
        ``get_tenant_id_from_context() or state_tenant`` ⇒ this goes red."""
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
            got, reads = self._call(CUSTOMER_TENANT_A)
        assert isinstance(got, ValueError), got
        assert "not_a_commercial_platform_tenant" in str(got)
        assert reads == []
        assert OWNER_KEY not in str(got)

    def test_unset_platform_set_refuses_a_proven_platform_tenant(
        self, monkeypatch, unset_platform_set
    ):
        monkeypatch.setenv("XAI_API_KEY", OWNER_KEY)
        with platform_tenant_context(PLATFORM_TENANT, provenance="license"):
            got, reads = self._call(PLATFORM_TENANT)
        assert isinstance(got, ValueError)
        assert "not_a_commercial_platform_tenant" in str(got)
        assert reads == []


# ================================================= worker routing (llm_config_resolver)


def _paid_ir(**extra: Any) -> dict[str, Any]:
    return {
        "agent_role": "r",
        "model_provider": "inference_router",
        "model": "deepseek-v4-pro",
        "ir_provider": "deepseek",
        **extra,
    }


class TestWorkerRoutingNeedsProof:
    """MUTATION: return ``config`` untouched for any platform MEMBER in
    ``enforce_customer_routing`` (skip the refusal) ⇒ the asserted tests go red."""

    @staticmethod
    def _route(config: dict[str, Any], tenant: str, provenance: str) -> dict[str, Any]:
        from src.worker.factory.llm_config_resolver import enforce_customer_routing

        with platform_tenant_context(tenant, provenance=provenance):
            return enforce_customer_routing(dict(config), tenant)

    @pytest.mark.parametrize("provenance", ["asserted", "default"])
    def test_asserted_platform_loses_pin_and_paid_ir_model(self, provenance):
        out = self._route(_paid_ir(), SYSTEM_TENANT, provenance)
        assert out["model_provider"] == "inference_router"
        assert out["model"] == get_settings().local_llm_model
        assert out["ir_provider"] is None

    @pytest.mark.parametrize("model", ["tier-medium", "default", "grok-4-fast"])
    def test_asserted_platform_tier_and_default_aliases_are_coerced(self, monkeypatch, model):
        monkeypatch.setenv("WM_TIER_ALIAS_PASSTHROUGH", "true")
        reset_settings_cache()
        out = self._route(_paid_ir(model=model), SYSTEM_TENANT, "asserted")
        assert out["model"] == get_settings().local_llm_model

    @pytest.mark.parametrize("provider", ["vllm", "mlx"])
    def test_asserted_platform_keeps_local_direct_providers(self, provider):
        cfg = {"agent_role": "r", "model_provider": provider, "model": "some-local-model"}
        out = self._route(cfg, SYSTEM_TENANT, "asserted")
        assert (out["model_provider"], out["model"]) == (provider, "some-local-model")

    def test_asserted_platform_keeps_ir_local_llm(self):
        local = get_settings().local_llm_model
        out = self._route(_paid_ir(model=local, ir_provider=None), SYSTEM_TENANT, "asserted")
        assert (out["model_provider"], out["model"]) == ("inference_router", local)

    def test_asserted_platform_external_provider_is_refused_at_build(self):
        """Not rewritten by routing; ModelFactory refuses it with the same predicate."""
        from pydantic_ai.exceptions import UserError

        from src.worker.agents.pydantic_ai_agent import ModelFactory

        cfg = {"agent_role": "r", "model_provider": "xai", "model": "grok-4"}
        out = self._route(cfg, SYSTEM_TENANT, "asserted")
        assert out["model_provider"] == "xai"
        with platform_tenant_context(SYSTEM_TENANT, provenance="asserted"):
            with pytest.raises(UserError):
                ModelFactory.enforce_local_only_provider(
                    "xai", call_site="test", tenant_id=SYSTEM_TENANT
                )

    @pytest.mark.parametrize("provenance", ["asserted", "default"])
    def test_asserted_platform_keeps_an_on_hardware_pin(self, provenance):
        """A pin to owner hardware reaches no commercial key: kept (fr#1385 CI
        fix). MUTATION: drop every stored pin again ⇒ red."""
        local = get_settings().local_llm_model
        out = self._route(_paid_ir(model=local, ir_provider="pgx-local"), SYSTEM_TENANT, provenance)
        assert out["ir_provider"] == "pgx-local"
        assert out["model"] == local

    @pytest.mark.parametrize("pin", ["deepseek", "xai", "unknown-upstream", "PGX-LOCAL"])
    def test_asserted_platform_drops_a_commercial_or_unknown_pin(self, pin):
        out = self._route(_paid_ir(ir_provider=pin), SYSTEM_TENANT, "asserted")
        assert out["ir_provider"] is None

    def test_proven_customer_on_hardware_pin_is_still_dropped(self):
        local = get_settings().local_llm_model
        out = self._route(
            _paid_ir(model=local, ir_provider="pgx-local"), CUSTOMER_TENANT_A, "license"
        )
        assert out["ir_provider"] is None

    def test_proven_platform_is_untouched(self):
        out = self._route(_paid_ir(), SYSTEM_TENANT, "license")
        assert out == _paid_ir()

    def test_proven_customer_is_coerced(self):
        out = self._route(_paid_ir(), CUSTOMER_TENANT_A, "license")
        assert out["model"] == get_settings().local_llm_model
        assert out["ir_provider"] is None

    def test_unset_platform_set_coerces_a_proven_system_tenant(self, unset_platform_set):
        out = self._route(_paid_ir(), SYSTEM_TENANT, "license")
        assert out["model_provider"] == "inference_router"
        assert out["model"] == get_settings().local_llm_model
        assert out["ir_provider"] is None


# ============================================================================ keep-out

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Every reader of an environment variable whose NAME is not a string literal
# (``os.environ.get(name)``, ``os.getenv(var)``, ``os.environ[x]``) — i.e. every
# place that could read an owner commercial key by a computed name. Key:
# "<path under src>::<qualified function>". A new one fails the test.
DYNAMIC_ENV_READERS: dict[str, str] = {
    "worker/graphs/nodes/inference_provider_node.py::_provider_env_key": (
        "owner key for a graph node; gated by owner_commercial_keys_refusal (fr#1385)"
    ),
    "worker/agents/pydantic_ai_agent.py::ModelFactory._owner_env_key": (
        "only get_api_key calls it, after enforce_local_only_provider admitted a PROVEN "
        "platform tenant (owner_commercial_keys_refusal)"
    ),
    "api/providers/_router.py::provider_is_available": (
        "presence bool only, evaluated after owner_commercial_keys_permitted"
    ),
    "api/providers/_router.py::_require_env": (
        "owner key for a model-list fetch; list_provider_models refuses an external "
        "provider with owner_commercial_keys_refusal before any fetcher runs"
    ),
    "api/workers/models.py::validate_effective_worker": (
        "presence bool only (config validation); the key value is never read"
    ),
    "security/secret_box.py::_master_key": "WM_SECRET_KEY (module constant), not an LLM key",
    "security/secret_box.py::has_secret_key": "WM_SECRET_KEY (module constant), not an LLM key",  # pragma: allowlist secret
    "worker/agents/cli_agent.py::CLIAgent.__init__": (
        "executor image name from a per-agent-class *_IMAGE env var, not an LLM key"
    ),
    "worker/mcp_utils.py::expand_environment_variables": (
        "KNOWN OPEN, out of scope of fr#1385: ${VAR} expansion in stdio MCP args can "
        "read ANY env var. Reported to the coordinator; do not add more like it"
    ),
}

# Every WHOLE-environment copy (``{**os.environ}``, ``dict(os.environ)``,
# ``os.environ.copy()``, ``.items()``/``.values()``, ``env=os.environ``,
# ``f(**os.environ)``) — each hands EVERY owner commercial key (``XAI_API_KEY``
# …) to whatever receives the copy, without naming one. A new one fails the test.
WHOLE_ENV_COPIES: dict[str, str] = {
    "worker/mcp_utils.py::_create_regular_stdio_server": (
        "stdio MCP child inherits the pod env (incl. owner keys). Not request-"
        "reachable: request mcp_config refuses a stdio ``command`` "
        "(api/workers/models.py validate_mcp_config -> check_request_mcp_config), a "
        "stored row/direct caller is refused by assert_mcp_config_allowed "
        "(worker/factory/mcp_config_resolver.py), and the image ships no stdio "
        "registry (Dockerfile does not COPY mcp_configs/). KNOWN OPEN if a stdio "
        "preset is ever shipped: pass an explicit env allowlist instead"
    ),
}

# Every call of the commercial-platform MEMBERSHIP predicate. Entries marked
# PROOF must also call the proof predicate in the same function (checked below),
# so a membership-only grant cannot come back at those sites.
_PROOF = "PROOF"
MEMBERSHIP_SITES: dict[str, str] = {
    "api/tenant_constants.py::is_commercial_platform_tenant": "the definition",
    "auth/tenant_proof.py::owner_commercial_keys_refusal": "the ONE predicate (membership+proof)",
    "api/providers/_chat_router.py::_resolve_tenant_access": _PROOF,
    "worker/factory/llm_config_resolver.py::enforce_customer_routing": _PROOF,
    "api/providers/_router.py::list_providers": (
        "catalog only; external availability is provider_is_available (proof-gated)"
    ),
    "api/providers/_router.py::list_provider_models": _PROOF,
    "worker/factory/llm_config_resolver.py::_apply_tier_alias_passthrough": (
        "the platform branch is the RESTRICTIVE one (no tier rewrite); an unproven "
        "platform tier alias is then coerced to local-llm by enforce_customer_routing"
    ),
    "api/workers/_globals.py::WorkerInstanceCache._create_worker_instance": (
        "DNS prewarm for the customer owner-infra check only (performance, no grant)"
    ),
    "api/tenant_config/_router.py::put_llm_config": (
        "write-side refusal for customers; storing a row grants nothing, use is gated "
        "by enforce_customer_routing / the chat router (both proof)"
    ),
    "api/tenant_config/_modality_router.py::_checked_base_url": (
        "lets a platform row keep a preset's own owner-GPU default_base_url; no owner "
        "commercial key is involved (modality keys have their own proof, _is_proven)"
    ),
}

_MEMBERSHIP_NAMES = frozenset({"is_commercial_platform_tenant", "commercial_platform_tenant_ids"})
_PROOF_NAMES = frozenset({"owner_commercial_keys_refusal", "owner_commercial_keys_permitted"})
# Every owner commercial key name the pod could hold (provider registry + IR's).
_OWNER_KEY_NAMES_EXTRA = frozenset({"MISTRAL_API_KEY", "OPENROUTER_API_KEY", "TOGETHER_API_KEY"})


def _owner_key_names() -> frozenset[str]:
    from src.worker.provider_registry import PROVIDER_API_KEY_ENV_VARS

    return frozenset(PROVIDER_API_KEY_ENV_VARS.values()) | _OWNER_KEY_NAMES_EXTRA


def _is_env_read(call: ast.Call) -> bool:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        if fn.attr == "getenv":
            return True
        if fn.attr == "get" and isinstance(fn.value, ast.Attribute | ast.Name):
            base = fn.value
            return (base.attr if isinstance(base, ast.Attribute) else base.id) == "environ"
    return isinstance(fn, ast.Name) and fn.id == "getenv"


def _is_environ(node: ast.AST | None) -> bool:
    """``os.environ`` or a bare ``environ`` (``from os import environ``)."""
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


_WHOLE_ENV_METHODS = frozenset({"copy", "items", "values"})
_WHOLE_ENV_WRAPPERS = frozenset({"dict", "copy", "deepcopy"})


def _is_whole_env_copy_call(call: ast.Call) -> bool:
    fn = call.func
    # os.environ.copy() / .items() / .values()
    if isinstance(fn, ast.Attribute) and fn.attr in _WHOLE_ENV_METHODS and _is_environ(fn.value):
        return True
    # dict(os.environ) / copy.copy(os.environ) / copy.deepcopy(os.environ)
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
    if name in _WHOLE_ENV_WRAPPERS and any(_is_environ(a) for a in call.args):
        return True
    # f(**os.environ) / subprocess.run(..., env=os.environ)
    return any(_is_environ(k.value) and (k.arg is None or k.arg == "env") for k in call.keywords)


def scan_source(rel: str, source: str) -> dict[str, list[tuple[str, int]]]:
    """Return {kind: [(site, lineno)]} for env reads/copies and membership/proof calls."""
    found: dict[str, list[tuple[str, int]]] = {
        "literal_owner_key_read": [],
        "dynamic_env_read": [],
        "whole_env_copy": [],
        "membership": [],
        "proof": [],
    }
    owner_keys = _owner_key_names()
    stack: list[str] = []

    def site() -> str:
        return f"{rel}::{'.'.join(stack) or '<module>'}"

    def env_arg(node: ast.AST | None) -> None:
        if node is None:
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in owner_keys:
                found["literal_owner_key_read"].append((site(), getattr(node, "lineno", 0)))
        else:
            found["dynamic_env_read"].append((site(), getattr(node, "lineno", 0)))

    class V(ast.NodeVisitor):
        def _scoped(self, node: Any) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scoped(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._scoped(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._scoped(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            v = node.value
            is_environ = (isinstance(v, ast.Attribute) and v.attr == "environ") or (
                isinstance(v, ast.Name) and v.id == "environ"
            )
            if is_environ and isinstance(node.ctx, ast.Load):
                env_arg(node.slice)
            self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            # ``{**os.environ}``: the unpacked mapping has a ``None`` key
            if any(
                k is None and _is_environ(v) for k, v in zip(node.keys, node.values, strict=True)
            ):
                found["whole_env_copy"].append((site(), node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_whole_env_copy_call(node):
                found["whole_env_copy"].append((site(), node.lineno))
            if _is_env_read(node):
                env_arg(node.args[0] if node.args else None)
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in _MEMBERSHIP_NAMES:
                found["membership"].append((site(), node.lineno))
            if name in _PROOF_NAMES:
                found["proof"].append((site(), node.lineno))
            self.generic_visit(node)

    V().visit(ast.parse(source))
    return found


def scan_src() -> dict[str, list[tuple[str, int]]]:
    total: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for kind, hits in scan_source(rel, path.read_text()).items():
            total.setdefault(kind, []).extend(hits)
    return total


class TestOwnerKeyKeepOut:
    def test_no_literal_read_of_an_owner_commercial_key(self):
        """``os.getenv("XAI_API_KEY")`` anywhere in src/ is an ungated reader."""
        assert scan_src()["literal_owner_key_read"] == []

    def test_every_dynamic_env_reader_is_allowlisted(self):
        sites = {s for s, _ in scan_src()["dynamic_env_read"]}
        unknown = sorted(sites - DYNAMIC_ENV_READERS.keys())
        assert unknown == [], (
            f"new env reader(s) {unknown}: a computed env-var name can read the OWNER's "
            "commercial keys. Gate it with src.auth.tenant_proof."
            "owner_commercial_keys_refusal, and only then add it with a reason."
        )
        stale = sorted(DYNAMIC_ENV_READERS.keys() - sites)
        assert stale == [], f"allowlist entries no longer present: {stale}"

    def test_every_whole_env_copy_is_allowlisted(self):
        sites = {s for s, _ in scan_src()["whole_env_copy"]}
        unknown = sorted(sites - WHOLE_ENV_COPIES.keys())
        assert unknown == [], (
            f"new whole-environment copy {unknown}: it hands EVERY owner commercial "
            "key to the receiver. Pass an explicit allowlist of variables instead."
        )
        stale = sorted(WHOLE_ENV_COPIES.keys() - sites)
        assert stale == [], f"allowlist entries no longer present: {stale}"

    def test_every_membership_check_is_allowlisted(self):
        sites = {s for s, _ in scan_src()["membership"]}
        unknown = sorted(sites - MEMBERSHIP_SITES.keys())
        assert unknown == [], (
            f"new platform-set membership check(s) {unknown}: membership is not proof "
            "(fr#1385). Use src.auth.tenant_proof.owner_commercial_keys_refusal."
        )
        stale = sorted(MEMBERSHIP_SITES.keys() - sites)
        assert stale == [], f"allowlist entries no longer present: {stale}"

    def test_proof_sites_still_call_the_proof_predicate(self):
        found = scan_src()
        proof_sites = {s for s, _ in found["proof"]}
        missing = sorted(
            s for s, why in MEMBERSHIP_SITES.items() if why == _PROOF and s not in proof_sites
        )
        assert missing == [], f"membership-only grant at {missing}"

    @pytest.mark.parametrize(
        "site",
        [
            # the four fr#1385 paths: each must call the proof predicate
            "worker/graphs/nodes/inference_provider_node.py::_provider_env_key",
            "api/providers/_chat_router.py::_resolve_tenant_access",
            "api/workers/message_endpoints.py::send_message",
            "worker/factory/llm_config_resolver.py::enforce_customer_routing",
        ],
    )
    def test_each_fr1385_path_calls_the_proof_predicate(self, site):
        assert site in {s for s, _ in scan_src()["proof"]}

    def test_scanner_catches_a_new_ungated_reader(self):
        """Positive control: the scanner is not blind."""
        src = (
            "import os\n"
            "def leak():\n    return os.environ.get('XAI_API_KEY')\n"
            "def leak2(n):\n    return os.getenv(n), os.environ[n]\n"
            "def member(t):\n    return is_commercial_platform_tenant(t)\n"
        )
        got = scan_source("x.py", src)
        assert got["literal_owner_key_read"] == [("x.py::leak", 3)]
        assert {s for s, _ in got["dynamic_env_read"]} == {"x.py::leak2"}
        assert len(got["dynamic_env_read"]) == 2
        assert got["membership"] == [("x.py::member", 7)]

    def test_scanner_catches_whole_env_copies(self):
        """Positive control: every whole-env copy shape is seen; a single-key read is not."""
        src = (
            "import os, copy, subprocess\n"
            "from os import environ\n"
            "def a():\n    return {**os.environ, 'X': '1'}\n"
            "def b():\n    return dict(os.environ)\n"
            "def c():\n    return os.environ.copy()\n"
            "def d():\n    return list(environ.items())\n"
            "def e():\n    return copy.deepcopy(os.environ)\n"
            "def f():\n    return subprocess.run(['x'], env=os.environ)\n"
            "def g():\n    return h(**os.environ)\n"
            "def ok():\n    return {'A': os.environ.get('HOME')}, dict(a=1)\n"
        )
        got = scan_source("x.py", src)["whole_env_copy"]
        assert {s for s, _ in got} == {f"x.py::{n}" for n in "abcdefg"}
        assert len(got) == 7
