"""InferenceProviderNode — state-driven provider fallback node.

This module provides the factory that builds per-schema
``InferenceProviderNode`` classes for the pydantic-graph execution engine.

Design
------
Each node is one link in a linear forward chain of LLM providers
(``provider_1 → provider_2 → … → provider_N → [eval_subgraph]``). The
FSM walks forward only, never backwards — the "each provider visited at
most once" invariant comes from graph *structure* (no back-edges), not
from a runtime "tried" set.

On success, the node returns either an ``End`` (if it is the last
provider in the chain) or the next NON-provider successor's class
instance (e.g. an eval_subgraph node).

On failure, the node records the failed attempt and returns the next
provider's class instance. If no further provider exists, it returns
``End({"status": "failed", ...})``.

Attempts are appended to ``ctx.state.framework_state["inference"]["attempts"]``
exactly once per provider tried (success OR failure), via the accessors
on :class:`WorkflowState` established in Task 1.

The inference-provider backend
------------------------------
We call an OpenAI-compatible HTTP endpoint via ``openai.AsyncOpenAI``
directly rather than reusing :class:`~src.worker.agents.pydantic_ai_agent.PydanticAIAgent`.
``PydanticAIAgent`` requires a heavy config (``db_path``, MCP servers,
``model_provider`` as a strict Literal, Tool instances, an exit-stack
lifecycle, …) that does not map onto a lean inference-provider config
of ``name/model/api_key_env/base_url/timeout/max_retries``. The direct
``AsyncOpenAI`` path keeps retry behaviour bounded (the FSM handles
cross-provider fallback; ``max_retries`` in config controls only
same-provider retries) and keeps the call site easy to mock in tests.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx
import openai
from pydantic_graph import BaseNode, End, GraphRunContext

from src.utils.xgrammar_schema import strip_xgrammar_unsupported
from src.worker.errors import classify_call_error as _classify_call_error
from src.worker.graphs.pydantic_graph_state import WorkflowState

if TYPE_CHECKING:
    from src.worker.graphs.graph_models import NodeSchema

logger = logging.getLogger(__name__)


def served_model_from_response(response: dict[str, Any]) -> str | None:
    """Return what inference-router REPORTED serving, read back from its response.

    Provenance comes from IR's read-back only: the top-level ``served_model_root``
    (the weights behind the served name), else ``served_model``. ⛔ Never the
    OpenAI-standard ``model`` field: vLLM/IR echo the name the CALLER sent (the
    alias, e.g. ``local-llm``), so falling back to it would record the configured
    name as provenance. ``None`` when IR reported neither — unknown stays unknown.
    """
    for key in ("served_model_root", "served_model"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _get_inference_config(schema: NodeSchema) -> dict[str, Any]:
    """Read and normalise the provider config from ``schema.framework_config``.

    Applies defaults for optional fields (``timeout``, ``max_retries``,
    ``base_url``) and validates that the required fields are present.

    Raises:
        ValueError: if required keys are missing.
    """
    raw = schema.framework_config or {}
    name = raw.get("name")
    model = raw.get("model")
    api_key_env = raw.get("api_key_env")

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Inference provider node '{schema.name}' missing 'name' in framework_config"
        )
    if not isinstance(model, str) or not model:
        raise ValueError(
            f"Inference provider node '{schema.name}' missing 'model' in framework_config"
        )
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError(
            f"Inference provider node '{schema.name}' missing 'api_key_env' in framework_config"
        )

    # Fail closed for a config that bypassed NodeSchema: the env var named by
    # api_key_env is sent as a Bearer to base_url (see request_overrides).
    from src.security.request_overrides import check_inference_provider_config

    check_inference_provider_config(raw)

    base_url = raw.get("base_url")
    # Resolve the default from settings so a single env-var
    # (``INFERENCE_PROVIDER_NODE_DEFAULT_TIMEOUT_S``) shifts every provider
    # node that doesn't explicitly override. Import is local to avoid
    # tripping the test paths that import this module without env wired.
    if "timeout" in raw:
        timeout = raw["timeout"]
    else:
        from src.settings import get_settings

        timeout = get_settings().inference_provider_node_default_timeout_s
    max_retries = raw.get("max_retries", 0)
    # Reasoning is OFF by default; per-provider opt-in via framework_config.
    # Callers can override per-request through state.variables on the node.
    enable_reasoning_default = bool(raw.get("enable_reasoning", False))

    return {
        "name": name,
        "model": model,
        "api_key_env": api_key_env,
        "base_url": base_url,
        "timeout": int(timeout),
        "max_retries": int(max_retries),
        "enable_reasoning": enable_reasoning_default,
    }


class CustomerEnvKeyRefusedError(ValueError):
    """A tenant that may not use the OWNER's keys had its graph name an env key.

    Refused unless the tenant is a PROVEN commercial-platform workspace
    (:func:`src.auth.tenant_proof.owner_commercial_keys_refusal`). A
    ``ValueError`` so the node's existing narrow failure handling records the
    attempt and moves to the next provider rather than crashing the run.
    """


def _provider_env_key(cfg: dict[str, Any], state_tenant: str | None) -> str:
    """Read the env var named by ``api_key_env`` — for a PROVEN platform tenant only.

    ⛔ Every pod env key is the OWNER's credential (``XAI_API_KEY`` …; owner
    ruling 2026-10-01, validator B1). Graph create is open to any tenant, and
    ``check_inference_provider_config`` only pins the NAME shape (``*_API_KEY``)
    with ``base_url`` optional, so a graph naming ``XAI_API_KEY`` spends the
    owner's key. Customers use their own keys only through tenant LLM config,
    never through this node.

    ⛔ Membership of the commercial-platform set is NOT enough (owner ruling
    2026-10-03, fr#1385): a tokenless caller can ASSERT a platform tenant, and an
    absent header defaults to one. The ONE predicate,
    :func:`src.auth.tenant_proof.owner_commercial_keys_refusal`, also requires
    THIS request to have proven the tenant (licence introspection, or a verified
    end-user token). The tenant is the graph state's (the run is FOR it), else
    the request's; the proof only holds for the REQUEST's tenant, so a state
    tenant other than the request's is refused, as is a run with no request
    identity at all (fail closed). A refusal never reads the variable.
    """
    from src.api.identity_context import get_tenant_id_from_context
    from src.auth.tenant_proof import owner_commercial_keys_refusal

    tenant = state_tenant or get_tenant_id_from_context() or None
    refusal = owner_commercial_keys_refusal(tenant)
    if refusal is not None:
        logger.warning(
            "inference_provider_env_key_refused provider=%s api_key_env=%s tenant=%s reason=%s",
            cfg.get("name"),
            cfg.get("api_key_env"),
            tenant,
            refusal,
        )
        raise CustomerEnvKeyRefusedError(
            f"Inference provider '{cfg.get('name')}' names a platform environment key; "
            f"tenant {tenant or '<unknown>'} may not use platform provider keys "
            f"({refusal}). Only a platform workspace that the request PROVED (a licence, "
            "or a verified end-user token) may; configure the tenant's own provider key "
            "in the tenant LLM settings instead."
        )
    return os.environ.get(cfg["api_key_env"], "")


def _extract_prompt(variables: dict[str, Any]) -> str:
    """Extract a flat prompt string from execution variables.

    Prefers explicit ``prompt``; falls back to the last user message in
    an OpenAI-style ``messages`` list; finally falls back to JSON-like
    repr. Kept simple on purpose — real request shaping belongs in the
    caller, not here.
    """
    prompt = variables.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt

    messages = variables.get("messages")
    if isinstance(messages, list) and messages:
        # last user message wins
        for raw_msg in reversed(messages):
            if not isinstance(raw_msg, dict):
                continue
            msg = cast("dict[str, Any]", raw_msg)
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        # fallback: concatenate all content fields
        parts: list[str] = []
        for raw_msg in messages:
            if not isinstance(raw_msg, dict):
                continue
            m = cast("dict[str, Any]", raw_msg)
            content = m.get("content")
            if content:
                parts.append(str(content))
        if parts:
            return "\n".join(parts)

    return str(variables)


def _extract_messages(variables: dict[str, Any]) -> list[dict[str, Any]]:
    """Return OpenAI-style messages payload from variables."""
    messages = variables.get("messages")
    if isinstance(messages, list) and messages:
        return [cast("dict[str, Any]", m) for m in messages if isinstance(m, dict)]
    prompt = _extract_prompt(variables)
    return [{"role": "user", "content": prompt}]


def _build_response_format(output_schema: Any) -> dict[str, Any] | None:
    """Build an OpenAI ``response_format`` dict from a worker ``output_schema``.

    Translates a JSON-Schema dict into
    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}``
    so inference-router can forward it to vLLM as a guided-decoding
    constraint. Returns ``None`` when ``output_schema`` is not a usable dict
    so the caller can omit ``response_format`` instead of sending a
    malformed payload.

    The schema is pruned to what xgrammar can compile, since vLLM silently
    disables structured decoding on a schema it cannot turn into a grammar.
    """
    if not isinstance(output_schema, dict) or not output_schema:
        return None
    # Read the name off the ORIGINAL schema — the strip removes ``title``.
    name = output_schema.get("title") or "output"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": str(name),
            "schema": strip_xgrammar_unsupported(output_schema),
        },
    }


def create_inference_provider_node_class(
    schema: NodeSchema,
    next_node_ids: list[str],
    all_nodes: dict[str, NodeSchema],
    edge_map: dict[str, list[dict[str, Any]]],
    next_edges: list[dict[str, Any]] | None = None,
) -> type[BaseNode[WorkflowState, None, Any]]:
    """Build a per-schema ``InferenceProviderNode`` class.

    Mirrors the closure-based pattern used by ``_create_worker_node_class``
    in ``graph_factory.py`` so schema metadata is captured in the class
    definition itself.
    """
    if next_edges is None:
        next_edges = [{"target": nid, "condition": None} for nid in next_node_ids]

    provider_config = _get_inference_config(schema)

    @dataclass
    class InferenceProviderNode(BaseNode[WorkflowState, None, Any]):
        """Per-schema inference-provider node.

        Calls an OpenAI-compatible endpoint; on failure falls back to the
        next provider in the chain via state-driven FSM transitions.
        """

        _node_schema_id: str = field(default_factory=lambda: str(schema.id), init=False)
        _next_node_ids: list[str] = field(default_factory=next_node_ids.copy, init=False)
        _all_nodes: dict[str, NodeSchema] = field(default_factory=all_nodes.copy, init=False)
        _edge_map: dict[str, list[dict[str, Any]]] = field(
            default_factory=edge_map.copy, init=False
        )
        _next_edges: list[dict[str, Any]] = field(
            default_factory=lambda: [e.copy() for e in next_edges], init=False
        )
        _provider_config: dict[str, Any] = field(
            default_factory=lambda: dict(provider_config), init=False
        )

        def _resolve_successor(self, kind: str) -> type[BaseNode[WorkflowState, None, Any]] | None:
            """Return the next node class by successor kind.

            Args:
                kind: Either ``"inference_provider"`` (next provider in
                    the fallback chain) or ``"other"`` (any non-provider
                    successor, e.g. a terminal eval_subgraph).

            Returns:
                The class, or None if no matching successor exists.
            """
            registry = getattr(type(self), "_node_class_registry", {})
            for next_id in self._next_node_ids:
                next_schema = self._all_nodes.get(next_id)
                if next_schema is None:
                    continue
                is_provider = next_schema.node_type == "inference_provider"
                if (kind == "inference_provider" and is_provider) or (
                    kind == "other" and not is_provider
                ):
                    cls_ = registry.get(next_id)
                    if cls_ is not None:
                        return cls_
            return None

        def _next_provider_fallback(
            self,
        ) -> type[BaseNode[WorkflowState, None, Any]] | None:
            """Next inference_provider successor, or None."""
            return self._resolve_successor("inference_provider")

        def _next_inference_successor(
            self,
        ) -> type[BaseNode[WorkflowState, None, Any]] | None:
            """Next non-inference_provider successor (e.g. eval_subgraph), or None."""
            return self._resolve_successor("other")

        async def _call_provider(
            self, ctx: GraphRunContext[WorkflowState]
        ) -> tuple[Any, int | None, int | None]:
            """Invoke the provider; return (response_payload, prompt_tokens, completion_tokens).

            Uses ``openai.AsyncOpenAI``. Uses any base_url and API key from
            the provider config; falls back to the env var named by
            ``api_key_env``. Same-provider retries are delegated to the
            OpenAI SDK (``max_retries``); cross-provider fallback is
            handled by the FSM.
            """
            cfg = self._provider_config
            api_key = _provider_env_key(cfg, getattr(ctx.state, "tenant_id", None))

            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": cfg["timeout"],
                "max_retries": cfg["max_retries"],
            }
            if cfg["base_url"]:
                client_kwargs["base_url"] = cfg["base_url"]

            variables = ctx.state.variables or {}
            messages = _extract_messages(variables)

            # Structured-output: per-request output_schema (variables) wins
            # over per-provider default (framework_config). Sent to IR as
            # OpenAI response_format=json_schema, which IR forwards to vLLM.
            output_schema = variables.get("output_schema") or cfg.get("output_schema")
            response_format = _build_response_format(output_schema)

            # Reasoning toggle: per-request variable wins, then framework_config.
            if "enable_reasoning" in variables:
                enable_reasoning = bool(variables["enable_reasoning"])
            else:
                enable_reasoning = bool(cfg.get("enable_reasoning", False))

            create_kwargs: dict[str, Any] = {
                "model": cfg["model"],
                # openai's stub expects a strongly-typed union (system/user/…
                # message param TypedDicts); we pass a generic list[dict] that
                # the runtime accepts unchanged. Cast through Any to express
                # "we have validated the shape enough for the wire protocol".
                "messages": cast("Any", messages),
            }
            if response_format is not None:
                create_kwargs["response_format"] = cast("Any", response_format)
            # IR strips ``enable_reasoning`` from the OpenAI payload and
            # rewrites it as ``chat_template_kwargs.enable_thinking``; we
            # only need to forward it once via extra_body.
            create_kwargs["extra_body"] = {"enable_reasoning": enable_reasoning}

            client = openai.AsyncOpenAI(**client_kwargs)
            try:
                completion = await client.chat.completions.create(**create_kwargs)
            finally:
                # AsyncOpenAI owns an httpx client; close it to avoid leaks.
                # close() is a coroutine on recent openai versions.
                close = getattr(client, "close", None)
                if close is not None:
                    maybe_coro = close()
                    if maybe_coro is not None:
                        await maybe_coro

            # Normalise the response into a plain dict so downstream code
            # (persistence, eval) does not need the openai type surface.
            response_dict: dict[str, Any]
            model_dump = getattr(completion, "model_dump", None)
            if callable(model_dump):
                response_dict = model_dump()
            elif isinstance(completion, dict):
                response_dict = completion
            else:
                response_dict = {"raw": str(completion)}

            usage = response_dict.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            return response_dict, prompt_tokens, completion_tokens

        async def run(
            self, ctx: GraphRunContext[WorkflowState]
        ) -> BaseNode[WorkflowState, None, Any] | End[Any]:
            """Attempt this provider and dispatch to the success or failure successor.

            On success hand off to the terminal successor (or End); on failure
            hand off to the next provider (or End with failure).
            """
            provider_name = self._provider_config["name"]

            # Stash the original request payload on first entry.
            bag = ctx.state.framework_state.get(WorkflowState.INFERENCE_KEY) or {}
            if bag.get("request_payload") is None:
                ctx.state.set_inference_request(ctx.state.variables or {})

            t0 = time.monotonic()
            try:
                response, prompt_tokens, completion_tokens = await self._call_provider(ctx)
            # Narrow exception surface: network, timeout, and the openai
            # SDK's own API errors. ValueError covers provider-config
            # surface (e.g. missing api key materialising as 401 parsing
            # in openai lib older versions) — still narrow.
            except (
                TimeoutError,
                httpx.HTTPError,
                openai.APIError,
                ValueError,
            ) as e:
                latency_ms = int((time.monotonic() - t0) * 1000)
                error_classification = _classify_call_error(e)
                # Severity: transient_network is retry-eligible (the FSM moves
                # to the next provider); upstream_timeout / upstream_5xx /
                # upstream_4xx / adapter_load are real failures and deserve
                # ERROR so they page through log alerting.
                log_level = (
                    logging.WARNING
                    if error_classification == "transient_network"
                    else logging.ERROR
                )
                logger.log(
                    log_level,
                    "InferenceProviderNode %s (%s) failed after %dms [classification=%s]: %s",
                    self._node_schema_id,
                    provider_name,
                    latency_ms,
                    error_classification,
                    e,
                )
                ctx.state.record_inference_attempt(
                    provider=provider_name,
                    status="error",
                    latency_ms=latency_ms,
                    error=str(e),
                    error_classification=error_classification,
                )
                ctx.state.mark_node_failed(self._node_schema_id, str(e))

                next_provider_cls = self._next_provider_fallback()
                if next_provider_cls is not None:
                    return next_provider_cls()

                attempts = ctx.state.framework_state.get(WorkflowState.INFERENCE_KEY, {}).get(
                    "attempts", []
                )
                return End(
                    {
                        "status": "failed",
                        "error": "all providers failed",
                        "attempts": len(attempts),
                    }
                )

            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "InferenceProviderNode %s (%s) succeeded in %dms",
                self._node_schema_id,
                provider_name,
                latency_ms,
            )
            served_model = served_model_from_response(response)
            ctx.state.record_inference_attempt(
                provider=provider_name,
                status="ok",
                latency_ms=latency_ms,
                tokens_prompt=prompt_tokens,
                tokens_completion=completion_tokens,
                served_model=served_model,
            )
            ctx.state.mark_inference_success(
                provider=provider_name,
                response=response,
                tokens_prompt=prompt_tokens or 0,
                tokens_completion=completion_tokens or 0,
                served_model=served_model,
            )
            ctx.state.set_node_output(
                self._node_schema_id,
                {"provider": provider_name, "served_model": served_model, "response": response},
            )

            # Hand off to a terminal (non-provider) successor if one
            # exists (e.g. eval_subgraph); otherwise end the run.
            terminal_cls = self._next_inference_successor()
            if terminal_cls is not None:
                return terminal_cls()
            return End(response)

    InferenceProviderNode.__name__ = f"InferenceProviderNode_{schema.id}".replace("-", "_")
    InferenceProviderNode.__qualname__ = InferenceProviderNode.__name__
    return InferenceProviderNode
