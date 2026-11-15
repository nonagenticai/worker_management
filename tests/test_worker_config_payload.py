"""The config payload POST and PUT hand to ``register_worker``.

Two defects are pinned here, both measured on production 2026-08-28:

1. **The two handlers had drifted.** They carried near-identical dict literals,
   but PUT's omitted ``container_id`` / ``executor_endpoint`` / ``workspace`` /
   ``work_tier`` — so a PUT *deleted* four keys a POST had set. 445 live rows
   carry ``container_id``; 0 of the 207 ``status='updated'`` rows carry
   ``work_tier``. Both now derive from one ``_CONFIG_FIELDS`` tuple, and this
   module fails if they diverge again.

2. **Omitted was indistinguishable from explicitly-null.** Every field defaults
   to ``None`` on ``WorkerCreateRequest`` and both handlers read
   ``request.<attr>`` unconditionally, so a bootstrapper sending 8 of 25 fields
   nulled the other 17. The payload is now built from ``model_fields_set``.

The DB-side half of the fix — that an absent key is preserved rather than
nulled — is what ``tests/integration/test_worker_config_merge.py`` proves,
against real Postgres. This module is only about what leaves the handler.
"""

from __future__ import annotations

import asyncio

import pytest

from src.api.workers.crud_endpoints import (
    _CONFIG_FIELDS,
    _already,
    _audit_config,
    _config_payload,
    _is_agent_arg,
    _validated_config_payload,
)
from src.api.workers.models import WorkerCreateRequest


def _request(**over) -> WorkerCreateRequest:
    return WorkerCreateRequest(worker_name="w", **over)


class TestOnlySetFieldsAreEmitted:
    def test_an_untouched_field_is_absent_not_null(self):
        payload = _config_payload(_request(model="local-llm"))
        assert payload == {"model": "local-llm"}
        assert "system_prompt" not in payload, (
            "an omitted field must be ABSENT — a null here is the clobber"
        )

    def test_a_validator_no_op_does_not_count_as_asserting_a_field(self):
        """``validate_agent_fields`` normalises ``agent_type`` after validation.

        Assigning to a field adds it to ``model_fields_set``, so an
        unconditional ``self.agent_type = None`` there would mark the field as
        asserted on every request and null a stored value for any caller that
        omits it — the clobber, reintroduced for one field. models.py guards the
        assignment; this fails if that guard is removed.
        """
        assert "agent_type" not in _config_payload(_request(model="m"))

    def test_an_explicit_null_is_present(self):
        """Deliberate clearing must keep working; only silence is non-destructive."""
        payload = _config_payload(
            WorkerCreateRequest.model_validate({"worker_name": "w", "model": None})
        )
        assert "model" in payload
        assert payload["model"] is None

    def test_the_kag_body_asserts_exactly_its_own_eight_keys(self):
        """knowledge/kag/src/wm_bootstrap.py:74-85, verbatim in shape."""
        payload = _config_payload(
            WorkerCreateRequest.model_validate(
                {
                    "worker_name": "kag_synthesize",
                    "agent_role": "kag /query/ask worker (kag_synthesize)",
                    "agent_type": "pydantic_ai",
                    "model_provider": "inference_router",
                    "model": "default",
                    "system_prompt": "Answer from the retrieved context.",
                    "base_url_override": "http://inference-router.inference-router.svc.cluster.local:8150/api",
                }
            )
        )
        assert set(payload) == {
            "agent_role",
            "agent_type",
            "model_provider",
            "model",
            "system_prompt",
            "base_url_override",
        }

    def test_only_set_false_emits_every_field(self):
        payload = _config_payload(_request(), only_set=False)
        assert set(payload) == set(_CONFIG_FIELDS)

    def test_an_empty_string_role_is_still_stored_as_none(self):
        """Preserved from the old literals: ``request.X if request.X else None``."""
        payload = _config_payload(_request(agent_role="", agent_type=""))
        assert payload["agent_role"] is None
        assert payload["agent_type"] is None


class TestAgentTypeIsDerivedNotMerelyDefaulted:
    """The one field where "omitted is left as it is" is FALSE, pinned deliberately.

    `validate_effective_worker` DERIVES `agent_type = "pydantic_ai"` when the
    effective worker carries `agent_role` + `model_provider` without it, and
    `_validated_config_payload` folds that back into the write — so such a
    request overwrites a stored `agent_type`.

    Not a regression: the old full replace wrote the same derived value. And the
    derivation cannot simply be discarded, because a CREATE has nothing to merge
    over and `_globals.py:276` reads `config.get("agent_type")` straight from
    the stored row — dropping it there would silently downgrade a new
    pydantic-ai worker to a basic one.

    The derivation moved out of `WorkerCreateRequest` when the cross-field rules
    did (they must see the merged row, not the request), so it is asserted on
    `_validated_config_payload` rather than on `_config_payload`.
    """

    def test_it_is_asserted_even_when_the_client_omits_it(self):
        payload = asyncio.run(
            _validated_config_payload(
                _request(agent_role="assistant", model_provider="mlx", model="mlx"),
                lambda: _already(None),
            )
        )
        assert payload["agent_type"] == "pydantic_ai"

    def test_it_is_not_asserted_without_a_model_provider(self):
        """The derivation is guarded, so the blast radius is that one shape.

        It needs agent_role AND model_provider AND model together — i.e. a
        caller that is already asserting the model and the provider anyway.
        """
        assert "agent_type" not in _config_payload(_request(agent_role="assistant"))

    def test_an_explicit_value_is_never_overwritten_by_the_derivation(self):
        payload = _config_payload(
            _request(
                agent_role="assistant",
                model_provider="mlx",
                model="mlx",
                agent_type="claude_code",
            )
        )
        assert payload["agent_type"] == "claude_code"


class TestTheFourKeysPutUsedToDelete:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("container_id", "c-1"),
            ("executor_endpoint", "http://coding-gateway.coding-gateway.svc.cluster.local:8687"),
            ("workspace", "/work"),
            ("work_tier", "high"),
        ],
    )
    def test_put_can_now_set_them(self, field, value):
        payload = _config_payload(_request(**{field: value}))
        assert payload[field] == value

    def test_they_are_in_the_shared_field_list(self):
        assert {"container_id", "executor_endpoint", "workspace", "work_tier"} <= set(
            _CONFIG_FIELDS
        )

    def test_every_config_field_exists_on_the_request_model(self):
        """The tuple is the single source of truth; a typo in it would silently drop a key."""
        missing = [f for f in _CONFIG_FIELDS if f not in WorkerCreateRequest.model_fields]
        assert missing == []


class TestIsAgentIsOnlyAssertedWithAgentRole:
    def test_omitting_agent_role_says_nothing(self):
        """698 of 750 live rows are is_agent=true; a silent request must not flip them."""
        assert _is_agent_arg(_request(model="m")) is None

    def test_sending_agent_role_asserts_true(self):
        assert _is_agent_arg(_request(agent_role="a helper")) is True

    def test_sending_a_null_agent_role_asserts_false(self):
        assert (
            _is_agent_arg(
                WorkerCreateRequest.model_validate({"worker_name": "w", "agent_role": None})
            )
            is False
        )


class TestAuditConfigReadsTheStoredRow:
    def test_it_reports_what_was_stored_not_what_was_sent(self):
        """The worker_states trail is the post-deploy verification signal.

        It is also the only reason the clobber was provable at all, so it has to
        record the row's contents rather than the caller's view of them.
        """
        stored = {"model": "kept-by-the-merge", "system_prompt": "kept too", "extra": "ignored"}
        assert _audit_config(stored) == {
            "agent_role": None,
            "agent_type": None,
            "model_provider": None,
            "model": "kept-by-the-merge",
            "system_prompt": "kept too",
            "instruction": None,
        }


class TestSeedIsPersistedLikeTemperature:
    """``seed`` rides the same only-set merge as every other sampling field.

    MUTATION: drop ``"seed"`` from ``_CONFIG_FIELDS`` ⇒ a PUT carrying it is
    accepted (200) and the key never reaches the write ⇒ red.
    """

    def test_a_sent_seed_is_in_the_payload(self):
        assert _config_payload(_request(seed=42)) == {"seed": 42}

    def test_seed_zero_is_zero_not_null(self):
        payload = _config_payload(_request(seed=0))
        assert payload["seed"] == 0
        assert payload["seed"] is not None

    def test_an_omitted_seed_is_absent_so_the_stored_one_survives(self):
        assert "seed" not in _config_payload(_request(temperature=0.2))

    def test_an_explicit_null_clears_it(self):
        payload = _config_payload(_request(seed=None))
        assert "seed" in payload
        assert payload["seed"] is None

    @pytest.mark.parametrize("bad", [True, "42", 1.5])
    def test_a_non_int_seed_is_refused(self, bad):
        with pytest.raises(ValueError):
            _request(seed=bad)
