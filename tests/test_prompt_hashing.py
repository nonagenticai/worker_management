"""The two digest definitions that make prompt attribution checkable.

``src/api/prompts/hashing.py`` is imported by BOTH the ledger (what a version
IS) and the live-call echo (what a call RAN). If the two sides ever compute
differently, every comparison answers "drifted" and the attribution silently
becomes noise — so these tests pin the definitions, including what is
deliberately NOT in each one.
"""

from __future__ import annotations

import hashlib

import pytest

from src.api.prompts.hashing import (
    format_prompt_revision,
    parse_prompt_revision,
    sha256_text,
    version_payload_sha256,
)


def _payload(**overrides):
    base = {
        "question": "Pick keywords for a card",
        "system": "You are a keyword picker.",
        "template": "Card: {{card}}",
        "output_schema": {"type": "object", "properties": {"keywords": {"type": "array"}}},
        "worker_names": ["worker-decide-keywords"],
        "purpose_tags": ["keywords", "retrieval"],
    }
    base.update(overrides)
    return base


class TestSha256Text:
    def test_is_plain_sha256_of_utf8(self):
        text = "You are a keyword picker."
        assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_does_not_normalise_whitespace(self):
        """A forgiving digest would call a drifted worker 'in sync'.

        The publisher converges a worker's prompt to EXACT bytes, so a trailing
        newline is a different prompt, not a formatting detail.
        """
        assert sha256_text("abc") != sha256_text("abc\n")
        assert sha256_text("abc") != sha256_text(" abc")

    def test_empty_string_is_a_real_digest(self):
        """'' is a prompt someone configured; None is 'no prompt'. Only the
        caller may conflate them, and the echo never does."""
        assert sha256_text("") == hashlib.sha256(b"").hexdigest()


class TestVersionPayloadSha256:
    def test_stable_across_key_order(self):
        a = version_payload_sha256(**_payload())
        reordered = dict(reversed(list(_payload().items())))
        assert version_payload_sha256(**reordered) == a

    @pytest.mark.parametrize(
        "field,value",
        [
            ("question", "A different question"),
            ("system", "You are something else."),
            ("template", "Card: {{other}}"),
            ("output_schema", {"type": "string"}),
            ("worker_names", ["worker-other"]),
            ("purpose_tags", ["other"]),
        ],
    )
    def test_every_payload_field_changes_the_digest(self, field, value):
        """Mutation control. A digest that ignored `template` would let a
        template-only edit read as 'unchanged' and never publish."""
        assert version_payload_sha256(**_payload(**{field: value})) != version_payload_sha256(
            **_payload()
        )

    def test_none_and_empty_list_are_the_same_payload(self):
        """A publisher that stops sending an empty list must not trip the
        immutability refusal."""
        assert version_payload_sha256(
            **_payload(worker_names=[], purpose_tags=[])
        ) == version_payload_sha256(**_payload(worker_names=None, purpose_tags=None))

    def test_payload_digest_is_not_the_system_digest(self):
        """The two digests answer different questions and must not collide —
        one is the version's identity, the other is the join key with a live
        call."""
        payload = _payload()
        assert version_payload_sha256(**payload) != sha256_text(payload["system"])


class TestParsePromptRevision:
    @pytest.mark.parametrize(
        "revision,expected",
        [
            ("worker-decide-keywords@3", ("worker-decide-keywords", 3)),
            ("chore.slice@1", ("chore.slice", 1)),
            ("  padded@12  ", ("padded", 12)),
        ],
    )
    def test_parses_the_s4_stamp(self, revision, expected):
        assert parse_prompt_revision(revision) == expected

    @pytest.mark.parametrize(
        "revision",
        [
            None,
            "",
            "   ",
            "v3",
            "3",
            "canary-worker-decide-keywords-v2",  # what swap_worker_prompt writes
            "worker@",
            "@3",
            "worker@abc",
            "worker@-1",
            "worker@1@2",
        ],
    )
    def test_anything_else_is_none_never_a_default(self, revision):
        """⛔ The bug this exists to prevent: the improvement-loop ETL does
        ``parseInt(config_revision)``, gets NaN, and records asset_version 1 —
        which is why 7,033 of 7,033 exec_quality rows read `legacy_default`.
        An unparseable value must surface as None, never as a plausible 1."""
        assert parse_prompt_revision(revision) == (None, None)

    def test_round_trips_with_format(self):
        assert parse_prompt_revision(format_prompt_revision("foo.bar", 7)) == ("foo.bar", 7)
