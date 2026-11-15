"""A malformed header must not beat a valid body correlation_id.

The join key reaches this service by two carriers — the ``X-Correlation-Id``
header and the request body's ``correlation_id``. ``graph_execution_api``
resolved them with a plain ``header or body``, which is a PRESENCE test rather
than a VALIDITY test. So a present-but-malformed header always won, and the
valid body value behind it was unreachable.

That is not hypothetical. Measured on prod 2026-08-14: the n8n Workers node
stamped the n8n execution id — a decimal string like ``529894`` — into the
header on every outbound call. Loki, ``namespace=workers``: 57 "Dropping
non-UUID correlation_id" warnings carrying 57 distinct decimal ids
(``527423..529894``), exactly 1:1 with the 57 ``graph_execution_sessions`` rows
written that day, every one of them with a NULL ``correlation_id``.

The n8n-side guard (``n8n_custom_nodes`` #1056) had already made the BODY carry
a well-formed UUID by 17:26:19Z that day, and it made no difference: 27 of those
57 drops landed *after* it went live, because the header kept winning. That is
what these tests pin — the server no longer depends on every caller getting the
header right.

    uv run pytest tests/test_correlation_id_precedence.py -v --no-cov
"""

import logging

from src.db.graph_execution_db import first_valid_correlation_id

# Observed being dropped in production on 2026-08-14.
BAD_HEADER = "529894"
GOOD_UUID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_UUID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


class TestMalformedHeaderNoLongerWins:
    def test_valid_body_survives_a_malformed_header(self):
        """The regression. Under `header or body` this returned '529894'."""
        assert first_valid_correlation_id(BAD_HEADER, GOOD_UUID) == GOOD_UUID

    def test_header_still_wins_when_both_are_valid(self):
        """Precedence is unchanged for real disagreements (spec 5.2.1)."""
        assert first_valid_correlation_id(GOOD_UUID, OTHER_UUID) == GOOD_UUID

    def test_absent_header_falls_through_to_the_body(self):
        assert first_valid_correlation_id(None, GOOD_UUID) == GOOD_UUID

    def test_both_malformed_yields_none_rather_than_garbage(self):
        """A key that cannot be stored must not be passed on as if it could."""
        assert first_valid_correlation_id(BAD_HEADER, "not-a-uuid") is None

    def test_nothing_supplied_is_none(self):
        assert first_valid_correlation_id(None, None) is None

    def test_a_valid_header_alone_is_preserved(self):
        assert first_valid_correlation_id(GOOD_UUID, None) == GOOD_UUID


class TestTheRejectedCallerIsStillNamed:
    def test_the_dropped_value_is_logged(self, caplog):
        """The warning is how the offending caller gets identified. Resolving
        precedence earlier must not cost us that signal."""
        with caplog.at_level(logging.WARNING, logger="src.db.graph_execution_db"):
            first_valid_correlation_id(BAD_HEADER, GOOD_UUID)

        assert BAD_HEADER in caplog.text
        assert "Dropping non-UUID correlation_id" in caplog.text

    def test_a_clean_pair_logs_nothing(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.db.graph_execution_db"):
            first_valid_correlation_id(GOOD_UUID, OTHER_UUID)

        assert "Dropping non-UUID correlation_id" not in caplog.text

    def test_absent_values_are_not_reported_as_drops(self, caplog):
        """`None` is an absence, not a malformed value — logging it would make
        the warning useless for finding real offenders."""
        with caplog.at_level(logging.WARNING, logger="src.db.graph_execution_db"):
            first_valid_correlation_id(None, None)

        assert "Dropping non-UUID correlation_id" not in caplog.text


class TestApiWiring:
    """The helper is worthless if the endpoints do not use it."""

    def test_both_endpoints_resolve_through_the_helper(self):
        import inspect

        from src.api import graph_execution_api

        source = inspect.getsource(graph_execution_api)

        # Exactly two endpoints resolve a correlation id: execute and
        # execute-stepwise. (The import is parenthesised over several lines, so
        # the name appears there without a trailing "(" and is not counted.)
        assert source.count("first_valid_correlation_id(") == 2
        # The presence-only form must be gone from both.
        assert "x_correlation_id or request.correlation_id" not in source
