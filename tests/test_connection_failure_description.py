"""Unit tests for `_describe_connection_failure`.

`test_send_message_retryability_status.py` proves the ENDPOINT returns 502 and
that the detail names the hop. It cannot cheaply prove the chain walk's edge
cases -- depth, cycles, an absent cause -- and those are exactly where a
diagnostic string turns into an unbounded response body or an infinite loop.
"""

from __future__ import annotations

from src.api.workers._helpers import _CAUSE_CHAIN_DEPTH, _describe_connection_failure


def _chain(depth: int) -> Exception:
    """Build a `__cause__` chain `depth` links deep below a RuntimeError."""
    root: Exception = ValueError("link-0")
    for i in range(1, depth):
        nxt: Exception = ValueError(f"link-{i}")
        nxt.__cause__ = root
        root = nxt
    top = RuntimeError("Connection error.")
    top.__cause__ = root
    return top


def test_never_loses_the_original_message():
    """Enriching may not discard information the caller had before."""
    out = _describe_connection_failure(
        RuntimeError("Connection error."), worker_name="w", attempts=1
    )
    assert "Connection error." in out
    assert "RuntimeError" in out


def test_names_the_worker_and_the_attempts_burned():
    out = _describe_connection_failure(
        RuntimeError("Connection error."), worker_name="worker-select-asset", attempts=7
    )
    assert "[worker=worker-select-asset, attempts=7]" in out


def test_surfaces_the_underlying_cause():
    """The cause is the whole point: ConnectTimeout and PoolTimeout are opposite fixes."""
    top = RuntimeError("Connection error.")
    top.__cause__ = TimeoutError("pool timeout waiting for a free connection")
    out = _describe_connection_failure(top, worker_name="w", attempts=1)
    assert "TimeoutError" in out
    assert "pool timeout waiting for a free connection" in out


def test_falls_back_to_context_when_there_is_no_explicit_cause():
    """A bare `raise` inside an `except` sets __context__, not __cause__."""
    top = RuntimeError("Connection error.")
    top.__context__ = OSError("connection refused")
    out = _describe_connection_failure(top, worker_name="w", attempts=1)
    assert "connection refused" in out


def test_absent_cause_yields_no_dangling_separator():
    out = _describe_connection_failure(
        RuntimeError("Connection error."), worker_name="w", attempts=1
    )
    assert "<-" not in out


def test_chain_is_capped():
    """An unbounded walk would put an arbitrarily long body on an error response."""
    out = _describe_connection_failure(_chain(10), worker_name="w", attempts=1)
    assert out.count("<-") == _CAUSE_CHAIN_DEPTH
    # The deepest links must be absent -- proving the cap truncates rather than
    # merely counting separators.
    assert "link-0" not in out


def test_a_cycle_terminates():
    """A self-referential chain must not hang the error path."""
    a = RuntimeError("Connection error.")
    b = ValueError("b")
    a.__cause__ = b
    b.__cause__ = a
    out = _describe_connection_failure(a, worker_name="w", attempts=1)
    assert "ValueError: b" in out


def test_a_cause_with_an_empty_message_still_names_its_type():
    top = RuntimeError("Connection error.")
    top.__cause__ = TimeoutError()
    out = _describe_connection_failure(top, worker_name="w", attempts=1)
    assert "TimeoutError" in out
    assert ": ]" not in out


def test_identical_consecutive_links_are_collapsed():
    """The retry ladder re-raises, so each attempt carries its twin as __context__.

    Undeduped, the detail rendered "APIConnectionError: Connection error. <-
    APIConnectionError: Connection error." and the depth cap then spent its budget
    on repeats instead of the one informative link. Caught by the real suite, not
    by a hand-built fixture -- my unit cases all used distinct messages.
    """
    a = RuntimeError("Connection error.")
    twin = RuntimeError("Connection error.")
    root = TimeoutError("pool timeout")
    twin.__context__ = root
    a.__context__ = twin

    out = _describe_connection_failure(a, worker_name="w", attempts=3)

    assert out.count("Connection error.") == 1
    # The informative link survives the collapse -- that is the point of collapsing.
    assert "pool timeout" in out


def test_a_repeat_that_is_not_consecutive_is_kept():
    """Collapse adjacent duplicates only; a genuine A -> B -> A chain is real."""
    a = RuntimeError("Connection error.")
    b = ValueError("middle")
    c = ValueError("edge")
    d = ValueError("middle")
    a.__cause__ = b
    b.__cause__ = c
    c.__cause__ = d
    out = _describe_connection_failure(a, worker_name="w", attempts=1)
    assert out.count("middle") == 2
