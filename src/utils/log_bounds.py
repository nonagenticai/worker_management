"""One place that decides how much of a payload may reach the log.

This is a log-RETENTION control, not a tidiness preference, and it lives in its
own module because two call sites need the SAME bound: the prompt going into an
agent (``worker/agents/pydantic_ai_agent.py``) and the response coming back out
(``api/workers/message_endpoints.py``). Bounding only one of them just moves the
problem to the other.

Measured on the live pod, 2026-08-13, over one 9-hour container log before it
rotated:

    total                    12,663 lines / 10,493,967 bytes
    real log records          6,075 lines /  1,868,634 bytes  (17.8%)
    unprefixed continuation   6,588 lines /  8,625,333 bytes  (82.2%)

The cost is retention. kubelet rotates the 10Mi container log and ``kubectl
logs`` serves only the CURRENT file, so an investigator sees a few hours of
history no matter what ``--since`` says; the rest needs root SSH on the owning
node. That has already misled two sessions -- one read a 5-hour window as 30
days, and one concluded the max_tokens cap-hit WARNINGs were never emitted when
they had merely rotated away.

Holding traffic fixed (145 agent runs in one hour), bounding the prompt alone
took the service from 7.71 MB/h to 2.22 MB/h and the 10Mi rotation interval from
1.3h to 4.5h.
"""

from __future__ import annotations

from typing import Any

# How much of a payload may reach the log.
#
# 2000 chars keeps the payload identifiable -- the agent, the task, the opening
# of the JSON -- while cutting the term that dominates the log. Raise it only
# with a fresh measurement of the rotation interval, not on the feeling that
# more context is better.
MAX_LOGGED_PAYLOAD_CHARS = 2000


def summarize_for_log(value: Any, *, limit: int = MAX_LOGGED_PAYLOAD_CHARS) -> str:
    """Bound a payload for logging, collapsing newlines so it stays ONE record.

    Collapsing is the load-bearing half. Python's logging prefixes only the
    FIRST line of a multi-line message, so a payload containing newlines lands
    as one prefixed record plus thousands of anonymous continuation lines --
    the shape that made 82% of this service's log unattributable to any logger,
    level, or timestamp, and which reads like a stray ``print()`` even though
    ``src/`` contains none.

    ``value`` is deliberately ``Any``: the prompt side passes ``str |
    list[Any]`` (pydantic-ai's multi-modal prompt) and the response side passes
    a parsed ``dict``/``list``. The non-string forms render LONGER, so they are
    the cases that most need bounding rather than ones to exclude.
    """
    rendered = value if isinstance(value, str) else str(value)
    collapsed = " ".join(rendered.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}... [truncated, {len(rendered)} chars total]"
