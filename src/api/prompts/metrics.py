"""Prometheus metrics for prompt-version lifecycle writes.

``wm_prompt_published_bound_total{prompt_id, bound}`` counts every COMMITTED
move of a version INTO ``PUBLISHED`` made by the service process: the status
route and the deprecated ``activate``. ``bound="true"`` when the prompt is
referenced by a ``worker_prompt_binding`` row at the moment of the move: under
``WM_PROMPT_SOURCE=published`` that move changes a live worker's system prompt
within the resolver TTL, so it is the series to alert on.

⚠️ The adoption job (``scripts/adopt_live_worker_prompts.py``) is NOT covered
by the scraped series. It is a separate CLI process, so its increments land in
that process's own registry, which nothing scrapes and which dies with it. It
counts after its per-worker transaction commits (a rolled-back worker, e.g. a
``_Skip("already_bound")``, is never counted), and always with
``bound="false"``: its publish precedes the binding insert, and the published
text is the worker's live text, so no worker's prompt changes.

Cardinality is bounded: the ``/prompts`` router is platform-only, so
``prompt_id`` ranges over the system prompt registry (~100 ids), times two.
"""

from __future__ import annotations

from prometheus_client import Counter

__all__ = ["PUBLISHED_TRANSITIONS", "count_published"]

PUBLISHED_TRANSITIONS = Counter(
    "wm_prompt_published_bound_total",
    "Prompt versions moved INTO PUBLISHED, by prompt and by whether a "
    "worker_prompt_binding references the prompt (bound='true': with "
    "WM_PROMPT_SOURCE=published this changes a live worker's system prompt).",
    ["prompt_id", "bound"],
)


def count_published(prompt_id: str, *, bound: bool) -> None:
    """Record one move of a version of ``prompt_id`` into PUBLISHED."""
    PUBLISHED_TRANSITIONS.labels(prompt_id=prompt_id, bound="true" if bound else "false").inc()
