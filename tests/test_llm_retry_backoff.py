"""Unit tests for the WM LLM-stage bounded backoff helper + retry settings."""

from __future__ import annotations

import random

from src.api.workers._helpers import _backoff_delay
from src.settings import get_settings, reset_settings_cache


def test_backoff_no_jitter_is_exponential() -> None:
    kw = {"base_delay_s": 0.5, "max_delay_s": 8.0, "jitter_ratio": 0.0}
    assert _backoff_delay(1, **kw) == 0.5
    assert _backoff_delay(2, **kw) == 1.0
    assert _backoff_delay(3, **kw) == 2.0


def test_backoff_caps_at_max_delay() -> None:
    assert _backoff_delay(10, base_delay_s=0.5, max_delay_s=8.0, jitter_ratio=0.0) == 8.0


def test_backoff_jitter_stays_in_band() -> None:
    d = _backoff_delay(
        1, base_delay_s=0.5, max_delay_s=8.0, jitter_ratio=0.25, rng=random.Random(0)
    )
    assert 0.375 <= d <= 0.625


def test_backoff_jitter_never_exceeds_max() -> None:
    for seed in range(20):
        d = _backoff_delay(
            5, base_delay_s=0.5, max_delay_s=8.0, jitter_ratio=0.25, rng=random.Random(seed)
        )
        assert 0.0 <= d <= 8.0


def test_llm_retry_settings_defaults() -> None:
    reset_settings_cache()
    s = get_settings()
    assert s.llm_retry_max_attempts == 3
    assert s.llm_retry_base_delay_s == 0.5
    assert s.llm_retry_max_delay_s == 8.0
    assert s.llm_retry_jitter_ratio == 0.25
