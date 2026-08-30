"""Tests for rate limiting and retry.

These matter more than they look. A silent 429 is indistinguishable from a
quality regression in the eval output -- this project's first end-to-end eval
reported 42% keyword coverage that was entirely caused by throttling. These
tests exist so that failure mode cannot come back unnoticed.
"""
from __future__ import annotations

import time

import pytest

from app.resilience import (
    RateLimiter,
    call_with_retry,
    is_rate_limit,
    is_transient,
    suggested_delay,
)

QUOTA_ERROR = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '8.977373066s'}]}}"
)


def test_detects_rate_limit():
    assert is_rate_limit(Exception(QUOTA_ERROR))
    assert is_rate_limit(Exception("429 Too Many Requests"))
    assert not is_rate_limit(Exception("404 NOT_FOUND"))


def test_transient_covers_server_errors_but_not_client_errors():
    assert is_transient(Exception("503 UNAVAILABLE"))
    assert is_transient(Exception(QUOTA_ERROR))
    # a bad key or unknown model must NOT be retried -- retrying hides the bug
    assert not is_transient(Exception("400 INVALID_ARGUMENT: API key not valid"))
    assert not is_transient(Exception("404 model not found"))


def test_prefers_the_servers_own_retry_delay():
    # honouring the API's number beats guessing: it knows when the window resets
    assert suggested_delay(Exception(QUOTA_ERROR), 0) == pytest.approx(9.477, abs=0.01)


def test_falls_back_to_exponential_backoff():
    plain = Exception("503 UNAVAILABLE")
    assert suggested_delay(plain, 0) == 1.0
    assert suggested_delay(plain, 2) == 4.0
    assert suggested_delay(plain, 10) == 30.0  # capped


def test_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("503 UNAVAILABLE")
        return "ok"

    assert call_with_retry(flaky, max_retries=4) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_non_transient_immediately(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad_key():
        calls["n"] += 1
        raise Exception("400 INVALID_ARGUMENT: API key not valid")

    with pytest.raises(Exception, match="API key not valid"):
        call_with_retry(bad_key, max_retries=4)
    assert calls["n"] == 1, "a non-transient error must not be retried"


def test_retry_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_limited():
        calls["n"] += 1
        raise Exception(QUOTA_ERROR)

    with pytest.raises(Exception):
        call_with_retry(always_limited, max_retries=2)
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_limiter_spaces_calls():
    limiter = RateLimiter(rpm=120)  # 0.5s apart
    limiter.wait()                  # first call is immediate
    t0 = time.monotonic()
    limiter.wait()
    assert time.monotonic() - t0 >= 0.4


def test_limiter_disabled_when_rpm_is_zero():
    limiter = RateLimiter(rpm=0)
    assert limiter.min_interval == 0.0
    t0 = time.monotonic()
    for _ in range(50):
        limiter.wait()
    assert time.monotonic() - t0 < 0.1
