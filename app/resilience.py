"""Rate limiting and retry for hosted API calls.

Hosted LLM APIs rate-limit aggressively -- the Gemini free tier allows 5
requests per minute. Without handling this, a batch job (an eval run, a nightly
re-index) silently fills with 429s that *look* like quality failures.

That is not hypothetical: the first end-to-end eval of this project reported
42% keyword coverage and 15 uncited answers. Retrieval was perfect throughout.
Every one of those failures was an HTTP 429. A system that cannot tell "the
model answered badly" from "the model was never called" cannot be evaluated at
all, which is why this lives in its own module and every provider uses it.

Two mechanisms, solving different problems:
    limiter -- spaces requests out so we mostly never hit the limit
    retry   -- recovers when we hit one anyway (shared quota, burst traffic)
"""
from __future__ import annotations

import re
import time

_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


class RateLimiter:
    """Minimum-interval limiter.

    Deliberately not thread-safe and deliberately in-process: a single API
    worker holds one. A multi-process deployment needs a *shared* limiter
    (Redis token bucket), and pretending an in-process one is sufficient there
    is the bug this comment exists to prevent.
    """

    def __init__(self, rpm: int = 0):
        self.rpm = rpm
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def is_transient(exc: Exception) -> bool:
    text = str(exc)
    return is_rate_limit(exc) or "503" in text or "UNAVAILABLE" in text or "500" in text


def suggested_delay(exc: Exception, attempt: int) -> float:
    """Prefer the delay the API itself asks for; fall back to exponential
    backoff. Honouring the server's number is both faster and politer than
    guessing -- it usually knows exactly when the quota window resets."""
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        return min(float(m.group(1)) + 0.5, 60.0)
    return min(2.0 ** attempt, 30.0)


def call_with_retry(fn, limiter: RateLimiter | None = None, max_retries: int = 4,
                    label: str = "api"):
    """Run `fn`, retrying on rate limits and transient server errors.

    Non-transient errors (a bad key, an unknown model) are re-raised
    immediately -- retrying those just wastes time and hides the real problem.
    """
    for attempt in range(max_retries + 1):
        if limiter:
            limiter.wait()
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc) or attempt == max_retries:
                raise
            delay = suggested_delay(exc, attempt)
            reason = "rate limited" if is_rate_limit(exc) else "unavailable"
            print(f"[{label}] {reason}, retrying in {delay:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
