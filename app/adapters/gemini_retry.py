from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_STATUS_TEXT = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "INTERNAL",
    "BAD_GATEWAY",
    "GATEWAY_TIMEOUT",
)


def _extract_error_code(error: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(error, "response", None)
    response_code = getattr(response, "status_code", None)
    if isinstance(response_code, int):
        return response_code

    return None


def is_retryable_gemini_error(error: Exception) -> bool:
    code = _extract_error_code(error)
    if code in _RETRYABLE_CODES:
        return True

    text = str(error).upper()
    if any(status in text for status in _RETRYABLE_STATUS_TEXT):
        return True

    return any(f"'CODE': {code}" in text or f'"CODE": {code}' in text for code in _RETRYABLE_CODES)


def call_gemini_with_retry(
    operation: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.75,
) -> T:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not is_retryable_gemini_error(exc):
                raise

            delay = min(base_delay_seconds * (2 ** (attempt - 1)), 4.0)
            jitter = random.uniform(0, delay * 0.25)
            time.sleep(delay + jitter)

    if last_error:
        raise last_error
    raise RuntimeError("Gemini operation was not executed")
