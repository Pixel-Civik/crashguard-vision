from unittest.mock import patch

import pytest

from app.adapters.gemini_retry import call_gemini_with_retry, is_retryable_gemini_error


def test_retryable_gemini_error_detects_503_unavailable_payload():
    error = RuntimeError(
        "503 UNAVAILABLE. {'error': {'code': 503, 'status': 'UNAVAILABLE'}}"
    )

    assert is_retryable_gemini_error(error)


def test_call_gemini_with_retry_retries_transient_error():
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError(
                "503 UNAVAILABLE. {'error': {'code': 503, 'status': 'UNAVAILABLE'}}"
            )
        return "ok"

    with patch("app.adapters.gemini_retry.time.sleep"):
        assert call_gemini_with_retry(operation, base_delay_seconds=0.01) == "ok"

    assert attempts["count"] == 2


def test_call_gemini_with_retry_does_not_retry_non_transient_error():
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        raise ValueError("Gemini returned non-JSON response")

    with patch("app.adapters.gemini_retry.time.sleep"), pytest.raises(ValueError):
        call_gemini_with_retry(operation, base_delay_seconds=0.01)

    assert attempts["count"] == 1
