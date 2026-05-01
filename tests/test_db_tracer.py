import pytest
from unittest.mock import MagicMock
from app.adapters.db_tracer import DbAnalysisTracer
import logging


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.create_analysis_call.return_value = "call-uuid-123"
    return repo


def test_record_success_returns_call_id(mock_repo):
    tracer = DbAnalysisTracer(repo=mock_repo)
    call_id = tracer.record(
        call_type="analyze_image",
        model="gemini-2.5-flash",
        latency_ms=1200,
        status="success",
        raw_response={"candidates": []},
        image_id="img-001",
    )
    assert call_id == "call-uuid-123"


def test_record_calls_repo_with_correct_args(mock_repo):
    tracer = DbAnalysisTracer(repo=mock_repo)
    tracer.record(
        call_type="aggregate_damages",
        model="gemini-2.5-flash",
        latency_ms=800,
        status="error",
        raw_response={},
        session_id="sess-001",
        error="timeout",
    )
    mock_repo.create_analysis_call.assert_called_once_with(
        call_type="aggregate_damages",
        model="gemini-2.5-flash",
        latency_ms=800,
        status="error",
        raw_response={},
        session_id="sess-001",
        image_id=None,
        prompt_tokens=None,
        response_tokens=None,
        error="timeout",
    )


def test_record_emits_structured_log(mock_repo, caplog):
    tracer = DbAnalysisTracer(repo=mock_repo)
    with caplog.at_level(logging.INFO, logger="vision.tracer"):
        tracer.record(
            call_type="analyze_image",
            model="gemini-2.5-flash",
            latency_ms=500,
            status="success",
            raw_response={},
        )
    assert any("analyze_image" in r.message for r in caplog.records)
