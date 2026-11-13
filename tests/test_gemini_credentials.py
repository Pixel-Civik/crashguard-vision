from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app import dependencies
from app.config import Settings


def make_settings(**overrides):
    return Settings(
        _env_file=None,
        **{
            "gemini_api_key": None,
            "vertex_ai_project": None,
            "vertex_ai_location": "global",
            "gemini_api_version": None,
            **overrides,
        },
    )


@pytest.fixture(autouse=True)
def clear_client_cache():
    dependencies._gemini_client.cache_clear()
    yield
    dependencies._gemini_client.cache_clear()


def test_adc_configuration_does_not_require_a_gemini_key():
    settings = make_settings(vertex_ai_project="test-project")
    assert settings.gemini_api_key is None


def test_missing_adc_project_and_key_fails_at_startup():
    with pytest.raises(ValidationError, match="VERTEX_AI_PROJECT"):
        make_settings()


def test_adc_takes_precedence_over_a_leftover_key(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "settings",
        make_settings(vertex_ai_project="test-project", gemini_api_key="test-old-key"),
    )
    with patch.object(dependencies.genai, "Client") as client:
        dependencies._gemini_client()

    kwargs = client.call_args.kwargs
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "test-project"
    assert kwargs["location"] == "global"
    assert kwargs["http_options"].api_version == "v1"
    assert "api_key" not in kwargs


def test_adc_keeps_explicit_location_and_api_version(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "settings",
        make_settings(
            vertex_ai_project="test-project",
            vertex_ai_location="us-east4",
            gemini_api_version="v1beta1",
        ),
    )
    with patch.object(dependencies.genai, "Client") as client:
        dependencies._gemini_client()

    assert client.call_args.kwargs["location"] == "us-east4"
    assert client.call_args.kwargs["http_options"].api_version == "v1beta1"


def test_developer_key_mode_still_works_for_local_and_dev(monkeypatch):
    monkeypatch.setattr(dependencies, "settings", make_settings(gemini_api_key="test-key"))
    with patch.object(dependencies.genai, "Client") as client:
        dependencies._gemini_client()

    client.assert_called_once_with(vertexai=False, api_key="test-key")
