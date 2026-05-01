import pytest
from fastapi import HTTPException
from app.auth import verify_api_key, hash_key
from app.config import settings


def test_hash_key_is_deterministic():
    h1 = hash_key("my-secret-key")
    h2 = hash_key("my-secret-key")
    assert h1 == h2


def test_hash_key_different_inputs():
    assert hash_key("key-a") != hash_key("key-b")


def test_verify_api_key_valid(monkeypatch):
    monkeypatch.setattr(settings, "vision_api_key", "test-key-123")
    result = verify_api_key("test-key-123")
    assert result == hash_key("test-key-123")


def test_verify_api_key_invalid(monkeypatch):
    monkeypatch.setattr(settings, "vision_api_key", "real-key")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key("wrong-key")
    assert exc_info.value.status_code == 401
