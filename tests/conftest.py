import os
import pytest

# Set required env vars before any app module is imported
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("SUPABASE_URL", "http://127.0.0.1:54321")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("VISION_API_KEY", "test-vision-key")

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app
from app.auth import verify_api_key
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
)


@pytest.fixture
def client():
    app.dependency_overrides[verify_api_key] = lambda: "test_key_hash"
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_damage():
    return Damage(
        id="dmg_01",
        type=DamageType.dent,
        zone=VehicleZone.hood,
        severity=Severity.medium,
        confidence=0.91,
        bbox=BoundingBox(x=0.12, y=0.34, w=0.18, h=0.09),
        description="Dent on hood",
    )
