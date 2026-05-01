import pytest
from unittest.mock import MagicMock
from app.main import app
from app.dependencies import get_session_service
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
    DamageMap, DamageMapSummary, SourceImageMeta, SkippedImage,
)


SESSION_ROW = {
    "id": "sess-uuid-001",
    "api_key_hash": "abc123",
    "vehicle_context": {"make": "Toyota"},
    "status": "open",
    "created_at": "2026-05-01T10:00:00+00:00",
    "expires_at": "2026-05-02T10:00:00+00:00",
}

IMAGE_ROW = {
    "id": "img-uuid-001",
    "session_id": "sess-uuid-001",
    "image_url": "https://example.com/car.jpg",
    "angle": "front",
    "image_width": 3024,
    "image_height": 4032,
    "status": "completed",
    "damages": [],
    "uploaded_at": "2026-05-01T10:05:00+00:00",
    "analyzed_at": "2026-05-01T10:05:02+00:00",
}

SAMPLE_DAMAGE = Damage(
    id="dmg_01",
    type=DamageType.dent,
    zone=VehicleZone.hood,
    severity=Severity.medium,
    confidence=0.91,
    bbox=BoundingBox(x=0.12, y=0.34, w=0.18, h=0.09),
    description="Dent on hood",
    source_image_id="img-uuid-001",
)

DAMAGE_MAP = DamageMap(
    session_id="sess-uuid-001",
    vehicle_context=None,
    images={"img-uuid-001": SourceImageMeta(url="https://example.com/car.jpg", width=3024, height=4032)},
    zones={z: [] for z in VehicleZone} | {VehicleZone.hood: [SAMPLE_DAMAGE]},
    summary=DamageMapSummary(
        total_images=1,
        images_analyzed=1,
        total_damages=1,
        zones_affected=[VehicleZone.hood],
        overall_severity=Severity.medium,
        damages_by_severity={Severity.medium: 1},
        damages_by_type={DamageType.dent: 1},
    ),
)


def test_create_session(client):
    mock_service = MagicMock()
    mock_service.create_session.return_value = SESSION_ROW
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.post(
            "/sessions",
            json={"vehicle_context": {"make": "Toyota"}},
            headers={"x-vision-key": "test"},
        )
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == "sess-uuid-001"
    assert "expires_at" in data


def test_create_session_no_context(client):
    mock_service = MagicMock()
    mock_service.create_session.return_value = {**SESSION_ROW, "vehicle_context": None}
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.post("/sessions", json={}, headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 201


def test_get_session_not_found(client):
    mock_service = MagicMock()
    mock_service.get_session.return_value = None
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.get("/sessions/nonexistent-id", headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 404


def test_get_session_wrong_tenant(client):
    mock_service = MagicMock()
    mock_service.get_session.side_effect = PermissionError("wrong tenant")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.get("/sessions/sess-uuid-001", headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 403


def test_add_image_to_session(client):
    mock_service = MagicMock()
    mock_service.add_image.return_value = (IMAGE_ROW, [SAMPLE_DAMAGE], 3024, 4032)
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.post(
            "/sessions/sess-uuid-001/images",
            json={"image_url": "https://example.com/car.jpg", "angle": "front"},
            headers={"x-vision-key": "test"},
        )
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 201
    data = response.json()
    assert data["image_id"] == "img-uuid-001"
    assert data["image_width"] == 3024
    assert data["status"] == "completed"
    assert len(data["damages"]) == 1


def test_add_image_session_not_found(client):
    mock_service = MagicMock()
    mock_service.add_image.side_effect = ValueError("Session not found")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.post(
            "/sessions/bad-id/images",
            json={"image_url": "https://example.com/car.jpg"},
            headers={"x-vision-key": "test"},
        )
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 404


def test_get_report(client):
    mock_service = MagicMock()
    mock_service.get_report.return_value = DAMAGE_MAP
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.get("/sessions/sess-uuid-001/report", headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess-uuid-001"
    assert "zones" in data
    assert "summary" in data
    assert data["summary"]["total_damages"] == 1


def test_get_report_session_not_found(client):
    mock_service = MagicMock()
    mock_service.get_report.side_effect = ValueError("Session not found")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.get("/sessions/bad-id/report", headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 404


def test_add_image_wrong_tenant(client):
    mock_service = MagicMock()
    mock_service.add_image.side_effect = PermissionError("wrong tenant")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.post(
            "/sessions/sess-uuid-001/images",
            json={"image_url": "https://example.com/car.jpg"},
            headers={"x-vision-key": "test"},
        )
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 403


def test_get_report_wrong_tenant(client):
    mock_service = MagicMock()
    mock_service.get_report.side_effect = PermissionError("wrong tenant")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    try:
        response = client.get("/sessions/sess-uuid-001/report", headers={"x-vision-key": "test"})
    finally:
        app.dependency_overrides.pop(get_session_service, None)

    assert response.status_code == 403
