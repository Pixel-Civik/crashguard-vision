import pytest
from unittest.mock import MagicMock, patch
from app.application.dto import ImageAnalysisResult


def test_analyze_returns_damages(client, sample_damage):
    from app.main import app
    from app.dependencies import get_analyze_service

    mock_service = MagicMock()
    mock_service.execute.return_value = ImageAnalysisResult(
        damages=[sample_damage],
        image_width=3024,
        image_height=4032,
        processing_ms=12,
    )

    app.dependency_overrides[get_analyze_service] = lambda: mock_service
    try:
        response = client.post(
            "/analyze",
            json={"image_url": "https://example.com/car.jpg"},
            headers={"x-vision-key": "test"},
        )
    finally:
        del app.dependency_overrides[get_analyze_service]

    assert response.status_code == 200
    data = response.json()
    assert data["image_width"] == 3024
    assert data["image_height"] == 4032
    assert len(data["damages"]) == 1
    assert data["damages"][0]["type"] == "dent"


def test_analyze_missing_image_url(client):
    response = client.post(
        "/analyze",
        json={},
        headers={"x-vision-key": "test"},
    )
    assert response.status_code == 422


def test_analyze_unauthorized():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        response = c.post("/analyze", json={"image_url": "https://x.com/car.jpg"})
    assert response.status_code == 422  # missing header
