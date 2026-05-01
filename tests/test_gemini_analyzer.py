import pytest
from unittest.mock import MagicMock, patch
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.domain.models import VehicleContext, DamageType, VehicleZone, Severity


GEMINI_RESPONSE_JSON = '[{"type":"dent","zone":"hood","severity":"medium","confidence":0.91,"bbox_x":0.12,"bbox_y":0.34,"bbox_w":0.18,"bbox_h":0.09,"description":"Dent on hood"}]'


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    response = MagicMock()
    response.text = GEMINI_RESPONSE_JSON
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    response.usage_metadata = usage
    client.models.generate_content.return_value = response
    return client


@pytest.fixture
def mock_image_bytes(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (3024, 4032), color=(100, 100, 100))
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return path.read_bytes()


def test_analyze_returns_damages(mock_gemini_client, mock_image_bytes):
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(
            client=mock_gemini_client, model="gemini-2.5-flash"
        )
        damages = analyzer.analyze(
            image_url="https://example.com/car.jpg",
            context=VehicleContext(make="Toyota"),
        )

    assert len(damages) == 1
    assert damages[0].type == DamageType.dent
    assert damages[0].zone == VehicleZone.hood
    assert damages[0].severity == Severity.medium
    assert damages[0].confidence == pytest.approx(0.91)
    assert damages[0].bbox.x == pytest.approx(0.12)


def test_analyze_unknown_zone_coerced(mock_gemini_client, mock_image_bytes):
    mock_gemini_client.models.generate_content.return_value.text = (
        '[{"type":"dent","zone":"zona_rara","severity":"low",'
        '"confidence":0.5,"bbox_x":0.1,"bbox_y":0.1,"bbox_w":0.1,"bbox_h":0.1,'
        '"description":"test"}]'
    )
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(client=mock_gemini_client, model="gemini-2.5-flash")
        damages = analyzer.analyze(image_url="https://example.com/car.jpg", context=None)

    assert damages[0].zone == VehicleZone.unknown


def test_analyze_returns_image_dimensions(mock_gemini_client, mock_image_bytes):
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(client=mock_gemini_client, model="gemini-2.5-flash")
        damages, width, height = analyzer.analyze_with_dimensions(
            image_url="https://example.com/car.jpg", context=None
        )

    assert width == 3024
    assert height == 4032


def test_analyze_raises_on_malformed_gemini_response(mock_gemini_client, mock_image_bytes):
    mock_gemini_client.models.generate_content.return_value.text = "Sorry, I cannot analyze this."
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(client=mock_gemini_client, model="gemini-2.5-flash")
        with pytest.raises(ValueError, match="non-JSON"):
            analyzer.analyze(image_url="https://example.com/car.jpg", context=None)
