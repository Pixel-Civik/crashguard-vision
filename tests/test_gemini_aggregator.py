import pytest
from unittest.mock import MagicMock
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
)


def make_damage(id: str, zone: VehicleZone, image_id: str) -> Damage:
    return Damage(
        id=id,
        type=DamageType.dent,
        zone=zone,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description="test",
        source_image_id=image_id,
    )


AGGREGATED_JSON = '[{"id":"dmg_01","type":"dent","zone":"hood","severity":"medium","confidence":0.91,"bbox_x":0.12,"bbox_y":0.34,"bbox_w":0.18,"bbox_h":0.09,"description":"Dent on hood","source_image_id":"img_001","also_seen_in":["img_002"]}]'


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    response = MagicMock()
    response.text = AGGREGATED_JSON
    usage = MagicMock()
    usage.prompt_token_count = 200
    usage.candidates_token_count = 80
    response.usage_metadata = usage
    client.models.generate_content.return_value = response
    return client


def test_aggregate_returns_consolidated_damages(mock_gemini_client):
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [
        [make_damage("d1", VehicleZone.hood, "img_001")],
        [make_damage("d2", VehicleZone.hood, "img_002")],
    ]
    result = aggregator.aggregate(damage_lists)

    assert len(result) == 1
    assert result[0].zone == VehicleZone.hood
    assert result[0].source_image_id == "img_001"
    assert "img_002" in result[0].also_seen_in


def test_aggregate_empty_input(mock_gemini_client):
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    result = aggregator.aggregate([])
    assert result == []
    mock_gemini_client.models.generate_content.assert_not_called()


def test_aggregate_raises_on_malformed_response(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = "Cannot process."
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[make_damage("d1", VehicleZone.hood, "img_001")]]
    with pytest.raises(ValueError, match="non-JSON"):
        aggregator.aggregate(damage_lists)


def test_aggregate_raises_on_non_array_response(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = '{"error": "too many damages"}'
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[make_damage("d1", VehicleZone.hood, "img_001")]]
    with pytest.raises(ValueError, match="Expected JSON array"):
        aggregator.aggregate(damage_lists)


def test_aggregate_empty_sublists_skips_gemini(mock_gemini_client):
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    result = aggregator.aggregate([[]])
    assert result == []
    mock_gemini_client.models.generate_content.assert_not_called()


def test_aggregate_raises_on_none_response(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = None
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[make_damage("d1", VehicleZone.hood, "img_001")]]
    with pytest.raises(ValueError, match="empty response"):
        aggregator.aggregate(damage_lists)
