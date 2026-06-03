import pytest
from unittest.mock import MagicMock
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
)


def make_damage(
    id: str,
    zone: VehicleZone,
    image_id: str,
    *,
    damage_type: DamageType = DamageType.dent,
    severity: Severity = Severity.medium,
    description: str = "test",
) -> Damage:
    return Damage(
        id=id,
        type=damage_type,
        zone=zone,
        severity=severity,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description=description,
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
    usage.thoughts_token_count = 25
    usage.total_token_count = 305
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
    assert aggregator.get_last_usage() == (200, 105)


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


def test_aggregate_filters_false_positive_inputs_without_gemini(mock_gemini_client):
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[
        make_damage(
            "d1",
            VehicleZone.wheel_front_left,
            "img_001",
            damage_type=DamageType.other,
            severity=Severity.high,
            description="Severely flat or deflated tire on the front left wheel.",
        ),
        make_damage(
            "d2",
            VehicleZone.roof,
            "img_002",
            damage_type=DamageType.stain,
            severity=Severity.low,
            description="Small organic stain or bird dropping on the roof surface.",
        ),
        make_damage(
            "d3",
            VehicleZone.roof,
            "img_003",
            damage_type=DamageType.stain,
            severity=Severity.low,
            description="Small debris on the rear section of the roof.",
        ),
        make_damage(
            "d4",
            VehicleZone.front_bumper,
            "img_004",
            damage_type=DamageType.other,
            severity=Severity.low,
            description="Water on the ground near the front bumper.",
        ),
    ]]

    result = aggregator.aggregate(damage_lists)

    assert result == []
    assert aggregator.get_last_usage() == (None, None)
    mock_gemini_client.models.generate_content.assert_not_called()


def test_aggregate_filters_false_positive_outputs(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = (
        "["
        '{"id":"dmg_01","type":"dent","zone":"front_bumper","severity":"medium",'
        '"confidence":0.91,"bbox_x":0.12,"bbox_y":0.34,"bbox_w":0.18,"bbox_h":0.09,'
        '"description":"Dent on front bumper","source_image_id":"img_001","also_seen_in":[]},'
        '{"id":"dmg_02","type":"other","zone":"wheel_front_left","severity":"high",'
        '"confidence":0.99,"bbox_x":0.10,"bbox_y":0.30,"bbox_w":0.20,"bbox_h":0.20,'
        '"description":"Severely flat or deflated tire on the front left wheel.",'
        '"source_image_id":"img_002","also_seen_in":[]},'
        '{"id":"dmg_03","type":"stain","zone":"roof","severity":"low",'
        '"confidence":0.85,"bbox_x":0.20,"bbox_y":0.20,"bbox_w":0.10,"bbox_h":0.10,'
        '"description":"Small organic stain or bird dropping on the roof surface.",'
        '"source_image_id":"img_003","also_seen_in":[]}'
        "]"
    )
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[make_damage("d1", VehicleZone.front_bumper, "img_001")]]

    result = aggregator.aggregate(damage_lists)

    assert len(result) == 1
    assert result[0].id == "dmg_01"
    assert result[0].description == "Dent on front bumper"


def test_aggregate_raises_on_none_response(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = None
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [[make_damage("d1", VehicleZone.hood, "img_001")]]
    with pytest.raises(ValueError, match="empty response"):
        aggregator.aggregate(damage_lists)
