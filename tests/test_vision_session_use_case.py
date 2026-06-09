from unittest.mock import MagicMock

import pytest

from app.application.use_cases.vision_sessions import VisionSessionUseCase
from app.domain.models import BoundingBox, Damage, DamageType, Severity, VehicleZone
from app.domain.services import DamageMapFactory


def _use_case(repo: MagicMock) -> VisionSessionUseCase:
    return VisionSessionUseCase(
        repo=repo,
        analyzer=MagicMock(),
        aggregator=MagicMock(),
        damage_map_builder=MagicMock(),
        tracer=MagicMock(),
        model_name="gemini-test",
    )


def test_create_inspection_session_reuses_active_session():
    repo = MagicMock()
    existing = {
        "id": "session-existing",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
        "mode": "inspection_damage_report",
        "expires_at": "2026-06-02T20:00:00+00:00",
    }
    repo.find_active_inspection_session.return_value = existing
    use_case = _use_case(repo)

    result = use_case.create_session(
        api_key_hash="hash-1",
        vehicle_context={"make": "Toyota"},
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        capture_session_id="capture-1",
        vehicle_id="vehicle-1",
        mode="inspection_damage_report",
    )

    assert result == existing
    repo.find_active_inspection_session.assert_called_once_with(
        api_key_hash="hash-1",
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        mode="inspection_damage_report",
    )
    repo.create_session.assert_not_called()


def test_create_lab_inspection_session_reuses_only_lab_mode():
    repo = MagicMock()
    existing = {
        "id": "lab-session-existing",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
        "mode": "vision_lab_inspection",
        "expires_at": "2026-06-02T20:00:00+00:00",
    }
    repo.find_active_inspection_session.return_value = existing
    use_case = _use_case(repo)

    result = use_case.create_session(
        api_key_hash="hash-1",
        vehicle_context={"make": "Toyota"},
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        capture_session_id="capture-1",
        vehicle_id="vehicle-1",
        mode="vision_lab_inspection",
    )

    assert result == existing
    repo.find_active_inspection_session.assert_called_once_with(
        api_key_hash="hash-1",
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        mode="vision_lab_inspection",
    )
    repo.create_session.assert_not_called()


def test_create_inspection_session_creates_when_no_active_session():
    repo = MagicMock()
    created = {
        "id": "session-new",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
        "mode": "inspection_damage_report",
        "expires_at": "2026-06-02T20:00:00+00:00",
    }
    repo.find_active_inspection_session.return_value = None
    repo.create_session.return_value = created
    use_case = _use_case(repo)

    result = use_case.create_session(
        api_key_hash="hash-1",
        vehicle_context={"make": "Toyota"},
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        capture_session_id="capture-1",
        vehicle_id="vehicle-1",
        mode="inspection_damage_report",
    )

    assert result == created
    repo.create_session.assert_called_once_with(
        api_key_hash="hash-1",
        vehicle_context={"make": "Toyota"},
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        capture_session_id="capture-1",
        vehicle_id="vehicle-1",
        mode="inspection_damage_report",
    )


def test_add_image_rejects_partial_inspection_metadata():
    repo = MagicMock()
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
    }
    use_case = _use_case(repo)

    with pytest.raises(ValueError, match="must include both"):
        use_case.add_image(
            session_id="session-1",
            api_key_hash="hash-1",
            image_url="https://example.com/car.jpg",
            angle="front",
            inspection_media_asset_id="asset-1",
        )

    repo.create_session_image.assert_not_called()


def test_add_image_rejects_inspection_metadata_on_lab_session():
    repo = MagicMock()
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "tenant_id": None,
        "inspection_id": None,
    }
    use_case = _use_case(repo)

    with pytest.raises(ValueError, match="inspection-linked session"):
        use_case.add_image(
            session_id="session-1",
            api_key_hash="hash-1",
            image_url="https://example.com/car.jpg",
            angle="front",
            inspection_media_asset_id="asset-1",
            inspection_item_id="item-1",
        )

    repo.create_session_image.assert_not_called()


def test_add_image_rejects_mismatched_inspection_metadata():
    repo = MagicMock()
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
    }
    repo.validate_inspection_image_link.return_value = False
    use_case = _use_case(repo)

    with pytest.raises(ValueError, match="does not match"):
        use_case.add_image(
            session_id="session-1",
            api_key_hash="hash-1",
            image_url="https://example.com/car.jpg",
            angle="front",
            inspection_media_asset_id="asset-1",
            inspection_item_id="item-1",
        )

    repo.validate_inspection_image_link.assert_called_once_with(
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        inspection_media_asset_id="asset-1",
        inspection_item_id="item-1",
    )
    repo.create_session_image.assert_not_called()


def test_add_image_reuses_existing_completed_inspection_image():
    repo = MagicMock()
    analyzer = MagicMock()
    tracer = MagicMock()
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
    }
    repo.validate_inspection_image_link.return_value = True
    repo.find_session_image_for_inspection_asset.return_value = {
        "id": "image-1",
        "session_id": "session-1",
        "image_url": "https://example.com/car.jpg",
        "status": "completed",
        "image_width": 100,
        "image_height": 80,
        "damages": [
            {
                "id": "dmg_01",
                "type": "dent",
                "zone": "hood",
                "severity": "low",
                "confidence": 0.82,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                "description": "small dent",
                "source_image_id": "image-1",
            }
        ],
    }
    use_case = VisionSessionUseCase(
        repo=repo,
        analyzer=analyzer,
        aggregator=MagicMock(),
        damage_map_builder=MagicMock(),
        tracer=tracer,
        model_name="gemini-test",
    )

    result = use_case.add_image(
        session_id="session-1",
        api_key_hash="hash-1",
        image_url="https://example.com/car.jpg",
        angle="front",
        inspection_media_asset_id="asset-1",
        inspection_item_id="item-1",
    )

    assert result.image_row["id"] == "image-1"
    assert result.image_row["status"] == "completed"
    assert result.image_width == 100
    assert result.image_height == 80
    assert len(result.damages) == 1
    repo.create_session_image.assert_not_called()
    analyzer.analyze_with_dimensions.assert_not_called()
    tracer.record.assert_not_called()


def test_add_image_reuses_existing_failed_inspection_image_without_duplicate():
    repo = MagicMock()
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
    }
    repo.validate_inspection_image_link.return_value = True
    repo.find_session_image_for_inspection_asset.return_value = {
        "id": "image-1",
        "session_id": "session-1",
        "image_url": "https://example.com/car.jpg",
        "status": "failed",
        "error": "503 UNAVAILABLE",
    }
    use_case = _use_case(repo)

    result = use_case.add_image(
        session_id="session-1",
        api_key_hash="hash-1",
        image_url="https://example.com/car.jpg",
        angle="front",
        inspection_media_asset_id="asset-1",
        inspection_item_id="item-1",
    )

    assert result.image_row["id"] == "image-1"
    assert result.image_row["status"] == "failed"
    assert result.image_row["error"] == "503 UNAVAILABLE"
    assert result.damages == []
    repo.create_session_image.assert_not_called()


def test_retry_image_reuses_existing_failed_image_with_fresh_url():
    repo = MagicMock()
    analyzer = MagicMock()
    tracer = MagicMock()
    damage = Damage(
        id="dmg_01",
        type=DamageType.dent,
        zone=VehicleZone.hood,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description="test",
        source_image_id="image-1",
    )
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "vehicle_context": None,
    }
    repo.get_session_image.return_value = {
        "id": "image-1",
        "session_id": "session-1",
        "image_url": "https://old.example.com/car.jpg",
        "status": "failed",
        "error": "503 UNAVAILABLE",
    }
    analyzer.analyze_with_dimensions.return_value = ([damage], 100, 80, 10, 5)
    tracer.record.return_value = "call-1"
    use_case = VisionSessionUseCase(
        repo=repo,
        analyzer=analyzer,
        aggregator=MagicMock(),
        damage_map_builder=MagicMock(),
        tracer=tracer,
        model_name="gemini-test",
    )

    result = use_case.retry_image(
        session_id="session-1",
        image_id="image-1",
        api_key_hash="hash-1",
        image_url="https://fresh.example.com/car.jpg",
    )

    assert result.image_row["status"] == "completed"
    assert result.damages == [damage]
    repo.create_session_image.assert_not_called()
    repo.update_image_url.assert_called_once_with("image-1", "https://fresh.example.com/car.jpg")
    analyzer.analyze_with_dimensions.assert_called_once_with(
        image_url="https://fresh.example.com/car.jpg",
        context=None,
        source_image_id="image-1",
    )
    repo.update_image_completed.assert_called_once()


def test_force_retry_reanalyzes_completed_image_and_invalidates_damage_map():
    repo = MagicMock()
    analyzer = MagicMock()
    tracer = MagicMock()
    damage = Damage(
        id="dmg_01",
        type=DamageType.dent,
        zone=VehicleZone.hood,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description="updated",
        source_image_id="image-1",
    )
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "vehicle_context": None,
    }
    repo.get_session_image.return_value = {
        "id": "image-1",
        "session_id": "session-1",
        "image_url": "https://old.example.com/car.jpg",
        "status": "completed",
        "damages": [],
    }
    analyzer.analyze_with_dimensions.return_value = ([damage], 120, 90, 11, 7)
    tracer.record.return_value = "call-1"
    use_case = VisionSessionUseCase(
        repo=repo,
        analyzer=analyzer,
        aggregator=MagicMock(),
        damage_map_builder=MagicMock(),
        tracer=tracer,
        model_name="gemini-test",
    )

    result = use_case.retry_image(
        session_id="session-1",
        image_id="image-1",
        api_key_hash="hash-1",
        image_url="https://fresh.example.com/car.jpg",
        force=True,
    )

    assert result.image_row["status"] == "completed"
    assert result.damages == [damage]
    repo.create_session_image.assert_not_called()
    repo.delete_damage_map.assert_called_once_with("session-1")
    repo.update_image_url.assert_called_once_with("image-1", "https://fresh.example.com/car.jpg")
    analyzer.analyze_with_dimensions.assert_called_once_with(
        image_url="https://fresh.example.com/car.jpg",
        context=None,
        source_image_id="image-1",
    )


def test_get_report_includes_aggregate_tokens_in_summary():
    repo = MagicMock()
    aggregator = MagicMock()
    tracer = MagicMock()
    damage = Damage(
        id="dmg_01",
        type=DamageType.dent,
        zone=VehicleZone.hood,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description="test",
        source_image_id="image-1",
    )
    repo.get_session.return_value = {
        "id": "session-1",
        "api_key_hash": "hash-1",
        "vehicle_context": None,
    }
    repo.get_all_images.return_value = [
        {
            "id": "image-1",
            "session_id": "session-1",
            "image_url": "https://example.com/car.jpg",
            "status": "completed",
            "image_width": 100,
            "image_height": 80,
            "damages": [damage.model_dump()],
            "processing_ms": 120,
            "prompt_tokens": 10,
            "response_tokens": 5,
        }
    ]
    repo.get_completed_images.return_value = repo.get_all_images.return_value
    repo.get_damage_map.return_value = None
    aggregator.aggregate.return_value = [damage]
    aggregator.get_last_usage.return_value = (30, 20)

    use_case = VisionSessionUseCase(
        repo=repo,
        analyzer=MagicMock(),
        aggregator=aggregator,
        damage_map_builder=DamageMapFactory(),
        tracer=tracer,
        model_name="gemini-test",
    )

    report = use_case.get_report("session-1", "hash-1")

    assert report.summary.total_prompt_tokens == 40
    assert report.summary.total_response_tokens == 25
    tracer.record.assert_called_once()
    assert tracer.record.call_args.kwargs["call_type"] == "aggregate_damages"
    assert tracer.record.call_args.kwargs["prompt_tokens"] == 30
    assert tracer.record.call_args.kwargs["response_tokens"] == 20
