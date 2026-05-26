from unittest.mock import MagicMock

import pytest

from app.application.use_cases.vision_sessions import VisionSessionUseCase


def _use_case(repo: MagicMock) -> VisionSessionUseCase:
    return VisionSessionUseCase(
        repo=repo,
        analyzer=MagicMock(),
        aggregator=MagicMock(),
        damage_map_builder=MagicMock(),
        tracer=MagicMock(),
        model_name="gemini-test",
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
