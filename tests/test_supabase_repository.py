from unittest.mock import MagicMock
import pytest

from app.db.supabase import SupabaseVisionRepository


def _chain(result):
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.lte.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.is_.return_value = query
    query.maybe_single.return_value = query
    query.execute.return_value = result
    return query


def test_get_damage_map_returns_none_when_maybe_single_returns_none():
    db = MagicMock()
    db.table.return_value = _chain(None)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    assert repo.get_damage_map("session-id") is None


def test_get_session_returns_none_when_maybe_single_returns_none():
    db = MagicMock()
    db.table.return_value = _chain(None)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    assert repo.get_session("session-id") is None


def test_validate_inspection_image_link_matches_asset_item_and_tenant():
    result = MagicMock()
    result.data = [{"id": "asset-1"}]
    db = MagicMock()
    db.table.return_value = _chain(result)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    assert repo.validate_inspection_image_link(
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        inspection_media_asset_id="asset-1",
        inspection_item_id="item-1",
    )

    query = db.table.return_value
    query.select.assert_called_once_with("id")
    query.eq.assert_any_call("id", "asset-1")
    query.eq.assert_any_call("tenant_id", "tenant-1")
    query.eq.assert_any_call("inspection_id", "inspection-1")
    query.eq.assert_any_call("inspection_item_id", "item-1")


def test_validate_inspection_image_link_rejects_missing_match():
    result = MagicMock()
    result.data = []
    db = MagicMock()
    db.table.return_value = _chain(result)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    assert not repo.validate_inspection_image_link(
        tenant_id="tenant-1",
        inspection_id="inspection-1",
        inspection_media_asset_id="asset-1",
        inspection_item_id="item-1",
    )


@pytest.mark.parametrize("method", [
    "get_session_image", "get_session_images", "get_completed_images",
    "find_session_image_for_inspection_asset",
])
def test_image_reads_load_usage_from_the_linked_call(method):
    row = {
        "id": "image-1", "gemini_call_id": "current-call",
        "analysis_usage": {"latency_ms": 140, "prompt_tokens": 110, "response_tokens": 80},
    }
    result = MagicMock()
    result.data = row if method == "get_session_image" else [row]
    query = _chain(result)
    db = MagicMock()
    db.table.return_value = query
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    if method == "find_session_image_for_inspection_asset":
        actual = repo.find_session_image_for_inspection_asset("session", "asset", "item")
    else:
        actual = getattr(repo, method)("id")
    image = actual[0] if isinstance(actual, list) else actual

    assert image["processing_ms"] == 140
    assert image["prompt_tokens"] == 110
    assert image["response_tokens"] == 80
    assert "analysis_usage" not in image
    query.select.assert_called_once_with(
        "*, analysis_usage:vision_analysis_calls!gemini_call_id(latency_ms,prompt_tokens,response_tokens)"
    )


def test_missing_image_call_usage_is_unknown_not_fabricated_zero():
    result = MagicMock()
    result.data = {"id": "legacy-image", "analysis_usage": None}
    db = MagicMock()
    db.table.return_value = _chain(result)
    row = SupabaseVisionRepository(db, 24).get_session_image("legacy-image")
    assert row["prompt_tokens"] is None
    assert row["response_tokens"] is None
    assert row["processing_ms"] is None


def test_cached_aggregation_usage_is_scoped_to_the_cache_build_time():
    result = MagicMock()
    result.data = [{"prompt_tokens": 30, "response_tokens": 20}]
    db = MagicMock()
    db.table.return_value = _chain(result)
    repo = SupabaseVisionRepository(db, 24)
    assert repo.get_cached_aggregation_usage("session", "2026-11-12T20:00:00Z") == result.data[0]
    query = db.table.return_value
    query.eq.assert_any_call("session_id", "session")
    query.eq.assert_any_call("call_type", "aggregate_damages")
    query.eq.assert_any_call("status", "success")
    query.lte.assert_called_once_with("created_at", "2026-11-12T20:00:00Z")
    query.order.assert_called_once_with("created_at", desc=True)
    query.limit.assert_called_once_with(1)


def test_create_ai_usage_event_writes_canonical_usage_with_pricing():
    pricing_result = MagicMock()
    pricing_result.data = [
        {
            "id": "pricing-1",
            "provider_key": "google",
            "model": "gemini-3-pro-preview",
            "api_version": None,
            "currency": "USD",
            "input_usd_per_1m": "2",
            "output_usd_per_1m": "12",
            "cached_input_usd_per_1m": None,
            "effective_from": "2026-05-01T00:00:00+00:00",
            "effective_to": None,
        }
    ]
    insert_result = MagicMock()
    insert_result.data = [{"id": "usage-1"}]
    pricing_chain = _chain(pricing_result)
    session_result = MagicMock()
    session_result.data = {
        "id": "session-1",
        "tenant_id": "tenant-1",
        "inspection_id": "inspection-1",
        "capture_session_id": "capture-1",
        "vehicle_id": "vehicle-1",
        "mode": "inspection_damage_report",
    }
    session_chain = _chain(session_result)
    usage_table = MagicMock()
    usage_table.insert.return_value = _chain(insert_result)
    db = MagicMock()
    db.table.side_effect = lambda table: {
        "ai_model_pricing": pricing_chain,
        "vision_sessions": session_chain,
        "ai_usage_events": usage_table,
    }.get(table, usage_table)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    repo.create_ai_usage_event(
        call_id="call-1",
        call_type="aggregate_damages",
        model="gemini-3-pro-preview",
        latency_ms=1500,
        status="success",
        session_id="session-1",
        prompt_tokens=1000,
        response_tokens=2000,
        request_metadata={"prompt_version": "vehicle-damage-consolidation-v2"},
    )

    usage_table.insert.assert_called_once()
    payload = usage_table.insert.call_args.args[0]
    assert payload["phase_key"] == "vision_consolidation"
    assert payload["tenant_id"] == "tenant-1"
    assert payload["inspection_id"] == "inspection-1"
    assert payload["capture_session_id"] == "capture-1"
    assert payload["request_metadata_json"]["vision_mode"] == "inspection_damage_report"
    assert (
        payload["request_metadata_json"]["prompt_version"]
        == "vehicle-damage-consolidation-v2"
    )
    assert payload["response_metadata_json"]["output_tokens_include_thinking"] is True
    assert payload["request_metadata_json"]["vehicle_id"] == "vehicle-1"
    assert payload["source_table"] == "vision_analysis_calls"
    assert payload["source_id"] == "call-1"
    assert payload["estimated_cost_usd"] == "0.02600000"
    assert payload["pricing_snapshot_json"]["pricing_id"] == "pricing-1"


def test_create_ai_usage_event_uses_missing_pricing_snapshot():
    pricing_result = MagicMock()
    pricing_result.data = []
    insert_result = MagicMock()
    insert_result.data = [{"id": "usage-1"}]
    pricing_chain = _chain(pricing_result)
    image_result = MagicMock()
    image_result.data = {
        "id": "image-1",
        "inspection_media_asset_id": "asset-1",
        "inspection_item_id": "item-1",
    }
    image_chain = _chain(image_result)
    usage_table = MagicMock()
    usage_table.insert.return_value = _chain(insert_result)
    db = MagicMock()
    db.table.side_effect = lambda table: {
        "ai_model_pricing": pricing_chain,
        "vision_session_images": image_chain,
        "ai_usage_events": usage_table,
    }.get(table, usage_table)
    repo = SupabaseVisionRepository(client=db, session_ttl_hours=24)

    repo.create_ai_usage_event(
        call_id="call-1",
        call_type="analyze_image",
        model="gemini-3-flash-preview",
        latency_ms=500,
        status="error",
        image_id="image-1",
        error="bad image",
    )

    payload = usage_table.insert.call_args.args[0]
    assert payload["phase_key"] == "vision_image_analysis"
    assert payload["media_asset_id"] == "asset-1"
    assert payload["request_metadata_json"]["inspection_item_id"] == "item-1"
    assert payload["outcome"] == "failed"
    assert payload["estimated_cost_usd"] is None
    assert payload["pricing_snapshot_json"] == {"status": "missing_pricing"}
