from unittest.mock import MagicMock

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
    usage_table = MagicMock()
    usage_table.insert.return_value = _chain(insert_result)
    db = MagicMock()
    db.table.side_effect = lambda table: (
        pricing_chain if table == "ai_model_pricing" else usage_table
    )
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
    )

    usage_table.insert.assert_called_once()
    payload = usage_table.insert.call_args.args[0]
    assert payload["phase_key"] == "vision_consolidation"
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
    usage_table = MagicMock()
    usage_table.insert.return_value = _chain(insert_result)
    db = MagicMock()
    db.table.side_effect = lambda table: (
        pricing_chain if table == "ai_model_pricing" else usage_table
    )
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
    assert payload["outcome"] == "failed"
    assert payload["estimated_cost_usd"] is None
    assert payload["pricing_snapshot_json"] == {"status": "missing_pricing"}
