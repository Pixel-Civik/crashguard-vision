from unittest.mock import MagicMock

from app.db.supabase import SupabaseVisionRepository


def _chain(result):
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
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
