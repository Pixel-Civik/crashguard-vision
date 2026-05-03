"""Composition Root — the only module that knows about config and concrete adapters."""

from functools import lru_cache
from google import genai
from supabase import create_client

from app.config import settings
from app.db.supabase import SupabaseVisionRepository
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.adapters.damage_map_builder import StandardDamageMapBuilder
from app.adapters.db_tracer import SupabaseAnalysisTracer
from app.services.analyze_service import AnalyzeService
from app.services.session_service import SessionService


@lru_cache
def _gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


@lru_cache
def _supabase_repo() -> SupabaseVisionRepository:
    client = create_client(settings.supabase_url, settings.supabase_service_key)
    return SupabaseVisionRepository(
        client=client,
        session_ttl_hours=settings.session_ttl_hours,
    )


def get_analyze_service() -> AnalyzeService:
    tracer = SupabaseAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    return AnalyzeService(
        analyzer=analyzer,
        tracer=tracer,
        model_name=settings.gemini_model,
    )


def get_session_service() -> SessionService:
    tracer = SupabaseAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    aggregator = GeminiDamageAggregator(client=_gemini_client(), model=settings.gemini_model)
    builder = StandardDamageMapBuilder()
    return SessionService(
        repo=_supabase_repo(),
        analyzer=analyzer,
        aggregator=aggregator,
        builder=builder,
        tracer=tracer,
        model_name=settings.gemini_model,
    )
