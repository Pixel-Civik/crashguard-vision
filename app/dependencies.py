"""Composition Root — the only module that knows about config and concrete adapters."""

from functools import lru_cache
from google import genai
from supabase import create_client

from app.application.use_cases.analyze_image import AnalyzeImageUseCase
from app.application.use_cases.vision_sessions import VisionSessionUseCase
from app.config import settings
from app.adapters.outbound.gemini import GeminiDamageAggregator, GeminiImageAnalyzer
from app.adapters.outbound.supabase import DbAnalysisTracer, SupabaseVisionRepository
from app.domain.services import DamageMapFactory


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


def get_analyze_image_use_case() -> AnalyzeImageUseCase:
    tracer = DbAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    return AnalyzeImageUseCase(
        analyzer=analyzer,
        tracer=tracer,
        model_name=settings.gemini_model,
    )


def get_vision_session_use_case() -> VisionSessionUseCase:
    tracer = DbAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    aggregator = GeminiDamageAggregator(client=_gemini_client(), model=settings.gemini_model)
    damage_map_builder = DamageMapFactory()
    return VisionSessionUseCase(
        repo=_supabase_repo(),
        analyzer=analyzer,
        aggregator=aggregator,
        damage_map_builder=damage_map_builder,
        tracer=tracer,
        model_name=settings.gemini_model,
    )


get_analyze_service = get_analyze_image_use_case
get_session_service = get_vision_session_use_case
