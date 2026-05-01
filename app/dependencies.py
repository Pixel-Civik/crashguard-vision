from functools import lru_cache
from google import genai
from app.config import settings
from app.db.supabase import get_client, VisionRepository
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.adapters.damage_map_builder import PythonDamageMapBuilder
from app.adapters.db_tracer import DbAnalysisTracer
from app.services.analyze_service import AnalyzeService
from app.services.session_service import SessionService


@lru_cache
def _gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


@lru_cache
def _supabase_repo() -> VisionRepository:
    return VisionRepository(get_client())


def get_analyze_service() -> AnalyzeService:
    tracer = DbAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    return AnalyzeService(analyzer=analyzer, tracer=tracer)


def get_session_service() -> SessionService:
    tracer = DbAnalysisTracer(repo=_supabase_repo())
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    aggregator = GeminiDamageAggregator(client=_gemini_client(), model=settings.gemini_model)
    builder = PythonDamageMapBuilder()
    return SessionService(
        repo=_supabase_repo(),
        analyzer=analyzer,
        aggregator=aggregator,
        builder=builder,
        tracer=tracer,
    )
