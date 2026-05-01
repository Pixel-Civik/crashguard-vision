from __future__ import annotations
import time
from app.domain.models import Damage, VehicleContext
from app.domain.ports import AnalysisTracer
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.config import settings


class AnalyzeService:
    def __init__(self, analyzer: GeminiImageAnalyzer, tracer: AnalysisTracer) -> None:
        self._analyzer = analyzer
        self._tracer = tracer

    def analyze(
        self,
        image_url: str,
        context: VehicleContext | None,
    ) -> tuple[list[Damage], int, int]:
        t0 = time.monotonic()
        try:
            damages, width, height = self._analyzer.analyze_with_dimensions(
                image_url=image_url,
                context=context,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=settings.gemini_model,
                latency_ms=latency_ms,
                status="success",
                raw_response={},
            )
            return damages, width, height
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=settings.gemini_model,
                latency_ms=latency_ms,
                status="error",
                raw_response={},
                error=str(exc),
            )
            raise
