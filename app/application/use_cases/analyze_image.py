from __future__ import annotations

import time

from app.application.dto import ImageAnalysisResult
from app.domain.models import VehicleContext
from app.domain.ports import AnalysisTracer, ImageAnalyzer


class AnalyzeImageUseCase:
    def __init__(
        self,
        analyzer: ImageAnalyzer,
        tracer: AnalysisTracer,
        model_name: str,
    ) -> None:
        self._analyzer = analyzer
        self._tracer = tracer
        self._model_name = model_name

    def execute(
        self,
        image_url: str,
        vehicle_context: VehicleContext | None,
    ) -> ImageAnalysisResult:
        t0 = time.monotonic()
        try:
            damages, width, height = self._analyzer.analyze_with_dimensions(
                image_url=image_url,
                context=vehicle_context,
            )
            processing_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=self._model_name,
                latency_ms=processing_ms,
                status="success",
                raw_response={},
            )
            return ImageAnalysisResult(
                damages=damages,
                image_width=width,
                image_height=height,
                processing_ms=processing_ms,
            )
        except Exception as exc:
            processing_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=self._model_name,
                latency_ms=processing_ms,
                status="error",
                raw_response={},
                error=str(exc),
            )
            raise
