from typing import Protocol
from app.domain.models import (
    Damage, DamageMap, SourceImageMeta, VehicleContext,
)


class ImageAnalyzer(Protocol):
    def analyze(
        self, image_url: str, context: VehicleContext | None
    ) -> list[Damage]: ...


class DamageAggregator(Protocol):
    def aggregate(self, damage_lists: list[list[Damage]]) -> list[Damage]: ...


class DamageMapBuilder(Protocol):
    def build(
        self,
        damages: list[Damage],
        images: dict[str, SourceImageMeta],
        session_id: str,
        vehicle_context: VehicleContext | None,
    ) -> DamageMap: ...


class AnalysisTracer(Protocol):
    def record(
        self,
        call_type: str,
        model: str,
        latency_ms: int,
        status: str,
        raw_response: dict,
        session_id: str | None = None,
        image_id: str | None = None,
        prompt_tokens: int | None = None,
        response_tokens: int | None = None,
        error: str | None = None,
    ) -> str: ...
