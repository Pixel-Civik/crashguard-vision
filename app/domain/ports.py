from typing import Protocol
from app.domain.models import (
    Damage, DamageMap, SourceImageMeta, VehicleContext,
)


class ImageAnalyzer(Protocol):
    def analyze(
        self, image_url: str, context: VehicleContext | None
    ) -> list[Damage]: ...

    def analyze_with_dimensions(
        self,
        image_url: str,
        context: VehicleContext | None,
        source_image_id: str | None = None,
    ) -> tuple[list[Damage], int, int, int | None, int | None]: ...


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


class VisionRepository(Protocol):
    def create_session(
        self,
        api_key_hash: str,
        vehicle_context: dict | None,
        tenant_id: str | None = None,
        inspection_id: str | None = None,
        capture_session_id: str | None = None,
        vehicle_id: str | None = None,
        mode: str | None = None,
    ) -> dict: ...
    def get_session(self, session_id: str) -> dict | None: ...
    def create_session_image(
        self,
        session_id: str,
        image_url: str,
        angle: str | None,
        inspection_media_asset_id: str | None = None,
        inspection_item_id: str | None = None,
    ) -> dict: ...
    def update_image_analyzing(self, image_id: str, width: int, height: int) -> None: ...
    def update_image_completed(self, image_id: str, damages: list[dict], gemini_call_id: str) -> None: ...
    def update_image_failed(self, image_id: str, error: str) -> None: ...
    def get_completed_images(self, session_id: str) -> list[dict]: ...
    def get_all_images(self, session_id: str) -> list[dict]: ...
    def get_damage_map(self, session_id: str) -> dict | None: ...
    def upsert_damage_map(self, session_id: str, images: dict, zones: dict, summary: dict, image_count: int) -> dict: ...
    def create_analysis_call(
        self, call_type: str, model: str, latency_ms: int, status: str, raw_response: dict,
        session_id: str | None = None, image_id: str | None = None,
        prompt_tokens: int | None = None, response_tokens: int | None = None, error: str | None = None
    ) -> str: ...
    def create_ai_usage_event(
        self, call_id: str, call_type: str, model: str, latency_ms: int, status: str,
        session_id: str | None = None, image_id: str | None = None,
        prompt_tokens: int | None = None, response_tokens: int | None = None, error: str | None = None
    ) -> None: ...
