from __future__ import annotations
import time
from app.domain.ports import (
    DamageAggregator, DamageMapBuilder, AnalysisTracer,
    VisionRepository, ImageAnalyzer,
)
from app.domain.models import (
    Damage, SourceImageMeta, VehicleContext, DamageMap,
    DamageMapSummary, SkippedImage, VehicleZone,
)


class SessionService:
    def __init__(
        self,
        repo: VisionRepository,
        analyzer: ImageAnalyzer,
        aggregator: DamageAggregator,
        builder: DamageMapBuilder,
        tracer: AnalysisTracer,
        model_name: str,
    ) -> None:
        self._repo = repo
        self._analyzer = analyzer
        self._aggregator = aggregator
        self._builder = builder
        self._tracer = tracer
        self._model_name = model_name

    def create_session(self, api_key_hash: str, vehicle_context: dict | None) -> dict:
        return self._repo.create_session(
            api_key_hash=api_key_hash,
            vehicle_context=vehicle_context,
        )

    def get_session(self, session_id: str, api_key_hash: str) -> dict | None:
        session = self._repo.get_session(session_id)
        if session is None:
            return None
        if session["api_key_hash"] != api_key_hash:
            raise PermissionError("Session belongs to another tenant")
        return session

    def add_image(
        self,
        session_id: str,
        api_key_hash: str,
        image_url: str,
        angle: str | None,
    ) -> tuple[dict, list[Damage], int, int]:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")

        image_row = self._repo.create_session_image(
            session_id=session_id,
            image_url=image_url,
            angle=angle,
        )
        image_id = image_row["id"]

        try:
            context_dict = session.get("vehicle_context")
            context = VehicleContext(**context_dict) if context_dict else None

            t0 = time.monotonic()
            damages, width, height = self._analyzer.analyze_with_dimensions(
                image_url=image_url,
                context=context,
                source_image_id=image_id,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            self._repo.update_image_analyzing(image_id, width, height)

            call_id = self._tracer.record(
                call_type="analyze_image",
                model=self._model_name,
                latency_ms=latency_ms,
                status="success",
                raw_response={},
                session_id=session_id,
                image_id=image_id,
            )

            damages_dicts = [d.model_dump() for d in damages]
            self._repo.update_image_completed(image_id, damages_dicts, call_id)

            updated_row = {
                **image_row,
                "status": "completed",
                "image_width": width,
                "image_height": height,
            }
            return updated_row, damages, width, height

        except Exception as exc:
            self._tracer.record(
                call_type="analyze_image",
                model=self._model_name,
                latency_ms=0,
                status="error",
                raw_response={},
                session_id=session_id,
                image_id=image_id,
                error=str(exc),
            )
            self._repo.update_image_failed(image_id, str(exc))
            error_row = {**image_row, "status": "failed", "error": str(exc)}
            return error_row, [], 0, 0

    def get_report(self, session_id: str, api_key_hash: str) -> DamageMap:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")

        all_images = self._repo.get_all_images(session_id)
        completed_images = self._repo.get_completed_images(session_id)
        cached_map = self._repo.get_damage_map(session_id)

        if cached_map and cached_map["image_count"] == len(completed_images):
            return DamageMap(
                session_id=session_id,
                vehicle_context=session.get("vehicle_context"),
                images={k: SourceImageMeta(**v) for k, v in cached_map["images"].items()},
                zones={
                    VehicleZone(k): [Damage(**d) for d in v]
                    for k, v in cached_map["zones"].items()
                },
                summary=DamageMapSummary(**cached_map["summary"]),
            )

        images_meta: dict[str, SourceImageMeta] = {
            row["id"]: SourceImageMeta(
                url=row["image_url"],
                width=row["image_width"],
                height=row["image_height"],
                angle=row.get("angle"),
            )
            for row in completed_images
        }

        damage_lists: list[list[Damage]] = []
        for row in completed_images:
            raw_damages = row.get("damages") or []
            damage_lists.append([Damage(**d) for d in raw_damages])

        t0 = time.monotonic()
        aggregated = self._aggregator.aggregate(damage_lists)
        latency_ms = int((time.monotonic() - t0) * 1000)

        self._tracer.record(
            call_type="aggregate_damages",
            model=self._model_name,
            latency_ms=latency_ms,
            status="success",
            raw_response={},
            session_id=session_id,
        )

        context_dict = session.get("vehicle_context")
        context = VehicleContext(**context_dict) if context_dict else None

        damage_map = self._builder.build(
            damages=aggregated,
            images=images_meta,
            session_id=session_id,
            vehicle_context=context,
        )

        completed_ids = {row["id"] for row in completed_images}
        skipped = [
            SkippedImage(image_id=img["id"], reason=img.get("status", "unknown"))
            for img in all_images
            if img["id"] not in completed_ids
        ]
        damage_map.summary.images_skipped = skipped
        damage_map.summary.total_images = len(all_images)

        self._repo.upsert_damage_map(
            session_id=session_id,
            images={k: v.model_dump() for k, v in images_meta.items()},
            zones={k.value: [d.model_dump() for d in v] for k, v in damage_map.zones.items()},
            summary=damage_map.summary.model_dump(),
            image_count=len(completed_images),
        )

        return damage_map
