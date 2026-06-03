from __future__ import annotations

import time

from app.application.dto import SessionImageAnalysisResult
from app.domain.models import (
    Damage,
    DamageMap,
    DamageMapSummary,
    SkippedImage,
    SourceImageMeta,
    VehicleContext,
    VehicleZone,
)
from app.domain.ports import (
    AnalysisTracer,
    DamageAggregator,
    DamageMapBuilder,
    ImageAnalyzer,
    VisionRepository,
)


class VisionSessionUseCase:
    def __init__(
        self,
        repo: VisionRepository,
        analyzer: ImageAnalyzer,
        aggregator: DamageAggregator,
        damage_map_builder: DamageMapBuilder,
        tracer: AnalysisTracer,
        model_name: str,
    ) -> None:
        self._repo = repo
        self._analyzer = analyzer
        self._aggregator = aggregator
        self._damage_map_builder = damage_map_builder
        self._tracer = tracer
        self._model_name = model_name

    def create_session(
        self,
        api_key_hash: str,
        vehicle_context: dict | None,
        tenant_id: str | None = None,
        inspection_id: str | None = None,
        capture_session_id: str | None = None,
        vehicle_id: str | None = None,
        mode: str | None = None,
    ) -> dict:
        normalized_mode = mode or "lab"
        if tenant_id and inspection_id and normalized_mode == "inspection_damage_report":
            existing = self._repo.find_active_inspection_session(
                api_key_hash=api_key_hash,
                tenant_id=tenant_id,
                inspection_id=inspection_id,
                mode=normalized_mode,
            )
            if existing is not None:
                return existing

        return self._repo.create_session(
            api_key_hash=api_key_hash,
            vehicle_context=vehicle_context,
            tenant_id=tenant_id,
            inspection_id=inspection_id,
            capture_session_id=capture_session_id,
            vehicle_id=vehicle_id,
            mode=normalized_mode,
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
        inspection_media_asset_id: str | None = None,
        inspection_item_id: str | None = None,
    ) -> SessionImageAnalysisResult:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")
        self._validate_inspection_metadata(
            session=session,
            inspection_media_asset_id=inspection_media_asset_id,
            inspection_item_id=inspection_item_id,
        )
        if inspection_media_asset_id and inspection_item_id:
            existing = self._repo.find_session_image_for_inspection_asset(
                session_id=session_id,
                inspection_media_asset_id=inspection_media_asset_id,
                inspection_item_id=inspection_item_id,
            )
            if isinstance(existing, dict):
                return self._result_from_stored_image(existing)

        image_row = self._repo.create_session_image(
            session_id=session_id,
            image_url=image_url,
            angle=angle,
            inspection_media_asset_id=inspection_media_asset_id,
            inspection_item_id=inspection_item_id,
        )
        image_id = image_row["id"]

        try:
            context_dict = session.get("vehicle_context")
            context = VehicleContext(**context_dict) if context_dict else None

            t0 = time.monotonic()
            damages, width, height, p_tokens, r_tokens = self._analyzer.analyze_with_dimensions(
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
                prompt_tokens=p_tokens,
                response_tokens=r_tokens,
            )
            self._repo.update_image_completed(
                image_id,
                [damage.model_dump() for damage in damages],
                call_id,
            )

            return SessionImageAnalysisResult(
                image_row={
                    **image_row,
                    "status": "completed",
                    "image_width": width,
                    "image_height": height,
                },
                damages=damages,
                image_width=width,
                image_height=height,
                processing_ms=latency_ms,
                prompt_tokens=p_tokens,
                response_tokens=r_tokens,
            )
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
            return SessionImageAnalysisResult(
                image_row={**image_row, "status": "failed", "error": str(exc)},
                damages=[],
                image_width=0,
                image_height=0,
            )

    def list_images(self, session_id: str, api_key_hash: str) -> list[dict]:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")
        return self._dedupe_session_image_rows(self._repo.get_session_images(session_id))

    def retry_image(
        self,
        session_id: str,
        image_id: str,
        api_key_hash: str,
        image_url: str | None = None,
    ) -> SessionImageAnalysisResult:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")

        image_row = self._repo.get_session_image(image_id)
        if image_row is None or image_row.get("session_id") != session_id:
            raise ValueError("Image not found")
        if image_row.get("status") != "failed":
            raise ValueError("Only failed images can be retried")

        if image_url:
            self._repo.update_image_url(image_id, image_url)
            image_row = {**image_row, "image_url": image_url}

        return self._analyze_existing_image(session, image_row)

    def get_report(self, session_id: str, api_key_hash: str) -> DamageMap:
        session = self.get_session(session_id, api_key_hash)
        if session is None:
            raise ValueError("Session not found")

        all_images = self._dedupe_session_image_rows(self._repo.get_all_images(session_id))
        completed_images = self._dedupe_session_image_rows(
            self._repo.get_completed_images(session_id)
        )
        cached_map = self._repo.get_damage_map(session_id)

        if cached_map:
            try:
                if cached_map.get("image_count") == len(completed_images):
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
            except Exception:
                pass  # Fallback to re-aggregation if cache schema is incompatible

        images_meta = {
            row["id"]: SourceImageMeta(
                url=row["image_url"],
                width=row["image_width"],
                height=row["image_height"],
                angle=row.get("angle"),
            )
            for row in completed_images
        }
        damage_lists = [
            [Damage(**damage) for damage in row.get("damages") or []]
            for row in completed_images
        ]

        t0 = time.monotonic()
        aggregated = self._aggregator.aggregate(damage_lists)
        latency_ms = int((time.monotonic() - t0) * 1000)
        aggregate_prompt_tokens = None
        aggregate_response_tokens = None
        get_aggregate_usage = getattr(self._aggregator, "get_last_usage", None)
        if callable(get_aggregate_usage):
            aggregate_prompt_tokens, aggregate_response_tokens = get_aggregate_usage()
        self._tracer.record(
            call_type="aggregate_damages",
            model=self._model_name,
            latency_ms=latency_ms,
            status="success",
            raw_response={},
            session_id=session_id,
            prompt_tokens=aggregate_prompt_tokens,
            response_tokens=aggregate_response_tokens,
        )

        context_dict = session.get("vehicle_context")
        context = VehicleContext(**context_dict) if context_dict else None

        total_processing_ms = sum(row.get("processing_ms") or 0 for row in completed_images)
        total_prompt_tokens = sum(row.get("prompt_tokens") or 0 for row in completed_images)
        total_response_tokens = sum(row.get("response_tokens") or 0 for row in completed_images)

        damage_map = self._damage_map_builder.build(
            damages=aggregated,
            images=images_meta,
            session_id=session_id,
            vehicle_context=context,
            total_processing_ms=total_processing_ms,
            total_prompt_tokens=total_prompt_tokens,
            total_response_tokens=total_response_tokens,
        )

        completed_ids = {row["id"] for row in completed_images}
        damage_map.summary.images_skipped = [
            SkippedImage(image_id=image["id"], reason=image.get("status", "unknown"))
            for image in all_images
            if image["id"] not in completed_ids
        ]
        damage_map.summary.total_images = len(all_images)

        self._repo.upsert_damage_map(
            session_id=session_id,
            images={k: v.model_dump() for k, v in images_meta.items()},
            zones={k.value: [d.model_dump() for d in v] for k, v in damage_map.zones.items()},
            summary=damage_map.summary.model_dump(),
            image_count=len(completed_images),
        )
        return damage_map

    def _validate_inspection_metadata(
        self,
        session: dict,
        inspection_media_asset_id: str | None,
        inspection_item_id: str | None,
    ) -> None:
        if inspection_media_asset_id is None and inspection_item_id is None:
            return
        if not inspection_media_asset_id or not inspection_item_id:
            raise ValueError("Inspection image metadata must include both media asset and item ids")

        tenant_id = session.get("tenant_id")
        inspection_id = session.get("inspection_id")
        if not tenant_id or not inspection_id:
            raise ValueError("Inspection image metadata requires an inspection-linked session")

        if not self._repo.validate_inspection_image_link(
            tenant_id=tenant_id,
            inspection_id=inspection_id,
            inspection_media_asset_id=inspection_media_asset_id,
            inspection_item_id=inspection_item_id,
        ):
            raise ValueError("Inspection image metadata does not match the Vision session")

    def _dedupe_session_image_rows(self, rows: list[dict]) -> list[dict]:
        by_key: dict[str, dict] = {}
        passthrough: list[dict] = []
        for row in rows:
            key = self._session_image_dedupe_key(row)
            if key is None:
                passthrough.append(row)
                continue
            current = by_key.get(key)
            if current is None or self._should_replace_session_image_row(current, row):
                by_key[key] = row
        return [*by_key.values(), *passthrough]

    def _session_image_dedupe_key(self, row: dict) -> str | None:
        media_asset_id = row.get("inspection_media_asset_id")
        inspection_item_id = row.get("inspection_item_id")
        if media_asset_id:
            return f"asset:{media_asset_id}"
        if inspection_item_id:
            return f"item:{inspection_item_id}"
        return None

    def _should_replace_session_image_row(self, current: dict, candidate: dict) -> bool:
        current_rank = self._session_image_status_rank(current.get("status"))
        candidate_rank = self._session_image_status_rank(candidate.get("status"))
        if candidate_rank != current_rank:
            return candidate_rank > current_rank
        current_time = str(current.get("analyzed_at") or current.get("uploaded_at") or "")
        candidate_time = str(candidate.get("analyzed_at") or candidate.get("uploaded_at") or "")
        return candidate_time > current_time

    def _session_image_status_rank(self, status: str | None) -> int:
        if status == "completed":
            return 4
        if status == "analyzing":
            return 3
        if status == "pending":
            return 2
        if status == "failed":
            return 1
        return 0

    def _result_from_stored_image(self, image_row: dict) -> SessionImageAnalysisResult:
        damages = []
        if image_row.get("status") == "completed":
            for item in image_row.get("damages") or []:
                try:
                    damages.append(Damage(**item))
                except Exception:
                    continue

        return SessionImageAnalysisResult(
            image_row=image_row,
            damages=damages,
            image_width=image_row.get("image_width") or 0,
            image_height=image_row.get("image_height") or 0,
            processing_ms=0,
            prompt_tokens=0,
            response_tokens=0,
        )

    def _analyze_existing_image(
        self,
        session: dict,
        image_row: dict,
    ) -> SessionImageAnalysisResult:
        image_id = image_row["id"]
        image_url = image_row["image_url"]
        session_id = session["id"]

        try:
            context_dict = session.get("vehicle_context")
            context = VehicleContext(**context_dict) if context_dict else None

            t0 = time.monotonic()
            damages, width, height, p_tokens, r_tokens = self._analyzer.analyze_with_dimensions(
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
                prompt_tokens=p_tokens,
                response_tokens=r_tokens,
            )
            self._repo.update_image_completed(
                image_id,
                [damage.model_dump() for damage in damages],
                call_id,
            )

            return SessionImageAnalysisResult(
                image_row={
                    **image_row,
                    "status": "completed",
                    "error": None,
                    "image_width": width,
                    "image_height": height,
                },
                damages=damages,
                image_width=width,
                image_height=height,
                processing_ms=latency_ms,
                prompt_tokens=p_tokens,
                response_tokens=r_tokens,
            )
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
            return SessionImageAnalysisResult(
                image_row={**image_row, "status": "failed", "error": str(exc)},
                damages=[],
                image_width=0,
                image_height=0,
            )
