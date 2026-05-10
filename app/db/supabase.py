from __future__ import annotations
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from supabase import Client


class SupabaseVisionRepository:
    def __init__(self, client: Client, session_ttl_hours: int) -> None:
        self._db = client
        self._session_ttl_hours = session_ttl_hours

    # --- sessions ---

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
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=self._session_ttl_hours
        )
        result = (
            self._db.table("vision_sessions")
            .insert({
                "api_key_hash": api_key_hash,
                "vehicle_context": vehicle_context,
                "tenant_id": tenant_id,
                "inspection_id": inspection_id,
                "capture_session_id": capture_session_id,
                "vehicle_id": vehicle_id,
                "mode": mode or "lab",
                "expires_at": expires_at.isoformat(),
            })
            .execute()
        )
        return result.data[0]

    def get_session(self, session_id: str) -> dict | None:
        result = (
            self._db.table("vision_sessions")
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data if result is not None else None

    # --- session images ---

    def create_session_image(
        self,
        session_id: str,
        image_url: str,
        angle: str | None,
        inspection_media_asset_id: str | None = None,
        inspection_item_id: str | None = None,
    ) -> dict:
        result = (
            self._db.table("vision_session_images")
            .insert({
                "session_id": session_id,
                "image_url": image_url,
                "angle": angle,
                "inspection_media_asset_id": inspection_media_asset_id,
                "inspection_item_id": inspection_item_id,
                "status": "pending",
            })
            .execute()
        )
        return result.data[0]

    def update_image_analyzing(
        self, image_id: str, width: int, height: int
    ) -> None:
        self._db.table("vision_session_images").update({
            "status": "analyzing",
            "image_width": width,
            "image_height": height,
        }).eq("id", image_id).execute()

    def update_image_completed(
        self,
        image_id: str,
        damages: list[dict],
        gemini_call_id: str,
    ) -> None:
        self._db.table("vision_session_images").update({
            "status": "completed",
            "damages": damages,
            "gemini_call_id": gemini_call_id,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", image_id).execute()

    def update_image_failed(self, image_id: str, error: str) -> None:
        self._db.table("vision_session_images").update({
            "status": "failed",
            "error": error,
        }).eq("id", image_id).execute()

    def get_completed_images(self, session_id: str) -> list[dict]:
        result = (
            self._db.table("vision_session_images")
            .select("*")
            .eq("session_id", session_id)
            .eq("status", "completed")
            .order("uploaded_at")
            .execute()
        )
        return result.data

    def get_all_images(self, session_id: str) -> list[dict]:
        result = (
            self._db.table("vision_session_images")
            .select("id, status, error")
            .eq("session_id", session_id)
            .execute()
        )
        return result.data

    # --- damage maps ---

    def get_damage_map(self, session_id: str) -> dict | None:
        result = (
            self._db.table("vision_damage_maps")
            .select("*")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data if result is not None else None

    def upsert_damage_map(
        self,
        session_id: str,
        images: dict,
        zones: dict,
        summary: dict,
        image_count: int,
    ) -> dict:
        result = (
            self._db.table("vision_damage_maps")
            .upsert({
                "session_id": session_id,
                "images": images,
                "zones": zones,
                "summary": summary,
                "image_count": image_count,
                "built_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="session_id")
            .execute()
        )
        return result.data[0]

    # --- analysis calls ---

    def create_analysis_call(
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
    ) -> str:
        result = (
            self._db.table("vision_analysis_calls")
            .insert({
                "call_type": call_type,
                "model": model,
                "latency_ms": latency_ms,
                "status": status,
                "raw_response": raw_response,
                "session_id": session_id,
                "image_id": image_id,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "error": error,
            })
            .execute()
        )
        return result.data[0]["id"]

    def create_ai_usage_event(
        self,
        call_id: str,
        call_type: str,
        model: str,
        latency_ms: int,
        status: str,
        session_id: str | None = None,
        image_id: str | None = None,
        prompt_tokens: int | None = None,
        response_tokens: int | None = None,
        error: str | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        pricing = self._find_model_pricing(
            provider_key="google",
            model=model,
            api_version=None,
            created_at=created_at,
        )
        estimated_cost = self._estimate_cost(
            pricing=pricing,
            input_tokens=prompt_tokens,
            output_tokens=response_tokens,
        )
        phase_key = (
            "vision_consolidation"
            if call_type == "aggregate_damages"
            else "vision_image_analysis"
        )
        outcome = self._map_usage_outcome(status)

        self._db.table("ai_usage_events").insert({
            "vision_session_id": session_id,
            "vision_image_id": image_id,
            "source_table": "vision_analysis_calls",
            "source_id": call_id,
            "service_key": "crashguard-vision",
            "phase_key": phase_key,
            "provider_key": "google",
            "model": model,
            "api_version": None,
            "operation_key": call_type,
            "input_tokens": prompt_tokens,
            "output_tokens": response_tokens,
            "latency_ms": latency_ms,
            "attempt": 1,
            "outcome": outcome,
            "error_message": error,
            "estimated_cost_usd": estimated_cost,
            "pricing_snapshot_json": (
                self._build_pricing_snapshot(pricing)
                if pricing is not None
                else {"status": "missing_pricing"}
            ),
            "response_metadata_json": {
                "legacy_status": status,
            },
            "created_at": created_at,
        }).execute()

    def _find_model_pricing(
        self,
        provider_key: str,
        model: str,
        api_version: str | None,
        created_at: str,
    ) -> dict | None:
        for candidate_api_version in (api_version, None):
            query = (
                self._db.table("ai_model_pricing")
                .select("*")
                .eq("provider_key", provider_key)
                .eq("model", model)
                .lte("effective_from", created_at)
                .order("effective_from", desc=True)
                .limit(10)
            )

            if candidate_api_version is None:
                query = query.is_("api_version", "null")
            else:
                query = query.eq("api_version", candidate_api_version)

            result = query.execute()
            for row in result.data or []:
                effective_to = row.get("effective_to")
                if effective_to is None or effective_to > created_at:
                    return row

        return None

    def _estimate_cost(
        self,
        pricing: dict | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> str | None:
        if pricing is None:
            return None

        input_rate = Decimal(str(pricing.get("input_usd_per_1m") or "0"))
        output_rate = Decimal(str(pricing.get("output_usd_per_1m") or "0"))
        cost = (
            (Decimal(input_tokens or 0) / Decimal(1_000_000)) * input_rate
            + (Decimal(output_tokens or 0) / Decimal(1_000_000)) * output_rate
        )
        return str(cost.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))

    def _build_pricing_snapshot(self, pricing: dict) -> dict:
        return {
            "pricing_id": pricing.get("id"),
            "provider_key": pricing.get("provider_key"),
            "model": pricing.get("model"),
            "api_version": pricing.get("api_version"),
            "currency": pricing.get("currency"),
            "input_usd_per_1m": pricing.get("input_usd_per_1m"),
            "output_usd_per_1m": pricing.get("output_usd_per_1m"),
            "cached_input_usd_per_1m": pricing.get("cached_input_usd_per_1m"),
            "effective_from": pricing.get("effective_from"),
            "effective_to": pricing.get("effective_to"),
        }

    def _map_usage_outcome(self, status: str) -> str:
        if status == "success":
            return "success"
        if status == "timeout":
            return "timeout"
        return "failed"
