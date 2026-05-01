from __future__ import annotations
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from app.config import settings


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


class VisionRepository:
    def __init__(self, client: Client) -> None:
        self._db = client

    # --- sessions ---

    def create_session(
        self,
        api_key_hash: str,
        vehicle_context: dict | None,
    ) -> dict:
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.session_ttl_hours
        )
        result = (
            self._db.table("vision_sessions")
            .insert({
                "api_key_hash": api_key_hash,
                "vehicle_context": vehicle_context,
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
        return result.data

    # --- session images ---

    def create_session_image(
        self,
        session_id: str,
        image_url: str,
        angle: str | None,
    ) -> dict:
        result = (
            self._db.table("vision_session_images")
            .insert({
                "session_id": session_id,
                "image_url": image_url,
                "angle": angle,
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
        return result.data

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
