from __future__ import annotations
import json
import uuid
import pydantic
from google import genai
from google.genai import types
from app.domain.models import Damage, BoundingBox


_SYSTEM_PROMPT = """You are a vehicle damage consolidation specialist.
You will receive a list of damages detected in multiple photos of the same vehicle.

Your task:
1. Identify which damages from different photos represent the same physical damage (seen from different angles)
2. Return a deduplicated list. For each canonical damage, use the one with highest confidence as the base
3. In "also_seen_in" list the source_image_id of other photos where the same damage appears
4. If uncertain, do NOT merge (keep them separate)

Rule: same damage = same zone + same type + similar bbox overlap

Return a JSON array with the same fields as the input plus "also_seen_in": list of image_ids.
If input is empty return [].
"""


class GeminiDamageAggregator:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    def aggregate(self, damage_lists: list[list[Damage]]) -> list[Damage]:
        all_damages = [d for sublist in damage_lists for d in sublist]
        if not all_damages:
            return []

        input_json = json.dumps([
            {
                "id": d.id,
                "type": d.type.value,
                "zone": d.zone.value,
                "severity": d.severity.value,
                "confidence": d.confidence,
                "bbox_x": d.bbox.x,
                "bbox_y": d.bbox.y,
                "bbox_w": d.bbox.w,
                "bbox_h": d.bbox.h,
                "description": d.description,
                "source_image_id": d.source_image_id,
            }
            for d in all_damages
        ], ensure_ascii=False)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Part.from_text(
                text=f"Consolidate these damages:\n{input_json}"
            )],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        return self._parse_response(response.text)

    def _parse_response(self, raw: str | None) -> list[Damage]:
        if raw is None:
            raise ValueError("Gemini returned empty response")
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned non-JSON response: {raw[:200]!r}") from exc
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array from Gemini, got {type(items).__name__}: {raw[:200]!r}")
        result = []
        for item in items:
            try:
                result.append(Damage(
                    id=item.get("id") or str(uuid.uuid4()),
                    type=item.get("type", "other"),
                    zone=item.get("zone", "unknown"),
                    severity=item.get("severity", "low"),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox=BoundingBox(
                        x=float(item.get("bbox_x", 0)),
                        y=float(item.get("bbox_y", 0)),
                        w=float(item.get("bbox_w", 0)),
                        h=float(item.get("bbox_h", 0)),
                    ),
                    description=item.get("description", ""),
                    source_image_id=item.get("source_image_id"),
                    also_seen_in=item.get("also_seen_in", []),
                ))
            except pydantic.ValidationError as exc:
                raise ValueError(f"Invalid damage data from Gemini: {item}") from exc
        return result
