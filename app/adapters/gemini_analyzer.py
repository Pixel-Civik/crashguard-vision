from __future__ import annotations
import base64
import io
import json
import re
import httpx
from PIL import Image
from google import genai
from google.genai import types
from app.domain.models import Damage, BoundingBox, VehicleContext

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.*)$", re.DOTALL)


_SYSTEM_PROMPT = """You are a vehicle damage assessor. Analyze the vehicle image and identify all visible damage.

For each damage return a JSON object with:
- type: one of [dent, scratch, crack, stain, rust, broken_glass, other]
- zone: vehicle zone, one of [hood, windshield, roof, trunk, front_door_left, rear_door_left, front_door_right, rear_door_right, side_left, side_right, front_bumper, rear_bumper, mirror_left, mirror_right, wheel_front_left, wheel_front_right, wheel_rear_left, wheel_rear_right]
- severity: low | medium | high
- confidence: float 0.0-1.0
- bbox_x, bbox_y, bbox_w, bbox_h: normalized bounding box [0-1], top-left origin
- description: brief description

Return a JSON array. If no damage found return [].
"""


class GeminiImageAnalyzer:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    def _download_image(self, image_url: str) -> tuple[bytes, int, int, str]:
        if image_url.startswith("data:"):
            match = _DATA_URL_RE.match(image_url)
            if not match:
                raise ValueError("Invalid data URL")
            image_bytes = base64.b64decode(match.group("data"))
        else:
            response = httpx.get(image_url, timeout=30, follow_redirects=False)
            response.raise_for_status()
            image_bytes = response.content
        if len(image_bytes) > 20 * 1024 * 1024:
            raise ValueError(f"Image too large: {len(image_bytes)} bytes")
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            mime_type = Image.MIME.get(img.format or "", "image/jpeg")
        return image_bytes, width, height, mime_type

    def _build_prompt(self, context: VehicleContext | None) -> str:
        if context and any((context.make, context.model, context.year, context.color)):
            parts = [p for p in [context.make, context.model, str(context.year) if context.year else None, context.color] if p]
            return f"Vehicle: {' '.join(parts)}. Analyze the damage."
        return "Analyze the damage on this vehicle."

    def _parse_response(self, raw: str, source_image_id: str | None) -> list[Damage]:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned non-JSON response: {raw[:200]!r}") from exc
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array from Gemini, got {type(items).__name__}: {raw[:200]!r}")
        damages = []
        for i, item in enumerate(items):
            damages.append(Damage(
                id=f"dmg_{i+1:02d}",
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
                source_image_id=source_image_id,
            ))
        return damages

    def analyze(
        self,
        image_url: str,
        context: VehicleContext | None,
    ) -> list[Damage]:
        damages, _, _ = self.analyze_with_dimensions(image_url, context)
        return damages

    def analyze_with_dimensions(
        self,
        image_url: str,
        context: VehicleContext | None,
        source_image_id: str | None = None,
    ) -> tuple[list[Damage], int, int]:
        image_bytes, width, height, mime_type = self._download_image(image_url)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                types.Part.from_text(text=self._build_prompt(context)),
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        damages = self._parse_response(response.text, source_image_id)
        return damages, width, height
