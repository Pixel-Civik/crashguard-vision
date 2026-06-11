from __future__ import annotations
import base64
import io
import json
import re
import httpx
from PIL import Image
from google import genai
from google.genai import types
from app.adapters.gemini_retry import call_gemini_with_retry
from app.adapters.gemini_usage import gemini_token_usage
from app.domain.damage_filters import is_non_damage_false_positive
from app.domain.models import Damage, BoundingBox, VehicleContext

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.*)$", re.DOTALL)


_SYSTEM_PROMPT = """You are a vehicle damage assessor. Analyze the vehicle image and identify only physical exterior vehicle damage.

Include: dents, scratches, cracks, rust, broken glass, broken or missing exterior parts, and permanent stains on the vehicle surface.

Exclude and do not report:
- flat, deflated, underinflated, or low-pressure tires unless there is visible structural tire or rim damage such as a cut, tear, cracked rim, bent rim, or missing tire material
- water, puddles, oil marks, dirt, dust, mud, leaves, bird droppings, organic debris, or temporary stains
- reflections, shadows, glare, background objects, stickers, logos, labels, normal wear, or normal tire pressure conditions
- anything on the ground, floor, road, sidewalk, or surrounding environment

For each damage return a JSON object with:
- type: one of [dent, scratch, crack, stain, rust, broken_glass, other]
- zone: vehicle zone, one of [hood, windshield, roof, trunk, front_door_left, rear_door_left, front_door_right, rear_door_right, side_left, side_right, front_bumper, rear_bumper, mirror_left, mirror_right, wheel_front_left, wheel_front_right, wheel_rear_left, wheel_rear_right]
- severity: low | medium | high
- confidence: float 0.0-1.0
- bbox_x, bbox_y, bbox_w, bbox_h: normalized bounding box [0-1], top-left origin
- description: brief description

Use the supplied source view as strong spatial context. Left and right always refer to the vehicle's own left and right sides, not the viewer's screen.
Do not guess a zone that is not visible. If the exact zone cannot be determined, omit the candidate rather than assigning an unrelated zone.
Every bounding box must tightly surround only the visible physical damage and stay inside the image.
Use "other" only for visible physical exterior vehicle damage that does not fit the other types. Never use "other" for tire pressure, dirt/debris, water/ground conditions, background objects, or image artifacts.
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
            
            supported_mimes = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
            if mime_type not in supported_mimes:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                out_io = io.BytesIO()
                img.save(out_io, format="JPEG", quality=90)
                image_bytes = out_io.getvalue()
                mime_type = "image/jpeg"
                
        return image_bytes, width, height, mime_type

    def _build_prompt(
        self,
        context: VehicleContext | None,
        source_view: str | None = None,
    ) -> str:
        prompt_parts = []
        if context and any((context.make, context.model, context.year, context.color)):
            parts = [p for p in [context.make, context.model, str(context.year) if context.year else None, context.color] if p]
            prompt_parts.append(f"Vehicle: {' '.join(parts)}.")
        if source_view:
            prompt_parts.append(f"Source view: {source_view}.")
        prompt_parts.append("Analyze only visible physical exterior vehicle damage.")
        return " ".join(prompt_parts)

    def _parse_response(self, raw: str, source_image_id: str | None) -> list[Damage]:
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned non-JSON response: {raw[:200]!r}") from exc
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array from Gemini, got {type(items).__name__}: {raw[:200]!r}")
        damages = []
        for i, item in enumerate(items):
            damage = Damage(
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
            )
            if not is_non_damage_false_positive(damage):
                damages.append(damage)
        return damages

    def analyze(
        self,
        image_url: str,
        context: VehicleContext | None,
    ) -> list[Damage]:
        damages, _, _, _, _ = self.analyze_with_dimensions(image_url, context)
        return damages

    def analyze_with_dimensions(
        self,
        image_url: str,
        context: VehicleContext | None,
        source_image_id: str | None = None,
        source_view: str | None = None,
    ) -> tuple[list[Damage], int, int, int | None, int | None]:
        image_bytes, width, height, mime_type = self._download_image(image_url)

        response = call_gemini_with_retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(
                        text=self._build_prompt(context, source_view)
                    ),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
            )
        )

        damages = self._parse_response(response.text, source_image_id)
        
        prompt_tokens, response_tokens = gemini_token_usage(response.usage_metadata)
        
        return damages, width, height, prompt_tokens, response_tokens
