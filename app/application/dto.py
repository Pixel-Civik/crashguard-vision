from pydantic import BaseModel

from app.domain.models import Damage


class ImageAnalysisResult(BaseModel):
    damages: list[Damage]
    image_width: int
    image_height: int
    processing_ms: int


class SessionImageAnalysisResult(BaseModel):
    image_row: dict
    damages: list[Damage]
    image_width: int
    image_height: int
