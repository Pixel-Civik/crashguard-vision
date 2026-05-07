from collections import Counter

from pydantic import BaseModel

from app.application.dto import ImageAnalysisResult
from app.domain.models import (
    AnalysisSummary,
    Damage,
    DamageMap,
    VehicleContext,
)


class AnalyzeImageRequest(BaseModel):
    image_url: str
    vehicle_context: VehicleContext | None = None


class AnalyzeImageResponse(BaseModel):
    image_width: int
    image_height: int
    damages: list[Damage]
    summary: AnalysisSummary

    @classmethod
    def from_result(cls, result: ImageAnalysisResult) -> "AnalyzeImageResponse":
        return cls(
            image_width=result.image_width,
            image_height=result.image_height,
            damages=result.damages,
            summary=AnalysisSummary(
                total_damages=len(result.damages),
                damages_by_severity=dict(Counter(d.severity for d in result.damages)),
                damages_by_type=dict(Counter(d.type for d in result.damages)),
                processing_ms=result.processing_ms,
                prompt_tokens=result.prompt_tokens,
                response_tokens=result.response_tokens,
            ),
        )


class CreateVisionSessionRequest(BaseModel):
    vehicle_context: VehicleContext | None = None


class CreateVisionSessionResponse(BaseModel):
    session_id: str
    expires_at: str


class AddSessionImageRequest(BaseModel):
    image_url: str
    angle: str | None = None


class SessionImageResponse(BaseModel):
    image_id: str
    image_width: int
    image_height: int
    status: str
    damages: list[Damage]
    error: str | None = None
    summary: AnalysisSummary | None = None


DamageReportResponse = DamageMap
