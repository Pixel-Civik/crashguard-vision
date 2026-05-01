from __future__ import annotations
import time
from collections import Counter
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.auth import verify_api_key
from app.domain.models import VehicleContext, AnalyzeResponse, AnalysisSummary
from app.services.analyze_service import AnalyzeService
from app.dependencies import get_analyze_service

router = APIRouter()


class AnalyzeRequest(BaseModel):
    image_url: str
    vehicle_context: VehicleContext | None = None


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: AnalyzeService = Depends(get_analyze_service),
) -> AnalyzeResponse:
    t0 = time.monotonic()
    damages, width, height = service.analyze(
        image_url=request.image_url,
        context=request.vehicle_context,
    )
    processing_ms = int((time.monotonic() - t0) * 1000)

    severity_counts = Counter(d.severity for d in damages)
    type_counts = Counter(d.type for d in damages)

    return AnalyzeResponse(
        image_width=width,
        image_height=height,
        damages=damages,
        summary=AnalysisSummary(
            total_damages=len(damages),
            damages_by_severity=dict(severity_counts),
            damages_by_type=dict(type_counts),
            processing_ms=processing_ms,
        ),
    )
