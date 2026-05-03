from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException

from app.adapters.inbound.http.schemas import (
    AddSessionImageRequest,
    CreateVisionSessionRequest,
    CreateVisionSessionResponse,
    DamageReportResponse,
    SessionImageResponse,
)
from app.application.use_cases.vision_sessions import VisionSessionUseCase
from app.auth import verify_api_key
from app.dependencies import get_vision_session_use_case
from app.domain.models import AnalysisSummary

router = APIRouter(prefix="/sessions")


@router.post("", status_code=201, response_model=CreateVisionSessionResponse)
def create_session(
    request: CreateVisionSessionRequest,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> CreateVisionSessionResponse:
    context_dict = request.vehicle_context.model_dump() if request.vehicle_context else None
    session = use_case.create_session(api_key_hash=api_key_hash, vehicle_context=context_dict)
    return CreateVisionSessionResponse(session_id=session["id"], expires_at=session["expires_at"])


@router.get("/{session_id}")
def get_session(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> dict:
    try:
        session = use_case.get_session(session_id, api_key_hash)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/images", status_code=201, response_model=SessionImageResponse)
def add_image(
    session_id: str,
    request: AddSessionImageRequest,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> SessionImageResponse:
    try:
        result = use_case.add_image(
            session_id=session_id,
            api_key_hash=api_key_hash,
            image_url=request.image_url,
            angle=request.angle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")

    summary = None
    if result.image_row["status"] == "completed":
        summary = AnalysisSummary(
            total_damages=len(result.damages),
            damages_by_severity=dict(Counter(d.severity for d in result.damages)),
            damages_by_type=dict(Counter(d.type for d in result.damages)),
            processing_ms=0,
        )

    return SessionImageResponse(
        image_id=result.image_row["id"],
        image_width=result.image_width,
        image_height=result.image_height,
        status=result.image_row["status"],
        damages=result.damages,
        error=result.image_row.get("error"),
        summary=summary,
    )


@router.get("/{session_id}/report", response_model=DamageReportResponse)
def get_report(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> DamageReportResponse:
    try:
        return use_case.get_report(session_id=session_id, api_key_hash=api_key_hash)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
