from __future__ import annotations
from collections import Counter
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth import verify_api_key
from app.domain.models import (
    VehicleContext, DamageMap, AnalysisSummary, SessionImageResponse,
)
from app.services.session_service import SessionService
from app.dependencies import get_session_service

router = APIRouter(prefix="/sessions")


class CreateSessionRequest(BaseModel):
    vehicle_context: VehicleContext | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    expires_at: str


class AddImageRequest(BaseModel):
    image_url: str
    angle: str | None = None


@router.post("", status_code=201, response_model=CreateSessionResponse)
def create_session(
    request: CreateSessionRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> CreateSessionResponse:
    context_dict = request.vehicle_context.model_dump() if request.vehicle_context else None
    session = service.create_session(api_key_hash=api_key_hash, vehicle_context=context_dict)
    return CreateSessionResponse(session_id=session["id"], expires_at=session["expires_at"])


@router.get("/{session_id}")
def get_session(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> dict:
    try:
        session = service.get_session(session_id, api_key_hash)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/images", status_code=201, response_model=SessionImageResponse)
def add_image(
    session_id: str,
    request: AddImageRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> SessionImageResponse:
    try:
        image_row, damages, width, height = service.add_image(
            session_id=session_id,
            api_key_hash=api_key_hash,
            image_url=request.image_url,
            angle=request.angle,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")

    summary = None
    if image_row["status"] == "completed":
        summary = AnalysisSummary(
            total_damages=len(damages),
            damages_by_severity=dict(Counter(d.severity for d in damages)),
            damages_by_type=dict(Counter(d.type for d in damages)),
            processing_ms=0,
        )

    return SessionImageResponse(
        image_id=image_row["id"],
        image_width=width,
        image_height=height,
        status=image_row["status"],
        damages=damages,
        error=image_row.get("error"),
        summary=summary,
    )


@router.get("/{session_id}/report", response_model=DamageMap)
def get_report(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> DamageMap:
    try:
        return service.get_report(session_id=session_id, api_key_hash=api_key_hash)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
