from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException

from app.adapters.inbound.http.schemas import (
    AddSessionImageRequest,
    CreateVisionSessionRequest,
    CreateVisionSessionResponse,
    DamageReportResponse,
    RetrySessionImageRequest,
    SessionImageResponse,
    StoredSessionImageResponse,
)
from app.application.use_cases.vision_sessions import VisionSessionUseCase
from app.auth import verify_api_key
from app.dependencies import get_vision_session_use_case
from app.domain.models import AnalysisSummary
from app.domain.prompt_metadata import (
    CONSOLIDATION_PROMPT_VERSION,
    IMAGE_PROMPT_VERSION,
)

router = APIRouter(prefix="/sessions")


def _summary_from_result(result) -> AnalysisSummary | None:
    if result.image_row["status"] != "completed":
        return None
    return AnalysisSummary(
        total_damages=len(result.damages),
        damages_by_severity=dict(Counter(d.severity for d in result.damages)),
        damages_by_type=dict(Counter(d.type for d in result.damages)),
        processing_ms=result.processing_ms or 0,
        prompt_tokens=result.prompt_tokens,
        response_tokens=result.response_tokens,
    )


def _result_to_response(result) -> SessionImageResponse:
    return SessionImageResponse(
        image_id=result.image_row["id"],
        image_width=result.image_width,
        image_height=result.image_height,
        status=result.image_row["status"],
        damages=result.damages,
        error=result.image_row.get("error"),
        summary=_summary_from_result(result),
    )


def _stored_image_to_response(row: dict) -> StoredSessionImageResponse:
    return StoredSessionImageResponse(
        image_id=row["id"],
        image_url=row["image_url"],
        angle=row.get("angle"),
        image_width=row.get("image_width"),
        image_height=row.get("image_height"),
        status=row["status"],
        damages=row.get("damages") or [],
        error=row.get("error"),
        uploaded_at=row.get("uploaded_at"),
        analyzed_at=row.get("analyzed_at"),
        inspection_media_asset_id=row.get("inspection_media_asset_id"),
        inspection_item_id=row.get("inspection_item_id"),
    )


@router.post("", status_code=201, response_model=CreateVisionSessionResponse)
def create_session(
    request: CreateVisionSessionRequest,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> CreateVisionSessionResponse:
    context_dict = request.vehicle_context.model_dump() if request.vehicle_context else None
    session = use_case.create_session(
        api_key_hash=api_key_hash,
        vehicle_context=context_dict,
        tenant_id=request.tenant_id,
        inspection_id=request.inspection_id,
        capture_session_id=request.capture_session_id,
        vehicle_id=request.vehicle_id,
        mode=request.mode,
    )
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


@router.get("/{session_id}/images", response_model=list[StoredSessionImageResponse])
def list_images(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> list[StoredSessionImageResponse]:
    try:
        images = use_case.list_images(session_id=session_id, api_key_hash=api_key_hash)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")

    return [_stored_image_to_response(row) for row in images]


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
            inspection_media_asset_id=request.inspection_media_asset_id,
            inspection_item_id=request.inspection_item_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")

    return _result_to_response(result)

@router.post("/{session_id}/images/{image_id}/retry", response_model=SessionImageResponse)
def retry_image(
    session_id: str,
    image_id: str,
    request: RetrySessionImageRequest,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> SessionImageResponse:
    try:
        result = use_case.retry_image(
            session_id=session_id,
            image_id=image_id,
            api_key_hash=api_key_hash,
            image_url=request.image_url,
            force=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")

    return _result_to_response(result)


@router.get("/{session_id}/report", response_model=DamageReportResponse)
def get_report(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    use_case: VisionSessionUseCase = Depends(get_vision_session_use_case),
) -> DamageReportResponse:
    try:
        from datetime import datetime, timezone
        dmap = use_case.get_report(session_id=session_id, api_key_hash=api_key_hash)
        return DamageReportResponse.from_domain(
            dmap=dmap,
            built_at=datetime.now(timezone.utc).isoformat(),
            image_count=len(dmap.images),
            model=use_case.model_name,
            prompt_versions={
                "image_analysis": IMAGE_PROMPT_VERSION,
                "consolidation": CONSOLIDATION_PROMPT_VERSION,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
