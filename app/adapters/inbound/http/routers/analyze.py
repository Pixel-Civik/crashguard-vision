from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException

from app.adapters.inbound.http.schemas import AnalyzeImageRequest, AnalyzeImageResponse
from app.application.use_cases.analyze_image import AnalyzeImageUseCase
from app.auth import verify_api_key
from app.dependencies import get_analyze_image_use_case

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeImageResponse)
def analyze(
    request: AnalyzeImageRequest,
    api_key_hash: str = Depends(verify_api_key),
    use_case: AnalyzeImageUseCase = Depends(get_analyze_image_use_case),
) -> AnalyzeImageResponse:
    try:
        result = use_case.execute(
            image_url=request.image_url,
            vehicle_context=request.vehicle_context,
        )
        return AnalyzeImageResponse.from_result(result)
    except Exception as e:
        logging.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
