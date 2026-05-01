import logging
from app.db.supabase import VisionRepository

logger = logging.getLogger("vision.tracer")


class DbAnalysisTracer:
    def __init__(self, repo: VisionRepository) -> None:
        self._repo = repo

    def record(
        self,
        call_type: str,
        model: str,
        latency_ms: int,
        status: str,
        raw_response: dict,
        session_id: str | None = None,
        image_id: str | None = None,
        prompt_tokens: int | None = None,
        response_tokens: int | None = None,
        error: str | None = None,
    ) -> str:
        # raw_response intentionally excluded from logs; persisted to DB only
        logger.info(
            call_type,
            extra={
                "call_type": call_type,
                "session_id": session_id,
                "image_id": image_id,
                "model": model,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "status": status,
                "error": error,
            },
        )
        return self._repo.create_analysis_call(
            call_type=call_type,
            model=model,
            latency_ms=latency_ms,
            status=status,
            raw_response=raw_response,
            session_id=session_id,
            image_id=image_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            error=error,
        )
