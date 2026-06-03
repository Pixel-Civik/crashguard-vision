from __future__ import annotations

from typing import Any


def _metadata_int(metadata: Any, *field_names: str) -> int | None:
    if metadata is None:
        return None
    for field_name in field_names:
        value = getattr(metadata, field_name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def gemini_token_usage(metadata: Any) -> tuple[int | None, int | None]:
    """Return input tokens and billable output tokens.

    Gemini 2.5 can report thinking tokens separately. CrashGuard stores only
    input/output tokens today, so thinking tokens are folded into output.
    """
    prompt_tokens = _metadata_int(metadata, "prompt_token_count")
    candidate_tokens = _metadata_int(metadata, "candidates_token_count")
    thought_tokens = _metadata_int(
        metadata,
        "thoughts_token_count",
        "thought_token_count",
    )
    total_tokens = _metadata_int(metadata, "total_token_count")

    response_tokens = None
    if candidate_tokens is not None or thought_tokens is not None:
        response_tokens = (candidate_tokens or 0) + (thought_tokens or 0)

    if total_tokens is not None and prompt_tokens is not None:
        total_response_tokens = max(total_tokens - prompt_tokens, 0)
        if response_tokens is None or total_response_tokens > response_tokens:
            response_tokens = total_response_tokens

    return prompt_tokens, response_tokens
