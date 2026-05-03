import hashlib
import hmac
from collections.abc import Callable
from fastapi import Header, HTTPException
from app.config import settings


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key_verifier(expected_key: str) -> Callable:
    """Factory that builds a FastAPI dependency closed over the expected key."""

    def verify_api_key(x_vision_key: str = Header(...)) -> str:
        if not hmac.compare_digest(x_vision_key, expected_key):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return hash_key(x_vision_key)

    return verify_api_key


def verify_api_key(x_vision_key: str = Header(...)) -> str:
    if not hmac.compare_digest(x_vision_key, settings.vision_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return hash_key(x_vision_key)
