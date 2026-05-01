import hashlib
import hmac
from fastapi import Header, HTTPException
from app.config import settings


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(x_vision_key: str = Header(...)) -> str:
    if not hmac.compare_digest(x_vision_key, settings.vision_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return hash_key(x_vision_key)
