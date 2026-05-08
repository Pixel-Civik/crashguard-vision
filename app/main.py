import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze, sessions

app = FastAPI(title="crashguard-vision", version="0.1.0")

DEFAULT_CORS_ORIGINS = {
    "https://crashguarddev.pixelcivik.com",
    "https://crashguard.pixelcivik.com",
}

_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
_allowed_origins = sorted(DEFAULT_CORS_ORIGINS.union(_cors_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(sessions.router)
