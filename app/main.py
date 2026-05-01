from fastapi import FastAPI
from app.routers import analyze

app = FastAPI(title="crashguard-vision", version="0.1.0")

app.include_router(analyze.router)
