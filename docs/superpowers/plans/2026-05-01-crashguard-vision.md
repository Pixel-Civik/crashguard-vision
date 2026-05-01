# crashguard-vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python/FastAPI service that analyzes vehicle damage from images using Gemini Vision, supporting single-image analysis and multi-image sessions with consolidated damage maps.

**Architecture:** Hexagonal architecture. FastAPI routers → Services → Ports (Protocols) → Adapters (Gemini SDK / Supabase). Four protocols: `ImageAnalyzer`, `DamageAggregator`, `DamageMapBuilder`, `AnalysisTracer`.

**Tech Stack:** Python 3.12, FastAPI, google-genai SDK, supabase-py, Pillow, httpx, pytest, pytest-asyncio

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Dependencies and project metadata |
| `.env.example` | Env var documentation |
| `app/config.py` | Pydantic-settings env var config |
| `app/main.py` | FastAPI app instance + router registration |
| `app/auth.py` | `X-Vision-Key` FastAPI dependency |
| `app/domain/models.py` | All Pydantic models and enums |
| `app/domain/ports.py` | Protocol interfaces |
| `app/db/supabase.py` | Supabase client + `VisionRepository` (all DB ops) |
| `app/db/migrations/001_initial.sql` | DB schema (4 tables) |
| `app/adapters/damage_map_builder.py` | `DamageMapBuilder` — pure Python |
| `app/adapters/db_tracer.py` | `AnalysisTracer` — writes to `vision_analysis_calls` |
| `app/adapters/gemini_analyzer.py` | `ImageAnalyzer` — Gemini structured output |
| `app/adapters/gemini_aggregator.py` | `DamageAggregator` — Gemini consolidation |
| `app/services/analyze_service.py` | Orchestrates `/analyze` flow |
| `app/services/session_service.py` | Orchestrates all session flows |
| `app/routers/analyze.py` | `POST /analyze` |
| `app/routers/sessions.py` | Session CRUD + report endpoints |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_models.py` | Model validation tests |
| `tests/test_auth.py` | Auth dependency tests |
| `tests/test_damage_map_builder.py` | DamageMapBuilder unit tests |
| `tests/test_gemini_analyzer.py` | GeminiImageAnalyzer tests (mocked Gemini) |
| `tests/test_gemini_aggregator.py` | GeminiDamageAggregator tests (mocked Gemini) |
| `tests/test_analyze_router.py` | `/analyze` endpoint integration tests |
| `tests/test_sessions_router.py` | Session endpoints integration tests |
| `Dockerfile` | Multi-stage Docker build |
| `cloudbuild.yaml` | Cloud Build / Cloud Run deploy |

---

## Task 1: Project scaffold

**Files:**
- Create: `crashguard-vision/pyproject.toml`
- Create: `crashguard-vision/.env.example`
- Create: `crashguard-vision/app/__init__.py`
- Create: `crashguard-vision/app/config.py`
- Create: `crashguard-vision/tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "crashguard-vision"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "google-genai>=1.14.0",
    "supabase>=2.10.0",
    "pillow>=11.0.0",
    "httpx>=0.27.0",
    "pydantic-settings>=2.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create .env.example**

```
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_KEY=your-service-role-key
VISION_API_KEY=your-internal-api-key
SESSION_TTL_HOURS=24
PORT=8080
```

- [ ] **Step 3: Create app/config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    supabase_url: str
    supabase_service_key: str
    vision_api_key: str
    session_ttl_hours: int = 24
    port: int = 8080

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
```

- [ ] **Step 4: Create empty __init__.py files**

```bash
touch crashguard-vision/app/__init__.py
touch crashguard-vision/app/domain/__init__.py
touch crashguard-vision/app/adapters/__init__.py
touch crashguard-vision/app/services/__init__.py
touch crashguard-vision/app/routers/__init__.py
touch crashguard-vision/app/db/__init__.py
touch crashguard-vision/tests/__init__.py
```

- [ ] **Step 5: Install dependencies**

Run from `crashguard-vision/`:
```bash
pip install -e ".[dev]"
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example app/ tests/
git commit -m "feat: scaffold crashguard-vision project"
```

---

## Task 2: Domain models

**Files:**
- Create: `crashguard-vision/app/domain/models.py`
- Create: `crashguard-vision/app/domain/ports.py`
- Create: `crashguard-vision/tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
import pytest
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
    VehicleContext, SourceImageMeta, AnalysisSummary, DamageMap,
    DamageMapSummary, SkippedImage,
)


def test_damage_type_fallback():
    d = Damage(
        id="dmg_01",
        type="unknown_type",
        zone="puerta_delantera_izq",
        severity="medium",
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.1),
        description="test",
    )
    assert d.type == DamageType.other


def test_vehicle_zone_fallback():
    d = Damage(
        id="dmg_01",
        type="abolladura",
        zone="zona_inexistente",
        severity="medium",
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.1),
        description="test",
    )
    assert d.zone == VehicleZone.unknown


def test_damage_defaults():
    d = Damage(
        id="dmg_01",
        type=DamageType.abolladura,
        zone=VehicleZone.capot,
        severity=Severity.low,
        confidence=0.8,
        bbox=BoundingBox(x=0.0, y=0.0, w=0.5, h=0.5),
        description="abolladura en capot",
    )
    assert d.source_image_id is None
    assert d.also_seen_in == []


def test_vehicle_context_all_optional():
    ctx = VehicleContext()
    assert ctx.make is None
    assert ctx.model is None


def test_damage_map_summary_fields():
    summary = DamageMapSummary(
        total_images=5,
        images_analyzed=4,
        total_damages=2,
        zones_affected=[VehicleZone.capot],
        overall_severity=Severity.medium,
        damages_by_severity={Severity.medium: 2},
        damages_by_type={DamageType.abolladura: 2},
    )
    assert summary.images_skipped == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Create app/domain/models.py**

```python
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, field_validator


class VehicleZone(str, Enum):
    capot = "capot"
    parabrisas = "parabrisas"
    techo = "techo"
    maletero = "maletero"
    puerta_delantera_izq = "puerta_delantera_izq"
    puerta_trasera_izq = "puerta_trasera_izq"
    puerta_delantera_der = "puerta_delantera_der"
    puerta_trasera_der = "puerta_trasera_der"
    lateral_izq = "lateral_izq"
    lateral_der = "lateral_der"
    paragolpes_del = "paragolpes_del"
    paragolpes_tras = "paragolpes_tras"
    espejo_izq = "espejo_izq"
    espejo_der = "espejo_der"
    rueda_del_izq = "rueda_del_izq"
    rueda_del_der = "rueda_del_der"
    rueda_tras_izq = "rueda_tras_izq"
    rueda_tras_der = "rueda_tras_der"
    unknown = "unknown"


class DamageType(str, Enum):
    abolladura = "abolladura"
    rayon = "rayon"
    quiebre = "quiebre"
    mancha = "mancha"
    corrosion = "corrosion"
    vidrio_roto = "vidrio_roto"
    other = "other"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class VehicleContext(BaseModel):
    make: str | None = None
    model: str | None = None
    year: int | None = None
    color: str | None = None


class SourceImageMeta(BaseModel):
    url: str
    width: int
    height: int
    angle: str | None = None


class Damage(BaseModel):
    id: str
    type: DamageType
    zone: VehicleZone
    severity: Severity
    confidence: float
    bbox: BoundingBox
    description: str
    source_image_id: str | None = None
    also_seen_in: list[str] = []

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: str) -> str:
        valid = {e.value for e in DamageType}
        return v if v in valid else DamageType.other.value

    @field_validator("zone", mode="before")
    @classmethod
    def coerce_zone(cls, v: str) -> str:
        valid = {e.value for e in VehicleZone}
        return v if v in valid else VehicleZone.unknown.value


class AnalysisSummary(BaseModel):
    total_damages: int
    damages_by_severity: dict[Severity, int]
    damages_by_type: dict[DamageType, int]
    processing_ms: int


class SkippedImage(BaseModel):
    image_id: str
    reason: str


class DamageMapSummary(BaseModel):
    total_images: int
    images_analyzed: int
    images_skipped: list[SkippedImage] = []
    total_damages: int
    zones_affected: list[VehicleZone]
    overall_severity: Severity
    damages_by_severity: dict[Severity, int]
    damages_by_type: dict[DamageType, int]


class DamageMap(BaseModel):
    session_id: str
    vehicle_context: VehicleContext | None
    images: dict[str, SourceImageMeta]
    zones: dict[VehicleZone, list[Damage]]
    summary: DamageMapSummary


class AnalyzeResponse(BaseModel):
    image_width: int
    image_height: int
    damages: list[Damage]
    summary: AnalysisSummary


class SessionImageResponse(BaseModel):
    image_id: str
    image_width: int
    image_height: int
    status: str
    damages: list[Damage]
    error: str | None = None
    summary: AnalysisSummary | None = None
```

- [ ] **Step 4: Create app/domain/ports.py**

```python
from typing import Protocol
from app.domain.models import (
    Damage, DamageMap, SourceImageMeta, VehicleContext,
)


class ImageAnalyzer(Protocol):
    def analyze(
        self, image_url: str, context: VehicleContext | None
    ) -> list[Damage]: ...


class DamageAggregator(Protocol):
    def aggregate(self, damage_lists: list[list[Damage]]) -> list[Damage]: ...


class DamageMapBuilder(Protocol):
    def build(
        self,
        damages: list[Damage],
        images: dict[str, SourceImageMeta],
    ) -> DamageMap: ...


class AnalysisTracer(Protocol):
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
    ) -> str: ...
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/domain/ tests/test_models.py
git commit -m "feat: add domain models and port protocols"
```

---

## Task 3: Database schema + Supabase client

**Files:**
- Create: `crashguard-vision/app/db/migrations/001_initial.sql`
- Create: `crashguard-vision/app/db/supabase.py`

- [ ] **Step 1: Create app/db/migrations/001_initial.sql**

```sql
-- Sessions
create table vision_sessions (
  id              uuid primary key default gen_random_uuid(),
  api_key_hash    text not null,
  vehicle_context jsonb,
  status          text not null default 'open',
  created_at      timestamptz not null default now(),
  expires_at      timestamptz not null
);

-- Analysis calls audit log (created before session_images to avoid circular FK)
create table vision_analysis_calls (
  id              uuid primary key default gen_random_uuid(),
  call_type       text not null,
  session_id      uuid references vision_sessions(id),
  image_id        uuid,           -- no FK: circular dependency with session_images
  model           text not null,
  prompt_tokens   int,
  response_tokens int,
  latency_ms      int,
  status          text not null,
  error           text,
  raw_response    jsonb,
  created_at      timestamptz not null default now()
);

-- Session images
create table vision_session_images (
  id               uuid primary key default gen_random_uuid(),
  session_id       uuid not null references vision_sessions(id),
  image_url        text not null,
  angle            text,
  image_width      int,
  image_height     int,
  status           text not null default 'pending',
  damages          jsonb,
  gemini_call_id   uuid,           -- no FK: circular dependency with analysis_calls
  error            text,
  verified_damages jsonb,
  verified_at      timestamptz,
  verified_by      text,
  uploaded_at      timestamptz not null default now(),
  analyzed_at      timestamptz
);

-- Consolidated damage maps
create table vision_damage_maps (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references vision_sessions(id) unique,
  images      jsonb not null,
  zones       jsonb not null,
  summary     jsonb not null,
  image_count int not null,
  built_at    timestamptz not null default now()
);
```

Run this against your local Supabase:
```bash
psql postgresql://postgres:postgres@127.0.0.1:54322/postgres -f app/db/migrations/001_initial.sql
```

- [ ] **Step 2: Create app/db/supabase.py**

```python
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Any
from supabase import create_client, Client
from app.config import settings


def get_client() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


class VisionRepository:
    def __init__(self, client: Client) -> None:
        self._db = client

    # --- sessions ---

    def create_session(
        self,
        api_key_hash: str,
        vehicle_context: dict | None,
    ) -> dict:
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.session_ttl_hours
        )
        result = (
            self._db.table("vision_sessions")
            .insert({
                "api_key_hash": api_key_hash,
                "vehicle_context": vehicle_context,
                "expires_at": expires_at.isoformat(),
            })
            .execute()
        )
        return result.data[0]

    def get_session(self, session_id: str) -> dict | None:
        result = (
            self._db.table("vision_sessions")
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    # --- session images ---

    def create_session_image(
        self,
        session_id: str,
        image_url: str,
        angle: str | None,
    ) -> dict:
        result = (
            self._db.table("vision_session_images")
            .insert({
                "session_id": session_id,
                "image_url": image_url,
                "angle": angle,
                "status": "pending",
            })
            .execute()
        )
        return result.data[0]

    def update_image_analyzing(
        self, image_id: str, width: int, height: int
    ) -> None:
        self._db.table("vision_session_images").update({
            "status": "analyzing",
            "image_width": width,
            "image_height": height,
        }).eq("id", image_id).execute()

    def update_image_completed(
        self,
        image_id: str,
        damages: list[dict],
        gemini_call_id: str,
    ) -> None:
        self._db.table("vision_session_images").update({
            "status": "completed",
            "damages": damages,
            "gemini_call_id": gemini_call_id,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", image_id).execute()

    def update_image_failed(self, image_id: str, error: str) -> None:
        self._db.table("vision_session_images").update({
            "status": "failed",
            "error": error,
        }).eq("id", image_id).execute()

    def get_completed_images(self, session_id: str) -> list[dict]:
        result = (
            self._db.table("vision_session_images")
            .select("*")
            .eq("session_id", session_id)
            .eq("status", "completed")
            .order("uploaded_at")
            .execute()
        )
        return result.data

    def get_all_images(self, session_id: str) -> list[dict]:
        result = (
            self._db.table("vision_session_images")
            .select("id, status, error")
            .eq("session_id", session_id)
            .execute()
        )
        return result.data

    # --- damage maps ---

    def get_damage_map(self, session_id: str) -> dict | None:
        result = (
            self._db.table("vision_damage_maps")
            .select("*")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    def upsert_damage_map(
        self,
        session_id: str,
        images: dict,
        zones: dict,
        summary: dict,
        image_count: int,
    ) -> dict:
        result = (
            self._db.table("vision_damage_maps")
            .upsert({
                "session_id": session_id,
                "images": images,
                "zones": zones,
                "summary": summary,
                "image_count": image_count,
                "built_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="session_id")
            .execute()
        )
        return result.data[0]

    # --- analysis calls ---

    def create_analysis_call(
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
        result = (
            self._db.table("vision_analysis_calls")
            .insert({
                "call_type": call_type,
                "model": model,
                "latency_ms": latency_ms,
                "status": status,
                "raw_response": raw_response,
                "session_id": session_id,
                "image_id": image_id,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "error": error,
            })
            .execute()
        )
        return result.data[0]["id"]
```

- [ ] **Step 3: Commit**

```bash
git add app/db/
git commit -m "feat: add DB schema and VisionRepository"
```

---

## Task 4: Auth dependency

**Files:**
- Create: `crashguard-vision/app/auth.py`
- Create: `crashguard-vision/tests/test_auth.py`

- [ ] **Step 1: Write failing tests**

`tests/test_auth.py`:
```python
import hashlib
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from app.auth import verify_api_key, hash_key
from app.config import settings


def test_hash_key_is_deterministic():
    h1 = hash_key("my-secret-key")
    h2 = hash_key("my-secret-key")
    assert h1 == h2


def test_hash_key_different_inputs():
    assert hash_key("key-a") != hash_key("key-b")


def test_verify_api_key_valid(monkeypatch):
    monkeypatch.setattr(settings, "vision_api_key", "test-key-123")
    result = verify_api_key("test-key-123")
    assert result == hash_key("test-key-123")


def test_verify_api_key_invalid(monkeypatch):
    monkeypatch.setattr(settings, "vision_api_key", "real-key")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key("wrong-key")
    assert exc_info.value.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_auth.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/auth.py**

```python
import hashlib
from fastapi import Header, HTTPException
from app.config import settings


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(x_vision_key: str = Header(...)) -> str:
    if x_vision_key != settings.vision_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return hash_key(x_vision_key)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_auth.py -v
```
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/auth.py tests/test_auth.py
git commit -m "feat: add X-Vision-Key auth dependency"
```

---

## Task 5: DamageMapBuilder

**Files:**
- Create: `crashguard-vision/app/adapters/damage_map_builder.py`
- Create: `crashguard-vision/tests/test_damage_map_builder.py`

- [ ] **Step 1: Write failing tests**

`tests/test_damage_map_builder.py`:
```python
import pytest
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
    SourceImageMeta, VehicleContext, DamageMap,
)
from app.adapters.damage_map_builder import PythonDamageMapBuilder


@pytest.fixture
def builder():
    return PythonDamageMapBuilder()


@pytest.fixture
def sample_damage():
    return Damage(
        id="dmg_01",
        type=DamageType.abolladura,
        zone=VehicleZone.capot,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.1),
        description="abolladura en capot",
        source_image_id="img_001",
    )


@pytest.fixture
def sample_images():
    return {
        "img_001": SourceImageMeta(url="https://example.com/img1.jpg", width=3024, height=4032),
    }


def test_build_populates_all_zones(builder, sample_damage, sample_images):
    result = builder.build(
        damages=[sample_damage],
        images=sample_images,
        session_id="sess_01",
        vehicle_context=None,
    )
    assert isinstance(result, DamageMap)
    assert VehicleZone.capot in result.zones
    assert len(result.zones[VehicleZone.capot]) == 1
    # All zones present, empty lists where no damages
    assert VehicleZone.techo in result.zones
    assert result.zones[VehicleZone.techo] == []


def test_build_summary_counts(builder, sample_damage, sample_images):
    result = builder.build(
        damages=[sample_damage],
        images=sample_images,
        session_id="sess_01",
        vehicle_context=None,
    )
    assert result.summary.total_damages == 1
    assert result.summary.zones_affected == [VehicleZone.capot]
    assert result.summary.overall_severity == Severity.medium
    assert result.summary.damages_by_severity[Severity.medium] == 1
    assert result.summary.damages_by_type[DamageType.abolladura] == 1


def test_build_empty_damages(builder, sample_images):
    result = builder.build(
        damages=[],
        images=sample_images,
        session_id="sess_01",
        vehicle_context=None,
    )
    assert result.summary.total_damages == 0
    assert result.summary.zones_affected == []
    all_empty = all(v == [] for v in result.zones.values())
    assert all_empty


def test_build_overall_severity_is_max(builder, sample_images):
    damages = [
        Damage(
            id="d1", type=DamageType.rayon, zone=VehicleZone.capot,
            severity=Severity.low, confidence=0.8,
            bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
            description="rayón", source_image_id="img_001",
        ),
        Damage(
            id="d2", type=DamageType.abolladura, zone=VehicleZone.techo,
            severity=Severity.high, confidence=0.9,
            bbox=BoundingBox(x=0.5, y=0.5, w=0.2, h=0.2),
            description="abolladura", source_image_id="img_001",
        ),
    ]
    result = builder.build(damages=damages, images=sample_images, session_id="sess_01", vehicle_context=None)
    assert result.summary.overall_severity == Severity.high
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_damage_map_builder.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/adapters/damage_map_builder.py**

```python
from collections import Counter
from app.domain.models import (
    Damage, DamageMap, DamageMapSummary, SourceImageMeta,
    VehicleContext, VehicleZone, Severity,
)

_SEVERITY_ORDER = {Severity.low: 0, Severity.medium: 1, Severity.high: 2}


class PythonDamageMapBuilder:
    def build(
        self,
        damages: list[Damage],
        images: dict[str, SourceImageMeta],
        session_id: str,
        vehicle_context: VehicleContext | None,
    ) -> DamageMap:
        zones: dict[VehicleZone, list[Damage]] = {z: [] for z in VehicleZone}

        for damage in damages:
            zones[damage.zone].append(damage)

        zones_affected = [z for z, dmgs in zones.items() if dmgs]

        severity_counts = Counter(d.severity for d in damages)
        type_counts = Counter(d.type for d in damages)

        overall = max(
            (d.severity for d in damages),
            key=lambda s: _SEVERITY_ORDER[s],
            default=Severity.low,
        )

        summary = DamageMapSummary(
            total_images=len(images),
            images_analyzed=len(images),
            total_damages=len(damages),
            zones_affected=zones_affected,
            overall_severity=overall,
            damages_by_severity=dict(severity_counts),
            damages_by_type=dict(type_counts),
        )

        return DamageMap(
            session_id=session_id,
            vehicle_context=vehicle_context,
            images=images,
            zones=zones,
            summary=summary,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_damage_map_builder.py -v
```
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/adapters/damage_map_builder.py tests/test_damage_map_builder.py
git commit -m "feat: add DamageMapBuilder pure Python adapter"
```

---

## Task 6: AnalysisTracer adapter

**Files:**
- Create: `crashguard-vision/app/adapters/db_tracer.py`
- Create: `crashguard-vision/tests/test_db_tracer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_db_tracer.py`:
```python
import pytest
from unittest.mock import MagicMock
from app.adapters.db_tracer import DbAnalysisTracer
import logging


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.create_analysis_call.return_value = "call-uuid-123"
    return repo


def test_record_success_returns_call_id(mock_repo):
    tracer = DbAnalysisTracer(repo=mock_repo)
    call_id = tracer.record(
        call_type="analyze_image",
        model="gemini-2.5-flash",
        latency_ms=1200,
        status="success",
        raw_response={"candidates": []},
        image_id="img-001",
    )
    assert call_id == "call-uuid-123"


def test_record_calls_repo_with_correct_args(mock_repo):
    tracer = DbAnalysisTracer(repo=mock_repo)
    tracer.record(
        call_type="aggregate_damages",
        model="gemini-2.5-flash",
        latency_ms=800,
        status="error",
        raw_response={},
        session_id="sess-001",
        error="timeout",
    )
    mock_repo.create_analysis_call.assert_called_once_with(
        call_type="aggregate_damages",
        model="gemini-2.5-flash",
        latency_ms=800,
        status="error",
        raw_response={},
        session_id="sess-001",
        image_id=None,
        prompt_tokens=None,
        response_tokens=None,
        error="timeout",
    )


def test_record_emits_structured_log(mock_repo, caplog):
    tracer = DbAnalysisTracer(repo=mock_repo)
    with caplog.at_level(logging.INFO, logger="vision.tracer"):
        tracer.record(
            call_type="analyze_image",
            model="gemini-2.5-flash",
            latency_ms=500,
            status="success",
            raw_response={},
        )
    assert any("analyze_image" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db_tracer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/adapters/db_tracer.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db_tracer.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/adapters/db_tracer.py tests/test_db_tracer.py
git commit -m "feat: add DbAnalysisTracer adapter"
```

---

## Task 7: GeminiImageAnalyzer adapter

**Files:**
- Create: `crashguard-vision/app/adapters/gemini_analyzer.py`
- Create: `crashguard-vision/tests/test_gemini_analyzer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_gemini_analyzer.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.domain.models import VehicleContext, DamageType, VehicleZone, Severity


GEMINI_RESPONSE_JSON = '[{"type":"abolladura","zone":"capot","severity":"medium","confidence":0.91,"bbox_x":0.12,"bbox_y":0.34,"bbox_w":0.18,"bbox_h":0.09,"description":"Abolladura en capot"}]'


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    response = MagicMock()
    response.text = GEMINI_RESPONSE_JSON
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    response.usage_metadata = usage
    client.models.generate_content.return_value = response
    return client


@pytest.fixture
def mock_image_bytes(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (3024, 4032), color=(100, 100, 100))
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return path.read_bytes()


def test_analyze_returns_damages(mock_gemini_client, mock_image_bytes):
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(
            client=mock_gemini_client, model="gemini-2.5-flash"
        )
        damages = analyzer.analyze(
            image_url="https://example.com/car.jpg",
            context=VehicleContext(make="Toyota"),
        )

    assert len(damages) == 1
    assert damages[0].type == DamageType.abolladura
    assert damages[0].zone == VehicleZone.capot
    assert damages[0].severity == Severity.medium
    assert damages[0].confidence == pytest.approx(0.91)
    assert damages[0].bbox.x == pytest.approx(0.12)


def test_analyze_unknown_zone_coerced(mock_gemini_client, mock_image_bytes):
    mock_gemini_client.models.generate_content.return_value.text = (
        '[{"type":"abolladura","zone":"zona_rara","severity":"low",'
        '"confidence":0.5,"bbox_x":0.1,"bbox_y":0.1,"bbox_w":0.1,"bbox_h":0.1,'
        '"description":"test"}]'
    )
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(client=mock_gemini_client, model="gemini-2.5-flash")
        damages = analyzer.analyze(image_url="https://example.com/car.jpg", context=None)

    assert damages[0].zone == VehicleZone.unknown


def test_analyze_returns_image_dimensions(mock_gemini_client, mock_image_bytes):
    with patch("app.adapters.gemini_analyzer.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = mock_image_bytes
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        analyzer = GeminiImageAnalyzer(client=mock_gemini_client, model="gemini-2.5-flash")
        damages, width, height = analyzer.analyze_with_dimensions(
            image_url="https://example.com/car.jpg", context=None
        )

    assert width == 3024
    assert height == 4032
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gemini_analyzer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/adapters/gemini_analyzer.py**

```python
from __future__ import annotations
import io
import json
import time
import uuid
import httpx
from PIL import Image
from google import genai
from google.genai import types
from app.domain.models import Damage, BoundingBox, VehicleContext


_SYSTEM_PROMPT = """Eres un perito de daños vehiculares. Analiza la imagen del vehículo e identifica todos los daños visibles.

Para cada daño retorna un objeto JSON con:
- type: uno de [abolladura, rayon, quiebre, mancha, corrosion, vidrio_roto, other]
- zone: zona del vehículo, uno de [capot, parabrisas, techo, maletero, puerta_delantera_izq, puerta_trasera_izq, puerta_delantera_der, puerta_trasera_der, lateral_izq, lateral_der, paragolpes_del, paragolpes_tras, espejo_izq, espejo_der, rueda_del_izq, rueda_del_der, rueda_tras_izq, rueda_tras_der]
- severity: low | medium | high
- confidence: float 0.0-1.0
- bbox_x, bbox_y, bbox_w, bbox_h: bounding box normalizado [0-1], origen top-left
- description: descripción breve en español

Retorna un array JSON. Si no hay daños retorna [].
"""


class GeminiImageAnalyzer:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    def _download_image(self, image_url: str) -> tuple[bytes, int, int]:
        response = httpx.get(image_url, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        return image_bytes, width, height

    def _build_prompt(self, context: VehicleContext | None) -> str:
        if context and any([context.make, context.model, context.year, context.color]):
            parts = [p for p in [context.make, context.model, str(context.year) if context.year else None, context.color] if p]
            return f"Vehículo: {' '.join(parts)}. Analiza los daños."
        return "Analiza los daños en este vehículo."

    def _parse_response(self, raw: str, source_image_id: str | None) -> list[Damage]:
        items = json.loads(raw)
        damages = []
        for i, item in enumerate(items):
            damages.append(Damage(
                id=f"dmg_{i+1:02d}",
                type=item.get("type", "other"),
                zone=item.get("zone", "unknown"),
                severity=item.get("severity", "low"),
                confidence=float(item.get("confidence", 0.0)),
                bbox=BoundingBox(
                    x=float(item.get("bbox_x", 0)),
                    y=float(item.get("bbox_y", 0)),
                    w=float(item.get("bbox_w", 0)),
                    h=float(item.get("bbox_h", 0)),
                ),
                description=item.get("description", ""),
                source_image_id=source_image_id,
            ))
        return damages

    def analyze(
        self,
        image_url: str,
        context: VehicleContext | None,
        source_image_id: str | None = None,
    ) -> list[Damage]:
        damages, _, _ = self.analyze_with_dimensions(image_url, context, source_image_id)
        return damages

    def analyze_with_dimensions(
        self,
        image_url: str,
        context: VehicleContext | None,
        source_image_id: str | None = None,
    ) -> tuple[list[Damage], int, int]:
        image_bytes, width, height = self._download_image(image_url)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=self._build_prompt(context)),
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        damages = self._parse_response(response.text, source_image_id)
        return damages, width, height

    def get_usage(self, response) -> tuple[int | None, int | None]:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return None, None
        return getattr(meta, "prompt_token_count", None), getattr(meta, "candidates_token_count", None)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gemini_analyzer.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/adapters/gemini_analyzer.py tests/test_gemini_analyzer.py
git commit -m "feat: add GeminiImageAnalyzer adapter"
```

---

## Task 8: GeminiDamageAggregator adapter

**Files:**
- Create: `crashguard-vision/app/adapters/gemini_aggregator.py`
- Create: `crashguard-vision/tests/test_gemini_aggregator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_gemini_aggregator.py`:
```python
import pytest
from unittest.mock import MagicMock
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity,
)


def make_damage(id: str, zone: VehicleZone, image_id: str) -> Damage:
    return Damage(
        id=id,
        type=DamageType.abolladura,
        zone=zone,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2),
        description="test",
        source_image_id=image_id,
    )


AGGREGATED_JSON = '[{"id":"dmg_01","type":"abolladura","zone":"capot","severity":"medium","confidence":0.91,"bbox_x":0.12,"bbox_y":0.34,"bbox_w":0.18,"bbox_h":0.09,"description":"Abolladura","source_image_id":"img_001","also_seen_in":["img_002"]}]'


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    response = MagicMock()
    response.text = AGGREGATED_JSON
    usage = MagicMock()
    usage.prompt_token_count = 200
    usage.candidates_token_count = 80
    response.usage_metadata = usage
    client.models.generate_content.return_value = response
    return client


def test_aggregate_returns_consolidated_damages(mock_gemini_client):
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    damage_lists = [
        [make_damage("d1", VehicleZone.capot, "img_001")],
        [make_damage("d2", VehicleZone.capot, "img_002")],
    ]
    result = aggregator.aggregate(damage_lists)

    assert len(result) == 1
    assert result[0].zone == VehicleZone.capot
    assert result[0].source_image_id == "img_001"
    assert "img_002" in result[0].also_seen_in


def test_aggregate_empty_input(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value.text = "[]"
    aggregator = GeminiDamageAggregator(client=mock_gemini_client, model="gemini-2.5-flash")
    result = aggregator.aggregate([])
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gemini_aggregator.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/adapters/gemini_aggregator.py**

```python
from __future__ import annotations
import json
from google import genai
from google.genai import types
from app.domain.models import Damage, BoundingBox


_SYSTEM_PROMPT = """Eres un especialista en consolidación de daños vehiculares.
Recibirás una lista de daños detectados en múltiples fotos del mismo vehículo.

Tu tarea:
1. Identificar qué daños de diferentes fotos representan el mismo daño físico (visto desde distintos ángulos)
2. Retornar una lista deduplicada. Para cada daño canonical, usa el de mayor confidence como base
3. En "also_seen_in" lista los source_image_id de otras fotos donde aparece el mismo daño
4. Si tienes duda, NO merges (conserva separados)

Regla: mismo daño = misma zone + mismo type + overlap similar de bbox

Retorna un array JSON con los mismos campos que la entrada más "also_seen_in": lista de image_ids.
Si la entrada está vacía retorna [].
"""


class GeminiDamageAggregator:
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    def aggregate(self, damage_lists: list[list[Damage]]) -> list[Damage]:
        all_damages = [d for sublist in damage_lists for d in sublist]
        if not all_damages:
            return []

        input_json = json.dumps([
            {
                "id": d.id,
                "type": d.type.value,
                "zone": d.zone.value,
                "severity": d.severity.value,
                "confidence": d.confidence,
                "bbox_x": d.bbox.x,
                "bbox_y": d.bbox.y,
                "bbox_w": d.bbox.w,
                "bbox_h": d.bbox.h,
                "description": d.description,
                "source_image_id": d.source_image_id,
            }
            for d in all_damages
        ], ensure_ascii=False)

        response = self._client.models.generate_content(
            model=self._model,
            contents=[types.Part.from_text(
                text=f"Consolida estos daños:\n{input_json}"
            )],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        return self._parse_response(response.text)

    def _parse_response(self, raw: str) -> list[Damage]:
        items = json.loads(raw)
        result = []
        for item in items:
            result.append(Damage(
                id=item.get("id", "dmg_00"),
                type=item.get("type", "other"),
                zone=item.get("zone", "unknown"),
                severity=item.get("severity", "low"),
                confidence=float(item.get("confidence", 0.0)),
                bbox=BoundingBox(
                    x=float(item.get("bbox_x", 0)),
                    y=float(item.get("bbox_y", 0)),
                    w=float(item.get("bbox_w", 0)),
                    h=float(item.get("bbox_h", 0)),
                ),
                description=item.get("description", ""),
                source_image_id=item.get("source_image_id"),
                also_seen_in=item.get("also_seen_in", []),
            ))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gemini_aggregator.py -v
```
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/adapters/gemini_aggregator.py tests/test_gemini_aggregator.py
git commit -m "feat: add GeminiDamageAggregator adapter"
```

---

## Task 9: AnalyzeService + /analyze router + FastAPI app

**Files:**
- Create: `crashguard-vision/app/services/analyze_service.py`
- Create: `crashguard-vision/app/routers/analyze.py`
- Create: `crashguard-vision/app/main.py`
- Create: `crashguard-vision/tests/test_analyze_router.py`
- Create: `crashguard-vision/tests/conftest.py`

- [ ] **Step 1: Write failing tests**

`tests/conftest.py`:
```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.main import app
from app.auth import verify_api_key
from app.domain.models import (
    Damage, BoundingBox, DamageType, VehicleZone, Severity, AnalysisSummary,
)


@pytest.fixture
def client():
    app.dependency_overrides[verify_api_key] = lambda: "test_key_hash"
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_damage():
    return Damage(
        id="dmg_01",
        type=DamageType.abolladura,
        zone=VehicleZone.capot,
        severity=Severity.medium,
        confidence=0.91,
        bbox=BoundingBox(x=0.12, y=0.34, w=0.18, h=0.09),
        description="Abolladura en capot",
    )
```

`tests/test_analyze_router.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from app.domain.models import AnalysisSummary, Severity, DamageType


def test_analyze_returns_damages(client, sample_damage):
    mock_service = MagicMock()
    mock_service.analyze.return_value = (
        [sample_damage],
        3024,
        4032,
    )

    with patch("app.routers.analyze.get_analyze_service", return_value=mock_service):
        response = client.post(
            "/analyze",
            json={"image_url": "https://example.com/car.jpg"},
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["image_width"] == 3024
    assert data["image_height"] == 4032
    assert len(data["damages"]) == 1
    assert data["damages"][0]["type"] == "abolladura"


def test_analyze_missing_image_url(client):
    response = client.post(
        "/analyze",
        json={},
        headers={"x-vision-key": "test"},
    )
    assert response.status_code == 422


def test_analyze_unauthorized():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        response = c.post("/analyze", json={"image_url": "https://x.com/car.jpg"})
    assert response.status_code == 422  # missing header
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_analyze_router.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/services/analyze_service.py**

```python
from __future__ import annotations
import time
from collections import Counter
from app.domain.models import (
    Damage, AnalysisSummary, VehicleContext, Severity, DamageType,
)
from app.domain.ports import ImageAnalyzer, AnalysisTracer
from app.config import settings


class AnalyzeService:
    def __init__(self, analyzer: ImageAnalyzer, tracer: AnalysisTracer) -> None:
        self._analyzer = analyzer
        self._tracer = tracer

    def analyze(
        self,
        image_url: str,
        context: VehicleContext | None,
    ) -> tuple[list[Damage], int, int]:
        t0 = time.monotonic()
        try:
            damages, width, height = self._analyzer.analyze_with_dimensions(
                image_url=image_url,
                context=context,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=settings.gemini_model,
                latency_ms=latency_ms,
                status="success",
                raw_response={},
            )
            return damages, width, height
        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            self._tracer.record(
                call_type="analyze_image",
                model=settings.gemini_model,
                latency_ms=latency_ms,
                status="error",
                raw_response={},
                error=str(exc),
            )
            raise
```

- [ ] **Step 4: Create app/routers/analyze.py**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from collections import Counter
from app.auth import verify_api_key
from app.domain.models import (
    VehicleContext, AnalyzeResponse, AnalysisSummary, Severity, DamageType,
)
from app.services.analyze_service import AnalyzeService
from app.dependencies import get_analyze_service

router = APIRouter()


class AnalyzeRequest(BaseModel):
    image_url: str
    vehicle_context: VehicleContext | None = None


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: AnalyzeService = Depends(get_analyze_service),
) -> AnalyzeResponse:
    import time
    t0 = time.monotonic()
    damages, width, height = service.analyze(
        image_url=request.image_url,
        context=request.vehicle_context,
    )
    processing_ms = int((time.monotonic() - t0) * 1000)

    severity_counts = Counter(d.severity for d in damages)
    type_counts = Counter(d.type for d in damages)

    return AnalyzeResponse(
        image_width=width,
        image_height=height,
        damages=damages,
        summary=AnalysisSummary(
            total_damages=len(damages),
            damages_by_severity=dict(severity_counts),
            damages_by_type=dict(type_counts),
            processing_ms=processing_ms,
        ),
    )
```

- [ ] **Step 5: Create app/dependencies.py**

```python
from functools import lru_cache
from google import genai
from app.config import settings
from app.db.supabase import get_client, VisionRepository
from app.adapters.gemini_analyzer import GeminiImageAnalyzer
from app.adapters.gemini_aggregator import GeminiDamageAggregator
from app.adapters.damage_map_builder import PythonDamageMapBuilder
from app.adapters.db_tracer import DbAnalysisTracer
from app.services.analyze_service import AnalyzeService
from app.services.session_service import SessionService


@lru_cache
def _gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def get_repo() -> VisionRepository:
    return VisionRepository(get_client())


def get_analyze_service() -> AnalyzeService:
    repo = get_repo()
    tracer = DbAnalysisTracer(repo=repo)
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    return AnalyzeService(analyzer=analyzer, tracer=tracer)


def get_session_service() -> SessionService:
    repo = get_repo()
    tracer = DbAnalysisTracer(repo=repo)
    analyzer = GeminiImageAnalyzer(client=_gemini_client(), model=settings.gemini_model)
    aggregator = GeminiDamageAggregator(client=_gemini_client(), model=settings.gemini_model)
    builder = PythonDamageMapBuilder()
    return SessionService(
        repo=repo,
        analyzer=analyzer,
        aggregator=aggregator,
        builder=builder,
        tracer=tracer,
    )
```

- [ ] **Step 6: Create app/main.py**

```python
from fastapi import FastAPI
from app.routers import analyze, sessions

app = FastAPI(title="crashguard-vision", version="0.1.0")

app.include_router(analyze.router)
app.include_router(sessions.router)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_analyze_router.py -v
```
Expected: 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add app/services/analyze_service.py app/routers/analyze.py app/dependencies.py app/main.py tests/conftest.py tests/test_analyze_router.py
git commit -m "feat: add /analyze endpoint"
```

---

## Task 10: Session CRUD endpoints

**Files:**
- Create: `crashguard-vision/app/services/session_service.py` (create + get)
- Create: `crashguard-vision/app/routers/sessions.py` (POST /sessions, GET /sessions/{id})
- Create: `crashguard-vision/tests/test_sessions_router.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sessions_router.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


SESSION_ROW = {
    "id": "sess-uuid-001",
    "api_key_hash": "abc123",
    "vehicle_context": {"make": "Toyota"},
    "status": "open",
    "created_at": "2026-05-01T10:00:00+00:00",
    "expires_at": "2026-05-02T10:00:00+00:00",
}


def test_create_session(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.create_session.return_value = SESSION_ROW
        mock_factory.return_value = mock_service

        response = client.post(
            "/sessions",
            json={"vehicle_context": {"make": "Toyota"}},
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["session_id"] == "sess-uuid-001"
    assert "expires_at" in data


def test_create_session_no_context(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.create_session.return_value = {**SESSION_ROW, "vehicle_context": None}
        mock_factory.return_value = mock_service

        response = client.post(
            "/sessions",
            json={},
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 201


def test_get_session_not_found(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.get_session.return_value = None
        mock_factory.return_value = mock_service

        response = client.get(
            "/sessions/nonexistent-id",
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 404


def test_get_session_wrong_tenant(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.get_session.side_effect = PermissionError("wrong tenant")
        mock_factory.return_value = mock_service

        response = client.get(
            "/sessions/sess-uuid-001",
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sessions_router.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create app/services/session_service.py (create + get)**

```python
from __future__ import annotations
from app.db.supabase import VisionRepository
from app.domain.ports import ImageAnalyzer, DamageAggregator, DamageMapBuilder, AnalysisTracer
from app.auth import hash_key


class SessionService:
    def __init__(
        self,
        repo: VisionRepository,
        analyzer: ImageAnalyzer,
        aggregator: DamageAggregator,
        builder: DamageMapBuilder,
        tracer: AnalysisTracer,
    ) -> None:
        self._repo = repo
        self._analyzer = analyzer
        self._aggregator = aggregator
        self._builder = builder
        self._tracer = tracer

    def create_session(
        self,
        api_key_hash: str,
        vehicle_context: dict | None,
    ) -> dict:
        return self._repo.create_session(
            api_key_hash=api_key_hash,
            vehicle_context=vehicle_context,
        )

    def get_session(self, session_id: str, api_key_hash: str) -> dict:
        session = self._repo.get_session(session_id)
        if session is None:
            return None
        if session["api_key_hash"] != api_key_hash:
            raise PermissionError("Session belongs to another tenant")
        return session
```

- [ ] **Step 4: Create app/routers/sessions.py (POST /sessions + GET /sessions/{id})**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth import verify_api_key
from app.domain.models import VehicleContext
from app.services.session_service import SessionService
from app.dependencies import get_session_service

router = APIRouter(prefix="/sessions")


class CreateSessionRequest(BaseModel):
    vehicle_context: VehicleContext | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    expires_at: str


@router.post("", status_code=201, response_model=CreateSessionResponse)
def create_session(
    request: CreateSessionRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> CreateSessionResponse:
    context_dict = request.vehicle_context.model_dump() if request.vehicle_context else None
    session = service.create_session(
        api_key_hash=api_key_hash,
        vehicle_context=context_dict,
    )
    return CreateSessionResponse(
        session_id=session["id"],
        expires_at=session["expires_at"],
    )


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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sessions_router.py::test_create_session tests/test_sessions_router.py::test_create_session_no_context tests/test_sessions_router.py::test_get_session_not_found tests/test_sessions_router.py::test_get_session_wrong_tenant -v
```
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/session_service.py app/routers/sessions.py tests/test_sessions_router.py
git commit -m "feat: add session CRUD endpoints"
```

---

## Task 11: Session image upload endpoint

**Files:**
- Modify: `crashguard-vision/app/services/session_service.py` (add `add_image`)
- Modify: `crashguard-vision/app/routers/sessions.py` (add `POST /sessions/{id}/images`)
- Modify: `crashguard-vision/tests/test_sessions_router.py` (add image upload tests)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sessions_router.py`:
```python
from app.domain.models import Damage, BoundingBox, DamageType, VehicleZone, Severity, AnalysisSummary


IMAGE_ROW = {
    "id": "img-uuid-001",
    "session_id": "sess-uuid-001",
    "image_url": "https://example.com/car.jpg",
    "angle": "lateral_izq",
    "image_width": 3024,
    "image_height": 4032,
    "status": "completed",
    "damages": [],
    "uploaded_at": "2026-05-01T10:05:00+00:00",
    "analyzed_at": "2026-05-01T10:05:02+00:00",
}

SAMPLE_DAMAGE = Damage(
    id="dmg_01",
    type=DamageType.abolladura,
    zone=VehicleZone.capot,
    severity=Severity.medium,
    confidence=0.91,
    bbox=BoundingBox(x=0.12, y=0.34, w=0.18, h=0.09),
    description="Abolladura en capot",
    source_image_id="img-uuid-001",
)


def test_add_image_to_session(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.add_image.return_value = (IMAGE_ROW, [SAMPLE_DAMAGE], 3024, 4032)
        mock_factory.return_value = mock_service

        response = client.post(
            "/sessions/sess-uuid-001/images",
            json={"image_url": "https://example.com/car.jpg", "angle": "lateral_izq"},
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["image_id"] == "img-uuid-001"
    assert data["image_width"] == 3024
    assert data["status"] == "completed"
    assert len(data["damages"]) == 1


def test_add_image_session_not_found(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.add_image.side_effect = ValueError("Session not found")
        mock_factory.return_value = mock_service

        response = client.post(
            "/sessions/bad-id/images",
            json={"image_url": "https://example.com/car.jpg"},
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sessions_router.py::test_add_image_to_session tests/test_sessions_router.py::test_add_image_session_not_found -v
```
Expected: FAIL (route not found)

- [ ] **Step 3: Add add_image to app/services/session_service.py**

Add this method to `SessionService`:
```python
def add_image(
    self,
    session_id: str,
    api_key_hash: str,
    image_url: str,
    angle: str | None,
) -> tuple[dict, list, int, int]:
    import time
    from app.domain.models import VehicleContext

    session = self.get_session(session_id, api_key_hash)
    if session is None:
        raise ValueError("Session not found")

    image_row = self._repo.create_session_image(
        session_id=session_id,
        image_url=image_url,
        angle=angle,
    )
    image_id = image_row["id"]

    try:
        context_dict = session.get("vehicle_context")
        context = VehicleContext(**context_dict) if context_dict else None

        damages, width, height = self._analyzer.analyze_with_dimensions(
            image_url=image_url,
            context=context,
            source_image_id=image_id,
        )
        self._repo.update_image_analyzing(image_id, width, height)

        t0 = time.monotonic()
        call_id = self._tracer.record(
            call_type="analyze_image",
            model="gemini-2.5-flash",
            latency_ms=int((time.monotonic() - t0) * 1000),
            status="success",
            raw_response={},
            session_id=session_id,
            image_id=image_id,
        )

        damages_dicts = [d.model_dump() for d in damages]
        self._repo.update_image_completed(image_id, damages_dicts, call_id)

        updated_row = {**image_row, "status": "completed", "image_width": width, "image_height": height}
        return updated_row, damages, width, height

    except Exception as exc:
        self._tracer.record(
            call_type="analyze_image",
            model="gemini-2.5-flash",
            latency_ms=0,
            status="error",
            raw_response={},
            session_id=session_id,
            image_id=image_id,
            error=str(exc),
        )
        self._repo.update_image_failed(image_id, str(exc))
        error_row = {**image_row, "status": "failed", "error": str(exc)}
        return error_row, [], 0, 0
```

- [ ] **Step 4: Add POST /sessions/{id}/images to app/routers/sessions.py**

Add these to `sessions.py`:
```python
class AddImageRequest(BaseModel):
    image_url: str
    angle: str | None = None


@router.post("/{session_id}/images", status_code=201)
def add_image(
    session_id: str,
    request: AddImageRequest,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
):
    from collections import Counter
    from app.domain.models import AnalysisSummary

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

    summary = AnalysisSummary(
        total_damages=len(damages),
        damages_by_severity=dict(Counter(d.severity for d in damages)),
        damages_by_type=dict(Counter(d.type for d in damages)),
        processing_ms=0,
    ) if image_row["status"] == "completed" else None

    from app.domain.models import SessionImageResponse
    return SessionImageResponse(
        image_id=image_row["id"],
        image_width=width,
        image_height=height,
        status=image_row["status"],
        damages=damages,
        error=image_row.get("error"),
        summary=summary,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sessions_router.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/session_service.py app/routers/sessions.py tests/test_sessions_router.py
git commit -m "feat: add POST /sessions/{id}/images endpoint"
```

---

## Task 12: Session report endpoint

**Files:**
- Modify: `crashguard-vision/app/services/session_service.py` (add `get_report`)
- Modify: `crashguard-vision/app/routers/sessions.py` (add `GET /sessions/{id}/report`)
- Modify: `crashguard-vision/tests/test_sessions_router.py` (add report tests)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sessions_router.py`:
```python
from app.domain.models import DamageMap, DamageMapSummary, VehicleZone, SourceImageMeta


DAMAGE_MAP = DamageMap(
    session_id="sess-uuid-001",
    vehicle_context=None,
    images={"img-uuid-001": SourceImageMeta(url="https://example.com/car.jpg", width=3024, height=4032)},
    zones={z: [] for z in VehicleZone} | {VehicleZone.capot: [SAMPLE_DAMAGE]},
    summary=DamageMapSummary(
        total_images=1,
        images_analyzed=1,
        total_damages=1,
        zones_affected=[VehicleZone.capot],
        overall_severity=Severity.medium,
        damages_by_severity={Severity.medium: 1},
        damages_by_type={DamageType.abolladura: 1},
    ),
)


def test_get_report(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.get_report.return_value = DAMAGE_MAP
        mock_factory.return_value = mock_service

        response = client.get(
            "/sessions/sess-uuid-001/report",
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess-uuid-001"
    assert "zones" in data
    assert "summary" in data
    assert data["summary"]["total_damages"] == 1


def test_get_report_session_not_found(client):
    with patch("app.routers.sessions.get_session_service") as mock_factory:
        mock_service = MagicMock()
        mock_service.get_report.side_effect = ValueError("Session not found")
        mock_factory.return_value = mock_service

        response = client.get(
            "/sessions/bad-id/report",
            headers={"x-vision-key": "test"},
        )

    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_sessions_router.py::test_get_report tests/test_sessions_router.py::test_get_report_session_not_found -v
```
Expected: FAIL (route not found)

- [ ] **Step 3: Add get_report to app/services/session_service.py**

Add this method to `SessionService`:
```python
def get_report(self, session_id: str, api_key_hash: str) -> "DamageMap":
    import time
    from app.domain.models import (
        Damage, SourceImageMeta, VehicleContext, DamageMapSummary,
        SkippedImage, VehicleZone,
    )

    session = self.get_session(session_id, api_key_hash)
    if session is None:
        raise ValueError("Session not found")

    # Check cache
    all_images = self._repo.get_all_images(session_id)
    completed_images = self._repo.get_completed_images(session_id)
    cached_map = self._repo.get_damage_map(session_id)

    if cached_map and cached_map["image_count"] == len(completed_images):
        # Reconstruct DamageMap from stored JSON
        from app.domain.models import DamageMap
        return DamageMap(**{
            "session_id": session_id,
            "vehicle_context": session.get("vehicle_context"),
            "images": {k: SourceImageMeta(**v) for k, v in cached_map["images"].items()},
            "zones": {
                VehicleZone(k): [Damage(**d) for d in v]
                for k, v in cached_map["zones"].items()
            },
            "summary": DamageMapSummary(**cached_map["summary"]),
        })

    # Build images registry
    images_meta: dict[str, SourceImageMeta] = {
        row["id"]: SourceImageMeta(
            url=row["image_url"],
            width=row["image_width"],
            height=row["image_height"],
            angle=row.get("angle"),
        )
        for row in completed_images
    }

    # Build damage lists
    damage_lists: list[list[Damage]] = []
    for row in completed_images:
        raw_damages = row.get("damages") or []
        damages = [Damage(**d) for d in raw_damages]
        damage_lists.append(damages)

    # Aggregate
    t0 = time.monotonic()
    aggregated = self._aggregator.aggregate(damage_lists)
    self._tracer.record(
        call_type="aggregate_damages",
        model="gemini-2.5-flash",
        latency_ms=int((time.monotonic() - t0) * 1000),
        status="success",
        raw_response={},
        session_id=session_id,
    )

    # Build skipped list
    completed_ids = {row["id"] for row in completed_images}
    skipped = [
        SkippedImage(image_id=img["id"], reason=img.get("status", "unknown"))
        for img in all_images
        if img["id"] not in completed_ids
    ]

    # Build map
    context_dict = session.get("vehicle_context")
    context = VehicleContext(**context_dict) if context_dict else None

    damage_map = self._builder.build(
        damages=aggregated,
        images=images_meta,
        session_id=session_id,
        vehicle_context=context,
    )
    damage_map.summary.images_skipped = skipped
    damage_map.summary.total_images = len(all_images)

    # Persist
    self._repo.upsert_damage_map(
        session_id=session_id,
        images={k: v.model_dump() for k, v in images_meta.items()},
        zones={k.value: [d.model_dump() for d in v] for k, v in damage_map.zones.items()},
        summary=damage_map.summary.model_dump(),
        image_count=len(completed_images),
    )

    return damage_map
```

- [ ] **Step 4: Add GET /sessions/{id}/report to app/routers/sessions.py**

Add this to `sessions.py`:
```python
@router.get("/{session_id}/report", response_model=DamageMap)
def get_report(
    session_id: str,
    api_key_hash: str = Depends(verify_api_key),
    service: SessionService = Depends(get_session_service),
) -> DamageMap:
    from app.domain.models import DamageMap
    try:
        return service.get_report(session_id=session_id, api_key_hash=api_key_hash)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
```

Add `from app.domain.models import DamageMap` to the imports at the top of `sessions.py`.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/session_service.py app/routers/sessions.py tests/test_sessions_router.py
git commit -m "feat: add GET /sessions/{id}/report endpoint"
```

---

## Task 13: Dockerfile + cloudbuild.yaml

**Files:**
- Create: `crashguard-vision/Dockerfile`
- Create: `crashguard-vision/cloudbuild.yaml`
- Create: `crashguard-vision/.dockerignore`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && pip install --no-cache-dir .

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app/ ./app/
ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Create .dockerignore**

```
__pycache__/
*.pyc
*.pyo
.env
.env.*
tests/
docs/
*.md
.git/
```

- [ ] **Step 3: Create cloudbuild.yaml**

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - build
      - -t
      - '$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPO/crashguard-vision:$COMMIT_SHA'
      - .

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - push
      - '$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPO/crashguard-vision:$COMMIT_SHA'

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - run
      - deploy
      - crashguard-vision
      - --image=$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPO/crashguard-vision:$COMMIT_SHA
      - --region=$_REGION
      - --platform=managed
      - --min-instances=0
      - --max-instances=10
      - --allow-unauthenticated
      - --set-secrets=GEMINI_API_KEY=GEMINI_API_KEY:latest,SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest,VISION_API_KEY=VISION_API_KEY:latest

substitutions:
  _REGION: us-central1
  _REPO: crashguard

images:
  - '$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPO/crashguard-vision:$COMMIT_SHA'
```

- [ ] **Step 4: Build Docker image locally to verify**

```bash
docker build -t crashguard-vision:local .
```
Expected: build succeeds

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore cloudbuild.yaml
git commit -m "feat: add Dockerfile and Cloud Build config"
```

---

## Self-Review Checklist

- [x] Spec §1 Propósito: Task 9 (`/analyze`) + Tasks 10-12 (sessions)
- [x] Spec §2 Stack: pyproject.toml uses google-genai, supabase, FastAPI, Pillow
- [x] Spec §3 Arquitectura hexagonal: ports.py → adapters → services → routers
- [x] Spec §4 Protocolos: all 4 protocols in ports.py, Task 2
- [x] Spec §5 Modelos: all models in models.py with enum fallbacks, Task 2
- [x] Spec §6 Responses: AnalyzeResponse, SessionImageResponse, DamageMap in models.py
- [x] Spec §7 Schema Supabase: migration in Task 3 (circular FK resolved: no FK on image_id/gemini_call_id)
- [x] Spec §8 Flujos: analyze flow Task 9, session image Task 11, report Task 12
- [x] Spec §9 Auth: X-Vision-Key dependency Task 4, api_key_hash stored in sessions
- [x] Spec §10 Estructura: matches file map exactly
- [x] Spec §11 Env vars: all in config.py Task 1
- [x] Spec §12 Observabilidad: DbAnalysisTracer logs + DB writes, Task 6
- [x] Spec §13 Training data: verified_damages columns in migration Task 3
- [x] Spec §14 Deploy: Dockerfile + cloudbuild.yaml Task 13
