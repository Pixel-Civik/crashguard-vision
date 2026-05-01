# crashguard-vision — Design Spec

**Date:** 2026-05-01  
**Status:** Approved

---

## 1. Propósito

Servicio standalone Python/FastAPI que analiza daños vehiculares en imágenes usando Gemini Vision. Produce JSON estructurado de daños con bounding boxes. Es el proveedor de inteligencia visual para `crashguard-api` (`inspection-analysis` module).

Dos productos:
1. **Análisis de imagen individual** — stateless, una foto → lista de daños con bboxes
2. **Sesión multi-foto** — 10-20 fotos de un vehículo → damage map consolidado por zona, persistido en DB

El frontend renderiza la visualización (3D o lo que decida). El backend solo produce JSON bien estructurado y lo persiste.

---

## 2. Stack

- Python 3.12 + FastAPI
- Gemini SDK directo (`google-genai`) — sin ADK
- Supabase (PostgreSQL) para sesiones y damage maps
- Cloud Run (mismo patrón que el resto del ecosistema)

**Decisión ADK:** No se usa ADK. El flujo de orquestación es siempre determinístico (Python lo controla). ADK agrega overhead para un pipeline fijo. Las interfaces (protocolos Python) permiten swapear la implementación de Gemini por otro modelo o ADK en el futuro sin cambiar rutas ni servicios.

---

## 3. Arquitectura de capas

```
FastAPI Routers
      ↓
  Services          — orquestación (qué llamar y cuándo)
      ↓
  Ports             — interfaces/protocolos Python
      ↓
  Adapters          — implementaciones concretas (hoy: Gemini SDK)
      ↓
Infrastructure      — Supabase, Gemini SDK, Pillow
```

Patrón hexagonal idéntico a `crashguard-api`.

---

## 4. Protocolos core (Ports)

```python
class ImageAnalyzer(Protocol):
    def analyze(self, image_url: str, context: VehicleContext | None) -> list[Damage]: ...

class DamageAggregator(Protocol):
    def aggregate(self, damage_lists: list[list[Damage]]) -> list[Damage]: ...

class DamageMapBuilder(Protocol):
    def build(self, damages: list[Damage], images: dict[str, SourceImageMeta]) -> DamageMap: ...

class AnalysisTracer(Protocol):
    def record(self,
               call_type: str,           # analyze_image | aggregate_damages
               model: str,
               latency_ms: int,
               status: str,              # success | error
               raw_response: dict,
               session_id: str | None = None,
               image_id: str | None = None,
               prompt_tokens: int | None = None,
               response_tokens: int | None = None,
               error: str | None = None) -> str: ...  # retorna call_id
```

- `ImageAnalyzer` → una llamada Gemini por imagen, responde JSON estructurado (structured output)
- `DamageAggregator` → una llamada Gemini con todos los daños de la sesión, consolida y deduplica (razona cuáles son el mismo golpe visto desde distintos ángulos)
- `DamageMapBuilder` → Python puro, mapea daños a zonas canónicas, construye el DamageMap
- `AnalysisTracer` → registra cada llamada a Gemini en DB y emite structured log. Implementado en `adapters/db_tracer.py`. Los adapters de Gemini lo reciben por inyección.

---

## 5. Modelos Pydantic

```python
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
    unknown = "unknown"             # fallback si Gemini devuelve zona no reconocida

class DamageType(str, Enum):
    abolladura = "abolladura"
    rayon = "rayon"
    quiebre = "quiebre"
    mancha = "mancha"
    corrosion = "corrosion"
    vidrio_roto = "vidrio_roto"
    other = "other"                 # fallback

class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class BoundingBox(BaseModel):
    x: float    # normalizado [0-1] relativo a la imagen fuente
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
    width: int          # píxeles originales
    height: int
    angle: str | None   # lateral_izq, frontal, etc.

class Damage(BaseModel):
    id: str
    type: DamageType
    zone: VehicleZone
    severity: Severity
    confidence: float           # 0.0 - 1.0
    bbox: BoundingBox           # normalizado relativo a source_image_id
    description: str
    source_image_id: str | None = None   # None solo en /analyze stateless
    also_seen_in: list[str] = []         # otros image_ids con el mismo daño (post-agregación)

class AnalysisSummary(BaseModel):
    total_damages: int
    damages_by_severity: dict[Severity, int]
    damages_by_type: dict[DamageType, int]
    processing_ms: int

class SkippedImage(BaseModel):
    image_id: str
    reason: str                                 # "analysis_failed" | "pending"

class DamageMapSummary(BaseModel):
    total_images: int
    images_analyzed: int                        # puede ser < total si hubo fallos
    images_skipped: list[SkippedImage] = []
    total_damages: int
    zones_affected: list[VehicleZone]
    overall_severity: Severity
    damages_by_severity: dict[Severity, int]
    damages_by_type: dict[DamageType, int]

class DamageMap(BaseModel):
    session_id: str
    vehicle_context: VehicleContext | None
    images: dict[str, SourceImageMeta]          # registro central de imágenes
    zones: dict[VehicleZone, list[Damage]]      # todas las zonas, lista vacía si sin daños
    summary: DamageMapSummary
```

**Nota bboxes:** Cada `Damage.bbox` está normalizado relativo a `source_image_id`. Para denormalizar en el frontend:
```js
const src = images[damage.source_image_id]
const px = { x: damage.bbox.x * src.width, y: damage.bbox.y * src.height, ... }
```
Las imágenes en una sesión pueden tener dimensiones distintas — cada daño lleva su referencia.

---

## 6. Responses de API

### POST /analyze (stateless)

```python
class AnalyzeResponse(BaseModel):
    image_width: int
    image_height: int
    damages: list[Damage]       # source_image_id siempre None
    summary: AnalysisSummary
```

### POST /sessions/{id}/images

```python
class SessionImageResponse(BaseModel):
    image_id: str
    image_width: int
    image_height: int
    status: str                 # completed | failed
    damages: list[Damage]       # source_image_id = image_id
    error: str | None
    summary: AnalysisSummary
```

### GET /sessions/{id}/report

Retorna `DamageMap` completo (ver modelo arriba).

Si hay imágenes `failed` o `pending`: el report se construye con las `completed` e informa las omitidas en `summary.images_skipped`. No falla — devuelve el mejor map posible con lo disponible.

**Reconstrucción:** Si la sesión tiene más imágenes que `damage_map.image_count`, el report se reconstruye. Sino devuelve el cacheado. Esto evita recalcular innecesariamente y previene race conditions entre requests concurrentes.

---

## 7. Schema Supabase

```sql
-- Sesiones multi-foto
create table vision_sessions (
  id              uuid primary key default gen_random_uuid(),
  api_key_hash    text not null,          -- hash de X-Vision-Key (scope de seguridad)
  vehicle_context jsonb,
  status          text not null default 'open',
    -- open | processing | completed | failed | expired
  created_at      timestamptz not null default now(),
  expires_at      timestamptz not null
);

-- Foto individual + sus daños
create table vision_session_images (
  id               uuid primary key default gen_random_uuid(),
  session_id       uuid not null references vision_sessions(id),
  image_url        text not null,
  angle            text,
  image_width      int,                      -- null hasta analizar
  image_height     int,
  status           text not null default 'pending',
    -- pending | analyzing | completed | failed
  damages          jsonb,                    -- list[Damage], null hasta completed
  gemini_call_id   uuid references vision_analysis_calls(id),
  error            text,                     -- razón si status = failed
  -- training data:
  verified_damages jsonb,                    -- damages corregidos por humano (ground truth)
  verified_at      timestamptz,
  verified_by      text,
  uploaded_at      timestamptz not null default now(),
  analyzed_at      timestamptz
);

-- Auditoría de cada llamada a Gemini (monitoreo + training)
create table vision_analysis_calls (
  id              uuid primary key default gen_random_uuid(),
  call_type       text not null,     -- analyze_image | aggregate_damages
  session_id      uuid references vision_sessions(id),
  image_id        uuid references vision_session_images(id),
  model           text not null,
  prompt_tokens   int,
  response_tokens int,
  latency_ms      int,
  status          text not null,     -- success | error
  error           text,
  raw_response    jsonb,             -- respuesta completa de Gemini
  created_at      timestamptz not null default now()
);

-- Damage map consolidado (output final)
create table vision_damage_maps (
  id          uuid primary key default gen_random_uuid(),
  session_id  uuid not null references vision_sessions(id) unique,
  images      jsonb not null,            -- dict[image_id, SourceImageMeta]
  zones       jsonb not null,            -- dict[VehicleZone, list[Damage]]
  summary     jsonb not null,
  image_count int not null,              -- fotos usadas para construir este map
  built_at    timestamptz not null default now()
);
```

**`api_key_hash` en sesiones:** Cada sesión está scoped a la key que la creó. Un tenant no puede acceder a sesiones de otro.

**`uploaded_at` vs `analyzed_at`:** `uploaded_at` preserva el orden de carga. `analyzed_at` refleja cuándo terminó el análisis (puede ser distinto por latencia de Gemini).

**Imágenes con distinto tamaño:** `image_width`/`image_height` se guardan por imagen en `vision_session_images` y también en el `images` dict del damage map — cada daño referencia su imagen fuente con sus propias dimensiones.

---

## 8. Flujos principales

### /analyze (stateless)
```
1. Descargar imagen → leer image_width, image_height (Pillow)
2. t0 = now()
3. GeminiImageAnalyzer.analyze(image_url, context) → list[Damage]
4. tracer.record(call_type="analyze_image", latency_ms=now()-t0, raw_response=...) → call_id
5. Retornar AnalyzeResponse (no persiste nada en DB)
```

### POST /sessions/{id}/images
```
1. Validar sesión existe y pertenece a la api_key (api_key_hash)
2. Insertar fila en vision_session_images (status=pending)
3. Descargar imagen → leer image_width, image_height (Pillow)
4. Actualizar fila (status=analyzing, image_width, image_height)
5. t0 = now()
6. GeminiImageAnalyzer.analyze() → list[Damage] con source_image_id
7. call_id = tracer.record(call_type="analyze_image", image_id=id, latency_ms=..., raw_response=...)
8. Actualizar fila (status=completed, damages, gemini_call_id=call_id, analyzed_at)
9. Retornar SessionImageResponse
   — si falla Gemini: tracer.record(status="error", error=msg), status=failed
```

### GET /sessions/{id}/report
```
1. Validar sesión y ownership
2. Cargar vision_damage_maps WHERE session_id = id
3. Cargar COUNT de vision_session_images WHERE status=completed
   — Si damage_map existe Y image_count == completed_count: devolver cacheado
   — Si no: reconstruir
4. Cargar todos los damages de imágenes completed (con image_width, image_height)
5. t0 = now()
6. GeminiDamageAggregator.aggregate(damage_lists) → list[Damage] consolidado
7. tracer.record(call_type="aggregate_damages", session_id=id, latency_ms=..., raw_response=...)
8. DamageMapBuilder.build(aggregated, images_meta) → DamageMap (Python puro)
9. Upsert vision_damage_maps
10. Retornar DamageMap
```

---

## 9. Autenticación

`X-Vision-Key` header en todos los endpoints. FastAPI dependency valida la key contra un hash almacenado en config. Se guarda `api_key_hash` en `vision_sessions` para scoping.

Sin JWT ni Supabase Auth — es un servicio interno.

---

## 10. Estructura de proyecto

```
crashguard-vision/
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── analyze.py           # POST /analyze
│   │   └── sessions.py          # CRUD + report
│   ├── domain/
│   │   ├── models.py            # todos los Pydantic models y enums
│   │   └── ports.py             # ImageAnalyzer, DamageAggregator, DamageMapBuilder
│   ├── adapters/
│   │   ├── gemini_analyzer.py   # implementa ImageAnalyzer
│   │   ├── gemini_aggregator.py # implementa DamageAggregator
│   │   └── db_tracer.py         # implementa AnalysisTracer → escribe a vision_analysis_calls
│   ├── services/
│   │   ├── analyze_service.py   # orquesta /analyze
│   │   └── session_service.py   # orquesta sesiones y report
│   ├── db/
│   │   └── supabase.py
│   └── config.py
├── tests/
├── Dockerfile
├── cloudbuild.yaml
└── pyproject.toml
```

---

## 11. Variables de entorno

```
GEMINI_API_KEY
GEMINI_MODEL          # default: gemini-2.5-flash
SUPABASE_URL
SUPABASE_SERVICE_KEY  # service role
VISION_API_KEY        # key para autenticar callers
SESSION_TTL_HOURS     # default: 24
PORT                  # default: 8080
```

---

## 12. Observabilidad

**Structured logging** en cada llamada a Gemini — Cloud Run exporta stdout a Cloud Logging automáticamente:

```python
logger.info("gemini_call", extra={
    "call_type": "analyze_image",
    "session_id": session_id,
    "image_id": image_id,
    "model": model,
    "latency_ms": latency,
    "prompt_tokens": tokens.input,
    "response_tokens": tokens.output,
    "status": "success",
})
```

Desde Cloud Logging se crean **log-based metrics** en Cloud Monitoring:
- Tasa de error por `call_type`
- Latencia p50/p95
- Volumen de llamadas por hora
- Tokens consumidos por día (proxy de costo)

**`vision_analysis_calls`** cubre el debugging detallado: para cualquier imagen fallida se puede buscar `WHERE image_id = X` y ver `raw_response` + `error` exacto.

---

## 13. Training data

Flujo para entrenar un modelo custom en el futuro:

1. Gemini genera labels iniciales → `vision_session_images.damages`
2. Humano revisa y corrige → `vision_session_images.verified_damages` + `verified_at` + `verified_by`
3. Export query:

```sql
select
  i.image_url,
  i.image_width,
  i.image_height,
  i.angle,
  coalesce(i.verified_damages, i.damages) as labels,
  i.verified_at is not null as is_ground_truth
from vision_session_images i
where i.status = 'completed'
  and i.damages is not null;
```

`coalesce(verified_damages, damages)` — si hay corrección humana la usa, sino usa la de Gemini. `is_ground_truth` permite filtrar solo los verificados para training de mayor calidad.

Sin endpoint de export por ahora — query directa a Supabase con service key.

---

## 14. Deploy

- Docker multi-stage: Python 3.12 slim
- Cloud Run: `min-instances: 0`, `max-instances: 10`
- `cloudbuild.yaml` igual al patrón de `crashguard-api`
- Agent Engine descartado: Cloud Run es $0.00/mes para el volumen esperado (dentro del free tier de 180k vCPU-segundos). Agent Engine no tiene scale-to-zero automático y cobra $0.25/1,000 session events.
