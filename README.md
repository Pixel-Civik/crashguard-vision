# crashguard-vision

Servicio de análisis de daños vehiculares con computer vision. Componente standalone de PixelCivic — recibe fotos de un vehículo y devuelve daños detectados con coordenadas (bounding boxes). Soporta análisis por foto individual y sesiones multi-foto para construir un mapa de fallas del vehículo.

Futuro consumidor: `crashguard-api` módulo `inspection-analysis` para detección de preexistencias.

---

## Stack

- **Runtime**: Python 3.12 + FastAPI
- **Agente**: Google ADK (open-source, optimizado para Gemini)
- **Modelo de visión**: Gemini 2.5 Flash
- **Sesiones**: Supabase (misma instancia que crashguard-api)
- **Deploy**: Cloud Run (mismo patrón que el resto del ecosistema)

---

## API

### POST `/analyze` — análisis single-photo (stateless)

Recibe una imagen, devuelve daños detectados. Internamente crea una sesión efímera de un solo paso.

```json
// Request
{
  "image_url": "https://...",   // URL pública o signed URL de Supabase Storage
  "vehicle_context": {          // opcional, mejora precisión
    "make": "Toyota",
    "model": "Corolla",
    "year": 2019,
    "color": "blanco"
  }
}

// Response
{
  "damages": [
    {
      "id": "dmg_01",
      "type": "abolladura",          // abolladura | rayón | quiebre | mancha | corrosión | vidrio_roto
      "zone": "puerta_delantera_izq",
      "severity": "medium",          // low | medium | high
      "confidence": 0.91,
      "bbox": {
        "x": 0.12,   // coordenadas normalizadas [0, 1]
        "y": 0.34,
        "w": 0.18,
        "h": 0.09
      },
      "description": "Abolladura visible en panel inferior de puerta delantera izquierda"
    }
  ],
  "summary": {
    "total_damages": 1,
    "max_severity": "medium",
    "processing_ms": 1840
  }
}
```

---

### POST `/sessions` — crear sesión multi-foto

```json
// Request
{
  "vehicle_context": { "make": "Toyota", "model": "Corolla", "year": 2019 }
}

// Response
{ "session_id": "sess_abc123", "expires_at": "2026-04-29T22:00:00Z" }
```

### POST `/sessions/{session_id}/images` — agregar foto a la sesión

```json
// Request
{ "image_url": "https://...", "angle": "lateral_izq" }  // angle opcional

// Response
{
  "image_id": "img_xyz",
  "damages_found": 2,        // análisis inmediato por imagen
  "damages": [ ... ]         // misma estructura que /analyze
}
```

### GET `/sessions/{session_id}/report` — mapa de fallas consolidado

Agrega todos los daños de la sesión, deduplica solapamientos entre fotos y construye el mapa por zona.

```json
{
  "session_id": "sess_abc123",
  "vehicle_context": { ... },
  "damage_map": {
    "capot": [],
    "puerta_delantera_izq": [ { ...damage } ],
    "puerta_trasera_izq": [],
    "lateral_der": [ { ...damage } ],
    "parabrisas": [],
    "techo": [],
    "maletero": [],
    "paragolpes_del": [],
    "paragolpes_tras": []
  },
  "summary": {
    "total_images": 5,
    "total_damages": 3,
    "zones_affected": ["puerta_delantera_izq", "lateral_der"],
    "overall_severity": "medium"
  }
}
```

---

## Arquitectura del agente (Google ADK)

El agente orquesta tres tools:

```
analyze_image(image_url, vehicle_context)
  → llama Gemini Vision con prompt estructurado
  → devuelve lista de damages con bboxes

aggregate_damages(damage_lists[])
  → consolida daños de múltiples imágenes
  → deduplica por zona + tipo + overlap de bbox
  → calcula severidad agregada por zona

build_damage_map(aggregated_damages, vehicle_zones)
  → mapea cada daño a zona canónica del vehículo
  → devuelve damage_map estructurado por zona
```

`/analyze` ejecuta solo `analyze_image`.  
`/sessions/{id}/report` ejecuta `aggregate_damages` → `build_damage_map`.

---

## Zonas canónicas del vehículo

```
capot | parabrisas | techo | maletero
puerta_delantera_izq | puerta_trasera_izq
puerta_delantera_der | puerta_trasera_der
lateral_izq | lateral_der
paragolpes_del | paragolpes_tras
espejo_izq | espejo_der
rueda_del_izq | rueda_del_der | rueda_tras_izq | rueda_tras_der
```

---

## Prompt de visión (Gemini)

El sistema instruction le indica a Gemini que actúe como perito de daños. Por cada daño detectado debe devolver:
- `type` — tipo de daño
- `zone` — zona del vehículo usando las zonas canónicas
- `severity` — low / medium / high
- `confidence` — 0 a 1
- `bbox` — x, y, w, h normalizados (0–1) relativo a la imagen
- `description` — descripción breve en español

El modelo responde siempre en JSON (`responseMimeType: application/json`).

---

## Sesiones en Supabase

Tabla `vision_sessions`:
```
id            uuid PK
vehicle_context_json  jsonb
status        text  (open | completed | expired)
created_at    timestamptz
expires_at    timestamptz
```

Tabla `vision_session_images`:
```
id            uuid PK
session_id    uuid FK → vision_sessions
image_url     text
angle         text nullable
damages_json  jsonb
analyzed_at   timestamptz
```

---

## Autenticación

Endpoint protegido por API Key (`X-Vision-Key` header). La key la emite crashguard-api (o se configura manualmente por tenant). Sin JWT ni Supabase auth — es un servicio interno.

---

## Estructura del proyecto

```
crashguard-vision/
├── app/
│   ├── main.py              # FastAPI app + routers
│   ├── routers/
│   │   ├── analyze.py       # POST /analyze
│   │   └── sessions.py      # CRUD de sesiones
│   ├── agent/
│   │   ├── agent.py         # ADK agent definition
│   │   └── tools/
│   │       ├── analyze_image.py
│   │       ├── aggregate_damages.py
│   │       └── build_damage_map.py
│   ├── models/
│   │   ├── damage.py        # Pydantic models
│   │   └── session.py
│   ├── db/
│   │   └── supabase.py      # Supabase client
│   └── config.py            # Env vars
├── tests/
├── Dockerfile
├── cloudbuild.yaml
├── pyproject.toml
└── README.md
```

---

## Variables de entorno

```
GEMINI_API_KEY          # o VERTEX_AI_PROJECT para Vertex
GEMINI_MODEL            # default: gemini-2.5-flash
SUPABASE_URL
SUPABASE_SERVICE_KEY    # service role, no anon
VISION_API_KEY          # key para autenticar callers internos
PORT                    # default: 8080
```

---

## Deploy

Mismo patrón que `crashguard-api`:
- Docker multi-stage: Python 3.12 slim → imagen final
- `cloudbuild.yaml` → Cloud Run
- Escala a cero fuera de horario (análisis no es tiempo real)
- `min-instances: 0`, `max-instances: 10`

---

## Integración futura con crashguard-api

Cuando `inspection-analysis` quiera usar este servicio:

```
POST https://crashguard-vision-xxx.run.app/sessions
  → crea sesión con vehicle_context de la inspección

POST /sessions/{id}/images  (x N fotos de la inspección)
  → análisis por foto

GET /sessions/{id}/report
  → damage_map → persiste como InspectionAIFinding en Supabase
```

El módulo `inspection-analysis` existente no cambia — solo agrega un step que llama a crashguard-vision.
