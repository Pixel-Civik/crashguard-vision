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
