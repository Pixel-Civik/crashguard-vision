from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, field_validator


class VehicleZone(str, Enum):
    hood = "hood"
    windshield = "windshield"
    roof = "roof"
    trunk = "trunk"
    front_door_left = "front_door_left"
    rear_door_left = "rear_door_left"
    front_door_right = "front_door_right"
    rear_door_right = "rear_door_right"
    side_left = "side_left"
    side_right = "side_right"
    front_bumper = "front_bumper"
    rear_bumper = "rear_bumper"
    mirror_left = "mirror_left"
    mirror_right = "mirror_right"
    wheel_front_left = "wheel_front_left"
    wheel_front_right = "wheel_front_right"
    wheel_rear_left = "wheel_rear_left"
    wheel_rear_right = "wheel_rear_right"
    unknown = "unknown"


class DamageType(str, Enum):
    dent = "dent"
    scratch = "scratch"
    crack = "crack"
    stain = "stain"
    rust = "rust"
    broken_glass = "broken_glass"
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
    prompt_tokens: int | None = None
    response_tokens: int | None = None


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
    total_processing_ms: int = 0
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0


class DamageMap(BaseModel):
    session_id: str
    vehicle_context: VehicleContext | None
    images: dict[str, SourceImageMeta]
    zones: dict[VehicleZone, list[Damage]]
    summary: DamageMapSummary

