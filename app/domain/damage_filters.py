from __future__ import annotations

import re

from app.domain.models import Damage, DamageType, VehicleZone


_WHEEL_ZONES = {
    VehicleZone.wheel_front_left,
    VehicleZone.wheel_front_right,
    VehicleZone.wheel_rear_left,
    VehicleZone.wheel_rear_right,
}

_TIRE_PRESSURE_RE = re.compile(
    r"\b(flat|deflated|underinflated|low\s+pressure|tire\s+pressure|"
    r"desinflad[ao]|baja\s+presi[oó]n)\b",
    re.IGNORECASE,
)
_TEMPORARY_SURFACE_RE = re.compile(
    r"\b(bird\s+droppings?|droppings?|organic\s+stains?|organic\s+debris|"
    r"debris|dirt|dust|mud|leaves?|leaf|sticker|logo|label)\b",
    re.IGNORECASE,
)
_ENVIRONMENT_RE = re.compile(
    r"\b(water|puddle|floor|ground|road|street|pavement|sidewalk|oil\s+mark|"
    r"agua|charco|piso|suelo|calle|vereda|pavimento)\b",
    re.IGNORECASE,
)
_IMAGE_ARTIFACT_RE = re.compile(
    r"\b(reflections?|shadows?|glare|background|overexposure|underexposure)\b",
    re.IGNORECASE,
)


def is_non_damage_false_positive(damage: Damage) -> bool:
    """Return true for conditions that are visible but not exterior vehicle damage."""
    description = damage.description or ""
    zone = damage.zone
    damage_type = damage.type

    if _TIRE_PRESSURE_RE.search(description):
        return zone in _WHEEL_ZONES or "tire" in description.lower() or "wheel" in description.lower()

    if _ENVIRONMENT_RE.search(description):
        return True

    if _IMAGE_ARTIFACT_RE.search(description):
        return True

    if _TEMPORARY_SURFACE_RE.search(description):
        return damage_type in {DamageType.stain, DamageType.other}

    return False
