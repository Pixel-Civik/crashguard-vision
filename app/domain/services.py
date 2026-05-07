from collections import Counter

from app.domain.models import (
    Damage,
    DamageMap,
    DamageMapSummary,
    Severity,
    SourceImageMeta,
    VehicleContext,
    VehicleZone,
)

_SEVERITY_ORDER = {Severity.low: 0, Severity.medium: 1, Severity.high: 2}


class DamageMapFactory:
    def build(
        self,
        damages: list[Damage],
        images: dict[str, SourceImageMeta],
        session_id: str,
        vehicle_context: VehicleContext | None,
        total_processing_ms: int = 0,
        total_prompt_tokens: int = 0,
        total_response_tokens: int = 0,
    ) -> DamageMap:
        zones: dict[VehicleZone, list[Damage]] = {z: [] for z in VehicleZone}

        for damage in damages:
            zones[damage.zone].append(damage)

        zones_affected = [zone for zone, zone_damages in zones.items() if zone_damages]
        severity_counts = Counter(damage.severity for damage in damages)
        type_counts = Counter(damage.type for damage in damages)
        overall_severity = max(
            (damage.severity for damage in damages),
            key=lambda severity: _SEVERITY_ORDER[severity],
            default=Severity.low,
        )

        return DamageMap(
            session_id=session_id,
            vehicle_context=vehicle_context,
            images=images,
            zones=zones,
            summary=DamageMapSummary(
                total_images=len(images),
                images_analyzed=len(images),
                total_damages=len(damages),
                zones_affected=zones_affected,
                overall_severity=overall_severity,
                damages_by_severity=dict(severity_counts),
                damages_by_type=dict(type_counts),
                total_processing_ms=total_processing_ms,
                total_prompt_tokens=total_prompt_tokens,
                total_response_tokens=total_response_tokens,
            ),
        )


PythonDamageMapBuilder = DamageMapFactory
