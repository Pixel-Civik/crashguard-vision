from collections import Counter
from app.domain.models import (
    Damage, DamageMap, DamageMapSummary, SourceImageMeta,
    VehicleContext, VehicleZone, Severity,
)

_SEVERITY_ORDER = {Severity.low: 0, Severity.medium: 1, Severity.high: 2}


class StandardDamageMapBuilder:
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
