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
