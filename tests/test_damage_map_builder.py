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
        type=DamageType.dent,
        zone=VehicleZone.hood,
        severity=Severity.medium,
        confidence=0.9,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.1),
        description="dent on hood",
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
    assert VehicleZone.hood in result.zones
    assert len(result.zones[VehicleZone.hood]) == 1
    # All zones present, empty lists where no damages
    assert VehicleZone.roof in result.zones
    assert result.zones[VehicleZone.roof] == []


def test_build_summary_counts(builder, sample_damage, sample_images):
    result = builder.build(
        damages=[sample_damage],
        images=sample_images,
        session_id="sess_01",
        vehicle_context=None,
    )
    assert result.summary.total_damages == 1
    assert result.summary.zones_affected == [VehicleZone.hood]
    assert result.summary.overall_severity == Severity.medium
    assert result.summary.damages_by_severity[Severity.medium] == 1
    assert result.summary.damages_by_type[DamageType.dent] == 1


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
            id="d1", type=DamageType.scratch, zone=VehicleZone.hood,
            severity=Severity.low, confidence=0.8,
            bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
            description="scratch", source_image_id="img_001",
        ),
        Damage(
            id="d2", type=DamageType.dent, zone=VehicleZone.roof,
            severity=Severity.high, confidence=0.9,
            bbox=BoundingBox(x=0.5, y=0.5, w=0.2, h=0.2),
            description="dent", source_image_id="img_001",
        ),
    ]
    result = builder.build(damages=damages, images=sample_images, session_id="sess_01", vehicle_context=None)
    assert result.summary.overall_severity == Severity.high
