"""Tests for validated configuration.

The theme here is that *invalid configuration must fail loudly at construction*.
A silently-accepted bad threshold is far more expensive than a crash: it produces
alerts that look plausible and are wrong.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vanachakshu.config import (
    MAX_AOI_SQ_KM,
    WESTERN_GHATS_CLEAR_SEASON,
    YELLAPUR_TALUK,
    AreaOfInterest,
    BoundingBox,
    OpticalDetectionConfig,
    SeasonWindow,
)


class TestBoundingBox:
    def test_accepts_a_well_formed_box(self) -> None:
        bbox = BoundingBox(west=74.55, south=14.80, east=74.90, north=15.15)
        assert bbox.west == 74.55
        assert bbox.area_sq_km == pytest.approx(1_475.0, rel=0.05)

    def test_is_immutable(self, tiny_bbox: BoundingBox) -> None:
        # Frozen config means nothing downstream can quietly mutate the AOI
        # halfway through a pipeline run.
        with pytest.raises(ValidationError):
            tiny_bbox.west = 5.0  # type: ignore[misc]

    def test_rejects_west_east_inversion(self) -> None:
        with pytest.raises(ValidationError, match="must be less than east"):
            BoundingBox(west=75.0, south=14.0, east=74.0, north=15.0)

    def test_rejects_south_north_inversion(self) -> None:
        with pytest.raises(ValidationError, match="must be less than north"):
            BoundingBox(west=74.0, south=15.0, east=75.0, north=14.0)

    def test_rejects_zero_width_box(self) -> None:
        with pytest.raises(ValidationError, match="must be less than east"):
            BoundingBox(west=74.0, south=14.0, east=74.0, north=15.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"west": -181.0, "south": 0.0, "east": 1.0, "north": 1.0},
            {"west": 0.0, "south": -91.0, "east": 1.0, "north": 1.0},
            {"west": 0.0, "south": 0.0, "east": 181.0, "north": 1.0},
            {"west": 0.0, "south": 0.0, "east": 1.0, "north": 91.0},
        ],
    )
    def test_rejects_impossible_coordinates(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(**kwargs)

    def test_rejects_aoi_above_quota_guardrail(self) -> None:
        # A 10x10 degree box is ~1.2 million km2 and would burn the monthly
        # Earth Engine allowance in a single run.
        with pytest.raises(ValidationError, match="guardrail"):
            BoundingBox(west=0.0, south=0.0, east=10.0, north=10.0)

    def test_guardrail_message_names_the_limit(self) -> None:
        with pytest.raises(ValidationError) as exc:
            BoundingBox(west=0.0, south=0.0, east=10.0, north=10.0)
        assert f"{MAX_AOI_SQ_KM:,.0f}" in str(exc.value)

    def test_centroid(self, western_ghats_bbox: BoundingBox) -> None:
        lon, lat = western_ghats_bbox.centroid
        assert lon == pytest.approx(74.65)
        assert lat == pytest.approx(14.95)

    def test_as_ee_coords_uses_ee_ordering(self, western_ghats_bbox: BoundingBox) -> None:
        # ee.Geometry.Rectangle expects [west, south, east, north]. Getting this
        # order wrong silently analyses the wrong patch of the planet.
        assert western_ghats_bbox.as_ee_coords() == pytest.approx([74.60, 14.90, 74.70, 15.00])


class TestAreaOfInterest:
    def test_derives_utm_from_centroid(self, western_ghats_bbox: BoundingBox) -> None:
        aoi = AreaOfInterest(name="Test", bbox=western_ghats_bbox)
        assert aoi.utm_epsg == 32643

    def test_explicit_utm_is_respected(self, western_ghats_bbox: BoundingBox) -> None:
        aoi = AreaOfInterest(name="Test", bbox=western_ghats_bbox, utm_epsg=32644)
        assert aoi.utm_epsg == 32644

    def test_rejects_empty_name(self, tiny_bbox: BoundingBox) -> None:
        with pytest.raises(ValidationError):
            AreaOfInterest(name="", bbox=tiny_bbox)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Yellapur Taluk", "yellapur-taluk"),
            ("Uttara_Kannada", "uttara-kannada"),
            ("SIRSI", "sirsi"),
        ],
    )
    def test_slug_is_filesystem_safe(
        self, name: str, expected: str, tiny_bbox: BoundingBox
    ) -> None:
        assert AreaOfInterest(name=name, bbox=tiny_bbox).slug == expected


class TestSeasonWindow:
    def test_date_range_is_end_exclusive(self) -> None:
        # End-exclusive matches ee.Filter.date semantics.
        assert SeasonWindow(start_month=1, end_month=3).date_range_for_year(2024) == (
            "2024-01-01",
            "2024-04-01",
        )

    def test_december_end_rolls_into_next_year(self) -> None:
        # The off-by-one that would otherwise silently drop December imagery.
        assert SeasonWindow(start_month=10, end_month=12).date_range_for_year(2024) == (
            "2024-10-01",
            "2025-01-01",
        )

    def test_single_month_window(self) -> None:
        assert SeasonWindow(start_month=6, end_month=6).date_range_for_year(2023) == (
            "2023-06-01",
            "2023-07-01",
        )

    def test_rejects_inverted_window(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed end_month"):
            SeasonWindow(start_month=11, end_month=2)

    @pytest.mark.parametrize("month", [0, 13, -1])
    def test_rejects_impossible_months(self, month: int) -> None:
        with pytest.raises(ValidationError):
            SeasonWindow(start_month=month, end_month=12)

    def test_years_share_the_same_months(self) -> None:
        # The whole point of the class: two different years must produce
        # windows over identical months, or the comparison is phenological
        # noise rather than change.
        window = SeasonWindow(start_month=1, end_month=3)
        start_a, end_a = window.date_range_for_year(2020)
        start_b, end_b = window.date_range_for_year(2025)
        assert start_a[4:] == start_b[4:]
        assert end_a[4:] == end_b[4:]


class TestOpticalDetectionConfig:
    def test_defaults_match_the_plan(self) -> None:
        cfg = OpticalDetectionConfig()
        assert cfg.hansen_treecover_min_pct == 30  # Global Forest Watch convention
        assert cfg.pixel_size_m == 10.0

    def test_detection_defaults_come_from_the_one_year_sweep(self) -> None:
        # Both values were re-tuned on ONE-year gaps, which is what the
        # scheduled monitor actually compares. The previous 0.25 / 0.5 ha pair
        # came from a four-year sweep and detected 0.0 ha across 1,463 km2 at
        # the deployed horizon — the live system could not raise an alert at all.
        #
        # At 0.5 ha the detector returns exactly zero on every one-year pair
        # tested; 0.2 ha (RADD's validated floor) returns real detections.
        #
        # If either changes, the sweep needs re-running: a threshold that drifts
        # away from its evidence is a guess again.
        cfg = OpticalDetectionConfig()
        assert cfg.ndvi_drop_threshold == 0.15
        assert cfg.min_clearing_ha == 0.2

    def test_default_floor_is_twenty_sentinel_pixels(self) -> None:
        # 0.2 ha at 10 m resolution.
        assert OpticalDetectionConfig().min_clearing_pixels == 20

    def test_min_clearing_pixels_tracks_resolution(self) -> None:
        # At Landsat's 30 m the same 0.2 ha floor is only ~2 pixels, which is
        # far too few to distinguish a clearing from speckle. This is precisely
        # why small-clearing detection needs Sentinel — and why Hansen, a 30 m
        # product, cannot adjudicate the detections this config now produces.
        cfg = OpticalDetectionConfig(pixel_size_m=30.0)
        assert cfg.min_clearing_pixels == 2

    def test_min_clearing_pixels_never_rounds_to_zero(self) -> None:
        cfg = OpticalDetectionConfig(min_clearing_ha=0.0001)
        assert cfg.min_clearing_pixels == 1

    @pytest.mark.parametrize("threshold", [0.0, -0.1])
    def test_rejects_non_positive_ndvi_drop(self, threshold: float) -> None:
        with pytest.raises(ValidationError):
            OpticalDetectionConfig(ndvi_drop_threshold=threshold)

    @pytest.mark.parametrize("ndvi", [-1.5, 1.5])
    def test_rejects_out_of_band_forest_ndvi(self, ndvi: float) -> None:
        # NDVI is bounded to [-1, 1] by construction.
        with pytest.raises(ValidationError):
            OpticalDetectionConfig(forest_ndvi_min=ndvi)

    @pytest.mark.parametrize("pct", [-1, 101])
    def test_rejects_impossible_treecover_percentage(self, pct: int) -> None:
        with pytest.raises(ValidationError):
            OpticalDetectionConfig(hansen_treecover_min_pct=pct)


class TestProjectDefaults:
    def test_default_aoi_is_within_quota_guardrail(self) -> None:
        assert YELLAPUR_TALUK.bbox.area_sq_km < MAX_AOI_SQ_KM

    def test_default_aoi_uses_utm_43n(self) -> None:
        assert YELLAPUR_TALUK.utm_epsg == 32643

    def test_default_season_is_the_clear_post_monsoon_window(self) -> None:
        assert WESTERN_GHATS_CLEAR_SEASON.start_month == 1
        assert WESTERN_GHATS_CLEAR_SEASON.end_month == 3
