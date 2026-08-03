"""Tests for pure geometry helpers.

None of these need Earth Engine, credentials, or a network — which is the point.
Area and projection arithmetic is where quiet, plausible-looking errors live, so
it is the part that gets pinned down hardest.
"""

from __future__ import annotations

import pytest

from vanachakshu.geometry import (
    pixel_count_to_hectares,
    spherical_bbox_area_sq_km,
    sq_m_to_hectares,
    utm_epsg_for_lon_lat,
)


class TestUtmEpsgForLonLat:
    def test_western_ghats_resolves_to_zone_43n(self) -> None:
        # The project AOI. If this ever changes, every hectare figure is wrong.
        assert utm_epsg_for_lon_lat(74.725, 14.975) == 32643

    @pytest.mark.parametrize(
        ("lon", "lat", "expected"),
        [
            (-177.0, 10.0, 32601),  # first northern zone
            (177.0, 10.0, 32660),  # last northern zone
            (0.0, 0.0, 32631),  # prime meridian, equator counts as northern
            (74.725, -14.975, 32743),  # southern hemisphere flips the prefix
            (-74.0, -10.0, 32718),  # Peru
        ],
    )
    def test_zone_boundaries(self, lon: float, lat: float, expected: int) -> None:
        assert utm_epsg_for_lon_lat(lon, lat) == expected

    def test_longitude_180_clamps_to_zone_60(self) -> None:
        # (180 + 180) / 6 + 1 == 61, which does not exist; must clamp.
        assert utm_epsg_for_lon_lat(180.0, 0.0) == 32660

    @pytest.mark.parametrize("lon", [-180.1, 180.1, 999.0])
    def test_rejects_out_of_range_longitude(self, lon: float) -> None:
        with pytest.raises(ValueError, match="longitude out of range"):
            utm_epsg_for_lon_lat(lon, 0.0)

    @pytest.mark.parametrize("lat", [-90.1, 90.1, 999.0])
    def test_rejects_out_of_range_latitude(self, lat: float) -> None:
        with pytest.raises(ValueError, match="latitude out of range"):
            utm_epsg_for_lon_lat(0.0, lat)


class TestSphericalBboxArea:
    def test_one_degree_box_at_equator(self) -> None:
        # A 1x1 degree box at the equator is ~12,360 km2. Known reference value.
        area = spherical_bbox_area_sq_km(0.0, 0.0, 1.0, 1.0)
        assert area == pytest.approx(12_363.0, rel=0.01)

    def test_area_shrinks_with_latitude(self) -> None:
        # Meridians converge toward the poles, so the same degree box covers
        # less ground further north. This is exactly why area must not be
        # computed in degrees.
        equatorial = spherical_bbox_area_sq_km(0.0, 0.0, 1.0, 1.0)
        temperate = spherical_bbox_area_sq_km(0.0, 60.0, 1.0, 61.0)
        assert temperate < equatorial / 1.5

    def test_project_aoi_is_roughly_1500_sq_km(self) -> None:
        area = spherical_bbox_area_sq_km(74.55, 14.80, 74.90, 15.15)
        assert 1_400 < area < 1_600

    def test_is_orientation_independent(self) -> None:
        # Swapped corners still yield a positive magnitude; ordering is enforced
        # by BoundingBox, not here.
        assert spherical_bbox_area_sq_km(1.0, 1.0, 0.0, 0.0) == pytest.approx(
            spherical_bbox_area_sq_km(0.0, 0.0, 1.0, 1.0)
        )


class TestAreaConversions:
    def test_sq_m_to_hectares(self) -> None:
        assert sq_m_to_hectares(10_000.0) == 1.0

    def test_hundred_ten_metre_pixels_is_one_hectare(self) -> None:
        # 100 pixels x 100 m2 = 10,000 m2 = 1 ha.
        assert pixel_count_to_hectares(100, 10.0) == pytest.approx(1.0)

    def test_fifty_pixels_is_the_half_hectare_floor(self) -> None:
        # The plan's 0.5 ha minimum clearing at Sentinel resolution.
        assert pixel_count_to_hectares(50, 10.0) == pytest.approx(0.5)

    def test_zero_pixels_is_zero_area(self) -> None:
        assert pixel_count_to_hectares(0, 10.0) == 0.0

    def test_rejects_negative_pixel_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pixel_count_to_hectares(-1, 10.0)

    @pytest.mark.parametrize("size", [0.0, -10.0])
    def test_rejects_non_positive_pixel_size(self, size: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            pixel_count_to_hectares(10, size)
