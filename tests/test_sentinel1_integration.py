"""Credentialed tests for radiometric terrain correction.

Run with ``pytest -m ee``.

**The central test here measures whether the correction actually works**, which
no fixture can answer. The method: a radar looks sideways, so backscatter
correlates with how the ground tilts *toward or away from the sensor*. That
correlation is the distortion. If the correction is right, it collapses; if the
implementation is wrong, it does not — and nothing else about the output would
look unusual.

Two mistakes this test caught while being written, both silent:

1. The DEM is mosaicked from tiles and so has no fixed projection. Without
   ``setDefaultProjection``, ``ee.Terrain.slope`` computes gradients at
   whatever resolution the enclosing request uses, producing a nonsense
   correction that still returns plausible-looking decibels.
2. Correlating against *unsigned* slope measures nothing. A slope tilted toward
   the radar brightens and one tilted away darkens, so the two cancel to
   roughly zero whether or not the correction works. The signed slope in the
   look direction is the variable that carries the effect.
"""

from __future__ import annotations

import math

import ee
import pytest

from vanachakshu import datasets
from vanachakshu.config import YELLAPUR_TALUK, RadarDetectionConfig
from vanachakshu.gee import initialize
from vanachakshu.sentinel1 import collection, slope_correction

pytestmark = pytest.mark.ee

_DEG = math.pi / 180.0
START, END = "2026-01-01", "2026-04-01"


@pytest.fixture(scope="module")
def geometry() -> ee.Geometry:
    initialize()
    return ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())


@pytest.fixture(scope="module")
def raw_scene(geometry: ee.Geometry) -> ee.Image:
    scenes = (
        ee.ImageCollection(datasets.SENTINEL1_GRD)
        .filterBounds(geometry)
        .filterDate(START, END)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", RadarDetectionConfig().orbit_pass))
    )
    return ee.Image(scenes.first())


@pytest.fixture(scope="module")
def corrected(raw_scene: ee.Image, geometry: ee.Geometry) -> ee.Image:
    return slope_correction(raw_scene, geometry, RadarDetectionConfig())


@pytest.fixture(scope="module")
def range_slope(raw_scene: ee.Image, geometry: ee.Geometry) -> ee.Image:
    """Signed terrain slope in the radar look direction, in degrees.

    Positive means tilted toward the sensor. This is the variable terrain
    distortion actually rides on.
    """
    elevation = (
        ee.ImageCollection(datasets.COPERNICUS_DEM)
        .select("DEM")
        .mosaic()
        .setDefaultProjection("EPSG:4326", None, 30)
    )
    theta = raw_scene.select("angle").multiply(_DEG)
    look_direction = ee.Number(
        ee.Terrain.aspect(theta)
        .reduceRegion(ee.Reducer.mean(), geometry, 1000, maxPixels=int(1e9))
        .get("aspect")
    ).multiply(_DEG)

    alpha_s = ee.Terrain.slope(elevation).select("slope").multiply(_DEG)
    phi_s = ee.Terrain.aspect(elevation).select("aspect").multiply(_DEG)
    phi_r = ee.Image.constant(look_direction).subtract(phi_s)
    return alpha_s.tan().multiply(phi_r.cos()).atan().divide(_DEG).rename("range_slope")


def _correlation(band: ee.Image, range_slope: ee.Image, geometry: ee.Geometry) -> float:
    value = (
        band.rename("x")
        .addBands(range_slope)
        .reduceRegion(
            reducer=ee.Reducer.pearsonsCorrelation(),
            geometry=geometry,
            scale=100,
            maxPixels=int(1e9),
            bestEffort=True,
        )
        .get("correlation")
        .getInfo()
    )
    # The reducer returns the string "NaN" when nothing survives masking, which
    # would otherwise crash formatting rather than reporting a real failure.
    assert value != "NaN", "no pixels survived; the correction emptied the image"
    return float(value)


class TestTerrainCorrectionRemovesTheDistortion:
    """The test that proves the implementation, not merely that it runs."""

    @pytest.mark.parametrize("polarisation", ["VV", "VH"])
    def test_distortion_exists_before_correction(
        self,
        polarisation: str,
        raw_scene: ee.Image,
        range_slope: ee.Image,
        geometry: ee.Geometry,
    ) -> None:
        # Sanity precondition. If uncorrected backscatter did NOT correlate with
        # range slope, there would be nothing to correct, and a "successful"
        # correction below would prove nothing at all.
        r = _correlation(raw_scene.select(polarisation), range_slope, geometry)
        assert abs(r) > 0.25, f"expected terrain distortion, measured r={r:+.4f}"

    @pytest.mark.parametrize("polarisation", ["VV", "VH"])
    def test_correction_collapses_the_distortion(
        self,
        polarisation: str,
        raw_scene: ee.Image,
        corrected: ee.Image,
        range_slope: ee.Image,
        geometry: ee.Geometry,
    ) -> None:
        before = _correlation(raw_scene.select(polarisation), range_slope, geometry)
        after = _correlation(corrected.select(polarisation), range_slope, geometry)

        # Measured at implementation time: VV 0.456 -> -0.055 (88% reduction),
        # VH 0.454 -> -0.098 (78%). The bound is deliberately looser than that,
        # so ordinary scene-to-scene variation does not cause false failures
        # while a broken correction still fails loudly.
        assert abs(after) < 0.20, f"{polarisation} still terrain-correlated: r={after:+.4f}"
        assert abs(after) < abs(before) * 0.5, (
            f"{polarisation} distortion only fell from {before:+.4f} to {after:+.4f}"
        )


class TestCorrectedOutput:
    def test_carries_the_expected_bands(self, corrected: ee.Image) -> None:
        assert set(corrected.bandNames().getInfo()) == {"VV", "VH", "valid"}

    def test_backscatter_is_physically_plausible(
        self, corrected: ee.Image, geometry: ee.Geometry
    ) -> None:
        stats = corrected.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=200,
            maxPixels=int(1e9),
            bestEffort=True,
        ).getInfo()
        # Forest gamma0 sits around -7 dB (VV) and -14 dB (VH). Anything far
        # outside this means the decibel/linear conversions are inverted
        # somewhere — a mistake that still produces finite, tidy-looking output.
        assert -20.0 < stats["VV"] < 0.0, f"VV={stats['VV']:.2f} dB is not forest-like"
        assert -25.0 < stats["VH"] < -5.0, f"VH={stats['VH']:.2f} dB is not forest-like"

    def test_cross_polarisation_is_weaker_than_co_polarisation(
        self, corrected: ee.Image, geometry: ee.Geometry
    ) -> None:
        # VH (cross-pol) is always weaker than VV (co-pol) over land. If this
        # inverted, the bands have been swapped somewhere.
        stats = corrected.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=200,
            maxPixels=int(1e9),
            bestEffort=True,
        ).getInfo()
        assert stats["VH"] < stats["VV"]

    def test_reports_where_the_geometry_is_unusable(
        self, corrected: ee.Image, geometry: ee.Geometry
    ) -> None:
        # Layover and shadow are permanent blind spots, and the project
        # publishes them rather than silently reporting no alerts there.
        fraction = (
            corrected.select("valid")
            .reduceRegion(ee.Reducer.mean(), geometry, 200, maxPixels=int(1e9), bestEffort=True)
            .get("valid")
            .getInfo()
        )
        assert 0.0 < float(fraction) <= 1.0


class TestCollection:
    def test_returns_multiple_passes_in_the_window(self, geometry: ee.Geometry) -> None:
        # Sentinel-1 revisits every 6-12 days, so a three-month window should
        # hold several passes. Too few would mean the orbit filter is wrong.
        count = collection(geometry, START, END).size().getInfo()
        assert count >= 5, f"only {count} passes; check the orbit_pass filter"

    def test_every_scene_shares_one_orbit_direction(self, geometry: ee.Geometry) -> None:
        passes = collection(geometry, START, END).aggregate_array("orbitProperties_pass").getInfo()
        assert set(passes) == {RadarDetectionConfig().orbit_pass}
