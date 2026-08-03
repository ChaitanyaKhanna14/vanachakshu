"""Credentialed tests for Sentinel-2 compositing.

Run with ``pytest -m ee``.

These assert that the composite is *physically plausible*, not that it equals
some fixed number. Satellite archives get reprocessed and the AOI is real, so
exact-value assertions would be brittle and would teach nothing. The useful
question is: does this look like a forest, and does it look like a real median?
"""

from __future__ import annotations

from typing import Any

import ee
import pytest

from vanachakshu.config import WESTERN_GHATS_CLEAR_SEASON, YELLAPUR_TALUK
from vanachakshu.gee import initialize
from vanachakshu.sentinel2 import seasonal_composite

pytestmark = pytest.mark.ee

# 2025 rather than the most recent year: a fully settled archive, and far enough
# from any reprocessing campaign to be stable.
TEST_YEAR = 2025


@pytest.fixture(scope="module")
def composite() -> ee.Image:
    initialize()
    geometry = ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())
    return seasonal_composite(geometry, WESTERN_GHATS_CLEAR_SEASON, TEST_YEAR)


@pytest.fixture(scope="module")
def stats(composite: ee.Image) -> dict[str, Any]:
    """Region statistics, fetched once — each getInfo() is a billed round-trip."""
    geometry = ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())
    result = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=200,  # coarse on purpose: this is a sanity check, not analysis
        maxPixels=int(1e8),
        bestEffort=True,
    ).getInfo()
    return dict(result)


class TestCompositeStructure:
    def test_has_the_expected_bands(self, composite: ee.Image) -> None:
        bands = composite.bandNames().getInfo()
        assert set(bands) == {"NDVI", "NBR", "n_obs"}

    def test_records_its_provenance(self, composite: ee.Image) -> None:
        # Without these, a composite in an export folder is unidentifiable.
        assert composite.get("vanachakshu:year").getInfo() == TEST_YEAR
        assert composite.get("vanachakshu:start").getInfo() == f"{TEST_YEAR}-01-01"
        assert composite.get("vanachakshu:end").getInfo() == f"{TEST_YEAR}-04-01"

    def test_is_built_from_multiple_scenes(self, composite: ee.Image) -> None:
        count = composite.get("vanachakshu:scene_count").getInfo()
        assert count > 5, f"only {count} scenes survived filtering — check cloud threshold"


class TestCompositeIsPhysicallyPlausible:
    def test_ndvi_is_in_the_valid_range(self, stats: dict[str, Any]) -> None:
        # NDVI is a normalised ratio and cannot leave [-1, 1]. Outside that
        # range means the band maths is wrong, not that the forest is unusual.
        assert -1.0 <= stats["NDVI"] <= 1.0

    def test_ndvi_looks_like_vegetation(self, stats: dict[str, Any]) -> None:
        # A largely forested AOI in the post-monsoon window should average well
        # into vegetated territory. A value near zero would mean the composite
        # is dominated by cloud, water or bare ground — i.e. masking failed.
        assert stats["NDVI"] > 0.4, f"mean NDVI {stats['NDVI']:.3f} is too low for forest"

    def test_nbr_is_in_the_valid_range(self, stats: dict[str, Any]) -> None:
        assert -1.0 <= stats["NBR"] <= 1.0

    def test_pixels_have_enough_cloud_free_looks(self, stats: dict[str, Any]) -> None:
        # A median over one or two observations is not a median. If this fails,
        # the clear-season window is not actually clear and the whole optical
        # approach needs revisiting for this AOI.
        assert stats["n_obs"] >= 3.0, f"mean n_obs {stats['n_obs']:.1f} is too low to trust"
