"""Credentialed tests for Hansen masks.

Run with ``pytest -m ee``.

These exist because of a real bug that unit tests structurally could not catch.
``lossyear`` stores no-loss pixels as **masked**, not as 0, despite the
documentation saying 0. In Earth Engine a masked pixel is not zero — every
comparison against it returns masked, and that propagates through ``.And()``
and ``.Or()``.

The result: ``forest_mask`` covered 109 ha of a 145,797 ha forested taluk, and
the detector found 7 clearings instead of 142. Nothing raised. The output was
simply, quietly, almost entirely empty.

The lesson generalises: for a masked-array API, assertions about *coverage* are
the ones that catch silent emptiness. Assertions about structure are not enough,
because an empty layer is structurally perfect.
"""

from __future__ import annotations

import ee
import pytest

from vanachakshu.config import YELLAPUR_TALUK
from vanachakshu.gee import initialize
from vanachakshu.hansen import forest_mask, loss_mask

pytestmark = pytest.mark.ee

BASELINE_YEAR = 2020
AOI_TOTAL_HA_APPROX = 145_797


@pytest.fixture(scope="module")
def geometry() -> ee.Geometry:
    initialize()
    return ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())


def _hectares(mask: ee.Image, geometry: ee.Geometry) -> float:
    """Area where ``mask`` is truthy, in hectares."""
    value = (
        ee.Image.pixelArea()
        .divide(10_000)
        .updateMask(mask.selfMask())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=30,  # Hansen's native resolution
            maxPixels=int(1e10),
        )
        .get("area")
    )
    return float(ee.Number(value).getInfo() or 0.0)


class TestForestMaskCoverage:
    """The regression guard. This is the test that would have caught the bug."""

    def test_covers_a_plausible_share_of_a_forested_taluk(self, geometry: ee.Geometry) -> None:
        hectares = _hectares(forest_mask(BASELINE_YEAR), geometry)
        share = hectares / AOI_TOTAL_HA_APPROX

        # Uttara Kannada is one of India's most heavily forested districts, but
        # the AOI is a rectangle that also catches towns, farmland and water.
        # Anywhere in 40-95% is credible; 0.07% was the bug.
        assert 0.40 < share < 0.95, (
            f"forest mask covers {share:.1%} of the AOI ({hectares:,.0f} ha). "
            "Far too low usually means a masked band silently emptied the layer; "
            "far too high means the tree-cover threshold is not being applied."
        )

    def test_is_not_empty(self, geometry: ee.Geometry) -> None:
        # Stated separately from the range check: an empty mask is the specific
        # failure that produces "no deforestation found" rather than an error.
        assert _hectares(forest_mask(BASELINE_YEAR), geometry) > 0


class TestLossYearUnmasking:
    def test_no_loss_pixels_read_as_zero_not_masked(self, geometry: ee.Geometry) -> None:
        # Directly pins the bug. Most of the AOI has never had recorded forest
        # loss, so `lossyear == 0` must cover most of it. Before the unmask()
        # fix this measured exactly 0 ha.
        from vanachakshu.hansen import _lossyear

        hectares = _hectares(_lossyear().eq(0), geometry)
        assert hectares > 0.5 * AOI_TOTAL_HA_APPROX, (
            f"'no loss' covers only {hectares:,.0f} ha — lossyear is masked "
            "rather than zero-filled again"
        )


class TestLossMask:
    def test_records_loss_in_the_period(self, geometry: ee.Geometry) -> None:
        assert _hectares(loss_mask(2020, 2025), geometry) > 0

    def test_longer_period_contains_at_least_as_much_loss(self, geometry: ee.Geometry) -> None:
        # Monotonicity: 2021-2025 cannot contain less loss than 2021-2023.
        # A failure here means the year-code arithmetic is off.
        short = _hectares(loss_mask(2020, 2023), geometry)
        long = _hectares(loss_mask(2020, 2025), geometry)
        assert long >= short

    def test_loss_is_a_small_fraction_of_forest(self, geometry: ee.Geometry) -> None:
        # Sanity bound: five years of loss should be a small slice of standing
        # forest. A large fraction would mean the year codes select far too
        # wide a window.
        loss = _hectares(loss_mask(2020, 2025), geometry)
        forest = _hectares(forest_mask(BASELINE_YEAR), geometry)
        assert 0 < loss < 0.10 * forest

    def test_rejects_inverted_period(self) -> None:
        with pytest.raises(ValueError, match="must be before"):
            loss_mask(2025, 2020)
