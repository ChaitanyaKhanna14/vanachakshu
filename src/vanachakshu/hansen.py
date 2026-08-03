"""Hansen Global Forest Change: forest masks and published loss labels.

Hansen serves two distinct roles, and confusing them is a real hazard:

1. **Forest mask** — an independent answer to "was this pixel forest before?"
   Used *inside* detection, to stop harvested cropland and drained reservoirs
   being reported as deforestation.
2. **Loss labels** — an independent record of where forest was actually lost,
   used *outside* detection, to score how well the detector did.

Role 2 must never leak into role 1's job of judging our own output, or we would
be marking our own homework.

The year encoding is the fiddly part. Hansen stores loss as a small integer:
``0`` means no loss recorded, ``1`` means 2001, ``2`` means 2002, and so on. An
off-by-one here silently shifts every label by a year, which quietly corrupts
every accuracy figure the project ever produces — so the conversion is a pure,
tested function rather than inline arithmetic.
"""

from __future__ import annotations

from typing import Final

import ee

from vanachakshu import datasets
from vanachakshu.config import OpticalDetectionConfig

__all__ = [
    "HANSEN_BASE_YEAR",
    "HANSEN_LAST_LOSS_YEAR",
    "forest_mask",
    "loss_code_for_year",
    "loss_mask",
    "year_for_loss_code",
]

# lossyear is stored as an offset from this year.
HANSEN_BASE_YEAR: Final = 2000

# Last year covered by the pinned asset. Must be bumped together with
# datasets.HANSEN_GFC — the encoding is relative to the collection's own end
# year, so a version bump changes what every code means.
HANSEN_LAST_LOSS_YEAR: Final = 2025


def loss_code_for_year(year: int) -> int:
    """Convert a calendar year to Hansen's ``lossyear`` encoding.

    2001 -> 1, 2002 -> 2, ..., 2025 -> 25.

    Raises for years the pinned asset cannot represent, rather than returning a
    code that would silently select the wrong pixels.
    """
    if not HANSEN_BASE_YEAR < year <= HANSEN_LAST_LOSS_YEAR:
        raise ValueError(
            f"year {year} is outside Hansen's coverage "
            f"({HANSEN_BASE_YEAR + 1}-{HANSEN_LAST_LOSS_YEAR}); "
            f"bump datasets.HANSEN_GFC and HANSEN_LAST_LOSS_YEAR together"
        )
    return year - HANSEN_BASE_YEAR


def year_for_loss_code(code: int) -> int:
    """Inverse of :func:`loss_code_for_year`. ``0`` (no loss) is not a year."""
    if not 1 <= code <= HANSEN_LAST_LOSS_YEAR - HANSEN_BASE_YEAR:
        raise ValueError(f"loss code {code} does not correspond to a covered year")
    return code + HANSEN_BASE_YEAR


def _lossyear() -> ee.Image:
    """The ``lossyear`` band with no-loss pixels set to 0 rather than masked.

    **This unmask is load-bearing, not defensive.** Hansen's documentation
    describes ``lossyear`` as "0 = no loss, 1-25 = 2001-2025", but in the Earth
    Engine asset the no-loss pixels are *masked*, not zero — the band's minimum
    observed value is 1.

    In Earth Engine a masked pixel is not zero: every comparison against it
    returns masked, and that propagates through ``.And()`` and ``.Or()``. Without
    this unmask, ``lossyear.eq(0)`` is masked everywhere, which masks the entire
    forest layer, which silently empties the detector while still returning a
    plausible-looking result.

    Cost of this bug when it was live: the forest mask covered 109 ha of a
    145,797 ha AOI, and detection found 7 patches instead of hundreds.
    """
    unmasked: ee.Image = ee.Image(datasets.HANSEN_GFC).select("lossyear").unmask(0)
    return unmasked


def forest_mask(as_of_year: int, config: OpticalDetectionConfig | None = None) -> ee.Image:
    """Pixels that were still forest at the start of ``as_of_year``.

    Two conditions, both required:

    * Canopy cover in 2000 was at least ``hansen_treecover_min_pct`` (30% by
      default — the Global Forest Watch convention, so results stay comparable
      with published work).
    * Hansen has not already recorded the forest being lost *before*
      ``as_of_year``. Without this, a patch cleared in 2005 would still count as
      forest in 2020, and any later regrowth-and-reclearing would be missed.
    """
    cfg = config if config is not None else OpticalDetectionConfig()
    hansen = ee.Image(datasets.HANSEN_GFC)

    had_canopy = hansen.select("treecover2000").gte(cfg.hansen_treecover_min_pct)

    # Loss recorded in as_of_year itself still counts as "was forest at the
    # start of it", so the comparison is >= rather than >.
    code = loss_code_for_year(as_of_year)
    lossyear = _lossyear()
    not_yet_lost = lossyear.eq(0).Or(lossyear.gte(code))

    # datamask: 0 = no data, 1 = mapped land, 2 = permanent water.
    is_land = hansen.select("datamask").eq(1)

    result: ee.Image = had_canopy.And(not_yet_lost).And(is_land).rename("forest")
    return result


def loss_mask(after_year: int, through_year: int) -> ee.Image:
    """Pixels where Hansen recorded loss in ``(after_year, through_year]``.

    This is the ground truth for scoring. Comparing composites from 2020 and
    2025 means asking about loss in 2021 through 2025 inclusive — the loss that
    happened *between* the two observations, not including 2020's own.
    """
    if after_year >= through_year:
        raise ValueError(f"after_year ({after_year}) must be before through_year ({through_year})")

    low = loss_code_for_year(after_year + 1)
    high = loss_code_for_year(through_year)

    # Unmasked for the same reason as in forest_mask: this must be a genuine
    # 0/1 layer, because scoring counts both the ones and the zeros.
    lossyear = _lossyear()
    result: ee.Image = lossyear.gte(low).And(lossyear.lte(high)).rename("hansen_loss")
    return result
