"""Sentinel-2 cloud masking and seasonal compositing.

The job of this module is to turn a few dozen individual satellite passes into
*one* image per year that can be honestly compared against another year.

Three things have to go right, and all three fail quietly rather than loudly:

1. **Clouds must be removed.** A cloud is bright in red and dark in near-infrared,
   which is spectrally almost identical to bare ground. Leave clouds in and the
   detector reports deforestation wherever the weather was bad.
2. **Only complete, same-season windows may be compared.** Enforced upstream by
   :class:`~vanachakshu.config.SeasonWindow`.
3. **Pixels with too few looks must be marked untrustworthy.** A median over two
   observations is not a median. The composite therefore carries an ``n_obs``
   band, and downstream code is expected to use it.

Constants and thresholds live here as tested Python values; only the functions
that build ``ee`` objects require credentials.
"""

from __future__ import annotations

from typing import Final

import ee

from vanachakshu import datasets
from vanachakshu.config import OpticalDetectionConfig, SeasonWindow

__all__ = [
    "SCL_CLASS_NAMES",
    "SCL_MASK_CLASSES",
    "add_indices",
    "mask_clouds",
    "seasonal_composite",
]

# Sentinel-2 Level-2A Scene Classification Layer, as published by ESA.
# Kept complete (not just the masked subset) so the intent of the mask below is
# readable without opening the ESA documentation.
SCL_CLASS_NAMES: Final[dict[int, str]] = {
    0: "no data",
    1: "saturated or defective",
    2: "dark area / topographic shadow",
    3: "cloud shadow",
    4: "vegetation",
    5: "bare soil",
    6: "water",
    7: "unclassified",
    8: "cloud, medium probability",
    9: "cloud, high probability",
    10: "thin cirrus",
    11: "snow or ice",
}

# Classes discarded before compositing.
#
# Note class 2 ("dark area / topographic shadow") is included. In flat terrain
# it is usually kept, but the Western Ghats is steep enough that hillside shadow
# is common, and a shadowed canopy depresses NDVI in exactly the way real
# clearing does. Discarding it costs valid pixels on shaded slopes; keeping it
# would manufacture disturbance on every north-facing hillside. Precision over
# recall, per the project's design rules.
#
# Classes 4, 5, 6 and 7 are deliberately kept: vegetation, bare soil, water and
# unclassified are all legitimate ground observations. Bare soil in particular
# must survive — it is what a freshly cleared patch looks like.
SCL_MASK_CLASSES: Final[frozenset[int]] = frozenset({0, 1, 2, 3, 8, 9, 10, 11})


def mask_clouds(image: ee.Image) -> ee.Image:
    """Mask cloud, shadow, cirrus and defective pixels using the SCL band."""
    scl = image.select("SCL")
    masked = sorted(SCL_MASK_CLASSES)
    # remap(from, to, default): masked classes -> 0, everything else -> 1.
    keep = scl.remap(masked, [0] * len(masked), 1)
    return image.updateMask(keep)


def add_indices(image: ee.Image) -> ee.Image:
    """Attach NDVI and NBR bands.

    NDVI = (NIR - Red) / (NIR + Red), using B8 and B4. Healthy canopy reflects
    strongly in near-infrared and absorbs red, so dense forest sits near 0.8-0.9
    and cleared ground drops sharply. This is the primary change signal.

    NBR = (NIR - SWIR2) / (NIR + SWIR2), using B8 and B12. Short-wave infrared
    responds to moisture and char, so NBR separates *burned* clearing from
    mechanical clearing — a distinction that matters when reporting cause, and
    one NDVI alone cannot make.

    Both are normalised ratios, so the reflectance scaling factor cancels and no
    unscaling is needed.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    return image.addBands(ndvi).addBands(nbr)


def seasonal_composite(
    geometry: ee.Geometry,
    season: SeasonWindow,
    year: int,
    config: OpticalDetectionConfig | None = None,
) -> ee.Image:
    """Build one cloud-free median composite for ``year``'s seasonal window.

    Returns an image carrying ``NDVI``, ``NBR`` and ``n_obs`` bands, plus
    provenance properties describing how it was built.

    Median rather than mean is deliberate: a median is robust to the handful of
    bright cloud pixels that always survive masking, whereas a mean is dragged
    upward by them.

    ``n_obs`` counts the cloud-free observations behind each pixel. It is not
    decoration — pixels below ``config.min_observations`` are unreliable and
    downstream code must exclude them rather than quietly trusting a median of
    one or two looks.
    """
    cfg = config if config is not None else OpticalDetectionConfig()
    start, end = season.date_range_for_year(year)

    collection = (
        ee.ImageCollection(datasets.SENTINEL2_SR)
        .filterBounds(geometry)
        .filterDate(start, end)
        # Drop mostly-cloud scenes before doing per-pixel work: they contribute
        # almost nothing but cost the same to process.
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.max_scene_cloud_pct))
        .map(mask_clouds)
        .map(add_indices)
    )

    indices = collection.select(["NDVI", "NBR"])
    composite = indices.median()
    # Counting on NDVI (not the raw bands) counts *usable* observations: any
    # pixel masked as cloud has no NDVI value to contribute.
    n_obs = indices.select("NDVI").count().rename("n_obs")

    # Annotated local rather than a direct return: ee.Image.set() is typed as
    # returning Any, and mypy's strict mode rejects returning Any from a
    # function declared to return Image.
    result: ee.Image = (
        composite.addBands(n_obs)
        .clip(geometry)
        .set(
            {
                "vanachakshu:year": year,
                "vanachakshu:start": start,
                "vanachakshu:end": end,
                "vanachakshu:scene_count": collection.size(),
                "vanachakshu:max_scene_cloud_pct": cfg.max_scene_cloud_pct,
            }
        )
    )
    return result
