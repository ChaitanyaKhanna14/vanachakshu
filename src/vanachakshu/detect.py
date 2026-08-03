"""Bi-temporal optical change detection.

Compares two seasonal composites and reports patches that look like forest loss.

The rule is deliberately conservative — a pixel is flagged only when **all four**
of these hold:

1. Its greenness fell by at least ``ndvi_drop_threshold``.
2. It was genuinely vegetated in the baseline year (``forest_ndvi_min``).
3. Hansen independently agrees it was forest then (:mod:`vanachakshu.hansen`).
4. Both composites had enough cloud-free looks to be trusted (``min_observations``).

Conditions 2 and 3 are what stop harvested cropland and drained reservoirs being
reported as deforestation, which is the largest source of naive false positives.
Condition 4 stops a median-of-two masquerading as a measurement.

Surviving pixels are then required to form a **connected patch** of at least
``min_clearing_ha``. Real clearings are contiguous; noise is scattered. This is
the cheapest large reduction in false positives available.

Limits worth restating: this says *something changed between two years*. It
cannot say when within that span, it cannot see through monsoon cloud, and it
cannot distinguish logging from fire or landslide. Those are Phase 3's problem.
"""

from __future__ import annotations

from typing import Final

import ee

from vanachakshu import hansen
from vanachakshu.config import OpticalDetectionConfig

__all__ = [
    "DISTURBANCE_BAND",
    "detect_disturbance",
    "disturbance_patches",
    "summarise_patches",
]

DISTURBANCE_BAND: Final = "disturbed"

# Cap for connected-component counting. Any patch at or above the minimum size
# passes regardless, so counting beyond a few hundred pixels buys nothing and
# costs compute.
_MAX_CONNECTED_PIXELS: Final = 256


def detect_disturbance(
    baseline: ee.Image,
    recent: ee.Image,
    baseline_year: int,
    config: OpticalDetectionConfig | None = None,
) -> ee.Image:
    """Return a mask of suspected forest loss between two composites.

    Both images must come from :func:`vanachakshu.sentinel2.seasonal_composite`
    and must cover the *same seasonal window* in different years — otherwise the
    difference measures the seasons, not the ground.

    The returned image carries the binary ``disturbed`` band plus ``ndvi_drop``
    (how far greenness fell, as a positive number) so downstream code can rank
    detections by severity rather than treating them all alike.
    """
    cfg = config if config is not None else OpticalDetectionConfig()

    baseline_ndvi = baseline.select("NDVI")
    recent_ndvi = recent.select("NDVI")

    # Positive number = greenness fell. Easier to reason about than a negative.
    ndvi_drop = baseline_ndvi.subtract(recent_ndvi).rename("ndvi_drop")

    dropped = ndvi_drop.gte(cfg.ndvi_drop_threshold)
    was_vegetated = baseline_ndvi.gte(cfg.forest_ndvi_min)
    was_forest = hansen.forest_mask(baseline_year, cfg)
    enough_looks = (
        baseline.select("n_obs")
        .gte(cfg.min_observations)
        .And(recent.select("n_obs").gte(cfg.min_observations))
    )

    candidate = dropped.And(was_vegetated).And(was_forest).And(enough_looks)

    # Drop patches smaller than the minimum clearing size. eightConnected means
    # diagonal neighbours count, which matches how a real clearing looks; four-
    # connected would split diagonal strips into separate specks.
    patch_pixels = candidate.selfMask().connectedPixelCount(
        maxSize=_MAX_CONNECTED_PIXELS, eightConnected=True
    )
    large_enough = patch_pixels.gte(cfg.min_clearing_pixels)

    disturbed = candidate.And(large_enough).rename(DISTURBANCE_BAND)

    result: ee.Image = disturbed.addBands(ndvi_drop.updateMask(disturbed))
    return result


def disturbance_patches(
    disturbance: ee.Image,
    geometry: ee.Geometry,
    config: OpticalDetectionConfig | None = None,
) -> ee.FeatureCollection:
    """Convert the disturbance mask into polygons carrying real hectare figures.

    Area comes from ``ee.Geometry.area()``, which computes on the ellipsoid and
    returns square metres. This is the authoritative figure — deliberately not
    "pixel count x 100", which would ignore the fact that a 10 m pixel is not
    exactly 100 m2 once projected.
    """
    cfg = config if config is not None else OpticalDetectionConfig()

    vectors = (
        disturbance.select(DISTURBANCE_BAND)
        .selfMask()
        .reduceToVectors(
            geometry=geometry,
            scale=cfg.pixel_size_m,
            geometryType="polygon",
            eightConnected=True,
            labelProperty="label",
            maxPixels=int(1e9),
        )
    )

    def _add_area(feature: ee.Feature) -> ee.Feature:
        # maxError of 1 m: the polygon is built from 10 m pixels, so sub-metre
        # precision is meaningless and only costs compute.
        hectares = feature.geometry().area(maxError=1).divide(10_000)
        # set() is typed as returning the generic Element; re-wrap so the
        # mapped collection stays a FeatureCollection to the type checker.
        with_ha: ee.Feature = ee.Feature(feature.set("area_ha", hectares))
        return with_ha

    with_area = vectors.map(_add_area)

    # The authoritative minimum-size rule. connectedPixelCount above is a cheap
    # pre-filter; this is the exact one, applied to true polygon area.
    result: ee.FeatureCollection = with_area.filter(ee.Filter.gte("area_ha", cfg.min_clearing_ha))
    return result


def summarise_patches(patches: ee.FeatureCollection) -> dict[str, float]:
    """Fetch headline numbers for a detection run.

    Performs one round-trip and returns plain Python values, so callers can
    print or serialise without touching Earth Engine.
    """
    areas = patches.aggregate_array("area_ha")
    stats: dict[str, float | None] = (
        ee.Dictionary(
            {
                "patch_count": patches.size(),
                "total_ha": areas.reduce(ee.Reducer.sum()),
                "largest_ha": areas.reduce(ee.Reducer.max()),
                "median_ha": areas.reduce(ee.Reducer.median()),
            }
        ).getInfo()
        or {}
    )

    # An empty collection makes the reducers return None rather than 0, so
    # every value needs a fallback — a run that finds nothing is a normal
    # outcome, not an error.
    return {
        "patch_count": float(stats.get("patch_count") or 0),
        "total_ha": float(stats.get("total_ha") or 0.0),
        "largest_ha": float(stats.get("largest_ha") or 0.0),
        "median_ha": float(stats.get("median_ha") or 0.0),
    }
