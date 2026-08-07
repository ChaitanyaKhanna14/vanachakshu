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

from datetime import date, timedelta
from typing import Final

import ee

from vanachakshu import embeddings, hansen, sentinel1
from vanachakshu.config import (
    EmbeddingDetectionConfig,
    OpticalDetectionConfig,
    RadarDetectionConfig,
)

__all__ = [
    "DISTURBANCE_BAND",
    "RADAR_DISTURBANCE_BAND",
    "baseline_window",
    "detect_disturbance",
    "detect_embedding_disturbance",
    "detect_radar_disturbance",
    "disturbance_patches",
    "summarise_patches",
]

DISTURBANCE_BAND: Final = "disturbed"
RADAR_DISTURBANCE_BAND: Final = "radar_disturbed"

# Cap for connected-component counting. Any patch at or above the minimum size
# passes regardless, so counting beyond a few hundred pixels buys nothing and
# costs compute.
_MAX_CONNECTED_PIXELS: Final = 256


def detect_embedding_disturbance(
    geometry: ee.Geometry,
    base_year: int,
    target_year: int,
    config: EmbeddingDetectionConfig | None = None,
    optical_config: OpticalDetectionConfig | None = None,
) -> ee.Image:
    """Detect forest loss from movement in AlphaEarth embedding space.

    **The primary detector.** Each 10 m pixel carries 64 numbers summarising a
    year of fused Sentinel-1 and Sentinel-2 observation; a pixel that changes
    land cover moves a long way in that space, and one that merely has a dry
    year does not.

    Measured against the NDVI detector it replaces, same AOI, same years, same
    tolerance, **both scored at 30 m**:

    ==================  =========  ========  =====
    Detector            Precision  Recall    F1
    ==================  =========  ========  =====
    NDVI drop >= 0.15   0.583      0.013     0.025
    Embedding L2 >=0.45 0.773      0.129     0.221
    ==================  =========  ========  =====

    Scored instead at the pipeline's own 10 m, the embedding detector gives
    precision 0.343 and recall 0.388 — a different balance and a better F1
    (0.364). See :class:`~vanachakshu.config.EmbeddingDetectionConfig` for why
    the two disagree; in short, 30 m sampling discards most of what this
    detector emits, and what survives is disproportionately correct.

    Simplicity is not an accident here: this thresholds a single derived band,
    and it outperformed a 130-feature random forest that could not be made to
    run inside Earth Engine's per-tile limits at all. The separability was
    already in the features; the classifier was an optimisation mistaken for
    the fix.

    Returns ``disturbed`` plus ``emb_distance``, so downstream code can rank
    detections by how far the pixel actually moved.
    """
    cfg = config if config is not None else EmbeddingDetectionConfig()
    optical_cfg = optical_config if optical_config is not None else OpticalDetectionConfig()

    distance = embeddings.euclidean_distance(
        embeddings.annual(geometry, base_year),
        embeddings.annual(geometry, target_year),
    ).rename("emb_distance")

    # Hansen still supplies the "was it forest" guard. Without it the detector
    # flags every land-cover change, including cropland turning over.
    was_forest = hansen.forest_mask(base_year, optical_cfg)
    candidate = distance.gte(cfg.distance_threshold).And(was_forest)

    patch_pixels = candidate.selfMask().connectedPixelCount(
        maxSize=_MAX_CONNECTED_PIXELS, eightConnected=True
    )
    disturbed = candidate.And(patch_pixels.gte(cfg.min_clearing_pixels)).rename(DISTURBANCE_BAND)

    result: ee.Image = disturbed.addBands(distance.updateMask(disturbed))
    return result


def landscape_normalised_drop(ndvi_drop: ee.Image, geometry: ee.Geometry) -> ee.Image:
    """Subtract the AOI-wide median shift from a per-pixel NDVI drop.

    **Deforestation is local; weather is not.** An absolute threshold cannot
    tell them apart, and in one year here that mattered enormously.

    Measured over Yellapur, median NDVI across the whole AOI moved by +0.008 in
    every year-on-year step except 2022→2023, where it fell by **0.055** — a
    landscape-wide drop, most likely the 2023 El Niño monsoon failure. The
    composites were not at fault: 34 scenes, ~28 cloud-free looks per pixel,
    0.1% thin. The ground really was less green everywhere.

    An absolute threshold reads that as deforestation across the entire
    district: the detector flagged **3,840 ha**, forty times all loss Hansen
    recorded, while neighbouring year-pairs at identical settings flagged ~20 ha.

    Referencing each pixel against its own landscape removes the common-mode
    signal. A dry year shifts everything together and cancels; a clearing still
    stands out against neighbours that stayed put.

    The median is used rather than the mean because it is not dragged by the
    disturbed pixels themselves — the very thing being measured must not define
    the reference it is measured against.

    **Measured effect**, threshold 0.15, 0.2 ha floor, 60 m tolerance:

    ==========  ==========  ==========  =========
    Pair        Absolute    Normalised  Precision
    ==========  ==========  ==========  =========
    2022→2023   304.4 ha    17.9 ha     0.000
    2023→2024   5.9 ha      5.9 ha      0.000
    2024→2025   2.1 ha      2.2 ha      0.565→0.583
    ==========  ==========  ==========  =========

    The drought-year artifact falls seventeenfold while the well-behaved years
    are left alone and precision on the best pair improves slightly. That
    asymmetry is the point: a correction that suppressed every year equally
    would have removed the signal along with the artifact.
    """
    median_shift = ee.Number(
        ndvi_drop.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=geometry,
            scale=200,  # coarse: this is one number for the whole AOI
            maxPixels=int(1e9),
            bestEffort=True,
        ).get("ndvi_drop")
    )
    result: ee.Image = ndvi_drop.subtract(ee.Image.constant(median_shift)).rename("ndvi_drop")
    return result


def detect_disturbance(
    baseline: ee.Image,
    recent: ee.Image,
    baseline_year: int,
    config: OpticalDetectionConfig | None = None,
    geometry: ee.Geometry | None = None,
) -> ee.Image:
    """Return a mask of suspected forest loss between two composites.

    Both images must come from :func:`vanachakshu.sentinel2.seasonal_composite`
    and must cover the *same seasonal window* in different years — otherwise the
    difference measures the seasons, not the ground.

    When ``geometry`` is supplied the drop is measured **relative to the
    landscape's own median shift**, which is strongly recommended. See
    :func:`landscape_normalised_drop` for why.

    The returned image carries the binary ``disturbed`` band plus ``ndvi_drop``
    (how far greenness fell, as a positive number) so downstream code can rank
    detections by severity rather than treating them all alike.
    """
    cfg = config if config is not None else OpticalDetectionConfig()

    baseline_ndvi = baseline.select("NDVI")
    recent_ndvi = recent.select("NDVI")

    # Positive number = greenness fell. Easier to reason about than a negative.
    ndvi_drop = baseline_ndvi.subtract(recent_ndvi).rename("ndvi_drop")
    if geometry is not None:
        ndvi_drop = landscape_normalised_drop(ndvi_drop, geometry)

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
    config: OpticalDetectionConfig | EmbeddingDetectionConfig | None = None,
) -> ee.FeatureCollection:
    """Convert the disturbance mask into polygons carrying real hectare figures.

    Area comes from ``ee.Geometry.area()``, which computes on the ellipsoid and
    returns square metres. This is the authoritative figure — deliberately not
    "pixel count x 100", which would ignore the fact that a 10 m pixel is not
    exactly 100 m2 once projected.

    Accepts either detector's config because the minimum patch size differs
    sharply between them: 0.2 ha for NDVI, 0.05 ha for embeddings. Sharing one
    default across both would silently zero out the embedding detector, which
    is exactly the mistake that nearly hid its improvement.
    """
    cfg = config if config is not None else EmbeddingDetectionConfig()

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
            # The embedding detector loads 128 bands (two years of 64) before
            # differencing, which exceeds the default memory budget over an AOI
            # this size. Splitting into more, smaller tiles is the standard
            # remedy and changes nothing about the result.
            tileScale=8,
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


def baseline_window(monitor_start: date, baseline_days: int) -> tuple[str, str]:
    """Dates defining "normal" for a pixel, ending where monitoring begins.

    The two windows must not overlap. If the baseline included the period being
    monitored, a real clearing would drag its own "normal" downward and partly
    hide itself — the detector would be grading its own homework.
    """
    if baseline_days < 1:
        raise ValueError(f"baseline_days must be positive, got {baseline_days}")
    start = monitor_start - timedelta(days=baseline_days)
    return start.isoformat(), monitor_start.isoformat()


def _has_consecutive_run(flagged: ee.ImageCollection, run_length: int) -> ee.Image:
    """1 where ``run_length`` *consecutive* passes all showed the drop.

    The first version of this took the last N passes of the window and required
    all of them to have dropped. That answers "is this pixel disturbed **now**",
    which is the right question for a live alert but the wrong one for "did a
    disturbance **occur** during this period" — a clearing in March only counted
    if its backscatter was still suppressed the following December, nine months
    and a monsoon later.

    A sliding window asks the right question, and window *length* then controls
    recency: a monitoring window of a few weeks makes "a run occurred in the
    window" and "it is disturbed now" the same statement.

    Implemented over the time axis as an array. Multiplying ``run_length``
    offset slices gives 1 only where every pass in the run dropped, and the
    maximum over the result asks whether any such run exists.
    """
    if run_length < 1:
        raise ValueError(f"run_length must be positive, got {run_length}")

    stack = flagged.select("dropped").toArray()

    product: ee.Image | None = None
    for offset in range(run_length):
        trailing = run_length - 1 - offset
        window = stack.arraySlice(0, offset, -trailing) if trailing else stack.arraySlice(0, offset)
        product = window if product is None else product.multiply(window)

    assert product is not None  # run_length >= 1 guarantees at least one slice

    # ImageCollection.toArray builds a 2-D array per pixel: [image, band].
    # Reducing along axis 0 collapses time but leaves shape [1, 1], so the band
    # axis has to be projected away before it can become an ordinary image.
    result: ee.Image = (
        product.arrayReduce(ee.Reducer.max(), [0])
        .arrayProject([0])
        .arrayFlatten([[RADAR_DISTURBANCE_BAND]])
    )
    return result


def detect_radar_disturbance(
    geometry: ee.Geometry,
    monitor_start: date,
    monitor_end: date,
    baseline_year: int,
    config: RadarDetectionConfig | None = None,
    optical_config: OpticalDetectionConfig | None = None,
) -> ee.Image:
    """Detect sustained backscatter loss — the radar detector.

    The idea in one line: work out what each pixel normally looks like to
    radar, then flag the ones that have got quieter and *stayed* quieter.

    Losing canopy reduces volume scattering, so VH backscatter falls. VH is used
    rather than VV because cross-polarised return comes mostly from multiple
    bounces inside the canopy, which is exactly what clearing destroys; VV is
    more sensitive to surface roughness and soil moisture.

    Persistence is what makes this trustworthy. Rain and wet soil change
    backscatter by several decibels — as much as real clearing — but recover by
    the next pass, whereas cut forest stays cut. So the drop is required in
    **every one of the last ``min_confirming_passes``** rather than merely
    somewhere in the window. A transient cannot satisfy that.

    Returns ``radar_disturbed`` (0/1), ``drop_db`` (how far backscatter fell),
    ``first_drop_millis`` (when it was first seen — the thing optical cannot
    tell you) and ``n_dropped`` (how many passes in total showed the drop).
    """
    cfg = config if config is not None else RadarDetectionConfig()

    baseline_start, baseline_end = baseline_window(monitor_start, cfg.baseline_days)
    baseline = (
        sentinel1.collection(geometry, baseline_start, baseline_end, cfg)
        .select("VH")
        .median()
        .rename("baseline_vh")
    )

    monitoring = sentinel1.collection(
        geometry, monitor_start.isoformat(), monitor_end.isoformat(), cfg
    )

    def _flag(image: ee.Image) -> ee.Image:
        # Positive means quieter than normal, which is the direction clearing
        # moves backscatter.
        fall = baseline.subtract(image.select("VH")).rename("drop_db")
        # unmask(0) so layover, shadow and other no-data read as "not dropped".
        # Left masked they would propagate through the run detection below and
        # silently void whole hillsides.
        dropped = fall.gte(cfg.drop_db).unmask(0).rename("dropped")
        # Acquisition time, kept only where the drop occurred, so reducing with
        # min() over the collection yields the first pass that saw it.
        stamp = (
            ee.Image.constant(ee.Number(image.get("system:time_start")))
            .updateMask(dropped)
            .rename("t")
            .toDouble()
        )
        # copyProperties returns the generic Element, so re-wrap to keep the
        # mapped collection an ImageCollection to the type checker.
        flagged_image: ee.Image = ee.Image(
            dropped.addBands(fall).addBands(stamp).copyProperties(image, ["system:time_start"])
        )
        return flagged_image

    flagged = ee.ImageCollection(monitoring.map(_flag).sort("system:time_start"))

    persistent = _has_consecutive_run(flagged, cfg.min_confirming_passes)

    first_drop = flagged.select("t").reduce(ee.Reducer.min()).rename("first_drop_millis")
    n_dropped = flagged.select("dropped").reduce(ee.Reducer.sum()).rename("n_dropped")
    depth = flagged.select("drop_db").reduce(ee.Reducer.mean()).rename("drop_db")

    # Same two guards as the optical detector: it must have been forest, and
    # the patch must be big enough to be a clearing rather than speckle.
    optical_cfg = optical_config if optical_config is not None else OpticalDetectionConfig()
    was_forest = hansen.forest_mask(baseline_year, optical_cfg)
    candidate = persistent.unmask(0).gt(0).And(was_forest)

    patch_pixels = candidate.selfMask().connectedPixelCount(
        maxSize=_MAX_CONNECTED_PIXELS, eightConnected=True
    )
    disturbed = candidate.And(patch_pixels.gte(optical_cfg.min_clearing_pixels)).rename(
        RADAR_DISTURBANCE_BAND
    )

    result: ee.Image = (
        disturbed.addBands(depth.updateMask(disturbed))
        .addBands(first_drop.updateMask(disturbed))
        .addBands(n_dropped.updateMask(disturbed))
    )
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
