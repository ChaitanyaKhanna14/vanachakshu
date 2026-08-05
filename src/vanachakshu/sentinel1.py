"""Sentinel-1 radar: terrain correction and cloud-free backscatter.

Radar is why this project can work in the Western Ghats. It carries its own
illumination and its wavelength passes through cloud, so it sees the ground
during the monsoon — the four months when the optical detector is blind and
when clearing peaks. It also revisits every 6-12 days, which is what lets an
alert say *when* rather than only *that*.

The price is geometry. A radar looks sideways, so a slope facing the sensor is
compressed and brightened while a slope facing away is stretched and darkened.
On flat ground this does not matter. In the Western Ghats it dominates: an
uncorrected hillside can differ from its neighbour by several decibels for no
reason but its aspect, which is the same magnitude as the signal from real
forest loss.

Correcting it is :func:`slope_correction`, an implementation of the
angular-based method of **Vollrath, Mullissa and Reiche (2020)**, *Angular-Based
Radiometric Slope Correction for Sentinel-1 on Google Earth Engine*, Remote
Sensing 12(11):1867 — validated in the Austrian Alps, which is why it is
trusted here rather than something written from scratch.

Reference implementation: https://github.com/ESA-PhiLab/radiometric-slope-correction
"""

from __future__ import annotations

import math
from typing import Final

import ee

from vanachakshu import datasets
from vanachakshu.config import RadarDetectionConfig

__all__ = [
    "add_speckle_filter",
    "collection",
    "slope_correction",
    "to_db",
    "to_linear",
]

_DEG_TO_RAD: Final = math.pi / 180.0

# Sentinel-1 GRD in Earth Engine is already terrain-*geometrically* corrected
# and calibrated to sigma0 in decibels. What it is not is radiometrically
# terrain-flattened, which is the gap this module fills.
_S1_BANDS: Final = ("VV", "VH")

# Native resolution of Copernicus GLO-30. Must be set explicitly on the
# mosaicked DEM — see slope_correction for why.
_DEM_SCALE_M: Final = 30.0


def to_linear(image: ee.Image) -> ee.Image:
    """Decibels to linear power.

    Every radiometric operation — averaging, ratios, the slope correction —
    must happen in linear power. Decibels are logarithmic, so averaging them
    computes a geometric mean, which is not the physical quantity wanted and
    is biased low.
    """
    result: ee.Image = ee.Image.constant(10).pow(image.divide(10))
    return result


def to_db(image: ee.Image) -> ee.Image:
    """Linear power back to decibels, for thresholding and display."""
    result: ee.Image = image.log10().multiply(10)
    return result


def slope_correction(
    image: ee.Image,
    geometry: ee.Geometry,
    config: RadarDetectionConfig | None = None,
) -> ee.Image:
    """Radiometrically flatten terrain, and mask where radar cannot see.

    Implements Vollrath et al. (2020). The steps, and why each exists:

    1. **Radar look direction.** Sentinel-1's ``angle`` band increases steadily
       across the swath, so the *aspect* of that band points along the look
       direction. Taking its mean over the AOI recovers the geometry without
       needing orbit metadata.
    2. **Project terrain into radar geometry.** A slope only matters in the
       direction the radar looks. ``alpha_r`` is the component of the slope in
       the range direction, ``alpha_az`` in azimuth.
    3. **Apply the correction model.** ``volume`` (Hoekman 1990) assumes volume
       scattering and is the right choice for canopy; ``surface`` (Ulander 1996)
       targets bare ground.
    4. **Mask layover and shadow.** Where a slope is steeper than the look
       angle, the radar receives several ground positions in one pixel
       (layover); where it faces away too steeply, it receives nothing
       (shadow). Neither is recoverable, so both are removed rather than
       corrected. In steep terrain these are permanent blind spots, and the
       project publishes them rather than silently reporting no alerts there.

    Returns VV and VH as terrain-flattened gamma0 in decibels, plus a
    ``valid`` band that is 1 where the geometry is usable.
    """
    cfg = config if config is not None else RadarDetectionConfig()

    # The DEM ships as a collection of tiles, so it must be mosaicked. Loading
    # it as an Image fails outright, which is the good case — the quiet failure
    # would have been a single tile silently covering part of the AOI.
    #
    # setDefaultProjection is not optional. A mosaic has no fixed projection,
    # and ee.Terrain.slope on such an image computes slope at whatever
    # resolution the enclosing request happens to use — which silently yields
    # nonsense gradients, and therefore a nonsense correction. Pinning it to
    # the DEM's native 30 m makes the slope mean what it says.
    elevation = (
        ee.ImageCollection(datasets.COPERNICUS_DEM)
        .select("DEM")
        .mosaic()
        .setDefaultProjection("EPSG:4326", None, _DEM_SCALE_M)
    )
    sigma0_pow = to_linear(image.select(list(_S1_BANDS)))

    ninety_rad = ee.Image.constant(90.0 * _DEG_TO_RAD)
    theta_i_rad = image.select("angle").multiply(_DEG_TO_RAD)

    # Step 1. Aspect of the incidence-angle band recovers the look direction.
    phi_i_rad = ee.Number(
        ee.Terrain.aspect(theta_i_rad)
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=1000, maxPixels=int(1e9))
        .get("aspect")
    ).multiply(_DEG_TO_RAD)

    # Step 2. Terrain slope and aspect, projected into radar geometry.
    alpha_s_rad = ee.Terrain.slope(elevation).select("slope").multiply(_DEG_TO_RAD)
    phi_s_rad = ee.Terrain.aspect(elevation).select("aspect").multiply(_DEG_TO_RAD)
    # phi_i_rad is a scalar (one look direction for the whole AOI) while
    # phi_s_rad varies per pixel. Promote the scalar to a constant image:
    # ee.Number.subtract(ee.Image) is not a valid operation.
    phi_r_rad = ee.Image.constant(phi_i_rad).subtract(phi_s_rad)

    alpha_r_rad = alpha_s_rad.tan().multiply(phi_r_rad.cos()).atan()
    alpha_az_rad = alpha_s_rad.tan().multiply(phi_r_rad.sin()).atan()

    # gamma0: sigma0 referenced to the plane perpendicular to the look
    # direction. This removes the flat-earth incidence-angle effect; the slope
    # correction below removes what terrain adds on top.
    gamma0 = sigma0_pow.divide(theta_i_rad.cos())

    # Step 3.
    if cfg.terrain_model == "volume":
        # Hoekman (1990): tan(90 - theta_i + alpha_r) / tan(90 - theta_i)
        numerator = ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).tan()
        denominator = ninety_rad.subtract(theta_i_rad).tan()
        correction = numerator.divide(denominator)
    else:
        # Ulander et al. (1996), for surface scattering.
        numerator = ninety_rad.subtract(theta_i_rad).cos()
        denominator = alpha_az_rad.cos().multiply(
            ninety_rad.subtract(theta_i_rad).add(alpha_r_rad).cos()
        )
        correction = numerator.divide(denominator)

    gamma0_flat = gamma0.divide(correction)

    # Step 4. Both conditions are TRUE where the geometry is usable, so the
    # combined mask reads as "valid" rather than "affected" — a naming trap in
    # the reference implementation worth stating explicitly.
    not_layover = alpha_r_rad.lt(theta_i_rad)
    not_shadow = alpha_r_rad.gt(ee.Image.constant(-1).multiply(ninety_rad.subtract(theta_i_rad)))
    valid = not_layover.And(not_shadow).rename("valid")

    # copyProperties returns the generic Element, so re-wrap to keep the
    # declared Image type honest rather than silently returning Any.
    corrected = (
        to_db(gamma0_flat)
        .updateMask(valid)
        .addBands(valid)
        .copyProperties(
            image, ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"]
        )
    )
    result: ee.Image = ee.Image(corrected)
    return result


def add_speckle_filter(image: ee.Image, config: RadarDetectionConfig | None = None) -> ee.Image:
    """Smooth speckle with a focal mean, applied in linear power.

    Speckle is not sensor noise — it is coherent interference between
    scatterers inside a resolution cell, and it makes single radar pixels
    almost meaningless. Averaging a neighbourhood is the standard remedy.

    Averaged in linear power, not decibels: averaging logarithms computes a
    geometric mean, which underestimates the true backscatter.
    """
    cfg = config if config is not None else RadarDetectionConfig()
    bands = image.select(list(_S1_BANDS))

    # focalMean, not reduceNeighborhood with Reducer.mean. They compute the
    # same thing, but focalMean is the optimised path; the generic form
    # exhausted Earth Engine's memory limit when mapped over a year of scenes.
    smoothed = to_db(to_linear(bands).focalMean(cfg.speckle_radius_m, "circle", "meters")).rename(
        list(_S1_BANDS)
    )

    result: ee.Image = image.addBands(smoothed, overwrite=True)
    return result


def collection(
    geometry: ee.Geometry,
    start: str,
    end: str,
    config: RadarDetectionConfig | None = None,
) -> ee.ImageCollection:
    """Terrain-corrected, speckle-filtered Sentinel-1 over ``geometry``.

    Restricted to a single orbit direction. Ascending and descending passes
    view a hillside from opposite sides, so their backscatter differs for
    reasons unrelated to the ground; mixing them into one baseline
    manufactures change where none happened.
    """
    cfg = config if config is not None else RadarDetectionConfig()

    raw = (
        ee.ImageCollection(datasets.SENTINEL1_GRD)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", cfg.orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    def _prepare(image: ee.Image) -> ee.Image:
        return add_speckle_filter(slope_correction(image, geometry, cfg), cfg)

    prepared: ee.ImageCollection = ee.ImageCollection(raw.map(_prepare))
    return prepared
