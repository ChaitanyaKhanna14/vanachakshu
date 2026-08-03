"""Pure geometry helpers. No Earth Engine, no network, no file I/O.

Everything here is deterministic and unit-testable, which is why the area and
projection logic lives in this module rather than being inlined into the Earth
Engine calls. Area arithmetic is the single easiest thing to get quietly wrong
in a geospatial project, so it gets its own tested home.
"""

from __future__ import annotations

import math
from typing import Final

# Mean Earth radius (WGS84 authalic sphere), metres.
EARTH_RADIUS_M: Final = 6_371_007.181

SQ_M_PER_HECTARE: Final = 10_000.0
SQ_M_PER_SQ_KM: Final = 1_000_000.0


def utm_epsg_for_lon_lat(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    Hectare figures must never be computed in EPSG:4326, because a degree of
    longitude is ~111 km at the equator and 0 km at the poles — areas computed
    in degrees are meaningless. Projecting to the local UTM zone gives metres,
    which is what stakeholders actually need ("2.4 ha cleared").

    At the Western Ghats (~74.7E, ~15N) this returns 32643, i.e. UTM zone 43N.
    """
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude out of range: {lon}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude out of range: {lat}")

    zone = math.floor((lon + 180.0) / 6.0) + 1
    # Longitude of exactly 180 lands in a 61st zone; clamp it back to 60.
    zone = min(zone, 60)
    # 326xx = northern hemisphere, 327xx = southern.
    return (32600 if lat >= 0 else 32700) + zone


def spherical_bbox_area_sq_km(west: float, south: float, east: float, north: float) -> float:
    """Approximate the area of a lon/lat bounding box in square kilometres.

    Uses the closed form for a spherical quadrangle, which is accurate to well
    under a percent at the box sizes we care about. This exists only to sanity
    check AOI size before spending Earth Engine quota on it — the authoritative
    per-alert areas come from projected geometry, not from this function.
    """
    lat_term = math.sin(math.radians(north)) - math.sin(math.radians(south))
    lon_term = math.radians(east - west)
    area_sq_m = (EARTH_RADIUS_M**2) * lon_term * lat_term
    return abs(area_sq_m) / SQ_M_PER_SQ_KM


def sq_m_to_hectares(area_sq_m: float) -> float:
    """Convert square metres to hectares (1 ha = 10,000 m²)."""
    return area_sq_m / SQ_M_PER_HECTARE


def pixel_count_to_hectares(pixel_count: int, pixel_size_m: float) -> float:
    """Convert a count of square pixels to hectares.

    Sentinel-1 and Sentinel-2 are handled at 10 m, so a single pixel is
    100 m² = 0.01 ha, and the plan's 0.5 ha minimum clearing is 50 pixels.
    Keeping this as a named function stops that factor being re-derived
    (and mis-derived) at each call site.
    """
    if pixel_count < 0:
        raise ValueError(f"pixel_count must be non-negative, got {pixel_count}")
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be positive, got {pixel_size_m}")
    return sq_m_to_hectares(pixel_count * pixel_size_m * pixel_size_m)
