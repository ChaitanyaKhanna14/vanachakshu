"""Dated sub-metre imagery from Esri Wayback.

The validation review needs two things that no single source provides:

* **Dated** imagery, or you cannot tell whether anything changed.
* **Sharp** imagery, or you cannot tell a plantation from regrowth — and in the
  Western Ghats that distinction decides whether a detection counts at all.

Sentinel-2 is dated but 10 m, so a half-hectare clearing is about seven pixels
and reviewers reported it as unusable. NICFI would solve this at 4.7 m but needs
a separate Planet registration that this project does not have. Esri's ordinary
World Imagery is sub-metre but carries no date.

**Wayback is Esri World Imagery with its history kept.** Every release is
archived and addressable, so picking one snapshot near each comparison year
gives a sub-metre before and after.

Cost: the releases are irregular — whatever date Esri happened to refresh that
area — so the pair will not line up exactly with the detection window. That is a
real caveat and the review page states each image's true date rather than
implying it matches.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image

__all__ = [
    "WaybackRelease",
    "list_releases",
    "nearest_release",
    "wayback_chip",
]

_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
_TILE_URL = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery"
    "/WMTS/1.0.0/default028mm/MapServer/tile/{release}/{z}/{row}/{col}"
)

_TILE_PIXELS = 256
# Web Mercator ground resolution at the equator, zoom 0.
_EQUATOR_M_PER_PIXEL = 156_543.03392

# Zoom 17 is ~1.15 m/pixel at 15N. Enough to resolve plantation rows and
# individual tree crowns, which is what the judgement actually needs.
_DEFAULT_ZOOM = 17

_TITLE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True)
class WaybackRelease:
    """One archived snapshot of Esri World Imagery."""

    release: int
    captured: date

    @property
    def label(self) -> str:
        return self.captured.isoformat()


@lru_cache(maxsize=1)
def list_releases(timeout: int = 60) -> tuple[WaybackRelease, ...]:
    """Every archived release, newest first.

    Cached because the catalogue is a few hundred entries and every chip would
    otherwise refetch it.
    """
    response = requests.get(_CONFIG_URL, timeout=timeout)
    response.raise_for_status()

    releases: list[WaybackRelease] = []
    for key, entry in response.json().items():
        match = _TITLE_DATE.search(str(entry.get("itemTitle", "")))
        if match:
            releases.append(
                WaybackRelease(release=int(key), captured=date.fromisoformat(match.group(1)))
            )
    return tuple(sorted(releases, key=lambda r: r.captured, reverse=True))


def nearest_release(target: date, timeout: int = 60) -> WaybackRelease | None:
    """The archived snapshot closest in time to ``target``.

    Nearest rather than most-recent-before, because Esri's refresh schedule is
    irregular: insisting on "before" can land on imagery years stale when a
    snapshot two months *after* the date is far more representative.
    """
    releases = list_releases(timeout=timeout)
    if not releases:
        return None
    return min(releases, key=lambda r: abs((r.captured - target).days))


def _tile_coords(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional Web Mercator tile coordinates for a point."""
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x * _TILE_PIXELS, y * _TILE_PIXELS


def wayback_chip(
    lon: float,
    lat: float,
    release: int,
    half_width_m: float,
    output_pixels: int,
    zoom: int = _DEFAULT_ZOOM,
    timeout: int = 60,
) -> Image.Image | None:
    """Fetch and stitch a square chip centred exactly on ``(lon, lat)``.

    Tiles are on a fixed grid, so a single tile would place the detection
    wherever it happened to fall. This fetches every tile the requested extent
    touches and crops to the exact centre, which matters when a reviewer is
    trusting a crosshair.

    Returns None if any tile is unavailable — a partially blank chip is worse
    than an honest gap, because the missing part still looks like data.
    """
    centre_x, centre_y = _tile_coords(lon, lat, zoom)
    metres_per_pixel = _EQUATOR_M_PER_PIXEL * math.cos(math.radians(lat)) / (2.0**zoom)
    half_pixels = half_width_m / metres_per_pixel

    left, top = centre_x - half_pixels, centre_y - half_pixels
    right, bottom = centre_x + half_pixels, centre_y + half_pixels

    col0, row0 = int(left // _TILE_PIXELS), int(top // _TILE_PIXELS)
    col1, row1 = int(right // _TILE_PIXELS), int(bottom // _TILE_PIXELS)

    canvas = Image.new(
        "RGB",
        ((col1 - col0 + 1) * _TILE_PIXELS, (row1 - row0 + 1) * _TILE_PIXELS),
    )

    for col in range(col0, col1 + 1):
        for row in range(row0, row1 + 1):
            url = _TILE_URL.format(release=release, z=zoom, row=row, col=col)
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                tile = Image.open(io.BytesIO(response.content)).convert("RGB")
            except (requests.RequestException, OSError):
                return None
            canvas.paste(tile, ((col - col0) * _TILE_PIXELS, (row - row0) * _TILE_PIXELS))

    crop = (
        int(left - col0 * _TILE_PIXELS),
        int(top - row0 * _TILE_PIXELS),
        int(right - col0 * _TILE_PIXELS),
        int(bottom - row0 * _TILE_PIXELS),
    )
    # Resampling.LANCZOS rather than Image.LANCZOS: the module-level alias was
    # removed in Pillow 10.
    return canvas.crop(crop).resize((output_pixels, output_pixels), Image.Resampling.LANCZOS)


def save_wayback_chip(
    lon: float,
    lat: float,
    release: int,
    destination: Path,
    half_width_m: float,
    output_pixels: int,
) -> Path | None:
    """Fetch a chip and write it to disk, or return None if unavailable."""
    image = wayback_chip(lon, lat, release, half_width_m, output_pixels)
    if image is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination
