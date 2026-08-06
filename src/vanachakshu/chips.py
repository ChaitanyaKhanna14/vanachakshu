"""Before-and-after image chips, so a human can judge a detection in one glance.

The validation worksheet ships with a Google Maps link, which shows what is
there *now* at very high resolution but carries no date — usually one to three
years stale. That answers "is there a clearing here?" when the question is
"did something change between these two years?"

This module answers the real question: two dated, cloud-masked composites of the
same patch, side by side, with the detection marked. Plus a NICFI chip at <5 m
where available, because a half-hectare clearing is only about seven Sentinel-2
pixels across and that is genuinely hard to judge.

Output is a single HTML page. Thirteen browser tabs and a spreadsheet is a
worse experience than one scrollable sheet, and a reviewer who finds the task
tedious makes worse judgements.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import ee
import requests

from vanachakshu import datasets
from vanachakshu.alerts import TrackedAlert
from vanachakshu.config import OpticalDetectionConfig, SeasonWindow
from vanachakshu.report import google_maps_url
from vanachakshu.sentinel2 import rgb_composite

__all__ = [
    "ChipSet",
    "chip_bounds",
    "download_chips",
    "write_contact_sheet",
]

# Half-width of the chip in metres. A 0.5 ha clearing is ~70 m across, so 250 m
# shows it with enough surrounding forest to judge context — whether the patch
# sits in continuous canopy, beside a road, or inside a plantation block.
_CHIP_HALF_WIDTH_M = 250.0

# Rendered pixel size. Deliberately larger than the native resolution supports:
# upsampling a small Sentinel-2 patch is easier on the eye than squinting at a
# 50 x 50 thumbnail, and no detail is invented that was not already there.
_CHIP_PIXELS = 384

# Sentinel-2 L2A surface reflectance is scaled by 10,000. 2,500 as the ceiling
# keeps vegetation from crushing to black while leaving bare soil visible.
_S2_VIS = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 2500, "gamma": 1.3}

# Planet's own recommended stretch for NICFI basemaps.
_NICFI_VIS = {"bands": ["R", "G", "B"], "min": 64, "max": 5454, "gamma": 1.8}


def chip_bounds(lon: float, lat: float, half_width_m: float = _CHIP_HALF_WIDTH_M) -> ee.Geometry:
    """Square region centred on a detection.

    Built by buffering a point and taking its bounds, so Earth Engine handles
    the metres-to-degrees conversion at this latitude rather than us
    approximating it.
    """
    region: ee.Geometry = ee.Geometry.Point([lon, lat]).buffer(half_width_m).bounds()
    return region


@dataclass(frozen=True)
class ChipSet:
    """The images gathered for one detection."""

    alert: TrackedAlert
    baseline_year: int
    recent_year: int
    baseline_path: Path | None
    recent_path: Path | None
    nicfi_path: Path | None

    @property
    def is_complete(self) -> bool:
        """Both dated chips present — enough to judge a change."""
        return self.baseline_path is not None and self.recent_path is not None


def _fetch(url: str, destination: Path, timeout: int = 120) -> Path | None:
    """Download one thumbnail, returning None rather than raising.

    A single missing chip must not abandon a batch of thirty. The contact sheet
    shows the gap explicitly, so a reviewer can see that imagery was
    unavailable rather than silently judging a blank square.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except (requests.RequestException, ee.EEException):
        return None

    destination.write_bytes(response.content)
    return destination


def _nicfi_image(geometry: ee.Geometry, year: int) -> ee.Image | None:
    """Highest-resolution chip available, or None if NICFI is not accessible.

    NICFI is free for noncommercial use but requires separate acceptance of
    Planet's terms. Treated as optional so validation still works without it —
    Sentinel-2 alone is coarse for half-hectare patches, but it is dated
    correctly, which matters more.
    """
    try:
        collection = (
            ee.ImageCollection(datasets.NICFI_BASEMAPS_ASIA)
            .filterBounds(geometry)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
        )
        if collection.size().getInfo() == 0:
            return None
        latest: ee.Image = ee.Image(collection.sort("system:time_start", False).first())
        return latest
    except ee.EEException:
        return None


def download_chips(
    alerts: list[TrackedAlert],
    baseline_year: int,
    recent_year: int,
    season: SeasonWindow,
    out_dir: Path,
    config: OpticalDetectionConfig | None = None,
) -> list[ChipSet]:
    """Fetch before/after chips for each detection."""
    cfg = config if config is not None else OpticalDetectionConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    chipsets: list[ChipSet] = []
    for alert in alerts:
        region = chip_bounds(alert.lon, alert.lat)
        paths: dict[str, Path | None] = {}

        for label, year in (("before", baseline_year), ("after", recent_year)):
            image = rgb_composite(region, season, year, cfg)
            url = image.getThumbURL(
                {**_S2_VIS, "region": region, "dimensions": _CHIP_PIXELS, "format": "png"}
            )
            paths[label] = _fetch(url, out_dir / f"{alert.alert_id}-{label}-{year}.png")

        nicfi = _nicfi_image(region, recent_year)
        nicfi_path: Path | None = None
        if nicfi is not None:
            url = nicfi.getThumbURL(
                {**_NICFI_VIS, "region": region, "dimensions": _CHIP_PIXELS, "format": "png"}
            )
            nicfi_path = _fetch(url, out_dir / f"{alert.alert_id}-nicfi-{recent_year}.png")

        chipsets.append(
            ChipSet(
                alert=alert,
                baseline_year=baseline_year,
                recent_year=recent_year,
                baseline_path=paths["before"],
                recent_path=paths["after"],
                nicfi_path=nicfi_path,
            )
        )

    return chipsets


_MISSING_CELL = '<div class="missing">no cloud-free<br>imagery</div>'


def _cell(path: Path | None, caption: str) -> str:
    if path is None:
        return f"<figure>{_MISSING_CELL}<figcaption>{html.escape(caption)}</figcaption></figure>"
    return (
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def write_contact_sheet(chipsets: list[ChipSet], path: Path) -> Path:
    """Write one scrollable HTML page for the whole sample.

    The alert id is shown in a copyable form beside each row, because the
    verdict still goes into the CSV and matching rows by eye across two windows
    is where mistakes happen.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for chips in chipsets:
        alert = chips.alert
        cells = [
            _cell(chips.baseline_path, f"{chips.baseline_year} (before)"),
            _cell(chips.recent_path, f"{chips.recent_year} (after)"),
        ]
        if chips.nicfi_path is not None:
            cells.append(_cell(chips.nicfi_path, f"NICFI {chips.recent_year} (<5 m)"))

        rows.append(
            f"""
    <section>
      <h2><code>{html.escape(alert.alert_id)}</code>
        <span class="meta">{alert.area_ha:.2f} ha &middot;
        {alert.lat:.5f}, {alert.lon:.5f} &middot;
        first seen {html.escape(alert.first_seen)}</span></h2>
      <div class="chips">{"".join(cells)}</div>
      <p class="links">
        <a href="{html.escape(google_maps_url(alert.lat, alert.lon))}" target="_blank">
          high-resolution view (undated)</a>
      </p>
    </section>"""
        )

    complete = sum(c.is_complete for c in chipsets)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Validation chips</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 2rem;
         max-width: 1200px; background: #111; color: #eee; }}
  h1 {{ font-size: 1.4rem; }}
  .intro {{ color: #aaa; max-width: 60ch; }}
  section {{ border-top: 1px solid #333; padding: 1.5rem 0; }}
  h2 {{ font-size: 1rem; margin: 0 0 .75rem; }}
  code {{ background: #222; padding: .15rem .4rem; border-radius: 3px; }}
  .meta {{ font-weight: normal; color: #999; margin-left: .75rem; }}
  .chips {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  figure {{ margin: 0; }}
  img {{ width: {_CHIP_PIXELS}px; max-width: 100%; border-radius: 4px;
         image-rendering: pixelated; background: #000; }}
  .missing {{ width: {_CHIP_PIXELS}px; height: {_CHIP_PIXELS}px; display: grid;
              place-items: center; background: #1a1a1a; border-radius: 4px;
              color: #666; text-align: center; }}
  figcaption {{ color: #999; font-size: .85rem; margin-top: .35rem; }}
  .links a {{ color: #6cf; }}
</style></head><body>
<h1>Validation chips &mdash; {len(chipsets)} detections</h1>
<p class="intro">Each row is one detection. Compare <strong>before</strong> and
<strong>after</strong>: you are judging whether something <em>changed</em>, not
whether a clearing exists today. Intact canopy looks bumpy and textured; cleared
ground looks smooth, often with a straight edge. Plantations look too regular.
Rock and riverbeds are irregular and follow natural lines.</p>
<p class="intro">Record <code>true_positive</code>, <code>false_positive</code>
or <code>unclear</code> against each id in the worksheet CSV. Use
<code>unclear</code> freely &mdash; it is excluded from the score, and guessing
corrupts the number you are trying to measure.</p>
<p class="intro">{complete} of {len(chipsets)} have both dated chips.</p>
{"".join(rows)}
</body></html>
"""
    path.write_text(document, encoding="utf-8")
    return path
