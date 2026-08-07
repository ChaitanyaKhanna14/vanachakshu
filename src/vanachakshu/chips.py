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

import hashlib
import html
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import ee
import requests
from PIL import Image, ImageChops, ImageStat

from vanachakshu import datasets
from vanachakshu.alerts import TrackedAlert
from vanachakshu.config import OpticalDetectionConfig, SeasonWindow
from vanachakshu.report import google_maps_url
from vanachakshu.sentinel2 import rgb_composite
from vanachakshu.validation import size_stratum
from vanachakshu.wayback import nearest_release, save_wayback_chip

__all__ = [
    "ChipSet",
    "chip_bounds",
    "download_chips",
    "esri_imagery_url",
    "write_contact_sheet",
]

# Esri's World Imagery service, free and key-less. Sub-metre over much of India,
# which is roughly twenty times sharper than Sentinel-2.
#
# It carries no date, so it cannot answer "did this change?" — only "what is
# this place?". Those are different questions and the review page shows both:
# the dated Sentinel-2 pair for the change, this for the identification. A
# reviewer squinting at 10 m pixels cannot tell a plantation from regrowth, and
# that distinction decides whether a detection counts at all.
_ESRI_EXPORT = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)

_METRES_PER_DEGREE_LAT = 111_320.0

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


def esri_imagery_url(
    lon: float, lat: float, half_width_m: float = _CHIP_HALF_WIDTH_M, pixels: int = 512
) -> str:
    """Static sub-metre aerial image covering the same ground as the chips.

    Deliberately the *same extent* as the Sentinel-2 pair, so the reviewer can
    move their eye between them without re-orienting.
    """
    lat_span = half_width_m / _METRES_PER_DEGREE_LAT
    lon_span = half_width_m / (_METRES_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    bbox = f"{lon - lon_span},{lat - lat_span},{lon + lon_span},{lat + lat_span}"
    return (
        f"{_ESRI_EXPORT}?bbox={bbox}&bboxSR=4326&imageSR=3857"
        f"&size={pixels},{pixels}&format=png&f=image"
    )


@dataclass(frozen=True)
class ChipSet:
    """The images gathered for one detection."""

    alert: TrackedAlert
    baseline_year: int
    recent_year: int
    baseline_path: Path | None
    recent_path: Path | None
    nicfi_path: Path | None
    highres_path: Path | None = None
    # Dated sub-metre pair, with the true capture dates rather than the
    # comparison years — Esri's refresh schedule is irregular and the labels
    # must not imply a match they do not have.
    wayback_before: tuple[Path, str] | None = None
    wayback_after: tuple[Path, str] | None = None
    # True when both Wayback releases returned the same acquisition, so the
    # sharp imagery cannot speak to change — only to what the place is.
    wayback_is_stale: bool = False

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


# Mean per-pixel difference below which two chips are the same acquisition.
# Measured across this AOI: identical imagery re-encoded differs by ~1.8, while
# genuinely different Sentinel-2 dates differ by ~7.1. Three sits between them.
_SAME_IMAGE_THRESHOLD = 3.0


def _pairs_match(before: tuple[Path, str] | None, after: tuple[Path, str] | None) -> bool:
    """True when two chips are the same acquisition rather than two dates."""
    if before is None or after is None:
        return False
    try:
        first = Image.open(before[0]).convert("L")
        second = Image.open(after[0]).convert("L")
    except OSError:
        return False
    difference = ImageStat.Stat(ImageChops.difference(first, second)).mean[0]
    return bool(difference < _SAME_IMAGE_THRESHOLD)


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

    # Resolved once for the batch: the catalogue is a few hundred entries and
    # the same two releases serve every detection.
    # Mid-season, since the Sentinel-2 composites cover January to March.
    before_release = nearest_release(date(baseline_year, 2, 15))
    after_release = nearest_release(date(recent_year, 2, 15))

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

        highres = _fetch(
            esri_imagery_url(alert.lon, alert.lat),
            out_dir / f"{alert.alert_id}-highres.png",
        )

        wayback: dict[str, tuple[Path, str] | None] = {"before": None, "after": None}
        for label, release in (("before", before_release), ("after", after_release)):
            if release is None:
                continue
            saved = save_wayback_chip(
                lon=alert.lon,
                lat=alert.lat,
                release=release.release,
                destination=out_dir / f"{alert.alert_id}-wayback-{label}.png",
                half_width_m=_CHIP_HALF_WIDTH_M,
                output_pixels=_CHIP_PIXELS,
            )
            if saved is not None:
                wayback[label] = (saved, release.label)

        # Wayback publishes dated *releases*, but a release only contains new
        # imagery where Esri actually re-flew. Over rural Uttara Kannada both
        # 2025 and 2026 releases serve the same acquisition, so the "pair" is
        # one image twice.
        #
        # Presenting that as before/after would be worse than showing nothing:
        # a reviewer comparing two identical images concludes nothing changed,
        # which is a false negative manufactured by the tooling rather than
        # observed in the data.
        stale = _pairs_match(wayback["before"], wayback["after"])

        chipsets.append(
            ChipSet(
                alert=alert,
                baseline_year=baseline_year,
                recent_year=recent_year,
                baseline_path=paths["before"],
                recent_path=paths["after"],
                nicfi_path=nicfi_path,
                wayback_before=None if stale else wayback["before"],
                wayback_after=None if stale else wayback["after"],
                # Keep one copy as undated context when the pair is redundant.
                highres_path=(wayback["after"][0] if stale and wayback["after"] else highres),
                wayback_is_stale=stale,
            )
        )

    return chipsets


_MISSING_CELL = '<div class="missing">no cloud-free<br>imagery</div>'


def _cell(path: Path | None, caption: str) -> str:
    if path is None:
        return f"<figure>{_MISSING_CELL}<figcaption>{html.escape(caption)}</figcaption></figure>"
    # The crosshair marks the detection's centre. Without it a reviewer has to
    # guess which of several bare patches in the frame was actually flagged,
    # and guessing wrong silently corrupts the verdict.
    return (
        f'<figure><div class="frame">'
        f'<img src="{html.escape(path.name)}" alt="{html.escape(caption)}">'
        f'<span class="crosshair"></span></div>'
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
        # Sharp dated pair first — it is the one that actually decides the
        # verdict. The Sentinel-2 pair follows as corroboration, since its dates
        # match the detection window exactly even though its pixels do not
        # resolve much.
        cells = []
        for label, entry in (("before", chips.wayback_before), ("after", chips.wayback_after)):
            if entry is not None:
                # Named chip_path, not path: unpacking into `path` shadowed this
                # function's output-file parameter, so the HTML was written into
                # the last chip PNG and that image was silently destroyed.
                chip_path, captured = entry
                cells.append(_cell(chip_path, f"{captured} ({label}, sub-metre)"))

        cells += [
            _cell(chips.baseline_path, f"{chips.baseline_year} Sentinel-2 (10 m)"),
            _cell(chips.recent_path, f"{chips.recent_year} Sentinel-2 (10 m)"),
        ]
        if chips.highres_path is not None:
            cells.append(_cell(chips.highres_path, "sub-metre (undated) — what is this place?"))
        if chips.nicfi_path is not None:
            cells.append(_cell(chips.nicfi_path, f"NICFI {chips.recent_year} (<5 m)"))

        aid = html.escape(alert.alert_id)
        rows.append(
            f"""
    <section data-id="{aid}" data-stratum="{html.escape(size_stratum(alert.area_ha))}"
             data-area="{alert.area_ha:.3f}">
      <h2><code>{aid}</code>
        <span class="meta">{alert.area_ha:.2f} ha &middot;
        {alert.lat:.5f}, {alert.lon:.5f} &middot;
        first seen {html.escape(alert.first_seen)}</span></h2>
      <div class="chips">{"".join(cells)}</div>
      <div class="verdict">
        <button data-v="true_positive">Real clearing</button>
        <button data-v="false_positive">Not clearing</button>
        <button data-v="unclear">Unclear</button>
        <input type="text"
               placeholder="note &mdash; e.g. plantation rows, already bare, quarry">
        <a href="{html.escape(google_maps_url(alert.lat, alert.lon))}"
           target="_blank">open in maps</a>
      </div>
    </section>"""
        )

    complete = sum(c.is_complete for c in chipsets)
    # Stable identifier for this set of detections, so saved verdicts belong to
    # one sample rather than leaking into the next.
    sample_id = hashlib.sha256(
        ",".join(sorted(c.alert.alert_id for c in chipsets)).encode("utf-8")
    ).hexdigest()[:12]

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
  .frame {{ position: relative; width: {_CHIP_PIXELS}px; max-width: 100%; }}
  img {{ width: 100%; display: block; border-radius: 4px;
         image-rendering: pixelated; background: #000; }}
  /* Marks the detection centre. Several bare patches often sit in one frame;
     without this the reviewer has to guess which was flagged. */
  .crosshair {{ position: absolute; inset: 0; pointer-events: none; }}
  .crosshair::before, .crosshair::after {{
    content: ""; position: absolute; background: rgba(255,80,80,.85); }}
  .crosshair::before {{ left: 50%; top: 42%; width: 1px; height: 16%; }}
  .crosshair::after  {{ top: 50%; left: 42%; height: 1px; width: 16%; }}
  .missing {{ width: {_CHIP_PIXELS}px; height: {_CHIP_PIXELS}px; display: grid;
              place-items: center; background: #1a1a1a; border-radius: 4px;
              color: #666; text-align: center; }}
  figcaption {{ color: #999; font-size: .85rem; margin-top: .35rem; }}
  .verdict {{ display: flex; gap: .5rem; align-items: center; margin-top: .9rem;
              flex-wrap: wrap; }}
  .verdict button {{ font: inherit; padding: .45rem .9rem; border-radius: 4px;
    border: 1px solid #444; background: #1c1c1c; color: #ddd; cursor: pointer; }}
  .verdict button:hover {{ border-color: #777; }}
  .verdict button.on {{ color: #fff; }}
  .verdict button.on[data-v="true_positive"]  {{ background:#1d4b25; border-color:#3a9c4a; }}
  .verdict button.on[data-v="false_positive"] {{ background:#5a2020; border-color:#b04a4a; }}
  .verdict button.on[data-v="unclear"]        {{ background:#4a411a; border-color:#a89234; }}
  .verdict input {{ font: inherit; flex: 1; min-width: 16rem; padding: .45rem .6rem;
    background:#1c1c1c; border:1px solid #444; border-radius:4px; color:#ddd; }}
  .verdict a {{ color:#6cf; font-size:.85rem; }}
  section.done {{ opacity: .55; }}
  section.done:hover {{ opacity: 1; }}
  #bar {{ position: sticky; bottom: 0; background:#181818; border-top:1px solid #333;
    padding: .8rem 1rem; display:flex; gap:1rem; align-items:center;
    margin: 2rem -2rem -2rem; }}
  #bar button {{ font: inherit; padding:.5rem 1rem; border-radius:4px; border:0;
    background:#2b6cb0; color:#fff; cursor:pointer; }}
  #bar button:disabled {{ background:#333; color:#777; cursor:not-allowed; }}
  #count {{ color:#aaa; }}
</style></head><body>
<h1>Validation chips &mdash; {len(chipsets)} detections</h1>
<p class="intro">Each row is one detection, marked with a red crosshair.</p>
<p class="intro"><strong>The Sentinel-2 pair is your evidence of change.</strong>
It is only 10&nbsp;m per pixel and looks blurry, but its two dates match the
detection window exactly. The sub-metre image is far sharper but
<strong>undated</strong>, so it answers only <em>what is this place</em> &mdash;
plantation rows, quarry, riverbed, settlement &mdash; not whether anything changed.</p>
<p class="intro">Sub-metre <em>dated</em> imagery was attempted via Esri Wayback
and is not available here: both the 2025 and 2026 releases return the same
acquisition, because Esri has not re-flown this area between them. Showing them
as a before/after pair would invite the conclusion that nothing changed, which
the imagery cannot support either way, so only one copy is shown and it is
labelled undated.</p>
<p class="intro">The commonest false positive: a bare patch that was
<em>already there</em> in the before image. If the crosshair sits on ground bare
in both Sentinel-2 chips, that is a <code>false_positive</code> however obvious
the patch looks. Also watch for plantations &mdash; regular rows or sharp
rectangular blocks in the sharp image &mdash; which are not forest loss.</p>
<p class="intro">The commonest false positive here: a bare patch that was
<em>already there</em> in the before image. If the crosshair sits on ground that
was bare in both years, that is a <code>false_positive</code> however obvious the
patch looks. Also watch for plantations &mdash; unnaturally regular rows or sharp
rectangular blocks &mdash; which are not forest loss.</p>
<p class="intro">Record <code>true_positive</code>, <code>false_positive</code>
or <code>unclear</code> against each id in the worksheet CSV. Use
<code>unclear</code> freely &mdash; it is excluded from the score, and guessing
corrupts the number you are trying to measure.</p>
<p class="intro">{complete} of {len(chipsets)} have both dated chips.</p>
{"".join(rows)}
<div id="bar">
  <span id="count"></span>
  <button id="save" disabled>Download verdicts CSV</button>
  <span style="color:#777;font-size:.85rem">
    Saved in this browser as you go &mdash; refreshing will not lose your work.</span>
</div>
<script>
const SAMPLE_ID = "{sample_id}";
// Verdicts are recorded here rather than in the spreadsheet because matching
// ids across two windows by eye is where transcription errors happen.
// localStorage means a closed tab does not discard a half-finished review.
//
// The key is scoped to this specific set of detections. A single shared key
// meant a second review inherited the first one's verdicts and exported both
// together — which happened, silently merging one detector's results into
// another's. The scope is derived from the ids present, so the same sample
// resumes and a different one starts clean.
const KEY = "vanachakshu-verdicts-" + SAMPLE_ID;
const state = JSON.parse(localStorage.getItem(KEY) || "{{}}");

function refresh() {{
  let done = 0;
  document.querySelectorAll("section[data-id]").forEach(s => {{
    const rec = state[s.dataset.id];
    s.querySelectorAll(".verdict button").forEach(b =>
      b.classList.toggle("on", !!rec && rec.verdict === b.dataset.v));
    const note = s.querySelector(".verdict input");
    if (rec && rec.note && note.value !== rec.note) note.value = rec.note;
    s.classList.toggle("done", !!rec);
    if (rec) done++;
  }});
  const total = document.querySelectorAll("section[data-id]").length;
  document.getElementById("count").textContent = done + " of " + total + " reviewed";
  document.getElementById("save").disabled = done === 0;
}}

document.querySelectorAll("section[data-id]").forEach(s => {{
  s.querySelectorAll(".verdict button").forEach(b => b.onclick = () => {{
    const cur = state[s.dataset.id] || {{}};
    state[s.dataset.id] = {{
      verdict: b.dataset.v, note: cur.note || "",
      stratum: s.dataset.stratum, area: s.dataset.area
    }};
    localStorage.setItem(KEY, JSON.stringify(state));
    refresh();
  }});
  s.querySelector(".verdict input").oninput = e => {{
    if (!state[s.dataset.id]) return;   // a note without a verdict scores nothing
    state[s.dataset.id].note = e.target.value;
    localStorage.setItem(KEY, JSON.stringify(state));
  }};
}});

document.getElementById("save").onclick = () => {{
  const lines = ["alert_id,stratum,area_ha,verdict,note"];
  for (const [id, r] of Object.entries(state)) {{
    // Quote the note: a comma in free text would otherwise shift every column.
    lines.push([id, r.stratum, r.area, r.verdict,
                '"' + String(r.note || "").replace(/"/g, '""') + '"'].join(","));
  }}
  const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "verdicts.csv";
  a.click();
}};

refresh();
</script>
</body></html>
"""
    path.write_text(document, encoding="utf-8")
    return path
