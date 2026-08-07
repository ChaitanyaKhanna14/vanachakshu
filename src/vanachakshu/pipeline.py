"""One detection cycle, end to end.

Glue between the Earth Engine side (composites, detection, vectorisation) and
the pure alert side (identity, confirmation, notify-once). Deliberately thin:
anything with real logic belongs in the module it came from, where it can be
tested without credentials.

The one boundary worth naming is :func:`fetch_patch_records` — it is where
Earth Engine stops. Everything after it is plain dictionaries, so the whole
alerting half of a cycle can be exercised in CI with no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import ee

from vanachakshu.alerts import AlertStore, TrackedAlert, detections_from_patch_records
from vanachakshu.config import (
    AreaOfInterest,
    BoundingBox,
    EmbeddingDetectionConfig,
    SeasonWindow,
)
from vanachakshu.detect import detect_embedding_disturbance, disturbance_patches

__all__ = [
    "RunResult",
    "default_comparison_years",
    "fetch_patch_records",
    "run_cycle",
    "store_path_for",
]


def default_comparison_years(
    season: SeasonWindow, today: date, gap_years: int = 1
) -> tuple[int, int]:
    """Pick which two years to compare when nobody says.

    The scheduled job has no one to tell it, so it takes the most recent
    *complete* seasonal window and looks back ``gap_years``.

    Completeness matters: compositing a season still in progress yields a
    thinner, cloudier image than the year it is compared against, which then
    reads as vegetation loss. See
    :meth:`~vanachakshu.config.SeasonWindow.most_recent_complete_year`.

    A one-year gap is the default because the Phase 1 finding showed gap length
    barely affects the score — so the shortest gap wins, since it at least
    narrows down *when* the loss happened.
    """
    if gap_years < 1:
        raise ValueError(f"gap_years must be at least 1, got {gap_years}")
    recent = season.most_recent_complete_year(today)
    return recent - gap_years, recent


# Grid divisions per axis when vectorising. Four gives sixteen tiles of roughly
# 90 km² each, which stays inside Earth Engine's per-request budget for a
# 128-band computation while keeping the number of round-trips manageable.
_VECTORISE_TILES: Final = 4


def _grid(bbox: BoundingBox, divisions: int) -> list[ee.Geometry]:
    """Split a bounding box into a ``divisions`` by ``divisions`` grid.

    Tiles share edges, so a clearing straddling a boundary is vectorised twice
    as two partial polygons. Both land within the alert store's dedup radius of
    each other and merge into one alert, so the seam does not produce duplicates
    — but it does mean a straddling clearing's reported area is the larger
    fragment rather than the whole, which is a small and deliberate
    underestimate rather than an error.
    """
    lon_step = (bbox.east - bbox.west) / divisions
    lat_step = (bbox.north - bbox.south) / divisions
    return [
        ee.Geometry.Rectangle(
            [
                bbox.west + i * lon_step,
                bbox.south + j * lat_step,
                bbox.west + (i + 1) * lon_step,
                bbox.south + (j + 1) * lat_step,
            ]
        )
        for i in range(divisions)
        for j in range(divisions)
    ]


def store_path_for(aoi: AreaOfInterest, root: Path | None = None) -> Path:
    """Where an AOI's alert store lives.

    Under ``data/alerts/`` inside the repository, because the scheduled job
    commits it — history and audit trail come from git rather than a database.
    """
    base = root if root is not None else Path("data") / "alerts"
    return base / f"{aoi.slug}.json"


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single cycle, in terms worth printing or emailing."""

    baseline_year: int
    recent_year: int
    patches_found: int
    patches_total_ha: float
    new_alerts: tuple[TrackedAlert, ...]
    pending_count: int
    notified_total: int
    dry_run: bool

    @property
    def new_alert_ha(self) -> float:
        return sum(alert.area_ha for alert in self.new_alerts)

    def summary_lines(self) -> list[str]:
        """Plain-text report. No markup, so the same text works in a terminal,
        a CI log, and the body of an alert email."""
        lines = [
            f"Compared {self.baseline_year} with {self.recent_year}",
            f"  disturbances detected : {self.patches_found} ({self.patches_total_ha:.1f} ha)",
            f"  newly confirmed       : {len(self.new_alerts)} ({self.new_alert_ha:.1f} ha)",
            f"  awaiting confirmation : {self.pending_count}",
            f"  announced to date     : {self.notified_total}",
        ]
        if self.dry_run:
            lines.append("  DRY RUN - alert store was not written")
        return lines


def fetch_patch_records(
    aoi: AreaOfInterest,
    season: SeasonWindow,
    baseline_year: int,
    recent_year: int,
    config: EmbeddingDetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Run detection on Earth Engine and bring back plain GeoJSON features.

    Uses the AlphaEarth embedding detector. Measured against the NDVI detector
    it replaced, on the same AOI, years, tolerance and scale:

    ==================  =========  ========  =====
    Detector            Precision  Recall    F1
    ==================  =========  ========  =====
    NDVI drop >= 0.15   0.583      0.013     0.025
    Embedding L2 >=0.45 0.773      0.129     0.221
    ==================  =========  ========  =====

    ``season`` is no longer used for detection — embeddings are annual and
    already seasonally aware — but is kept in the signature because the alert
    store, reports and chip generation are all organised around it, and because
    the optical detector remains available for comparison.

    This is the last function in a cycle that needs credentials. It returns
    dictionaries rather than ``ee`` objects on purpose, so everything
    downstream is ordinary Python.
    """
    if baseline_year >= recent_year:
        raise ValueError(
            f"baseline_year ({baseline_year}) must be earlier than recent_year ({recent_year})"
        )

    cfg = config if config is not None else EmbeddingDetectionConfig()

    # Fetched tile by tile rather than in one request. The embedding detector
    # loads 128 bands — two years of 64 — before differencing, and vectorising
    # that across the whole AOI exceeds Earth Engine's memory budget; raising
    # tileScale far enough to fix that makes the request time out instead.
    #
    # Splitting into separate requests is the same remedy that fixed training
    # sample extraction. Detections are sparse — a couple of hectares in
    # 146,000 — so most tiles are empty and cheap.
    features: list[dict[str, Any]] = []
    for tile in _grid(aoi.bbox, _VECTORISE_TILES):
        disturbance = detect_embedding_disturbance(tile, baseline_year, recent_year, cfg)
        patches = disturbance_patches(disturbance, tile, cfg)
        collection: dict[str, Any] = patches.getInfo() or {}
        features.extend(collection.get("features", []))

    return features


def run_cycle(
    aoi: AreaOfInterest,
    season: SeasonWindow,
    baseline_year: int,
    recent_year: int,
    today: date,
    store: AlertStore,
    config: EmbeddingDetectionConfig | None = None,
    dry_run: bool = False,
) -> RunResult:
    """Detect, confirm, record, and report.

    ``store`` is passed in rather than constructed here so a caller can supply
    a temporary one — which is what makes a full cycle testable end to end.

    On ``dry_run`` the store is loaded and updated in memory but never written.
    Useful for trying a threshold without burning the confirmation state, which
    is not recoverable once an alert has been marked announced.
    """
    store.load()

    records = fetch_patch_records(aoi, season, baseline_year, recent_year, config)
    detections = detections_from_patch_records(records, today)
    new_alerts = store.ingest(detections, today)

    if not dry_run:
        store.save(today)

    return RunResult(
        baseline_year=baseline_year,
        recent_year=recent_year,
        patches_found=len(records),
        patches_total_ha=sum(float(r.get("properties", {}).get("area_ha", 0.0)) for r in records),
        new_alerts=tuple(new_alerts),
        pending_count=len(store.pending()),
        notified_total=len(store.notified()),
        dry_run=dry_run,
    )
