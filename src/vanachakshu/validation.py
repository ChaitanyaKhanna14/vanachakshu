"""Manual validation against high-resolution imagery.

Scoring against Hansen has reached its limit. Hansen is a 30 m annual product
with its own detection floor, and in the test area it records 10.5 ha of loss
across 366 km² — a base rate of 0.03%. At that sparsity a detection that misses
it is equally consistent with a false positive, with real clearing Hansen never
recorded, with plantation harvest Hansen correctly declines to call forest loss,
and with truth sitting in radar layover. **Tuning against a reference that
sparse fits noise.**

The way out is to look. NICFI Planet basemaps are <5 m and free for
noncommercial use, which is fine enough to judge a half-hectare clearing by eye.
This module builds the worksheet for that, and does the statistics on the way
back.

Everything here is pure — no Earth Engine, no network — so the sampling scheme
and the interval maths are tested rather than trusted.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from vanachakshu.alerts import TrackedAlert
from vanachakshu.report import google_maps_url, openstreetmap_url

__all__ = [
    "SIZE_STRATA",
    "ValidationRecord",
    "Verdict",
    "load_verdicts",
    "precision_report",
    "size_stratum",
    "stratified_sample",
    "wilson_interval",
    "worksheet_alert_ids",
    "write_worksheet",
]


class Verdict(StrEnum):
    """What a human decided when they looked at the imagery."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    # Kept deliberately. Forcing a binary call on genuinely ambiguous imagery
    # manufactures certainty; these are excluded from the denominator and
    # reported separately, because how *often* the answer is unclear is itself a
    # finding about the detector.
    UNCLEAR = "unclear"


# Boundaries in hectares. Stratifying by size matters because small clearings
# are the hard case and the interesting one: a headline precision figure
# dominated by a few large obvious clearcuts hides the performance that
# actually determines whether the system is useful.
SIZE_STRATA: Final[tuple[tuple[str, float, float], ...]] = (
    ("under_0.5ha", 0.0, 0.5),
    ("0.5_to_1ha", 0.5, 1.0),
    ("1_to_5ha", 1.0, 5.0),
    ("over_5ha", 5.0, math.inf),
)


def size_stratum(area_ha: float) -> str:
    """Name the size band an area falls into."""
    if area_ha <= 0:
        raise ValueError(f"area_ha must be positive, got {area_ha}")
    for name, low, high in SIZE_STRATA:
        if low <= area_ha < high:
            return name
    raise AssertionError(f"no stratum matched {area_ha}")  # pragma: no cover


def stratified_sample(
    alerts: Sequence[TrackedAlert],
    per_stratum: int,
    seed: int,
) -> list[TrackedAlert]:
    """Sample up to ``per_stratum`` alerts from each size band.

    Stratified rather than uniform, because detections are heavily skewed toward
    small patches. A uniform sample of 300 would be almost entirely sub-hectare
    and would say nothing about large clearings — or, if the skew ran the other
    way, would hide poor performance on exactly the small ones that matter.

    ``seed`` is required, not optional. A validation sample that cannot be
    reproduced cannot be audited, and "I checked 300 random alerts" is only
    meaningful if someone else can draw the same 300.
    """
    if per_stratum < 1:
        raise ValueError(f"per_stratum must be positive, got {per_stratum}")

    rng = random.Random(seed)
    chosen: list[TrackedAlert] = []

    for name, _, _ in SIZE_STRATA:
        members = [a for a in alerts if size_stratum(a.area_ha) == name]
        # Sorted first so the sample depends only on the seed, never on the
        # order the store happened to return.
        members.sort(key=lambda a: a.alert_id)
        chosen.extend(rng.sample(members, min(per_stratum, len(members))))

    return chosen


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, default 95%.

    Wilson rather than the textbook ``p ± z·sqrt(p(1-p)/n)``. The normal
    approximation breaks exactly where this project needs it most: near 0 or 1
    it produces intervals extending below 0 or above 1, and at small ``n`` it is
    far too narrow. Observing 0 successes in 30 gives a naive interval of exactly
    zero width — an infinitely confident claim from thirty samples.

    Returns ``(0.0, 1.0)`` for an empty sample: with no observations, every
    proportion remains possible.
    """
    if successes < 0 or total < 0:
        raise ValueError("counts cannot be negative")
    if successes > total:
        raise ValueError(f"successes ({successes}) cannot exceed total ({total})")
    if total == 0:
        return 0.0, 1.0

    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z / denominator * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class ValidationRecord:
    """One human judgement about one detection."""

    alert_id: str
    area_ha: float
    stratum: str
    verdict: Verdict
    note: str = ""


@dataclass(frozen=True)
class StratumResult:
    """Precision within one size band, with its uncertainty."""

    stratum: str
    true_positive: int
    false_positive: int
    unclear: int

    @property
    def judged(self) -> int:
        """Samples where a call was actually made."""
        return self.true_positive + self.false_positive

    @property
    def precision(self) -> float:
        return self.true_positive / self.judged if self.judged else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.true_positive, self.judged)

    def as_row(self) -> str:
        low, high = self.interval
        unclear_note = f", {self.unclear} unclear" if self.unclear else ""
        if not self.judged:
            return f"{self.stratum:>12}: no judged samples{unclear_note}"
        return (
            f"{self.stratum:>12}: precision {self.precision:.2f} "
            f"[{low:.2f}-{high:.2f}] on n={self.judged}{unclear_note}"
        )


def precision_report(records: Sequence[ValidationRecord]) -> list[StratumResult]:
    """Precision per size band, in the order of :data:`SIZE_STRATA`.

    Every stratum is returned, including empty ones. A silently absent band
    would read as "nothing to report" when it actually means "never sampled".
    """
    results: list[StratumResult] = []
    for name, _, _ in SIZE_STRATA:
        in_stratum = [r for r in records if r.stratum == name]
        results.append(
            StratumResult(
                stratum=name,
                true_positive=sum(r.verdict is Verdict.TRUE_POSITIVE for r in in_stratum),
                false_positive=sum(r.verdict is Verdict.FALSE_POSITIVE for r in in_stratum),
                unclear=sum(r.verdict is Verdict.UNCLEAR for r in in_stratum),
            )
        )
    return results


_WORKSHEET_FIELDS: Final = (
    "alert_id",
    "stratum",
    "area_ha",
    "lat",
    "lon",
    "first_seen",
    "confirmations",
    "satellite_view",
    "map_view",
    "verdict",
    "note",
)


def write_worksheet(alerts: Sequence[TrackedAlert], path: Path) -> Path:
    """Write a CSV for a human to fill in, one row per detection.

    ``verdict`` is left blank on purpose. Pre-filling it — even with a
    "suggested" value — biases the reviewer toward agreeing with the detector,
    which would quietly convert an independent check into a rubber stamp.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_WORKSHEET_FIELDS))
        writer.writeheader()
        for alert in alerts:
            writer.writerow(
                {
                    "alert_id": alert.alert_id,
                    "stratum": size_stratum(alert.area_ha),
                    "area_ha": round(alert.area_ha, 3),
                    "lat": round(alert.lat, 6),
                    "lon": round(alert.lon, 6),
                    "first_seen": alert.first_seen,
                    "confirmations": alert.confirmations,
                    "satellite_view": google_maps_url(alert.lat, alert.lon),
                    "map_view": openstreetmap_url(alert.lat, alert.lon),
                    "verdict": "",
                    "note": "",
                }
            )
    return path


def worksheet_alert_ids(path: Path) -> list[str]:
    """Every alert id in a worksheet, filled in or not.

    Distinct from :func:`load_verdicts`, which deliberately skips unreviewed
    rows. Fetching imagery is the step that happens *before* anyone reviews
    anything, so it needs the whole sample.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["alert_id"] for row in csv.DictReader(handle) if row.get("alert_id")]


def load_verdicts(path: Path) -> list[ValidationRecord]:
    """Read a filled-in worksheet.

    Rows with a blank verdict are skipped rather than treated as anything: a
    half-finished worksheet must not silently score as though the unreviewed
    rows were false positives.
    """
    records: list[ValidationRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("verdict") or "").strip().lower()
            if not raw:
                continue
            try:
                verdict = Verdict(raw)
            except ValueError as exc:
                raise ValueError(
                    f"row {row.get('alert_id')!r} has verdict {raw!r}; "
                    f"expected one of {[v.value for v in Verdict]}"
                ) from exc
            records.append(
                ValidationRecord(
                    alert_id=row["alert_id"],
                    area_ha=float(row["area_ha"]),
                    stratum=row["stratum"],
                    verdict=verdict,
                    note=(row.get("note") or "").strip(),
                )
            )
    return records
