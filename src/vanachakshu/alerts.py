"""Alert identity, confirmation, and delivery-once guarantees.

This is what separates a script that prints detections from a service someone
can rely on. It is pure Python — no Earth Engine, no network — so all of it runs
in CI, which matters because these are the rules that decide whether a real
person is told something.

Three guarantees, in order of importance:

1. **Nothing is announced twice.** An alert carries a stable identity derived
   from its location, and once notified it is never notified again. A system
   that re-sends last month's clearing every fortnight is worse than no system:
   it trains its users to ignore it.

2. **Nothing is announced on a single sighting.** A disturbance must be seen on
   ``min_confirmations`` separate passes first. Wet ground and thin haze look
   like clearing and recover by the next pass; cut forest stays cut. This costs
   one revisit of latency and removes most weather-driven false alarms.

3. **Nothing is silently forgotten.** Every disturbance ever seen stays in the
   store with its history, including the ones never confirmed. Those are the
   record of what the detector got wrong, which is what tuning needs.

The store is a plain JSON file, deliberately. It is committed to the repository
by the scheduled job, so history, backup and an audit trail come free from git
rather than from a database that has to be paid for and kept alive.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final

from vanachakshu.config import AlertConfig

__all__ = [
    "STORE_FORMAT_VERSION",
    "AlertStore",
    "Detection",
    "TrackedAlert",
    "haversine_m",
    "new_alert_id",
]

STORE_FORMAT_VERSION: Final = 1

_EARTH_RADIUS_M: Final = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two coordinates, in metres.

    Used to decide whether a new detection is the same clearing as a known one.

    An earlier version of this module gave each detection an identity by
    snapping it to a grid cell and hashing the cell. That is subtly broken:
    grids have boundaries, and two detections 20 m apart can fall on opposite
    sides of one. A clearing whose centroid drifted across that line would be
    announced a second time — precisely the failure the module exists to
    prevent. Distance has no boundaries, so matching by distance does not have
    that failure mode.
    """
    lon1_r, lat1_r, lon2_r, lat2_r = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def new_alert_id(lon: float, lat: float, first_seen: date) -> str:
    """Mint an identifier for a newly discovered disturbance.

    Derived from where and when it was *first* seen, then stored. Later
    sightings inherit the stored id by proximity rather than recomputing one,
    so the identity cannot drift as the patch boundary changes.

    A hash rather than raw coordinates so ids are fixed-width and safe in
    filenames, URLs and message subjects.
    """
    key = f"{lon:.6f},{lat:.6f},{first_seen.isoformat()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Detection:
    """One disturbance found by a single detection run."""

    lon: float
    lat: float
    area_ha: float
    observed_on: date

    def __post_init__(self) -> None:
        if self.area_ha <= 0:
            raise ValueError(f"area_ha must be positive, got {self.area_ha}")


@dataclass(frozen=True)
class TrackedAlert:
    """A disturbance's accumulated history across runs."""

    alert_id: str
    lon: float
    lat: float
    area_ha: float
    first_seen: str
    last_seen: str
    confirmations: int
    notified_on: str | None = None

    @property
    def is_notified(self) -> bool:
        return self.notified_on is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrackedAlert:
        return cls(
            alert_id=str(raw["alert_id"]),
            lon=float(raw["lon"]),
            lat=float(raw["lat"]),
            area_ha=float(raw["area_ha"]),
            first_seen=str(raw["first_seen"]),
            last_seen=str(raw["last_seen"]),
            confirmations=int(raw["confirmations"]),
            notified_on=raw.get("notified_on"),
        )


class AlertStore:
    """Persistent record of every disturbance seen, and what was announced."""

    def __init__(self, path: Path, config: AlertConfig | None = None) -> None:
        self.path = path
        self.config = config if config is not None else AlertConfig()
        self._alerts: dict[str, TrackedAlert] = {}

    def load(self) -> None:
        """Read the store. A missing file is an empty store, not an error.

        The first run of a new deployment has no state, and that must not be
        treated as a failure — otherwise the scheduled job can never start.
        """
        if not self.path.exists():
            self._alerts = {}
            return

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        version = raw.get("version")
        if version != STORE_FORMAT_VERSION:
            raise ValueError(
                f"alert store at {self.path} is format version {version!r}, "
                f"expected {STORE_FORMAT_VERSION}. Migrate it rather than "
                f"deleting it — it is the record of everything already announced."
            )
        self._alerts = {
            entry["alert_id"]: TrackedAlert.from_dict(entry) for entry in raw.get("alerts", [])
        }

    def save(self, updated_on: date) -> None:
        """Write the store, sorted for a readable diff.

        Sorting matters because this file is committed by the scheduled job:
        a stable order means each commit's diff shows what actually changed
        rather than a reshuffle.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_FORMAT_VERSION,
            "updated": updated_on.isoformat(),
            "alerts": [self._alerts[key].to_dict() for key in sorted(self._alerts)],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @property
    def alerts(self) -> list[TrackedAlert]:
        """All tracked disturbances, in stable id order."""
        return [self._alerts[key] for key in sorted(self._alerts)]

    def ingest(self, detections: Iterable[Detection], today: date) -> list[TrackedAlert]:
        """Fold a run's detections in, and return only what to announce now.

        Returns alerts that have *just* reached the confirmation threshold and
        have never been notified. Re-running the same detections returns an
        empty list on every run after the first — that is the notify-once
        guarantee, and it is the property most worth testing.
        """
        to_notify: list[TrackedAlert] = []

        for detection in detections:
            existing = self._nearest_within_radius(detection.lon, detection.lat)

            if existing is None:
                key = new_alert_id(detection.lon, detection.lat, detection.observed_on)
                merged = TrackedAlert(
                    alert_id=key,
                    lon=detection.lon,
                    lat=detection.lat,
                    area_ha=detection.area_ha,
                    first_seen=detection.observed_on.isoformat(),
                    last_seen=detection.observed_on.isoformat(),
                    confirmations=1,
                )
            else:
                key = existing.alert_id
                merged = replace(
                    existing,
                    last_seen=detection.observed_on.isoformat(),
                    confirmations=existing.confirmations + 1,
                    # Clearings grow. Keep the largest extent seen so the
                    # reported area never shrinks below what was announced.
                    area_ha=max(existing.area_ha, detection.area_ha),
                )

            if not merged.is_notified and merged.confirmations >= self.config.min_confirmations:
                merged = replace(merged, notified_on=today.isoformat())
                to_notify.append(merged)

            self._alerts[key] = merged

        return to_notify

    def _nearest_within_radius(self, lon: float, lat: float) -> TrackedAlert | None:
        """Closest known alert within the dedup radius, if any.

        Nearest rather than first-found: with several alerts inside the radius,
        attaching to the closest is the only choice that does not depend on
        insertion order, which would make the store's behaviour depend on the
        order Earth Engine happened to return polygons in.
        """
        best: TrackedAlert | None = None
        best_distance = self.config.dedup_radius_m

        for candidate in self._alerts.values():
            distance = haversine_m(lon, lat, candidate.lon, candidate.lat)
            if distance <= best_distance:
                best, best_distance = candidate, distance
        return best

    def pending(self) -> list[TrackedAlert]:
        """Seen at least once but not yet confirmed enough to announce."""
        return [a for a in self.alerts if not a.is_notified]

    def notified(self) -> list[TrackedAlert]:
        """Already announced. Never announced again."""
        return [a for a in self.alerts if a.is_notified]


def detections_from_patch_records(
    records: Sequence[dict[str, Any]], observed_on: date
) -> list[Detection]:
    """Convert Earth Engine patch features into detections.

    Kept here rather than in :mod:`vanachakshu.detect` so that the alert
    pipeline can be exercised end-to-end in CI from plain dictionaries, with no
    Earth Engine involved.
    """
    detections: list[Detection] = []
    for record in records:
        coordinates = record["geometry"]["coordinates"]
        properties = record.get("properties", record)
        lon, lat = _polygon_centroid(coordinates)
        detections.append(
            Detection(
                lon=lon,
                lat=lat,
                area_ha=float(properties["area_ha"]),
                observed_on=observed_on,
            )
        )
    return detections


def _polygon_centroid(coordinates: Any) -> tuple[float, float]:
    """Mean vertex position of a GeoJSON polygon's outer ring.

    Deliberately the vertex mean rather than the true area centroid: the result
    is only used to pick a grid cell, and the cell is far larger than the
    difference between the two.
    """
    ring = coordinates[0]
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return sum(lons) / len(lons), sum(lats) / len(lats)
