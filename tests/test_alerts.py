"""Tests for alert identity, confirmation, and delivery-once guarantees.

Entirely pure — no Earth Engine, no network — so every one of these runs in CI.

That matters more here than anywhere else in the codebase. These rules decide
whether a real person is told something. The two failures that would destroy
trust in the system are announcing the same clearing repeatedly, and announcing
a rain shower as deforestation. Both are prevented here, so both are tested here.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from vanachakshu.alerts import (
    STORE_FORMAT_VERSION,
    AlertStore,
    Detection,
    detections_from_patch_records,
    haversine_m,
    new_alert_id,
)
from vanachakshu.config import AlertConfig

RADIUS_M = 150.0
DAY1 = date(2026, 1, 1)
DAY2 = date(2026, 1, 13)
DAY3 = date(2026, 1, 25)

# Inside the project AOI.
LON, LAT = 74.70, 14.95


def _detection(lon: float = LON, lat: float = LAT, area: float = 1.2, *, on: date) -> Detection:
    return Detection(lon=lon, lat=lat, area_ha=area, observed_on=on)


class TestHaversine:
    def test_zero_distance_to_itself(self) -> None:
        assert haversine_m(LON, LAT, LON, LAT) == pytest.approx(0.0, abs=1e-9)

    def test_is_symmetric(self) -> None:
        assert haversine_m(74.70, 14.95, 74.71, 14.96) == pytest.approx(
            haversine_m(74.71, 14.96, 74.70, 14.95)
        )

    def test_one_degree_of_latitude_is_about_111_km(self) -> None:
        # Known reference value; anchors the units as metres.
        assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_195, rel=0.01)

    def test_small_offsets_measure_tens_of_metres(self) -> None:
        # 0.00018 degrees of latitude is ~20 m — the scale of centroid drift
        # between runs that dedup has to absorb.
        assert haversine_m(LON, LAT, LON, LAT + 0.00018) == pytest.approx(20, abs=3)

    def test_longitude_distance_shrinks_away_from_the_equator(self) -> None:
        # Meridians converge toward the poles, so the same degree of longitude
        # is a shorter distance at higher latitude. A flat degree-based
        # threshold would silently change meaning with latitude.
        at_equator = haversine_m(0.0, 0.0, 1.0, 0.0)
        at_sixty = haversine_m(0.0, 60.0, 1.0, 60.0)
        assert at_sixty < at_equator / 1.8


class TestNewAlertId:
    def test_is_deterministic(self) -> None:
        assert new_alert_id(LON, LAT, DAY1) == new_alert_id(LON, LAT, DAY1)

    def test_differs_by_location(self) -> None:
        assert new_alert_id(LON, LAT, DAY1) != new_alert_id(74.75, 14.99, DAY1)

    def test_is_fixed_width_and_filename_safe(self) -> None:
        alert_id = new_alert_id(LON, LAT, DAY1)
        assert len(alert_id) == 16
        assert alert_id.isalnum()


class TestIdentityIsStableUnderDrift:
    """The property the grid-based first attempt could not deliver.

    Grids have boundaries: two detections 20 m apart can land on opposite sides
    of one and receive different ids, so a clearing whose centroid drifted
    across that line would be announced twice. Distance-based matching has no
    boundaries, so it does not fail that way.
    """

    def test_a_drifting_centroid_keeps_its_original_id(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        store.ingest([_detection(lon=74.70000, lat=14.95000, on=DAY1)], DAY1)
        original_id = store.alerts[0].alert_id

        # Same clearing, centroid moved ~20 m as the patch boundary shifted.
        store.ingest([_detection(lon=74.70018, lat=14.95015, on=DAY2)], DAY2)

        assert len(store.alerts) == 1
        assert store.alerts[0].alert_id == original_id
        assert store.alerts[0].confirmations == 2

    def test_drift_across_many_runs_never_forks(self) -> None:
        # Cumulative drift in one direction is the case a naive radius check
        # against the *original* point would eventually fail.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        for step in range(6):
            store.ingest([_detection(lon=74.70000 + step * 0.00018, lat=14.95000, on=DAY1)], DAY1)
        assert len(store.alerts) == 1

    def test_genuinely_separate_clearings_stay_separate(self) -> None:
        # ~1 km apart: far outside the dedup radius, must not be merged.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        store.ingest(
            [
                _detection(lon=74.70, lat=14.95, on=DAY1),
                _detection(lon=74.71, lat=14.96, on=DAY1),
            ],
            DAY1,
        )
        assert len(store.alerts) == 2

    def test_matches_the_nearest_alert_not_the_first_found(self) -> None:
        # With several alerts in range, attaching to the closest is the only
        # order-independent choice — otherwise behaviour would depend on the
        # order Earth Engine happened to return polygons in.
        #
        # At 14.95N one degree of longitude is ~107,554 m. Geometry below:
        #   A ---- 200 m ---- B      (apart by more than the 150 m radius,
        #   A --120 m-- P            so they stay separate)
        #               P --80 m-- B (P is in range of both, closer to B)
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1, dedup_radius_m=150.0))
        store.ingest([_detection(lon=74.70000, lat=14.95, on=DAY1)], DAY1)
        store.ingest([_detection(lon=74.70186, lat=14.95, on=DAY1)], DAY1)
        assert len(store.alerts) == 2, "seeds are 200 m apart and must not merge"

        nearer_id = next(a.alert_id for a in store.alerts if a.lon == pytest.approx(74.70186))
        store.ingest([_detection(lon=74.70112, lat=14.95, on=DAY2)], DAY2)

        assert len(store.alerts) == 2, "probe was in range of both; no third alert"
        matched = next(a for a in store.alerts if a.confirmations == 2)
        assert matched.alert_id == nearer_id


class TestConfirmationRule:
    def test_a_single_sighting_is_not_announced(self) -> None:
        # The rain-shower case. One drop in greenness is not evidence.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        assert store.ingest([_detection(on=DAY1)], DAY1) == []

    def test_second_sighting_confirms_and_announces(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        store.ingest([_detection(on=DAY1)], DAY1)
        notified = store.ingest([_detection(on=DAY2)], DAY2)
        assert len(notified) == 1
        assert notified[0].confirmations == 2

    def test_unconfirmed_sightings_are_kept_not_discarded(self) -> None:
        # These are the record of what the detector got wrong, which is what
        # threshold tuning needs. Throwing them away destroys that evidence.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        store.ingest([_detection(on=DAY1)], DAY1)
        assert len(store.pending()) == 1
        assert store.notified() == []

    def test_threshold_of_three_needs_three_passes(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=3))
        assert store.ingest([_detection(on=DAY1)], DAY1) == []
        assert store.ingest([_detection(on=DAY2)], DAY2) == []
        assert len(store.ingest([_detection(on=DAY3)], DAY3)) == 1

    def test_threshold_of_one_announces_immediately(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        assert len(store.ingest([_detection(on=DAY1)], DAY1)) == 1


class TestNotifyOnceGuarantee:
    """The most important behaviour in the module."""

    def test_a_confirmed_alert_is_never_announced_twice(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        store.ingest([_detection(on=DAY1)], DAY1)
        assert len(store.ingest([_detection(on=DAY2)], DAY2)) == 1

        # Every subsequent run sees the same clearing and must stay silent.
        for day in (DAY3, date(2026, 2, 6), date(2026, 2, 18)):
            assert store.ingest([_detection(on=day)], day) == []

    def test_rerunning_an_identical_detection_announces_nothing_new(self) -> None:
        # Idempotency. The scheduled job may be retried after a failure, or run
        # twice by accident; neither may produce duplicate messages.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        detections = [_detection(on=DAY1)]
        assert len(store.ingest(detections, DAY1)) == 1
        assert store.ingest(detections, DAY1) == []
        assert store.ingest(detections, DAY1) == []

    def test_a_retry_on_the_same_day_does_not_falsely_confirm(self) -> None:
        # The scheduled job retries after transient failures. Counting the same
        # imagery twice would confirm an alert that has genuinely been seen
        # once, defeating the entire point of the confirmation rule.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        assert store.ingest([_detection(on=DAY1)], DAY1) == []
        assert store.ingest([_detection(on=DAY1)], DAY1) == [], "retry must not confirm"
        assert store.ingest([_detection(on=DAY1)], DAY1) == []
        assert store.alerts[0].confirmations == 1

        # A genuinely new observation still confirms.
        assert len(store.ingest([_detection(on=DAY2)], DAY2)) == 1

    def test_a_retry_still_updates_the_recorded_area(self) -> None:
        # Not counting a retry as confirmation must not mean ignoring it: a
        # re-run with a better composite may measure the clearing more fully.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=2))
        store.ingest([_detection(area=1.0, on=DAY1)], DAY1)
        store.ingest([_detection(area=4.0, on=DAY1)], DAY1)
        assert store.alerts[0].confirmations == 1
        assert store.alerts[0].area_ha == 4.0

    def test_notification_date_is_recorded(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        notified = store.ingest([_detection(on=DAY1)], DAY1)
        assert notified[0].notified_on == DAY1.isoformat()


class TestGrowingClearings:
    def test_reported_area_never_shrinks(self) -> None:
        # A clearing that expands must not be re-announced, but its recorded
        # extent should reflect the largest seen — never less than what was
        # already reported to someone.
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        store.ingest([_detection(area=2.0, on=DAY1)], DAY1)
        store.ingest([_detection(area=5.0, on=DAY2)], DAY2)
        store.ingest([_detection(area=3.0, on=DAY3)], DAY3)
        assert store.alerts[0].area_ha == 5.0

    def test_last_seen_advances(self) -> None:
        store = AlertStore(Path("unused"), AlertConfig(min_confirmations=1))
        store.ingest([_detection(on=DAY1)], DAY1)
        store.ingest([_detection(on=DAY3)], DAY3)
        alert = store.alerts[0]
        assert alert.first_seen == DAY1.isoformat()
        assert alert.last_seen == DAY3.isoformat()


class TestPersistence:
    def test_survives_a_round_trip(self, tmp_path: Path) -> None:
        # The scheduled job is a fresh process each run, so state that does not
        # survive serialisation means the notify-once guarantee is worthless.
        path = tmp_path / "alerts.json"
        store = AlertStore(path, AlertConfig(min_confirmations=1))
        store.ingest([_detection(on=DAY1)], DAY1)
        store.save(DAY1)

        reloaded = AlertStore(path, AlertConfig(min_confirmations=1))
        reloaded.load()
        assert len(reloaded.notified()) == 1
        # And the reloaded store must still refuse to re-announce.
        assert reloaded.ingest([_detection(on=DAY2)], DAY2) == []

    def test_missing_file_is_an_empty_store_not_an_error(self, tmp_path: Path) -> None:
        # The first run of a new deployment has no state. Treating that as a
        # failure would mean the job could never start.
        store = AlertStore(tmp_path / "does-not-exist.json")
        store.load()
        assert store.alerts == []

    def test_refuses_an_unknown_format_version(self, tmp_path: Path) -> None:
        # Silently ignoring an unrecognised store would re-announce everything
        # it contains. Better to stop and demand a migration.
        path = tmp_path / "alerts.json"
        path.write_text(json.dumps({"version": 999, "alerts": []}), encoding="utf-8")
        store = AlertStore(path)
        with pytest.raises(ValueError, match="Migrate it rather than deleting"):
            store.load()

    def test_written_file_is_sorted_for_readable_diffs(self, tmp_path: Path) -> None:
        # The store is committed by the scheduled job; stable ordering means a
        # commit diff shows what changed rather than a reshuffle.
        path = tmp_path / "alerts.json"
        store = AlertStore(path, AlertConfig(min_confirmations=1))
        store.ingest(
            [
                _detection(lon=74.80, lat=14.99, on=DAY1),
                _detection(lon=74.70, lat=14.95, on=DAY1),
                _detection(lon=74.75, lat=14.97, on=DAY1),
            ],
            DAY1,
        )
        store.save(DAY1)
        written = json.loads(path.read_text(encoding="utf-8"))
        ids = [entry["alert_id"] for entry in written["alerts"]]
        assert ids == sorted(ids)

    def test_records_the_format_version(self, tmp_path: Path) -> None:
        path = tmp_path / "alerts.json"
        store = AlertStore(path)
        store.save(DAY1)
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == STORE_FORMAT_VERSION


class TestDetectionValidation:
    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_rejects_non_positive_area(self, area: float) -> None:
        with pytest.raises(ValueError, match="area_ha must be positive"):
            Detection(lon=LON, lat=LAT, area_ha=area, observed_on=DAY1)


class TestPatchRecordConversion:
    def test_converts_geojson_features(self) -> None:
        records = [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[74.70, 14.95], [74.71, 14.95], [74.71, 14.96], [74.70, 14.96]]
                    ],
                },
                "properties": {"area_ha": 1.5},
            }
        ]
        detections = detections_from_patch_records(records, DAY1)
        assert len(detections) == 1
        assert detections[0].area_ha == 1.5
        assert detections[0].lon == pytest.approx(74.705)
        assert detections[0].lat == pytest.approx(14.955)

    def test_empty_input_gives_empty_output(self) -> None:
        # A run that finds nothing is normal, not an error.
        assert detections_from_patch_records([], DAY1) == []
