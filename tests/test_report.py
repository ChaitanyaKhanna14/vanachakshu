"""Tests for alert formatting.

Pure — no network, no Earth Engine.

Several of these assert on *wording*, which is unusual for a test suite and
deliberate here. The project's responsible-use rules are binding: an alert must
never assert that an offence occurred. That is a property of the output, so it
belongs in the tests rather than only in a document nobody re-reads.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from vanachakshu.alerts import TrackedAlert
from vanachakshu.config import YELLAPUR_TALUK
from vanachakshu.report import (
    GROUND_VERIFICATION_NOTICE,
    alerts_to_geojson,
    format_alert,
    format_digest,
    google_maps_url,
    openstreetmap_url,
    write_reports,
)

ISSUED = date(2026, 8, 4)

# Words that must never appear in the *substance* of a message. Each asserts a
# cause the imagery cannot establish.
#
# Checked against the text with the disclaimer removed, because the disclaimer
# legitimately uses some of these while negating them ("not evidence of any
# offence"). A blanket scan flagged its own safety notice — the check has to
# distinguish asserting a cause from explicitly disclaiming one.
ACCUSATORY_TERMS = ["illegal", "logging", "encroach", "crime", "offence", "culprit", "poach"]


def _substance_of(text: str) -> str:
    """Message text with the standard disclaimer stripped out."""
    return text.replace(GROUND_VERIFICATION_NOTICE, "").lower()


def _alert(
    alert_id: str = "3aad293b1f7961ed",
    area_ha: float = 2.4,
    lat: float = 15.0265,
    lon: float = 74.6358,
    confirmations: int = 2,
) -> TrackedAlert:
    return TrackedAlert(
        alert_id=alert_id,
        lon=lon,
        lat=lat,
        area_ha=area_ha,
        first_seen="2026-07-05",
        last_seen="2026-08-04",
        confirmations=confirmations,
        notified_on="2026-08-04",
    )


class TestResponsibleWording:
    """Binding project rule, enforced in code rather than only in docs."""

    def test_single_alert_makes_no_accusation(self) -> None:
        text = _substance_of(format_alert(_alert(), YELLAPUR_TALUK))
        for term in ACCUSATORY_TERMS:
            assert term not in text, f"alert text asserts a cause it cannot know: {term!r}"

    def test_digest_makes_no_accusation(self) -> None:
        text = _substance_of(format_digest([_alert()], YELLAPUR_TALUK, ISSUED))
        for term in ACCUSATORY_TERMS:
            assert term not in text, f"digest asserts a cause it cannot know: {term!r}"

    def test_empty_digest_makes_no_accusation(self) -> None:
        text = _substance_of(format_digest([], YELLAPUR_TALUK, ISSUED))
        for term in ACCUSATORY_TERMS:
            assert term not in text

    def test_the_disclaimer_is_what_carries_the_negation(self) -> None:
        # Guards the exemption above from being abused: the notice may mention
        # these words only because it denies them. If it ever stopped denying,
        # stripping it before scanning would hide a real accusation.
        notice = GROUND_VERIFICATION_NOTICE.lower()
        assert "not evidence" in notice
        assert "not confirmed" in notice

    def test_digest_carries_the_verification_notice(self) -> None:
        assert GROUND_VERIFICATION_NOTICE in format_digest([_alert()], YELLAPUR_TALUK, ISSUED)

    def test_notice_says_possible_and_demands_verification(self) -> None:
        notice = GROUND_VERIFICATION_NOTICE.lower()
        assert "possible" in notice
        assert "ground verification" in notice
        assert "not evidence" in notice

    def test_geojson_status_is_hedged(self) -> None:
        feature = alerts_to_geojson([_alert()], YELLAPUR_TALUK)["features"][0]
        assert "requires ground verification" in feature["properties"]["status"]


class TestGeoJsonStatusDistinguishesConfidence:
    """A GPS file that treated confirmed and unconfirmed alike would send
    someone to locations seen exactly once — where the weather-driven false
    positives live."""

    def test_confirmed_alert_says_so(self) -> None:
        confirmed = _alert()  # notified_on is set
        status = alerts_to_geojson([confirmed], YELLAPUR_TALUK)["features"][0]["properties"][
            "status"
        ]
        assert "confirmed on multiple passes" in status
        assert "UNCONFIRMED" not in status

    def test_unconfirmed_alert_is_flagged_loudly(self) -> None:
        pending = TrackedAlert(
            alert_id="pending01",
            lon=74.6358,
            lat=15.0265,
            area_ha=1.1,
            first_seen="2026-08-04",
            last_seen="2026-08-04",
            confirmations=1,
            notified_on=None,
        )
        status = alerts_to_geojson([pending], YELLAPUR_TALUK)["features"][0]["properties"]["status"]
        assert status.startswith("UNCONFIRMED")
        assert "1 pass" in status

    def test_unconfirmed_plural_is_grammatical(self) -> None:
        pending = TrackedAlert(
            alert_id="pending02",
            lon=74.6358,
            lat=15.0265,
            area_ha=1.1,
            first_seen="2026-07-01",
            last_seen="2026-08-04",
            confirmations=2,
            notified_on=None,
        )
        status = alerts_to_geojson([pending], YELLAPUR_TALUK)["features"][0]["properties"]["status"]
        assert "2 passes" in status


class TestFormatAlert:
    def test_contains_what_someone_needs_to_act(self) -> None:
        text = format_alert(_alert(), YELLAPUR_TALUK)
        assert "3aad293b1f7961ed" in text  # which alert
        assert "2.40 ha" in text  # how big
        assert "15.02650" in text  # where
        assert "2026-08-04" in text  # when
        assert "2 separate satellite passes" in text  # how confident

    def test_includes_both_map_links(self) -> None:
        text = format_alert(_alert(), YELLAPUR_TALUK)
        assert "google.com/maps" in text
        assert "openstreetmap.org" in text

    def test_names_the_area_of_interest(self) -> None:
        assert YELLAPUR_TALUK.name in format_alert(_alert(), YELLAPUR_TALUK)

    @pytest.mark.parametrize(
        ("confirmations", "expected"),
        [(1, "1 separate satellite pass"), (2, "2 separate satellite passes")],
    )
    def test_confirmation_count_is_grammatical(self, confirmations: int, expected: str) -> None:
        # Cosmetic, but this text is read by the people the project is for, and
        # sloppiness in an alert undermines confidence in the thing reporting it.
        assert expected in format_alert(_alert(confirmations=confirmations), YELLAPUR_TALUK)

    def test_is_plain_text(self) -> None:
        # The same string goes to a terminal, an email, and Telegram.
        text = format_alert(_alert(), YELLAPUR_TALUK)
        assert "<" not in text
        assert "[bold" not in text


class TestSizeDescription:
    @pytest.mark.parametrize(
        ("area_ha", "expected_fragment"),
        [
            (0.55, "square metres"),
            (2.4, "football pitches"),
            (25.0, "hectares"),
        ],
    )
    def test_scale_is_described_in_familiar_terms(
        self, area_ha: float, expected_fragment: str
    ) -> None:
        # Hectares mean little to a journalist or a community member. An alert
        # that only a forest officer can parse reaches fewer people.
        assert expected_fragment in format_alert(_alert(area_ha=area_ha), YELLAPUR_TALUK)


class TestFormatDigest:
    def test_largest_alert_comes_first(self) -> None:
        # With limited time, the biggest clearing is the one worth visiting.
        text = format_digest(
            [_alert(alert_id="small", area_ha=0.6), _alert(alert_id="large", area_ha=9.9)],
            YELLAPUR_TALUK,
            ISSUED,
        )
        assert text.index("large") < text.index("small")

    def test_reports_count_and_total(self) -> None:
        text = format_digest(
            [_alert(alert_id="a", area_ha=1.5), _alert(alert_id="b", area_ha=2.5)],
            YELLAPUR_TALUK,
            ISSUED,
        )
        assert "2 newly confirmed" in text
        assert "4.00 ha" in text

    def test_singular_wording_for_one_alert(self) -> None:
        assert "1 newly confirmed possible disturbance," in format_digest(
            [_alert()], YELLAPUR_TALUK, ISSUED
        )

    def test_empty_cycle_says_so_plainly(self) -> None:
        # A quiet cycle is normal. The message must read as "nothing found",
        # not as an error or an empty template.
        text = format_digest([], YELLAPUR_TALUK, ISSUED)
        assert "No newly confirmed forest disturbances" in text
        assert YELLAPUR_TALUK.name in text

    def test_includes_the_issue_date(self) -> None:
        assert "2026-08-04" in format_digest([], YELLAPUR_TALUK, ISSUED)


class TestMapLinks:
    def test_google_link_requests_satellite_imagery(self) -> None:
        # Without the data parameter this opens the road map, which shows
        # nothing useful over forest.
        assert "!3m1!1e3" in google_maps_url(15.0265, 74.6358)

    def test_links_carry_the_coordinates(self) -> None:
        for url in (google_maps_url(15.0265, 74.6358), openstreetmap_url(15.0265, 74.6358)):
            assert "15.026500" in url
            assert "74.635800" in url

    def test_negative_coordinates_survive(self) -> None:
        # Nothing about the formatter should assume the northern or eastern
        # hemisphere, even though this AOI is in both.
        url = google_maps_url(-15.0265, -74.6358)
        assert "-15.026500" in url
        assert "-74.635800" in url


class TestGeoJsonExport:
    def test_is_valid_geojson(self) -> None:
        data = alerts_to_geojson([_alert()], YELLAPUR_TALUK)
        assert data["type"] == "FeatureCollection"
        assert data["features"][0]["geometry"]["type"] == "Point"

    def test_coordinates_are_lon_lat_not_lat_lon(self) -> None:
        # GeoJSON mandates [longitude, latitude]. Reversing it is the single
        # most common export bug, and it silently puts every alert in the
        # wrong hemisphere rather than failing.
        coords = alerts_to_geojson([_alert()], YELLAPUR_TALUK)["features"][0]["geometry"][
            "coordinates"
        ]
        assert coords == [pytest.approx(74.6358), pytest.approx(15.0265)]

    def test_carries_the_fields_a_field_visit_needs(self) -> None:
        props = alerts_to_geojson([_alert()], YELLAPUR_TALUK)["features"][0]["properties"]
        for key in ("alert_id", "area_ha", "first_seen", "confirmations", "status"):
            assert key in props

    def test_is_json_serialisable(self) -> None:
        # It gets written to a file for QGIS or a handheld GPS, so anything
        # unserialisable would fail at the last step.
        json.dumps(alerts_to_geojson([_alert()], YELLAPUR_TALUK))

    def test_empty_input_gives_an_empty_collection(self) -> None:
        data = alerts_to_geojson([], YELLAPUR_TALUK)
        assert data["features"] == []
        assert data["type"] == "FeatureCollection"


class TestWriteReports:
    def test_writes_both_files(self, tmp_path: Path) -> None:
        files = write_reports([_alert()], [_alert()], YELLAPUR_TALUK, tmp_path, ISSUED)
        assert files.digest.exists()
        assert files.geojson.exists()

    def test_creates_the_output_directory(self, tmp_path: Path) -> None:
        # The scheduled job starts from a fresh checkout with no output dir.
        target = tmp_path / "nested" / "output"
        write_reports([], [], YELLAPUR_TALUK, target, ISSUED)
        assert target.is_dir()

    def test_digest_is_dated_but_geojson_is_not(self, tmp_path: Path) -> None:
        # The digest is a message from one cycle, so it is archived per date.
        # The GeoJSON is the current picture, so anything pointing at it — a
        # map, a bookmark — should not need updating every cycle.
        files = write_reports([_alert()], [_alert()], YELLAPUR_TALUK, tmp_path, ISSUED)
        assert "2026-08-04" in files.digest.name
        assert "2026-08-04" not in files.geojson.name

    def test_digest_covers_only_new_alerts(self, tmp_path: Path) -> None:
        # Someone must not be re-told what they were told last cycle.
        new = _alert(alert_id="brandnew", area_ha=3.0)
        old = _alert(alert_id="oldnews", area_ha=9.0)
        files = write_reports([new], [new, old], YELLAPUR_TALUK, tmp_path, ISSUED)
        text = files.digest.read_text(encoding="utf-8")
        assert "brandnew" in text
        assert "oldnews" not in text

    def test_geojson_covers_everything_tracked(self, tmp_path: Path) -> None:
        # The working file needs the full picture, with uncertainty attached.
        new = _alert(alert_id="brandnew", area_ha=3.0)
        old = _alert(alert_id="oldnews", area_ha=9.0)
        files = write_reports([new], [new, old], YELLAPUR_TALUK, tmp_path, ISSUED)
        data = json.loads(files.geojson.read_text(encoding="utf-8"))
        assert {f["properties"]["alert_id"] for f in data["features"]} == {
            "brandnew",
            "oldnews",
        }

    def test_a_quiet_cycle_still_writes_both_files(self, tmp_path: Path) -> None:
        # Absent files would look like a crashed run rather than a quiet month.
        files = write_reports([], [_alert()], YELLAPUR_TALUK, tmp_path, ISSUED)
        assert "No newly confirmed" in files.digest.read_text(encoding="utf-8")
        assert json.loads(files.geojson.read_text(encoding="utf-8"))["features"]

    def test_geojson_is_valid_json_on_disk(self, tmp_path: Path) -> None:
        files = write_reports([_alert()], [_alert()], YELLAPUR_TALUK, tmp_path, ISSUED)
        json.loads(files.geojson.read_text(encoding="utf-8"))
