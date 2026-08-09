"""Tests for the public map page.

No network, no credentials. The page is a pure function of the alert GeoJSON,
which is what makes the parts that matter — escaping, and the honesty of the
published claims — testable offline.

Counts and totals are rendered by the page's own JavaScript from an inlined
config object, so the assertions read that object rather than scraping numbers
out of the markup. Scraping would pass on a page whose script never runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from vanachakshu.webmap import build_page, meeting_current_config, write_page

CENTRE = (74.71, 14.96)
BOUNDS = {"west": 74.5, "south": 14.7, "east": 74.9, "north": 15.2}


def alerts(*props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [74.7 + i / 100, 14.9 + i / 100]},
                "properties": {"alert_id": f"id{i}", **p},
            }
            for i, p in enumerate(props)
        ],
    }


def config_of(page: str) -> dict[str, Any]:
    """Pull the inlined config back out of the page, as a browser would."""
    payload = page.split("const CFG = ")[1].split(";\n")[0]
    return json.loads(payload)  # type: ignore[no-any-return]


class TestBuildPage:
    def test_renders_every_alert(self) -> None:
        cfg = config_of(build_page(alerts({"area_ha": 1.0}, {"area_ha": 2.0}), "Yellapur", CENTRE))
        assert len(cfg["alerts"]["features"]) == 2

    def test_handles_an_empty_alert_store(self) -> None:
        """A month with no detections is the expected case, not an error."""
        page = build_page(alerts(), "Yellapur", CENTRE)
        assert config_of(page)["alerts"]["features"] == []
        assert "fitBounds" in page  # guarded at runtime, not omitted

    def test_missing_area_does_not_break_the_page(self) -> None:
        page = build_page(alerts({"first_seen": "2026-08-07"}), "Yellapur", CENTRE)
        assert len(config_of(page)["alerts"]["features"]) == 1

    def test_carries_the_aoi_name(self) -> None:
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Yellapur Taluk", CENTRE))
        assert cfg["aoiName"] == "Yellapur Taluk"


class TestComparedPeriod:
    """Which years of imagery the map says it is showing.

    An earlier version derived this from today's date and displayed
    "2025-2026" for detections actually made from 2024 and 2025 imagery.
    AlphaEarth is annual and published months in arrears, so the current year is
    never the compared year. On a page built to be honest about its limits,
    inventing the data's date is close to the worst small lie available.
    """

    def test_read_from_the_detections_not_the_calendar(self) -> None:
        cfg = config_of(
            build_page(
                alerts({"area_ha": 1.0, "baseline_year": 2024, "recent_year": 2025}), "Y", CENTRE
            )
        )
        assert cfg["period"] == "2024\u20132025"

    def test_spans_every_year_pair_present(self) -> None:
        """A store filled over several cycles holds more than one pair, and
        quoting only the newest overstates how current the older markers are."""
        cfg = config_of(
            build_page(
                alerts(
                    {"area_ha": 1.0, "baseline_year": 2024, "recent_year": 2025},
                    {"area_ha": 1.0, "baseline_year": 2022, "recent_year": 2023},
                ),
                "Y",
                CENTRE,
            )
        )
        assert cfg["period"] == "2022\u20132025"

    def test_is_null_rather_than_guessed_when_unrecorded(self) -> None:
        """True of every alert written before the field existed. The page shows
        its build date instead, labelled as such — a plausible wrong year is
        worse than an obviously different fact, because nobody checks it."""
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Y", CENTRE))
        assert cfg["period"] is None

    def test_the_page_says_so_rather_than_showing_a_bare_date(self) -> None:
        """An earlier build printed the month alone — the tile read 'Aug'."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert "Imagery years not recorded" in page
        assert "Imagery compared: " in page

    def test_partial_years_are_not_used(self) -> None:
        """Half a pair is not a period."""
        cfg = config_of(build_page(alerts({"area_ha": 1.0, "baseline_year": 2024}), "Y", CENTRE))
        assert cfg["period"] is None


class TestMonitoredArea:
    """Blank space on the map is ambiguous unless the boundary is drawn.

    Recall is 0.32 over one taluk, so an area with no marker may mean nothing
    was found, or that nothing was ever looked at. Only the boundary separates
    the second case, and it is the case that would mislead someone worst.
    """

    def test_bounds_are_published_when_given(self) -> None:
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Y", CENTRE, bounds=BOUNDS))
        assert cfg["bounds"] == BOUNDS

    def test_falls_back_to_a_box_around_the_centre(self) -> None:
        """Never omit the boundary — a missing one is indistinguishable from
        unlimited coverage, which is the wrong way to be wrong."""
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Y", CENTRE))
        b = cfg["bounds"]
        assert b["west"] < CENTRE[0] < b["east"]
        assert b["south"] < CENTRE[1] < b["north"]

    def test_the_page_draws_it(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE, bounds=BOUNDS)
        assert "L.rectangle" in page
        assert "monitored area" in page


class TestEscaping:
    """Data crosses into a <script> element, so it must not be able to escape it."""

    def test_closing_script_tag_in_data_cannot_break_out(self) -> None:
        """json.dumps alone does not escape '<', so this once broke out for real."""
        page = build_page(alerts({"note": "</script><script>alert(1)</script>"}), "X", CENTRE)

        assert "</script><script>" not in page
        # Only the page's own two script elements may close.
        assert page.count("</script>") == 2

    def test_escaping_survives_a_round_trip(self) -> None:
        """Escaping must not corrupt the value — JSON.parse has to see it intact."""
        hostile = "</script><script>alert(1)</script>"
        cfg = config_of(build_page(alerts({"note": hostile}), "X", CENTRE))
        assert cfg["alerts"]["features"][0]["properties"]["note"] == hostile

    def test_quotes_in_data_do_not_terminate_the_literal(self) -> None:
        cfg = config_of(build_page(alerts({"note": 'he said "hi"'}), "X", CENTRE))
        assert cfg["alerts"]["features"][0]["properties"]["note"] == 'he said "hi"'

    def test_line_separators_do_not_break_the_literal(self) -> None:
        """U+2028/9 are legal in JSON but terminate a JavaScript string."""
        page = build_page(alerts({"note": "a\u2028b\u2029c"}), "X", CENTRE)
        payload = page.split("const CFG = ")[1].split(";\n")[0]

        assert "\u2028" not in payload and "\u2029" not in payload
        assert (
            json.loads(payload)["alerts"]["features"][0]["properties"]["note"] == "a\u2028b\u2029c"
        )


class TestPublishedClaims:
    """The page makes claims to the public. These pin the honest ones in place.

    Precision alone would flatter the system: someone reading '80% accurate'
    naturally assumes the map is near-complete. Recall is the number that stops
    an empty area being read as a safe one, so it has to be published too.
    """

    def test_states_that_most_real_clearing_is_missing(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        assert "0.32" in page
        assert "has <em>not</em> been shown to be undisturbed" in page

    def test_states_precision_with_its_interval(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        assert "0.80" in page
        assert "0.53" in page and "0.97" in page

    def test_discloses_the_year_lag(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        assert "lag" in page.lower()

    def test_says_nothing_is_monitored_outside_the_boundary(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE, bounds=BOUNDS)
        assert "Outside the dashed boundary nothing is monitored" in page

    @pytest.mark.parametrize("forbidden", ["illegal logging", "illegal felling", "encroacher"])
    def test_never_accuses_anyone_of_a_crime(self, forbidden: str) -> None:
        """A model output is not evidence of an offence, and clearing may be lawful."""
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        assert forbidden not in page.lower()

    def test_requires_ground_verification_in_the_popup_not_only_the_sidebar(self) -> None:
        """The sidebar is easy to collapse; the popup is what gets screenshotted."""
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        popup = page.split("function popupHtml(")[1].split("const markers")[0]
        assert "Requires ground verification" in popup
        assert "may be lawful" in popup


class TestUsability:
    """The page has to work as a tool, not only as a picture."""

    def test_offers_place_names(self) -> None:
        """Esri imagery carries no labels, so a bare pin cannot say where it is."""
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Y", CENTRE))
        assert "Boundaries_and_Places" in cfg["labels"]

    def test_offers_a_download(self) -> None:
        """Principle 3 promised a downloadable file for someone's GPS."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert "geojson" in page and "Blob" in page

    def test_status_differs_by_more_than_colour(self) -> None:
        """Roughly 8% of men cannot separate the amber from the crimson reliably.

        Confirmed detections carry an extra halo, which survives greyscale and
        colour blindness in a way hue alone does not.
        """
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert "rings.set" in page
        assert "radius:r + 4" in page

    def test_no_detection_is_drawn_faintly(self) -> None:
        """Every alert is pending until a second pass confirms it.

        An earlier build drew pending as a hollow ring at 8% fill, so with every
        stored alert unconfirmed the entire map read as empty. Markers are now
        solid regardless of status; only the halo distinguishes them.
        """
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        marker = page.split("pointToLayer:")[1].split("onEachFeature")[0]

        opacities = [float(v) for v in re.findall(r"fillOpacity:\.?(\d*\.?\d+)", marker)]
        assert opacities, "no fillOpacity found — check the marker style"
        assert min(opacities) >= 0.5

    def test_place_labels_come_off_the_topographic_basemap(self) -> None:
        """Topographic already has names baked in; leaving the overlay on
        renders every label twice, offset."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert "baselayerchange" in page
        assert "removeLayer(labels)" in page

    def test_selection_survives_being_opened_from_a_file(self) -> None:
        """history.replaceState throws a SecurityError on file:// in some
        browsers, which would break the whole click handler when previewing."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert "try { history.replaceState" in page

    def test_list_rows_contain_only_phrasing_content(self) -> None:
        """A <button> may not contain a <div>; screen readers mishandle it."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        row = page.split("row.innerHTML =")[1].split("row.addEventListener")[0]
        assert "<div" not in row

    def test_has_a_favicon(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert 'rel="icon"' in page

    def test_filter_buttons_carry_their_counts(self) -> None:
        """So an empty result is explained before it is clicked, not after."""
        page = build_page(alerts({"area_ha": 1.0}), "Y", CENTRE)
        assert 'id="n-con"' in page and 'id="n-pen"' in page

    def test_every_alert_carries_an_id_for_linking(self) -> None:
        cfg = config_of(build_page(alerts({"area_ha": 1.0}), "Y", CENTRE))
        assert cfg["alerts"]["features"][0]["properties"]["alert_id"]


class TestOnlyPublishWhatTheDetectorStandsBehind:
    """The alert store outlives the settings that filled it.

    When min_clearing_ha moved 0.05 -> 0.20 — taking measured precision from
    0.31 to 0.80 — 42 of 57 stored alerts fell below the new floor. Publishing
    them beside a claim of 0.80 precision overstated accuracy on three quarters
    of the map. That shipped once; these stop it shipping again.
    """

    def test_drops_detections_below_the_current_floor(self) -> None:
        cfg = config_of(
            build_page(
                alerts({"area_ha": 0.06}, {"area_ha": 0.19}, {"area_ha": 0.25}),
                "Yellapur",
                CENTRE,
                min_area_ha=0.20,
            )
        )
        assert len(cfg["alerts"]["features"]) == 1

    def test_a_detection_exactly_on_the_floor_is_published(self) -> None:
        """0.20 ha is 20 pixels, which the detector emits — the boundary is inclusive."""
        cfg = config_of(build_page(alerts({"area_ha": 0.20}), "Yellapur", CENTRE, min_area_ha=0.20))
        assert len(cfg["alerts"]["features"]) == 1

    def test_filter_keeps_the_rest_of_the_collection_intact(self) -> None:
        source = {**alerts({"area_ha": 0.5}), "name": "yellapur", "crs": {"type": "name"}}
        out = meeting_current_config(source, 0.2)
        assert out["name"] == "yellapur" and out["crs"] == {"type": "name"}

    def test_write_page_reports_how_many_were_withheld(self, tmp_path: Path) -> None:
        """Silently dropping most of a map is not something to do without saying so."""
        src = tmp_path / "alerts.geojson"
        src.write_text(
            json.dumps(alerts({"area_ha": 0.06}, {"area_ha": 0.5}, {"area_ha": 0.1})),
            encoding="utf-8",
        )

        _, shown, withheld = write_page(
            src, tmp_path / "index.html", "Yellapur", CENTRE, min_area_ha=0.20
        )

        assert (shown, withheld) == (1, 2)


class TestWritePage:
    def test_writes_and_creates_parent_directories(self, tmp_path: Path) -> None:
        src = tmp_path / "alerts.geojson"
        src.write_text(json.dumps(alerts({"area_ha": 1.0})), encoding="utf-8")

        out, shown, withheld = write_page(src, tmp_path / "site" / "index.html", "Yellapur", CENTRE)

        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
        assert (shown, withheld) == (1, 0)

    def test_page_is_self_contained_apart_from_leaflet_and_tiles(self, tmp_path: Path) -> None:
        """No sibling data file to fall out of sync with the page."""
        src = tmp_path / "alerts.geojson"
        src.write_text(json.dumps(alerts({"area_ha": 1.0})), encoding="utf-8")
        page = write_page(src, tmp_path / "index.html", "Yellapur", CENTRE)[0].read_text("utf-8")

        assert "fetch(" not in page
        assert "alerts.geojson" not in page
