"""Tests for the public map page.

No network, no credentials. The page is a pure function of the alert GeoJSON,
which is what makes the parts that matter — escaping, and the honesty of the
published claims — testable offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vanachakshu.webmap import build_page, write_page

CENTRE = (74.71, 14.96)


def alerts(*props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [74.7 + i / 100, 14.9 + i / 100]},
                "properties": p,
            }
            for i, p in enumerate(props)
        ],
    }


class TestBuildPage:
    def test_renders_every_alert(self) -> None:
        page = build_page(alerts({"area_ha": 1.0}, {"area_ha": 2.0}), "Yellapur", CENTRE)
        assert "95f11e59338ab270" not in page
        assert page.count('"type":"Feature"') == 2

    def test_reports_the_count_and_total_area(self) -> None:
        page = build_page(alerts({"area_ha": 1.25}, {"area_ha": 2.5}), "Yellapur", CENTRE)
        assert ">2<" in page
        assert "3.8" in page  # 1.25 + 2.5, one decimal

    def test_handles_an_empty_alert_store(self) -> None:
        """A month with no detections is the expected case, not an error."""
        page = build_page(alerts(), "Yellapur", CENTRE)
        assert ">0<" in page
        assert "fitBounds" in page  # guarded at runtime, not omitted

    def test_missing_area_does_not_break_the_page(self) -> None:
        page = build_page(alerts({"first_seen": "2026-08-07"}), "Yellapur", CENTRE)
        assert "0.0" in page


class TestEscaping:
    """Data crosses into a <script> element, so it must not be able to escape it."""

    def test_closing_script_tag_in_data_cannot_break_out(self) -> None:
        """json.dumps alone does not escape '<', so this once broke out for real."""
        hostile = "</script><script>alert(1)</script>"
        page = build_page(alerts({"note": hostile}), "X", CENTRE)

        assert "</script><script>" not in page
        # Only the page's own two script elements may close.
        assert page.count("</script>") == 2

    def test_escaping_survives_a_round_trip(self) -> None:
        """Escaping must not corrupt the value — JSON.parse has to see it intact."""
        hostile = "</script><script>alert(1)</script>"
        page = build_page(alerts({"note": hostile}), "X", CENTRE)
        payload = page.split("const ALERTS = ")[1].split(";\n")[0]

        # < is valid JSON, so json.loads decodes it exactly as a browser would.
        assert json.loads(payload)["features"][0]["properties"]["note"] == hostile

    def test_quotes_in_data_do_not_terminate_the_literal(self) -> None:
        page = build_page(alerts({"note": 'he said "hi"'}), "X", CENTRE)
        assert page.count("const ALERTS = ") == 1
        payload = page.split("const ALERTS = ")[1].split(";\n")[0]
        assert json.loads(payload)["features"][0]["properties"]["note"] == 'he said "hi"'

    def test_line_separators_do_not_break_the_literal(self) -> None:
        """U+2028/9 are legal in JSON but terminate a JavaScript string."""
        page = build_page(alerts({"note": "a\u2028b\u2029c"}), "X", CENTRE)
        payload = page.split("const ALERTS = ")[1].split(";\n")[0]

        assert "\u2028" not in payload and "\u2029" not in payload
        assert json.loads(payload)["features"][0]["properties"]["note"] == "a\u2028b\u2029c"


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

    @pytest.mark.parametrize("forbidden", ["illegal logging", "illegal felling", "encroacher"])
    def test_never_accuses_anyone_of_a_crime(self, forbidden: str) -> None:
        """A model output is not evidence of an offence, and clearing may be lawful."""
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        assert forbidden not in page.lower()

    def test_requires_ground_verification_in_the_popup_not_only_the_sidebar(self) -> None:
        """The sidebar is easy to collapse; the popup is what gets screenshotted."""
        page = build_page(alerts({"area_ha": 1.0}), "Yellapur", CENTRE)
        popup = page.split("function popup(")[1].split("const layer")[0]
        assert "Requires ground verification" in popup
        assert "may be lawful" in popup


class TestWritePage:
    def test_writes_and_creates_parent_directories(self, tmp_path: Path) -> None:
        src = tmp_path / "alerts.geojson"
        src.write_text(json.dumps(alerts({"area_ha": 1.0})), encoding="utf-8")

        out = write_page(src, tmp_path / "site" / "index.html", "Yellapur", CENTRE)

        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!doctype html>")

    def test_page_is_self_contained_apart_from_leaflet_and_tiles(self, tmp_path: Path) -> None:
        """No sibling data file to fall out of sync with the page."""
        src = tmp_path / "alerts.geojson"
        src.write_text(json.dumps(alerts({"area_ha": 1.0})), encoding="utf-8")
        page = write_page(src, tmp_path / "index.html", "Yellapur", CENTRE).read_text("utf-8")

        assert "fetch(" not in page
        assert "alerts.geojson" not in page
