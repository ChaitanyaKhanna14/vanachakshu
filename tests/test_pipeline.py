"""Tests for a full detection cycle.

Earth Engine is replaced with a stub, so a complete cycle — detect, confirm,
record, report — runs in CI with no credentials.

That is only possible because ``fetch_patch_records`` is the single place Earth
Engine stops. Everything after it is plain dictionaries. Stubbing one function
therefore exercises the entire alerting half of the system, including the
guarantees that decide whether a person is told something.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from vanachakshu import pipeline
from vanachakshu.alerts import AlertStore
from vanachakshu.config import (
    WESTERN_GHATS_CLEAR_SEASON,
    YELLAPUR_TALUK,
    AlertConfig,
    AreaOfInterest,
    BoundingBox,
)
from vanachakshu.pipeline import RunResult, run_cycle, store_path_for

DAY1 = date(2026, 1, 1)
DAY2 = date(2026, 1, 13)
DAY3 = date(2026, 1, 25)


def _patch(lon: float, lat: float, area_ha: float) -> dict[str, Any]:
    """A GeoJSON feature shaped like what reduceToVectors returns."""
    d = 0.0005
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon - d, lat - d],
                    [lon + d, lat - d],
                    [lon + d, lat + d],
                    [lon - d, lat + d],
                    [lon - d, lat - d],
                ]
            ],
        },
        "properties": {"area_ha": area_ha, "label": 1},
    }


TWO_PATCHES = [_patch(74.70, 14.95, 1.4), _patch(74.75, 14.99, 0.8)]


@pytest.fixture
def stub_earth_engine(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    """Replace the Earth Engine boundary with a scripted sequence of results.

    Returns a mutable list; set its contents to control what the next cycle
    "detects".
    """
    scripted: list[list[dict[str, Any]]] = [list(TWO_PATCHES)]

    def _fake_fetch(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return scripted[0]

    monkeypatch.setattr(pipeline, "fetch_patch_records", _fake_fetch)
    return scripted


@pytest.fixture
def store(tmp_path: Path) -> AlertStore:
    return AlertStore(tmp_path / "alerts.json", AlertConfig(min_confirmations=2))


def _cycle(store: AlertStore, today: date, dry_run: bool = False) -> RunResult:
    return run_cycle(
        aoi=YELLAPUR_TALUK,
        season=WESTERN_GHATS_CLEAR_SEASON,
        baseline_year=2020,
        recent_year=2025,
        today=today,
        store=store,
        dry_run=dry_run,
    )


class TestStorePath:
    def test_uses_the_aoi_slug(self) -> None:
        assert store_path_for(YELLAPUR_TALUK).name == "yellapur-taluk.json"

    def test_defaults_under_data_alerts(self) -> None:
        # Inside the repo, so the scheduled job's commit carries the history.
        assert store_path_for(YELLAPUR_TALUK).parent == Path("data") / "alerts"

    def test_respects_an_explicit_root(self, tmp_path: Path) -> None:
        assert store_path_for(YELLAPUR_TALUK, tmp_path).parent == tmp_path

    def test_distinct_aois_get_distinct_stores(self, tmp_path: Path) -> None:
        # Two AOIs must never share a store, or one's alerts would suppress
        # the other's.
        other = AreaOfInterest(
            name="Sirsi Taluk",
            bbox=BoundingBox(west=74.60, south=14.55, east=74.70, north=14.65),
        )
        assert store_path_for(YELLAPUR_TALUK, tmp_path) != store_path_for(other, tmp_path)


class TestFullCycle:
    def test_first_cycle_confirms_nothing(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        result = _cycle(store, DAY1)
        assert result.patches_found == 2
        assert result.new_alerts == ()
        assert result.pending_count == 2

    def test_second_cycle_confirms_both(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        _cycle(store, DAY1)
        result = _cycle(store, DAY2)
        assert len(result.new_alerts) == 2
        assert result.pending_count == 0
        assert result.notified_total == 2

    def test_third_cycle_announces_nothing_new(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        # The notify-once guarantee, surviving a full cycle rather than just a
        # direct call to the store.
        _cycle(store, DAY1)
        _cycle(store, DAY2)
        result = _cycle(store, DAY3)
        assert result.new_alerts == ()
        assert result.notified_total == 2

    def test_a_new_clearing_appearing_later_is_confirmed_separately(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        _cycle(store, DAY1)
        _cycle(store, DAY2)  # both confirmed

        stub_earth_engine[0] = [*TWO_PATCHES, _patch(74.80, 15.02, 2.1)]
        assert _cycle(store, DAY3).new_alerts == ()  # third seen once

        assert len(_cycle(store, date(2026, 2, 6)).new_alerts) == 1

    def test_total_area_is_reported(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        assert _cycle(store, DAY1).patches_total_ha == pytest.approx(2.2)

    def test_finding_nothing_is_a_normal_outcome(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        # A quiet fortnight must not look like a failure.
        stub_earth_engine[0] = []
        result = _cycle(store, DAY1)
        assert result.patches_found == 0
        assert result.new_alerts == ()


class TestPersistenceAcrossProcesses:
    def test_state_survives_a_fresh_store_object(
        self, stub_earth_engine: list[list[dict[str, Any]]], tmp_path: Path
    ) -> None:
        # The scheduled job is a new process each run. If confirmation state did
        # not survive that, every run would re-announce everything.
        path = tmp_path / "alerts.json"
        config = AlertConfig(min_confirmations=2)

        _cycle(AlertStore(path, config), DAY1)
        assert len(_cycle(AlertStore(path, config), DAY2).new_alerts) == 2
        assert _cycle(AlertStore(path, config), DAY3).new_alerts == ()

    def test_store_file_is_written(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        _cycle(store, DAY1)
        assert store.path.exists()
        assert json.loads(store.path.read_text(encoding="utf-8"))["alerts"]


class TestDryRun:
    def test_does_not_write_the_store(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        _cycle(store, DAY1, dry_run=True)
        assert not store.path.exists()

    def test_does_not_burn_confirmation_state(
        self, stub_earth_engine: list[list[dict[str, Any]]], tmp_path: Path
    ) -> None:
        # The reason dry-run exists. Marking an alert announced is not
        # reversible, so experimenting must not consume that state.
        path = tmp_path / "alerts.json"
        config = AlertConfig(min_confirmations=2)

        _cycle(AlertStore(path, config), DAY1, dry_run=True)
        _cycle(AlertStore(path, config), DAY2, dry_run=True)

        # Real first cycle: still nothing confirmed, because the dry runs left
        # no trace.
        assert _cycle(AlertStore(path, config), DAY3).new_alerts == ()

    def test_is_reported_in_the_summary(
        self, stub_earth_engine: list[list[dict[str, Any]]], store: AlertStore
    ) -> None:
        text = "\n".join(_cycle(store, DAY1, dry_run=True).summary_lines())
        assert "DRY RUN" in text


class TestRunResult:
    def test_summary_has_no_markup(self) -> None:
        # The same text goes to a terminal, a CI log and an email body.
        result = RunResult(2020, 2025, 3, 4.5, (), 3, 0, dry_run=False)
        text = "\n".join(result.summary_lines())
        assert "[" not in text.replace("[dim]", "")

    def test_summary_names_both_years(self) -> None:
        text = "\n".join(RunResult(2020, 2025, 0, 0.0, (), 0, 0, dry_run=False).summary_lines())
        assert "2020" in text and "2025" in text


class TestInputValidation:
    def test_rejects_a_baseline_after_the_recent_year(self) -> None:
        # Silently comparing backwards would invert every drop into a rise and
        # report no disturbance anywhere.
        with pytest.raises(ValueError, match="must be earlier than"):
            pipeline.fetch_patch_records(YELLAPUR_TALUK, WESTERN_GHATS_CLEAR_SEASON, 2025, 2020)

    def test_rejects_identical_years(self) -> None:
        with pytest.raises(ValueError, match="must be earlier than"):
            pipeline.fetch_patch_records(YELLAPUR_TALUK, WESTERN_GHATS_CLEAR_SEASON, 2025, 2025)
