"""Tests for manual-validation sampling and statistics.

Pure — no Earth Engine, no network.

These matter more than usual because this module produces the number the project
will eventually publish. A sampling scheme that is subtly biased, or an interval
that is too narrow, would not fail loudly — it would produce a confident claim
that happens to be wrong.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vanachakshu.alerts import TrackedAlert
from vanachakshu.validation import (
    SIZE_STRATA,
    ValidationRecord,
    Verdict,
    load_verdicts,
    precision_report,
    size_stratum,
    stratified_sample,
    wilson_interval,
    write_worksheet,
)


def _alert(alert_id: str, area_ha: float) -> TrackedAlert:
    return TrackedAlert(
        alert_id=alert_id,
        lon=74.70,
        lat=14.95,
        area_ha=area_ha,
        first_seen="2026-07-05",
        last_seen="2026-08-04",
        confirmations=2,
        notified_on="2026-08-04",
    )


class TestSizeStratum:
    @pytest.mark.parametrize(
        ("area_ha", "expected"),
        [
            (0.2, "under_0.5ha"),
            (0.49, "under_0.5ha"),
            (0.5, "0.5_to_1ha"),  # boundary belongs to the upper band
            (0.99, "0.5_to_1ha"),
            (1.0, "1_to_5ha"),
            (4.99, "1_to_5ha"),
            (5.0, "over_5ha"),
            (500.0, "over_5ha"),
        ],
    )
    def test_boundaries(self, area_ha: float, expected: str) -> None:
        assert size_stratum(area_ha) == expected

    def test_every_positive_area_lands_somewhere(self) -> None:
        # An unmatched area would raise mid-validation, after the reviewer had
        # already done the work.
        for area in (0.001, 0.5, 1.0, 5.0, 1e6):
            assert size_stratum(area)

    @pytest.mark.parametrize("area", [0.0, -1.0])
    def test_rejects_non_positive_area(self, area: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            size_stratum(area)


class TestStratifiedSample:
    @pytest.fixture
    def skewed_alerts(self) -> list[TrackedAlert]:
        # Deliberately skewed the way real detections are: mostly tiny.
        small = [_alert(f"s{i:03d}", 0.3) for i in range(200)]
        medium = [_alert(f"m{i:03d}", 2.0) for i in range(20)]
        large = [_alert(f"l{i:03d}", 9.0) for i in range(3)]
        return [*small, *medium, *large]

    def test_covers_every_populated_stratum(self, skewed_alerts: list[TrackedAlert]) -> None:
        # A uniform sample of this population would be ~90% sub-hectare and say
        # nothing at all about large clearings.
        sample = stratified_sample(skewed_alerts, per_stratum=5, seed=1)
        strata = {size_stratum(a.area_ha) for a in sample}
        assert strata == {"under_0.5ha", "1_to_5ha", "over_5ha"}

    def test_takes_everything_when_a_stratum_is_small(
        self, skewed_alerts: list[TrackedAlert]
    ) -> None:
        # Only 3 large alerts exist; asking for 5 must not raise.
        sample = stratified_sample(skewed_alerts, per_stratum=5, seed=1)
        assert sum(size_stratum(a.area_ha) == "over_5ha" for a in sample) == 3

    def test_is_reproducible_from_the_seed(self, skewed_alerts: list[TrackedAlert]) -> None:
        # "I checked 300 random alerts" is only meaningful if someone else can
        # draw the same 300.
        a = stratified_sample(skewed_alerts, per_stratum=5, seed=42)
        b = stratified_sample(skewed_alerts, per_stratum=5, seed=42)
        assert [x.alert_id for x in a] == [x.alert_id for x in b]

    def test_different_seeds_give_different_samples(
        self, skewed_alerts: list[TrackedAlert]
    ) -> None:
        a = stratified_sample(skewed_alerts, per_stratum=5, seed=1)
        b = stratified_sample(skewed_alerts, per_stratum=5, seed=2)
        assert [x.alert_id for x in a] != [x.alert_id for x in b]

    def test_does_not_depend_on_input_order(self, skewed_alerts: list[TrackedAlert]) -> None:
        # Otherwise the sample would silently change whenever the store's
        # ordering changed, breaking reproducibility without any seed change.
        forward = stratified_sample(skewed_alerts, per_stratum=5, seed=7)
        backward = stratified_sample(list(reversed(skewed_alerts)), per_stratum=5, seed=7)
        assert {a.alert_id for a in forward} == {a.alert_id for a in backward}

    def test_empty_input_gives_empty_sample(self) -> None:
        assert stratified_sample([], per_stratum=5, seed=1) == []

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_non_positive_size(self, n: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            stratified_sample([], per_stratum=n, seed=1)


class TestWilsonInterval:
    def test_contains_the_observed_proportion(self) -> None:
        low, high = wilson_interval(80, 100)
        assert low < 0.8 < high

    def test_narrows_as_the_sample_grows(self) -> None:
        narrow = wilson_interval(800, 1000)
        wide = wilson_interval(8, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_successes_does_not_give_a_zero_width_interval(self) -> None:
        # The reason Wilson is used instead of the textbook formula. The normal
        # approximation gives exactly zero width here — an infinitely confident
        # claim from thirty samples.
        low, high = wilson_interval(0, 30)
        assert low == 0.0
        assert high > 0.05, "0/30 must not imply certainty that precision is 0"

    def test_all_successes_does_not_give_a_zero_width_interval(self) -> None:
        low, high = wilson_interval(30, 30)
        assert high == 1.0
        assert low < 0.95, "30/30 must not imply certainty that precision is 1"

    def test_never_escapes_zero_to_one(self) -> None:
        # The naive interval routinely produces bounds below 0 or above 1 near
        # the extremes, which is nonsense for a proportion.
        for successes, total in ((0, 5), (5, 5), (1, 3), (99, 100)):
            low, high = wilson_interval(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_empty_sample_admits_every_proportion(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_rejects_more_successes_than_trials(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            wilson_interval(11, 10)

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            wilson_interval(-1, 10)


class TestPrecisionReport:
    def test_unclear_verdicts_leave_the_denominator(self) -> None:
        # Counting "unclear" as either outcome would invent a judgement the
        # reviewer explicitly declined to make.
        records = [
            ValidationRecord("a", 2.0, "1_to_5ha", Verdict.TRUE_POSITIVE),
            ValidationRecord("b", 2.0, "1_to_5ha", Verdict.FALSE_POSITIVE),
            ValidationRecord("c", 2.0, "1_to_5ha", Verdict.UNCLEAR),
        ]
        result = next(r for r in precision_report(records) if r.stratum == "1_to_5ha")
        assert result.judged == 2
        assert result.precision == pytest.approx(0.5)
        assert result.unclear == 1

    def test_reports_every_stratum_even_when_unsampled(self) -> None:
        # An absent band would read as "nothing to report" when it actually
        # means "never looked at".
        report = precision_report([ValidationRecord("a", 2.0, "1_to_5ha", Verdict.TRUE_POSITIVE)])
        assert [r.stratum for r in report] == [name for name, _, _ in SIZE_STRATA]

    def test_an_unjudged_stratum_says_so_rather_than_reporting_zero(self) -> None:
        report = precision_report([])
        for result in report:
            assert result.judged == 0
            assert "no judged samples" in result.as_row()

    def test_row_includes_the_interval_and_sample_size(self) -> None:
        records = [
            ValidationRecord(str(i), 2.0, "1_to_5ha", Verdict.TRUE_POSITIVE) for i in range(9)
        ]
        records.append(ValidationRecord("x", 2.0, "1_to_5ha", Verdict.FALSE_POSITIVE))
        row = next(r for r in precision_report(records) if r.stratum == "1_to_5ha").as_row()
        assert "0.90" in row
        assert "n=10" in row
        assert "[" in row and "]" in row


class TestWorksheet:
    def test_verdict_column_is_left_blank(self, tmp_path: Path) -> None:
        # Pre-filling it, even with a suggestion, biases the reviewer toward
        # agreeing with the detector and turns an independent check into a
        # rubber stamp.
        path = write_worksheet([_alert("a1", 2.0)], tmp_path / "w.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows[0]["verdict"] == ""

    def test_includes_links_for_the_reviewer(self, tmp_path: Path) -> None:
        path = write_worksheet([_alert("a1", 2.0)], tmp_path / "w.csv")
        row = next(iter(csv.DictReader(path.open(encoding="utf-8"))))
        assert "google.com/maps" in row["satellite_view"]
        assert "openstreetmap.org" in row["map_view"]

    def test_records_the_stratum(self, tmp_path: Path) -> None:
        path = write_worksheet([_alert("a1", 2.0)], tmp_path / "w.csv")
        row = next(iter(csv.DictReader(path.open(encoding="utf-8"))))
        assert row["stratum"] == "1_to_5ha"

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        path = write_worksheet([_alert("a1", 2.0)], tmp_path / "nested" / "w.csv")
        assert path.exists()


class TestLoadVerdicts:
    def test_round_trips_a_completed_worksheet(self, tmp_path: Path) -> None:
        path = write_worksheet([_alert("a1", 2.0), _alert("a2", 0.3)], tmp_path / "w.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        rows[0]["verdict"] = "true_positive"
        rows[1]["verdict"] = "false_positive"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        records = load_verdicts(path)
        assert {r.alert_id: r.verdict for r in records} == {
            "a1": Verdict.TRUE_POSITIVE,
            "a2": Verdict.FALSE_POSITIVE,
        }

    def test_blank_verdicts_are_skipped_not_counted(self, tmp_path: Path) -> None:
        # A half-finished worksheet must not silently score the unreviewed rows
        # as false positives, which would understate precision.
        path = write_worksheet([_alert("a1", 2.0), _alert("a2", 2.0)], tmp_path / "w.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        rows[0]["verdict"] = "true_positive"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        assert len(load_verdicts(path)) == 1

    def test_rejects_an_unrecognised_verdict(self, tmp_path: Path) -> None:
        # A typo like "yes" must stop the run rather than be silently dropped,
        # which would quietly shrink the sample the published figure rests on.
        path = write_worksheet([_alert("a1", 2.0)], tmp_path / "w.csv")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        rows[0]["verdict"] = "yes"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="expected one of"):
            load_verdicts(path)
