"""Tests for the pure parts of detection.

Almost all of the detection logic is Earth Engine expressions and is validated
by the credentialed suites. The baseline window is the exception: it is plain
date arithmetic, and getting it wrong would corrupt every radar detection in a
way that is invisible in the output.
"""

from __future__ import annotations

from datetime import date

import pytest

from vanachakshu.config import RadarDetectionConfig
from vanachakshu.detect import DISTURBANCE_BAND, RADAR_DISTURBANCE_BAND, baseline_window


class TestBaselineWindow:
    def test_ends_exactly_where_monitoring_begins(self) -> None:
        # The two windows must not overlap. If the baseline included the period
        # being monitored, a real clearing would drag its own "normal" downward
        # and partly hide itself — the detector grading its own homework.
        _, end = baseline_window(date(2025, 1, 1), 365)
        assert end == "2025-01-01"

    def test_spans_the_requested_number_of_days(self) -> None:
        start, end = baseline_window(date(2025, 1, 1), 365)
        assert (date.fromisoformat(end) - date.fromisoformat(start)).days == 365

    def test_365_days_is_not_the_same_as_one_year(self) -> None:
        # Counting back 365 days from 1 Jan 2025 lands on 2 Jan 2024, not
        # 1 Jan — the span contains 29 Feb 2024, so a calendar year is 366 days
        # here. The window is defined in days on purpose: satellite revisit
        # cycles are counted in days, and "a year ago" is ambiguous.
        start, _ = baseline_window(date(2025, 1, 1), 365)
        assert start == "2024-01-02"

    def test_a_span_that_misses_the_leap_day_is_a_plain_year(self) -> None:
        # From 1 Mar 2024 to 1 Mar 2025 the leap day has already passed, so 365
        # days lands exactly a calendar year back. Contrast with the case above:
        # whether the offset is a whole year depends on which months it spans,
        # which is precisely why this is not computed by subtracting 1 from the
        # year number.
        start, _ = baseline_window(date(2025, 3, 1), 365)
        assert start == "2024-03-01"

    def test_returns_iso_dates(self) -> None:
        # Earth Engine's date filter wants ISO strings.
        start, end = baseline_window(date(2025, 6, 15), 180)
        assert date.fromisoformat(start) and date.fromisoformat(end)

    @pytest.mark.parametrize("days", [0, -1, -365])
    def test_rejects_a_non_positive_baseline(self, days: int) -> None:
        # A zero-length baseline would give every pixel an undefined "normal",
        # and a negative one would put the baseline *after* monitoring.
        with pytest.raises(ValueError, match="must be positive"):
            baseline_window(date(2025, 1, 1), days)

    def test_default_config_baseline_is_a_year(self) -> None:
        start, end = baseline_window(date(2025, 1, 1), RadarDetectionConfig().baseline_days)
        assert (date.fromisoformat(end) - date.fromisoformat(start)).days == 365


class TestBandNames:
    def test_optical_and_radar_bands_are_distinct(self) -> None:
        # They are combined in the same scoring runs; a collision would make one
        # detector silently overwrite the other.
        assert DISTURBANCE_BAND != RADAR_DISTURBANCE_BAND

    def test_band_names_are_earth_engine_safe(self) -> None:
        for name in (DISTURBANCE_BAND, RADAR_DISTURBANCE_BAND):
            assert name.replace("_", "").isalnum()
            assert not name[0].isdigit()
