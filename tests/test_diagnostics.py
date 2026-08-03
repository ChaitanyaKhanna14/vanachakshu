"""Tests for the diagnostics framework.

The framework — result type, pass/fail aggregation, report rendering — is pure
and fully tested here. The individual probes make network calls and are covered
by the credentialed suite instead.
"""

from __future__ import annotations

from datetime import date

import pytest

from vanachakshu.config import SeasonWindow
from vanachakshu.diagnostics import (
    CheckResult,
    all_passed,
    format_report,
    run_diagnostics,
)


class TestCheckResult:
    def test_is_immutable(self) -> None:
        result = CheckResult(name="x", ok=True, detail="fine")
        with pytest.raises(AttributeError):
            result.ok = False  # type: ignore[misc]

    def test_remediation_defaults_to_none(self) -> None:
        assert CheckResult(name="x", ok=True, detail="fine").remediation is None


class TestAllPassed:
    def test_true_when_every_check_passes(self) -> None:
        results = [
            CheckResult("a", True, "ok"),
            CheckResult("b", True, "ok"),
        ]
        assert all_passed(results) is True

    def test_false_when_any_check_fails(self) -> None:
        results = [
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "broken"),
        ]
        assert all_passed(results) is False

    def test_empty_run_passes_vacuously(self) -> None:
        # Matters because run_diagnostics uses all_passed to decide whether to
        # continue; an empty list must not read as failure.
        assert all_passed([]) is True


class TestFormatReport:
    def test_marks_pass_and_fail(self) -> None:
        report = format_report(
            [CheckResult("Alpha", True, "fine"), CheckResult("Beta", False, "broke")]
        )
        assert "[PASS] Alpha: fine" in report
        assert "[FAIL] Beta: broke" in report

    def test_appends_remediation_for_failures(self) -> None:
        report = format_report(
            [CheckResult("Beta", False, "broke", remediation="turn it off and on")]
        )
        assert "How to fix 'Beta':" in report
        assert "turn it off and on" in report

    def test_omits_remediation_for_passes(self) -> None:
        report = format_report([CheckResult("Alpha", True, "fine", remediation="unused")])
        assert "How to fix" not in report

    def test_empty_results_render_as_empty_string(self) -> None:
        assert format_report([]) == ""

    def test_contains_no_markup(self) -> None:
        # Rendering must stay plain so the same text works in a terminal, a CI
        # log, and an email from the scheduled job.
        report = format_report([CheckResult("Alpha", True, "fine")])
        assert "[green]" not in report
        assert "[/" not in report


class TestMissingConfigIsDiagnosedNotLeaked:
    """Regression: a missing project id must produce remediation, not a traceback.

    The first real run of ``doctor`` printed a raw pydantic ValidationError
    here, because the config probe called ``Settings()`` directly instead of
    routing through the error classifier. Showing users a stack trace is exactly
    what this command exists to prevent.

    Runs offline: configuration fails before anything reaches the network.
    """

    @pytest.fixture(autouse=True)
    def _isolated_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
        monkeypatch.delenv("VANACHAKSHU_EE_PROJECT", raising=False)
        # Empty cwd, so a developer's real .env cannot mask the failure.
        monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]

    def test_config_check_fails(self) -> None:
        results = run_diagnostics()
        config_check = next(r for r in results if r.name == "Configuration")
        assert config_check.ok is False

    def test_config_failure_carries_remediation(self) -> None:
        results = run_diagnostics()
        config_check = next(r for r in results if r.name == "Configuration")
        assert config_check.remediation is not None
        assert "VANACHAKSHU_EE_PROJECT" in config_check.remediation

    def test_stops_before_attempting_network_checks(self) -> None:
        # Nothing downstream should run, and nothing should reach Google.
        names = [r.name for r in run_diagnostics()]
        assert names == ["Area of interest", "Configuration"]

    def test_aoi_check_still_passes_first(self) -> None:
        # The pure check runs before configuration, so a broken setup still
        # confirms the AOI is sane.
        results = run_diagnostics()
        assert results[0].name == "Area of interest"
        assert results[0].ok is True


class TestMostRecentCompleteYear:
    """The 'is this season actually finished?' rule.

    Compositing a half-finished season yields a thinner, cloudier image than the
    years it is compared against, which then reads as vegetation loss. This is a
    quiet, plausible-looking failure, so the boundary is pinned precisely.
    """

    @pytest.fixture
    def jan_to_mar(self) -> SeasonWindow:
        return SeasonWindow(start_month=1, end_month=3)

    def test_after_season_ends_uses_current_year(self, jan_to_mar: SeasonWindow) -> None:
        assert jan_to_mar.most_recent_complete_year(date(2026, 8, 3)) == 2026

    def test_first_day_after_season_uses_current_year(self, jan_to_mar: SeasonWindow) -> None:
        # 1 April: March is over, so this year's window is complete.
        assert jan_to_mar.most_recent_complete_year(date(2026, 4, 1)) == 2026

    def test_during_season_falls_back_to_previous_year(self, jan_to_mar: SeasonWindow) -> None:
        # Mid-March: this year's window is still filling up.
        assert jan_to_mar.most_recent_complete_year(date(2026, 3, 15)) == 2025

    def test_last_day_of_season_still_falls_back(self, jan_to_mar: SeasonWindow) -> None:
        # The off-by-one that would silently composite a partial season.
        assert jan_to_mar.most_recent_complete_year(date(2026, 3, 31)) == 2025

    def test_before_season_starts_falls_back(self, jan_to_mar: SeasonWindow) -> None:
        assert jan_to_mar.most_recent_complete_year(date(2026, 1, 1)) == 2025

    def test_december_window_only_completes_at_year_end(self) -> None:
        window = SeasonWindow(start_month=10, end_month=12)
        assert window.most_recent_complete_year(date(2026, 12, 31)) == 2025
        assert window.most_recent_complete_year(date(2027, 1, 1)) == 2026
