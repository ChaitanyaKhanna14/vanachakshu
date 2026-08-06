"""Tests for the command-line interface.

The point of these is **exit codes**, not pretty output. The scheduled job will
use ``vanachakshu doctor`` as a precondition, so "did it exit non-zero when
something was broken" is load-bearing behaviour, not cosmetics.

``run_diagnostics`` is replaced with a stub, so these run in CI with no
credentials and no network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vanachakshu import cli
from vanachakshu.alerts import AlertStore, Detection
from vanachakshu.config import AlertConfig
from vanachakshu.diagnostics import CheckResult

runner = CliRunner()


@pytest.fixture
def passing_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: [
            CheckResult("Area of interest", True, "1,475 km2"),
            CheckResult("Earth Engine initialises", True, "initialised"),
        ],
    )


@pytest.fixture
def failing_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "run_diagnostics",
        lambda: [
            CheckResult("Area of interest", True, "1,475 km2"),
            CheckResult(
                "Earth Engine initialises",
                False,
                "permission denied on project",
                remediation="Register the project at code.earthengine.google.com/register",
            ),
        ],
    )


class TestVersion:
    def test_prints_version_and_exits_cleanly(self) -> None:
        result = runner.invoke(cli.app, ["version"])
        assert result.exit_code == 0
        assert "vanachakshu" in result.stdout


class TestDoctor:
    def test_exits_zero_when_all_checks_pass(self, passing_checks: None) -> None:
        result = runner.invoke(cli.app, ["doctor"])
        assert result.exit_code == 0

    def test_exits_non_zero_when_any_check_fails(self, failing_checks: None) -> None:
        # Cron and CI branch on this. If it ever returns 0 on failure, a broken
        # pipeline reports itself as healthy — the worst possible outcome for a
        # system people are meant to trust.
        result = runner.invoke(cli.app, ["doctor"])
        assert result.exit_code == 1

    def test_reports_each_check_by_name(self, passing_checks: None) -> None:
        result = runner.invoke(cli.app, ["doctor"])
        assert "Area of interest" in result.stdout
        assert "Earth Engine initialises" in result.stdout

    def test_shows_remediation_when_a_check_fails(self, failing_checks: None) -> None:
        result = runner.invoke(cli.app, ["doctor"])
        assert "How to fix" in result.stdout

    def test_confirms_success_explicitly(self, passing_checks: None) -> None:
        result = runner.invoke(cli.app, ["doctor"])
        assert "All checks passed" in result.stdout


class TestAppSurface:
    def test_bare_invocation_shows_help_rather_than_erroring(self) -> None:
        result = runner.invoke(cli.app, [])
        assert "doctor" in result.stdout


class TestValidateSample:
    """Phase 4 worksheet generation. No Earth Engine involved."""

    @pytest.fixture
    def populated_store(self, tmp_path: Path) -> Path:
        store = AlertStore(tmp_path / "yellapur-taluk.json", AlertConfig(min_confirmations=1))
        store.ingest(
            [
                Detection(
                    lon=74.70 + i * 0.01,
                    lat=14.95 + i * 0.01,
                    area_ha=area,
                    observed_on=date(2026, 1, 1),
                )
                for i, area in enumerate([0.3, 0.4, 0.7, 2.0, 3.5, 9.0])
            ],
            date(2026, 1, 1),
        )
        store.save(date(2026, 1, 1))
        return tmp_path

    def test_empty_store_exits_non_zero(self, tmp_path: Path) -> None:
        # Sampling nothing and reporting success would look like a completed
        # validation that found no problems.
        result = runner.invoke(
            cli.app, ["validate-sample", "--store-dir", str(tmp_path), "--out-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "empty" in result.stdout.lower()

    def test_writes_a_worksheet(self, populated_store: Path, tmp_path: Path) -> None:
        out = tmp_path / "sheets"
        result = runner.invoke(
            cli.app,
            ["validate-sample", "--store-dir", str(populated_store), "--out-dir", str(out)],
        )
        assert result.exit_code == 0
        assert list(out.glob("*.csv"))

    def test_filename_records_the_seed(self, populated_store: Path, tmp_path: Path) -> None:
        # The sample must be redrawable by someone else; a seed that is not
        # recorded alongside the output cannot be audited.
        out = tmp_path / "sheets"
        runner.invoke(
            cli.app,
            [
                "validate-sample",
                "--store-dir",
                str(populated_store),
                "--out-dir",
                str(out),
                "--seed",
                "7",
            ],
        )
        assert any("seed7" in p.name for p in out.glob("*.csv"))

    def test_reports_the_stratum_breakdown(self, populated_store: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            cli.app,
            ["validate-sample", "--store-dir", str(populated_store), "--out-dir", str(tmp_path)],
        )
        assert "under_0.5ha" in result.stdout
        assert "over_5ha" in result.stdout


class TestValidateReport:
    def test_missing_worksheet_exits_non_zero(self, tmp_path: Path) -> None:
        result = runner.invoke(cli.app, ["validate-report", str(tmp_path / "nope.csv")])
        assert result.exit_code == 1

    def test_unfilled_worksheet_exits_non_zero(self, tmp_path: Path) -> None:
        # An unreviewed worksheet must not report precision 0.00 as though the
        # detector had been checked and failed.
        path = tmp_path / "sheet.csv"
        path.write_text(
            "alert_id,stratum,area_ha,verdict,note\na,1_to_5ha,2.0,,\n", encoding="utf-8"
        )
        result = runner.invoke(cli.app, ["validate-report", str(path)])
        assert result.exit_code == 1
        assert "no verdicts" in result.stdout.lower()

    def test_reports_precision_with_an_interval(self, tmp_path: Path) -> None:
        path = tmp_path / "sheet.csv"
        rows = "\n".join(
            f"a{i},1_to_5ha,2.0,{'true_positive' if i < 8 else 'false_positive'},"
            for i in range(10)
        )
        path.write_text(f"alert_id,stratum,area_ha,verdict,note\n{rows}\n", encoding="utf-8")

        result = runner.invoke(cli.app, ["validate-report", str(path)])
        assert result.exit_code == 0
        assert "0.80" in result.stdout
        assert "n=8" in result.stdout or "n=10" in result.stdout
        # An interval must always accompany the point estimate.
        assert "[" in result.stdout
