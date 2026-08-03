"""Tests for the command-line interface.

The point of these is **exit codes**, not pretty output. The scheduled job will
use ``vanachakshu doctor`` as a precondition, so "did it exit non-zero when
something was broken" is load-bearing behaviour, not cosmetics.

``run_diagnostics`` is replaced with a stub, so these run in CI with no
credentials and no network.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from vanachakshu import cli
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
