"""Integration tests that require real Earth Engine credentials.

Excluded from CI. Run locally, after ``earthengine authenticate``, with::

    pytest -m ee

Their job is to notice when Google's behaviour drifts away from what the
offline classifier in ``test_gee.py`` assumes — a pure unit test cannot see an
upstream error message being reworded.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from vanachakshu.gee import healthcheck, initialize

pytestmark = pytest.mark.ee


def test_initialize_and_round_trip() -> None:
    """The full happy path: credentials load and the server evaluates something."""
    initialize()
    healthcheck()  # raises if the round-trip fails


def test_nonexistent_project_is_diagnosed_usefully() -> None:
    """A typo'd project id must produce an actionable error, not ``UNKNOWN``.

    Runs in a **subprocess**, which is not incidental. Earth Engine's
    initialisation is process-global: once this process has initialised
    successfully, a later call naming a different project is silently ignored
    and raises nothing. Testing a bad project therefore requires a clean
    interpreter, or the test passes for entirely the wrong reason.
    """
    script = textwrap.dedent("""
        from vanachakshu.config import Settings
        from vanachakshu.gee import EarthEngineSetupError, initialize

        bogus = Settings(
            ee_project="vanachakshu-definitely-not-real-9f3a2",
            ee_service_account_key=None,
        )
        try:
            initialize(bogus)
        except EarthEngineSetupError as exc:
            print(f"KIND={exc.kind}")
        else:
            print("KIND=did-not-raise")
    """)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    stdout = completed.stdout.strip()

    assert "KIND=" in stdout, f"subprocess produced no verdict:\n{completed.stderr}"
    kind = stdout.rsplit("KIND=", 1)[1].strip()

    # The precise diagnosis varies with how Google reports it, and that is fine.
    # UNKNOWN is not fine: it is the one outcome that leaves a user with an
    # opaque error and nothing to act on.
    assert kind in {"project_not_registered", "permission_denied"}, (
        f"unhelpful diagnosis {kind!r} for a nonexistent project. "
        f"Google may have reworded the message; update _PROBES in gee.py.\n"
        f"stdout: {stdout}\nstderr: {completed.stderr}"
    )
