"""Tests for Earth Engine error classification.

No credentials, no network. The classifier is a pure function over error text,
which is precisely why it was written as one — the diagnosis logic is the part
that needs to be right, and it can be pinned down completely offline.

The error strings below are representative of what Earth Engine and the
underlying Google API client actually emit.
"""

from __future__ import annotations

import pytest

from vanachakshu.gee import (
    EarthEngineSetupError,
    InitFailureKind,
    classify_init_error,
    initialize,
    remediation_for,
)


class TestClassifyInitError:
    @pytest.mark.parametrize(
        "message",
        [
            "Please authorize access to your Earth Engine account by running\n\n"
            "earthengine authenticate\n\nin your command line, and then retry.",
            "Earth Engine client library not initialized. Run ee.Initialize()",
            "invalid_grant: Token has been expired or revoked.",
            "Credentials file not found.",
            "HttpError 401 when requesting ...",
        ],
    )
    def test_detects_authentication_failures(self, message: str) -> None:
        assert classify_init_error(message) == InitFailureKind.NOT_AUTHENTICATED

    @pytest.mark.parametrize(
        "message",
        [
            "Earth Engine API has not been used in project 482913 before or it is disabled.",
            "Project 'my-proj' is not registered to use Earth Engine.",
            "The caller is not signed up for Earth Engine.",
            "project not found: vanachakshu-typo",
            # Verbatim from Earth Engine for a typo'd project id. Captured from
            # a real run: the quoted id between "Project" and "not found" is
            # what made an earlier version of the probe list miss this.
            "Project 'projects/vanachakshu-typo-9f3a2' not found or deleted.",
        ],
    )
    def test_detects_unregistered_project(self, message: str) -> None:
        assert classify_init_error(message) == InitFailureKind.PROJECT_NOT_REGISTERED

    @pytest.mark.parametrize(
        "message",
        [
            "Caller does not have required permission to use project my-project.",
            "HttpError 403 when requesting ... returned 'Forbidden'",
            "Access denied to asset.",
        ],
    )
    def test_detects_permission_failures(self, message: str) -> None:
        assert classify_init_error(message) == InitFailureKind.PERMISSION_DENIED

    @pytest.mark.parametrize(
        "message",
        [
            "Quota exceeded for quota metric 'EECU seconds'.",
            "429 Too Many Requests",
            "Project has entered restricted mode; EECU allowance consumed.",
            "RESOURCE_EXHAUSTED",
        ],
    )
    def test_detects_quota_exhaustion(self, message: str) -> None:
        assert classify_init_error(message) == InitFailureKind.QUOTA_EXHAUSTED

    @pytest.mark.parametrize(
        "message",
        [
            "Connection aborted.",
            "The read operation timed out",
            "Temporary failure in name resolution",
        ],
    )
    def test_detects_network_failures(self, message: str) -> None:
        assert classify_init_error(message) == InitFailureKind.NETWORK

    def test_unrecognised_message_is_unknown_not_a_guess(self) -> None:
        # Falling back to UNKNOWN is deliberate: a confidently wrong diagnosis
        # costs more time than an honest "I don't know, here is the raw error".
        assert classify_init_error("something went sideways") == InitFailureKind.UNKNOWN

    def test_classification_is_case_insensitive(self) -> None:
        assert (
            classify_init_error("QUOTA EXCEEDED")
            == classify_init_error("quota exceeded")
            == InitFailureKind.QUOTA_EXHAUSTED
        )

    def test_empty_message_is_unknown(self) -> None:
        assert classify_init_error("") == InitFailureKind.UNKNOWN

    def test_quota_wins_over_permission(self) -> None:
        # Real quota errors frequently also mention permissions. Quota is the
        # more specific and more actionable diagnosis, so it must win.
        message = "Permission denied: quota exceeded for this project"
        assert classify_init_error(message) == InitFailureKind.QUOTA_EXHAUSTED

    def test_unregistered_project_wins_over_permission(self) -> None:
        # This is the single most common first-run failure, and Google reports
        # it using the word "permission". Misdiagnosing it sends the user to
        # fix IAM roles when they actually just need to register the project.
        message = (
            "Caller does not have required permission; "
            "Earth Engine API has not been used in project 12345 before."
        )
        assert classify_init_error(message) == InitFailureKind.PROJECT_NOT_REGISTERED


class TestRemediation:
    def test_every_failure_kind_has_remediation(self) -> None:
        # Guards the obvious regression: adding an enum member and forgetting
        # the message, which would raise KeyError at the worst possible moment.
        for kind in InitFailureKind:
            assert remediation_for(kind).strip()

    def test_auth_remediation_names_the_command(self) -> None:
        text = remediation_for(InitFailureKind.NOT_AUTHENTICATED)
        assert "earthengine authenticate" in text

    def test_project_remediation_points_at_contributor_tier(self) -> None:
        # 150 vs 1,000 EECU-hours for the same zero cost. Users routinely land
        # on Community by default and hit the ceiling, so the fix is named here.
        text = remediation_for(InitFailureKind.PROJECT_NOT_REGISTERED)
        assert "Contributor" in text
        assert "1,000" in text

    def test_quota_remediation_explains_the_monthly_reset(self) -> None:
        text = remediation_for(InitFailureKind.QUOTA_EXHAUSTED)
        assert "reset" in text.lower()


class TestEarthEngineSetupError:
    def test_carries_the_classified_kind(self) -> None:
        # The scheduled job branches on this: quota failures should retry next
        # cycle, authentication failures should stop and notify a human.
        err = EarthEngineSetupError(InitFailureKind.QUOTA_EXHAUSTED, "raw text")
        assert err.kind == InitFailureKind.QUOTA_EXHAUSTED

    def test_preserves_the_original_message(self) -> None:
        err = EarthEngineSetupError(InitFailureKind.UNKNOWN, "raw text")
        assert err.original == "raw text"

    def test_message_contains_both_remediation_and_original(self) -> None:
        err = EarthEngineSetupError(InitFailureKind.NOT_AUTHENTICATED, "boom")
        rendered = str(err)
        assert "earthengine authenticate" in rendered
        assert "boom" in rendered

    def test_is_a_runtime_error(self) -> None:
        assert isinstance(EarthEngineSetupError(InitFailureKind.UNKNOWN, "x"), RuntimeError)


class TestInitializeWithoutConfig:
    """``initialize`` must fail on missing config *before* touching the network.

    These run in CI with no credentials because configuration is validated
    first — nothing reaches Google.
    """

    def test_missing_project_raises_config_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.delenv("VANACHAKSHU_EE_PROJECT", raising=False)
        # Run from an empty directory so a developer's real .env cannot leak in
        # and make this pass for the wrong reason.
        monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]

        with pytest.raises(EarthEngineSetupError) as exc:
            initialize()

        assert exc.value.kind == InitFailureKind.CONFIG_MISSING

    def test_config_missing_message_names_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.delenv("VANACHAKSHU_EE_PROJECT", raising=False)
        monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]

        with pytest.raises(EarthEngineSetupError) as exc:
            initialize()

        assert "VANACHAKSHU_EE_PROJECT" in str(exc.value)
