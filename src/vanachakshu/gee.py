"""The only module that talks to Earth Engine.

Everything that needs Google credentials is confined here, which is what keeps
the rest of the package testable in CI without secrets.

The bulk of this module is error handling rather than API calls, deliberately.
Earth Engine's setup failures are genuinely hard to read: an unregistered
project, an expired token, a typo'd project id and an exhausted quota all
surface as some variation of "permission denied", usually wrapped in a stack
trace from inside an HTTP client. The classification below turns those into
messages that say what to actually do next.

Classification is a pure function over the error text, so it is unit-tested
without credentials. Only ``initialize`` and ``healthcheck`` touch the network.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import ee
from pydantic import ValidationError

from vanachakshu.config import Settings

__all__ = [
    "EarthEngineSetupError",
    "InitFailureKind",
    "classify_init_error",
    "healthcheck",
    "initialize",
    "remediation_for",
]

_DOCS_HINT: Final = "See the 'Earth Engine access' section of README.md."


class InitFailureKind(StrEnum):
    """Why Earth Engine initialisation failed, in terms a user can act on."""

    CONFIG_MISSING = "config_missing"
    NOT_AUTHENTICATED = "not_authenticated"
    PROJECT_NOT_REGISTERED = "project_not_registered"
    PERMISSION_DENIED = "permission_denied"
    QUOTA_EXHAUSTED = "quota_exhausted"
    NETWORK = "network"
    UNKNOWN = "unknown"


class EarthEngineSetupError(RuntimeError):
    """Earth Engine could not be initialised, with actionable remediation.

    Carries the classified ``kind`` so callers (the CLI, the scheduled job) can
    branch on it — for example, a quota failure should retry next cycle, while
    an authentication failure should stop and notify a human.
    """

    def __init__(self, kind: InitFailureKind, original: str) -> None:
        self.kind = kind
        self.original = original
        super().__init__(f"{remediation_for(kind)}\n\nOriginal error:\n  {original}")


# Substring probes, checked in order. Ordering matters: an unregistered project
# often reports itself as a permission problem, so the more specific project and
# quota probes run before the generic permission one.
_PROBES: Final[tuple[tuple[InitFailureKind, tuple[str, ...]], ...]] = (
    (
        InitFailureKind.QUOTA_EXHAUSTED,
        (
            "quota",
            "eecu",
            "restricted mode",
            "rate limit",
            "429",
            # Google reports this as the gRPC status name, so the underscore
            # form is what actually arrives; the spaced form appears in prose.
            "resource_exhausted",
            "resource exhausted",
        ),
    ),
    (
        InitFailureKind.PROJECT_NOT_REGISTERED,
        (
            "not registered",
            "not signed up",
            "is not a valid project",
            "no project found",
            "project not found",
            "has not been used",
            # Observed verbatim for a typo'd id:
            #   "Project 'projects/foo' not found or deleted."
            # The quoted id sits between "project" and "not found", so the
            # probe above cannot match it.
            "not found or deleted",
        ),
    ),
    # Checked before NOT_AUTHENTICATED. Transport failures happen while talking
    # to the OAuth endpoint, so their messages contain the URL ".../token" and
    # would otherwise be misread as a credentials problem. Observed for real: an
    # SSL certificate error was reported as "no valid credentials", sending the
    # user to re-authenticate when the actual cause was TLS interception.
    (
        InitFailureKind.NETWORK,
        (
            "certificate verify failed",
            "certificate_verify_failed",
            "sslerror",
            "ssl:",
            "max retries exceeded",
            "connection",
            "timed out",
            "timeout",
            "temporary failure",
            "unreachable",
            "name resolution",
        ),
    ),
    (
        InitFailureKind.NOT_AUTHENTICATED,
        (
            "authenticate",
            "authorize",
            "credential",
            "not initialized",
            "invalid_grant",
            # Deliberately specific. A bare "token" also matches the OAuth
            # endpoint URL, which appears in transport errors that have nothing
            # to do with credentials.
            "token has been expired",
            "token has been revoked",
            "invalid token",
            "401",
        ),
    ),
    (
        InitFailureKind.PERMISSION_DENIED,
        ("permission", "forbidden", "403", "access denied", "caller does not have"),
    ),
)

_REMEDIATION: Final[dict[InitFailureKind, str]] = {
    InitFailureKind.CONFIG_MISSING: (
        "Required configuration is missing.\n"
        "Set VANACHAKSHU_EE_PROJECT to your Earth Engine Cloud project id, either as an "
        "environment variable or in a .env file at the repository root:\n"
        "    echo 'VANACHAKSHU_EE_PROJECT=your-project-id' > .env\n" + _DOCS_HINT
    ),
    InitFailureKind.NOT_AUTHENTICATED: (
        "Earth Engine has no valid credentials for this machine.\n"
        "Run:  earthengine authenticate\n"
        "If you are running unattended (cron/CI), set VANACHAKSHU_EE_SERVICE_ACCOUNT_KEY "
        "to a service-account JSON key path instead. " + _DOCS_HINT
    ),
    InitFailureKind.PROJECT_NOT_REGISTERED: (
        "Earth Engine is not usable on this Cloud project yet. THREE separate things "
        "must all be true, and they are easy to confuse:\n"
        "  1. VANACHAKSHU_EE_PROJECT is the project *id*, not the display name.\n"
        "  2. The Earth Engine API is enabled on that project. The original error "
        "below usually contains a direct console link — that is the fastest route.\n"
        "  3. The project is registered for noncommercial use at "
        "https://code.earthengine.google.com/register. Choose the Contributor tier: "
        "free, and 1,000 EECU-hours/month instead of Community's 150.\n"
        "A newly enabled API can take a few minutes to propagate, so retry before "
        "assuming step 2 failed. " + _DOCS_HINT
    ),
    InitFailureKind.PERMISSION_DENIED: (
        "Earth Engine refused access to this project.\n"
        "Most often this means the project is not registered for Earth Engine (register "
        "at https://code.earthengine.google.com/register), or the authenticated account "
        "is not a member of it. Confirm you authenticated as the same Google account "
        "that owns the project. " + _DOCS_HINT
    ),
    InitFailureKind.QUOTA_EXHAUSTED: (
        "Earth Engine compute quota is exhausted for this month.\n"
        "Noncommercial projects are metered since 27 April 2026. The Community tier is "
        "150 EECU-hours/month; the Contributor tier is 1,000 and is equally free — it "
        "only requires attaching a billing account for verification, with no charge for "
        "noncommercial use. Upgrade the tier, shrink the AOI, or wait for the monthly "
        "reset (1st of the month, Pacific time). " + _DOCS_HINT
    ),
    InitFailureKind.NETWORK: (
        "Could not reach Earth Engine. This is a transport problem, not a "
        "configuration one — re-authenticating will not help.\n"
        "If the error mentions a certificate ('CERTIFICATE_VERIFY_FAILED'), something "
        "is intercepting TLS. Usual causes, in order of likelihood: corporate or "
        "campus network proxy, antivirus HTTPS scanning, or a VPN. Try another "
        "network, pause the interceptor, or point REQUESTS_CA_BUNDLE at the "
        "intercepting root certificate.\n"
        "Otherwise check connectivity and retry — transient failures are common and "
        "the scheduled job should simply run again next cycle."
    ),
    InitFailureKind.UNKNOWN: (
        "Earth Engine initialisation failed for an unrecognised reason. The original "
        "error is below. " + _DOCS_HINT
    ),
}


def classify_init_error(message: str) -> InitFailureKind:
    """Map an Earth Engine error message to an actionable failure kind.

    Pure and case-insensitive, so it is fully unit-tested without credentials.
    Unrecognised messages fall back to :attr:`InitFailureKind.UNKNOWN` rather
    than guessing — a wrong diagnosis wastes more time than no diagnosis.
    """
    text = message.lower()
    for kind, needles in _PROBES:
        if any(needle in text for needle in needles):
            return kind
    return InitFailureKind.UNKNOWN


def remediation_for(kind: InitFailureKind) -> str:
    """Return the human-facing 'what to do next' text for a failure kind."""
    return _REMEDIATION[kind]


def _credentials_from_key_file(key_path: str) -> Any:
    # Return type is Any because earthengine-api ships no type stubs, so
    # ee.ServiceAccountCredentials is a plain function to mypy, not a type.
    """Build service-account credentials, failing clearly if the key is missing."""
    path = Path(key_path)
    if not path.is_file():
        raise EarthEngineSetupError(
            InitFailureKind.NOT_AUTHENTICATED,
            f"service-account key file not found at {path}",
        )
    # email=None lets the client read client_email out of the key JSON itself,
    # so the path is the single source of truth.
    return ee.ServiceAccountCredentials(None, key_file=str(path))


def initialize(settings: Settings | None = None) -> None:
    """Initialise Earth Engine, raising :class:`EarthEngineSetupError` on failure.

    Uses a service-account key when ``ee_service_account_key`` is set (the
    unattended path, for cron and CI), and otherwise falls back to the
    credentials written by ``earthengine authenticate`` (the local path).

    .. warning::
       Earth Engine's initialisation is **process-global**, and a second call
       with *different* settings is silently ignored rather than re-initialising.
       Verified empirically: after a successful init, calling this again with a
       nonexistent project raises nothing at all.

       Consequences worth remembering:

       * A successful ``initialize`` proves very little on its own — the project
         is not validated until a request is actually made. Always follow it
         with :func:`healthcheck`.
       * Anything that needs to test against a different project must do so in a
         fresh process, not merely by calling this again.
    """
    if settings is not None:
        resolved = settings
    else:
        try:
            # Fields are populated from the environment, which mypy cannot see.
            resolved = Settings()  # type: ignore[call-arg]
        except ValidationError as exc:
            raise EarthEngineSetupError(InitFailureKind.CONFIG_MISSING, str(exc)) from exc

    try:
        if resolved.ee_service_account_key:
            credentials = _credentials_from_key_file(resolved.ee_service_account_key)
            ee.Initialize(credentials, project=resolved.ee_project)
        else:
            ee.Initialize(project=resolved.ee_project)
    except EarthEngineSetupError:
        # Already classified by _credentials_from_key_file; do not re-wrap.
        raise
    except Exception as exc:
        # Deliberately broad: Earth Engine raises EEException, HttpError,
        # google.auth errors and bare OSErrors depending on how it fails. The
        # classifier works off the message text, so the exception type is not
        # what we branch on.
        raise EarthEngineSetupError(classify_init_error(str(exc)), str(exc)) from exc


def healthcheck() -> None:
    """Verify a real server round-trip, raising on failure.

    ``ee.Initialize`` can succeed locally and still leave you unable to compute
    anything — the credential check and the compute permission check are
    separate. This forces one trivial server-side evaluation so setup problems
    surface now rather than midway through a scheduled run. The computation is
    deliberately tiny and costs no meaningful quota.
    """
    try:
        result = ee.Number(1).add(1).getInfo()
    except Exception as exc:
        # Broad for the same reason as in initialize().
        raise EarthEngineSetupError(classify_init_error(str(exc)), str(exc)) from exc

    if result != 2:
        raise EarthEngineSetupError(
            InitFailureKind.UNKNOWN,
            f"server round-trip returned {result!r}, expected 2",
        )
