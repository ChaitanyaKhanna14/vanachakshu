"""Setup verification — the machinery behind ``vanachakshu doctor``.

Earth Engine setup has several independent things that can be wrong, and by
default they all fail at the same place with the same unhelpful message. This
module checks them one at a time, in dependency order, and stops at the first
failure so the report names the actual problem rather than its consequences.

The check *framework* (result type, ordering, summarising, exit codes) is pure
and unit-tested. Only the individual probe bodies touch the network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import ee
from pydantic import ValidationError

from vanachakshu import datasets
from vanachakshu.config import (
    WESTERN_GHATS_CLEAR_SEASON,
    YELLAPUR_TALUK,
    AreaOfInterest,
    SeasonWindow,
    Settings,
)
from vanachakshu.gee import (
    EarthEngineSetupError,
    InitFailureKind,
    initialize,
    remediation_for,
)

__all__ = [
    "CheckResult",
    "all_passed",
    "format_report",
    "run_diagnostics",
]

# A probe returns a human-readable detail string on success, or raises.
Probe = Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single setup check."""

    name: str
    ok: bool
    detail: str
    remediation: str | None = None


def all_passed(results: Sequence[CheckResult]) -> bool:
    """True when every check succeeded (vacuously true for an empty run)."""
    return all(result.ok for result in results)


def format_report(results: Sequence[CheckResult]) -> str:
    """Render results as plain text, remediation last.

    Kept free of Rich markup so it is equally usable in a terminal, a CI log,
    and the body of a failure notification from the scheduled job.
    """
    lines: list[str] = []
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        lines.append(f"[{marker}] {result.name}: {result.detail}")

    failed = [r for r in results if not r.ok and r.remediation]
    for result in failed:
        lines.append("")
        lines.append(f"How to fix '{result.name}':")
        lines.append(result.remediation or "")

    return "\n".join(lines)


def _run(name: str, probe: Probe) -> CheckResult:
    """Execute one probe, converting any failure into a CheckResult."""
    try:
        return CheckResult(name=name, ok=True, detail=probe())
    except EarthEngineSetupError as exc:
        return CheckResult(
            name=name, ok=False, detail=exc.original, remediation=remediation_for(exc.kind)
        )
    except Exception as exc:
        # Broad by design: this is a diagnostic tool, and an unexpected
        # exception type is itself a finding worth reporting rather than
        # crashing on.
        return CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")


def run_diagnostics(
    settings: Settings | None = None,
    aoi: AreaOfInterest = YELLAPUR_TALUK,
    season: SeasonWindow = WESTERN_GHATS_CLEAR_SEASON,
    today: date | None = None,
) -> list[CheckResult]:
    """Run setup checks in dependency order, stopping at the first failure.

    Stopping early is deliberate. If Earth Engine never initialised, every
    later check also fails, and a wall of red hides the one line that matters.
    """
    now = today if today is not None else date.today()
    year = season.most_recent_complete_year(now)
    start, end = season.date_range_for_year(year)

    results: list[CheckResult] = []

    # 1. Pure, no network: confirms the AOI is sane before spending anything.
    results.append(
        _run(
            "Area of interest",
            lambda: (
                f"{aoi.name} — {aoi.bbox.area_sq_km:,.0f} km2, area maths in EPSG:{aoi.utm_epsg}"
            ),
        )
    )

    # 2. Configuration and credentials.
    resolved: Settings | None = settings

    def _check_config() -> str:
        nonlocal resolved
        if resolved is None:
            try:
                resolved = Settings()  # type: ignore[call-arg]
            except ValidationError as exc:
                # Without this the user sees a raw pydantic traceback instead of
                # "set VANACHAKSHU_EE_PROJECT" — which is the whole point of
                # this command.
                raise EarthEngineSetupError(InitFailureKind.CONFIG_MISSING, str(exc)) from exc
        mode = "service account" if resolved.ee_service_account_key else "user credentials"
        return f"project '{resolved.ee_project}', authenticating with {mode}"

    results.append(_run("Configuration", _check_config))
    if not all_passed(results):
        return results

    results.append(_run("Earth Engine initialises", lambda: _init(resolved)))
    if not all_passed(results):
        return results

    # 3. Server round-trip: credentials can load and still not compute.
    results.append(_run("Server round-trip", _round_trip))
    if not all_passed(results):
        return results

    # 4. Real data access. These are the checks that prove the account can do
    #    the actual job, not merely authenticate.
    geometry = ee.Geometry.Rectangle(aoi.bbox.as_ee_coords())

    results.append(
        _run(
            f"Sentinel-2 imagery ({start} to {end})",
            lambda: _count_sentinel2(geometry, start, end),
        )
    )
    results.append(_run("Hansen forest-loss labels", lambda: _read_hansen(geometry)))

    return results


def _init(settings: Settings | None) -> str:
    initialize(settings)
    return "initialised"


def _round_trip() -> str:
    value = ee.Number(1).add(1).getInfo()
    if value != 2:
        raise RuntimeError(f"server returned {value!r}, expected 2")
    return "server evaluated a trivial computation"


def _count_sentinel2(geometry: ee.Geometry, start: str, end: str) -> str:
    collection = (
        ee.ImageCollection(datasets.SENTINEL2_SR).filterBounds(geometry).filterDate(start, end)
    )
    count = int(collection.size().getInfo())
    if count == 0:
        raise RuntimeError(
            "no Sentinel-2 scenes found for this AOI and date range — check the "
            "bounding box is where you think it is"
        )
    return f"{count} scenes available"


def _read_hansen(geometry: ee.Geometry) -> str:
    image = ee.Image(datasets.HANSEN_GFC)
    bands = image.bandNames().getInfo()
    required = {"treecover2000", "lossyear"}
    missing = required - set(bands)
    if missing:
        raise RuntimeError(f"Hansen asset is missing expected bands: {sorted(missing)}")

    # Mean loss-year over the AOI at coarse scale: cheap, and proves the asset
    # is readable *here* rather than merely existing.
    stats = (
        image.select("treecover2000")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=300,
            maxPixels=int(1e8),
            bestEffort=True,
        )
        .getInfo()
    )
    cover = stats.get("treecover2000")
    if cover is None:
        raise RuntimeError("could not sample treecover2000 over the AOI")
    return f"readable; mean tree cover in AOI is {float(cover):.0f}% (year 2000 baseline)"
