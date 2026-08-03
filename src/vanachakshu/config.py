"""Validated configuration for the alert pipeline.

Design rule: **every value is validated at construction time.** A malformed
bounding box or an out-of-range threshold raises immediately at startup, rather
than producing silently wrong alerts three hours into a scheduled run. Because
these objects are frozen, nothing downstream can mutate them mid-pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vanachakshu.geometry import spherical_bbox_area_sq_km, utm_epsg_for_lon_lat

# Quota guardrail, not a physical limit. Dense Sentinel-1 time series over a
# large AOI will exhaust the 1,000 EECU-hour monthly Earth Engine allowance
# fast. The plan starts at one taluk (~500-1,000 km2) and scales only after
# precision is demonstrated. Raise this deliberately, never casually.
MAX_AOI_SQ_KM: Final = 5_000.0

Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]


class BoundingBox(BaseModel):
    """An axis-aligned lon/lat box in EPSG:4326."""

    model_config = ConfigDict(frozen=True)

    west: Longitude
    south: Latitude
    east: Longitude
    north: Latitude

    @model_validator(mode="after")
    def _check_ordering_and_size(self) -> Self:
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) must be less than east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be less than north ({self.north})")

        area = self.area_sq_km
        if area > MAX_AOI_SQ_KM:
            raise ValueError(
                f"AOI is {area:,.0f} km2, above the {MAX_AOI_SQ_KM:,.0f} km2 guardrail. "
                "Large AOIs exhaust the Earth Engine monthly quota. Shrink the box, "
                "or raise MAX_AOI_SQ_KM deliberately if you have quota headroom."
            )
        return self

    @property
    def area_sq_km(self) -> float:
        """Approximate area, used for quota sanity checks only."""
        return spherical_bbox_area_sq_km(self.west, self.south, self.east, self.north)

    @property
    def centroid(self) -> tuple[float, float]:
        """Return ``(lon, lat)`` of the box centre."""
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def as_ee_coords(self) -> list[float]:
        """Return ``[west, south, east, north]``, the order ``ee.Geometry.Rectangle`` wants."""
        return [self.west, self.south, self.east, self.north]


class AreaOfInterest(BaseModel):
    """A named region to monitor, with the projection used for area maths."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    bbox: BoundingBox
    utm_epsg: int | None = Field(
        default=None,
        description="Projected CRS for area computation. Derived from the centroid if omitted.",
    )

    @model_validator(mode="after")
    def _derive_utm(self) -> Self:
        if self.utm_epsg is None:
            lon, lat = self.bbox.centroid
            # Frozen models need object.__setattr__ to fill a derived field.
            object.__setattr__(self, "utm_epsg", utm_epsg_for_lon_lat(lon, lat))
        return self

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, e.g. ``yellapur-taluk``."""
        return self.name.lower().replace(" ", "-").replace("_", "-")


class SeasonWindow(BaseModel):
    """An inclusive month range used to build same-season composites.

    Comparing a June image against a December image manufactures phenological
    change that is indistinguishable from deforestation — the single most common
    way these projects produce confident nonsense. Restricting every composite
    to the same months is the fix.

    January-March is the Western Ghats' clear post-monsoon window: the sky is
    reliably open and the canopy has not yet entered pre-monsoon stress.
    """

    model_config = ConfigDict(frozen=True)

    start_month: int = Field(ge=1, le=12)
    end_month: int = Field(ge=1, le=12)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.start_month > self.end_month:
            raise ValueError(
                f"start_month ({self.start_month}) must not exceed end_month "
                f"({self.end_month}); year-wrapping windows are not supported yet"
            )
        return self

    def date_range_for_year(self, year: int) -> tuple[str, str]:
        """Return ``(start, end)`` ISO dates for ``year``, end-exclusive.

        End-exclusive because that is the convention ``ee.Filter.date`` uses.
        Rolling to 1 January of the next year avoids month-length arithmetic.
        """
        start = f"{year:04d}-{self.start_month:02d}-01"
        if self.end_month == 12:
            end = f"{year + 1:04d}-01-01"
        else:
            end = f"{year:04d}-{self.end_month + 1:02d}-01"
        return start, end

    def most_recent_complete_year(self, today: date) -> int:
        """Latest year whose window has fully elapsed as of ``today``.

        Compositing a season that is still in progress silently produces a
        thinner, more cloud-contaminated image than the years it is compared
        against — which then reads as vegetation loss. Always compare complete
        seasons.
        """
        return today.year if today.month > self.end_month else today.year - 1


class OpticalDetectionConfig(BaseModel):
    """Thresholds for the Phase 1 Sentinel-2 baseline detector."""

    model_config = ConfigDict(frozen=True)

    pixel_size_m: float = Field(default=10.0, gt=0.0)

    ndvi_drop_threshold: float = Field(
        default=0.25,
        gt=0.0,
        le=2.0,
        description=(
            "Absolute NDVI fall required to flag a pixel as disturbed. 0.25 was "
            "chosen empirically, not guessed: it gives the best IoU and F1 in the "
            "threshold sweep. Raising it to 0.35 buys precision 0.54 (from 0.38) "
            "but collapses recall to 0.21. See "
            "docs/findings/2026-08-phase1-optical-baseline.md."
        ),
    )
    forest_ndvi_min: float = Field(
        default=0.60,
        ge=-1.0,
        le=1.0,
        description="Baseline NDVI a pixel must reach to be treated as forest at all.",
    )
    hansen_treecover_min_pct: int = Field(
        default=30,
        ge=0,
        le=100,
        description=(
            "Hansen treecover2000 percentage defining 'forest'. 30% is the "
            "convention used by Global Forest Watch, so results stay comparable."
        ),
    )
    min_clearing_ha: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Minimum patch size reported. RADD's validated floor is 0.2 ha in flat "
            "terrain; 0.5 ha is the honest target for the steep, fragmented "
            "Western Ghats. Raising this trades recall for precision."
        ),
    )
    max_scene_cloud_pct: float = Field(
        default=60.0,
        gt=0.0,
        le=100.0,
        description=(
            "Skip scenes cloudier than this before compositing. Mostly-cloud scenes "
            "contribute almost no valid pixels but cost the same to process."
        ),
    )
    min_observations: int = Field(
        default=3,
        ge=1,
        description=(
            "Cloud-free observations a pixel needs before its composite value is "
            "trusted. A median over one or two looks is not a median, it is a "
            "coin flip, and thin cloud that slipped the mask will survive it."
        ),
    )

    @property
    def min_clearing_pixels(self) -> int:
        """``min_clearing_ha`` expressed as a whole number of pixels."""
        px_area_ha = (self.pixel_size_m**2) / 10_000.0
        return max(1, round(self.min_clearing_ha / px_area_ha))


class Settings(BaseSettings):
    """Environment-driven settings, primarily secrets and account identifiers.

    Read from the environment or a local ``.env`` file, both of which are
    gitignored. Nothing here is ever committed.
    """

    model_config = SettingsConfigDict(
        env_prefix="VANACHAKSHU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ee_project: str = Field(
        description=(
            "Google Cloud project registered for Earth Engine noncommercial use. "
            "Set VANACHAKSHU_EE_PROJECT. See README for Contributor-tier signup."
        )
    )
    ee_service_account_key: str | None = Field(
        default=None,
        description="Path to a service-account JSON key. Unset locally; set in CI/cron.",
    )


# --- Default AOI -------------------------------------------------------------
# Approximate bounding box around Yellapur taluk, Uttara Kannada, Karnataka.
# Deliberately a rectangle for now: replace with the real administrative
# boundary (data/aoi/*.geojson) before quoting any area figure publicly, since
# a bbox includes non-forest and neighbouring taluks.
YELLAPUR_TALUK: Final = AreaOfInterest(
    name="Yellapur Taluk",
    bbox=BoundingBox(west=74.55, south=14.80, east=74.90, north=15.15),
)

WESTERN_GHATS_CLEAR_SEASON: Final = SeasonWindow(start_month=1, end_month=3)
