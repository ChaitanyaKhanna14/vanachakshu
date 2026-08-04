"""Tests for radar configuration.

Pure — no Earth Engine. The correction maths itself is validated in
``test_sentinel1_integration.py``, because whether it works is a question about
real terrain and cannot be answered with fixtures.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vanachakshu.config import RadarDetectionConfig


class TestDefaults:
    def test_volume_model_is_the_default(self) -> None:
        # The AOI is forest. The volume model assumes volume scattering, which
        # is what a canopy does; the surface model targets bare ground.
        assert RadarDetectionConfig().terrain_model == "volume"

    def test_single_orbit_direction_by_default(self) -> None:
        # Ascending and descending passes view a hillside from opposite sides.
        # Mixing them into one baseline manufactures change.
        assert RadarDetectionConfig().orbit_pass == "DESCENDING"

    def test_baseline_covers_a_full_year(self) -> None:
        # A shorter baseline would compare a wet-season pass against a
        # dry-season normal and read the seasonal cycle as disturbance.
        assert RadarDetectionConfig().baseline_days == 365

    def test_confirmation_is_required(self) -> None:
        # The biggest false-positive filter in radar: wet ground changes
        # backscatter dramatically but recovers by the next pass.
        assert RadarDetectionConfig().min_confirming_passes >= 2


class TestValidation:
    @pytest.mark.parametrize("value", ["ascending", "BOTH", "", "UP"])
    def test_rejects_unknown_orbit_pass(self, value: str) -> None:
        # A typo must not silently fall through to "no filter", which would mix
        # viewing geometries and corrupt every baseline built from it.
        with pytest.raises(ValidationError):
            RadarDetectionConfig(orbit_pass=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["Volume", "vegetation", "none"])
    def test_rejects_unknown_terrain_model(self, value: str) -> None:
        with pytest.raises(ValidationError):
            RadarDetectionConfig(terrain_model=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0.0, -10.0])
    def test_rejects_non_positive_speckle_radius(self, value: float) -> None:
        with pytest.raises(ValidationError):
            RadarDetectionConfig(speckle_radius_m=value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_non_positive_drop_threshold(self, value: float) -> None:
        # A zero or negative threshold would flag every pixel on every pass.
        with pytest.raises(ValidationError):
            RadarDetectionConfig(drop_db=value)

    def test_rejects_a_baseline_too_short_to_cover_seasons(self) -> None:
        with pytest.raises(ValidationError):
            RadarDetectionConfig(baseline_days=30)

    def test_rejects_zero_confirming_passes(self) -> None:
        with pytest.raises(ValidationError):
            RadarDetectionConfig(min_confirming_passes=0)

    def test_is_immutable(self) -> None:
        cfg = RadarDetectionConfig()
        with pytest.raises(ValidationError):
            cfg.drop_db = 99.0  # type: ignore[misc]


class TestAcceptedValues:
    @pytest.mark.parametrize("value", ["ASCENDING", "DESCENDING"])
    def test_both_orbit_directions_are_selectable(self, value: str) -> None:
        assert RadarDetectionConfig(orbit_pass=value).orbit_pass == value  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["volume", "surface"])
    def test_both_terrain_models_are_selectable(self, value: str) -> None:
        assert RadarDetectionConfig(terrain_model=value).terrain_model == value  # type: ignore[arg-type]
