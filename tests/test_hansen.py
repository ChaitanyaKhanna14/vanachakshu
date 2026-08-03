"""Tests for Hansen's loss-year encoding.

Hansen stores forest loss as a small integer offset, not a year: 1 means 2001,
25 means 2025. Getting that conversion wrong by one shifts every label by a
year — the detector would then be scored against the wrong ground truth, and
every accuracy figure the project ever publishes would be quietly wrong while
looking entirely reasonable.

Nothing here touches the network.
"""

from __future__ import annotations

import pytest

from vanachakshu.hansen import (
    HANSEN_BASE_YEAR,
    HANSEN_LAST_LOSS_YEAR,
    loss_code_for_year,
    year_for_loss_code,
)


class TestLossCodeForYear:
    @pytest.mark.parametrize(
        ("year", "code"),
        [
            (2001, 1),  # first year Hansen can represent
            (2002, 2),
            (2020, 20),
            (2024, 24),
            (2025, 25),  # last year in the pinned v1.13 asset
        ],
    )
    def test_known_conversions(self, year: int, code: int) -> None:
        assert loss_code_for_year(year) == code

    def test_base_year_itself_is_rejected(self) -> None:
        # 2000 is the canopy-cover baseline, not a loss year. Code 0 means
        # "no loss recorded", so accepting 2000 here would make "no loss"
        # indistinguishable from "lost in 2000".
        with pytest.raises(ValueError, match="outside Hansen's coverage"):
            loss_code_for_year(HANSEN_BASE_YEAR)

    @pytest.mark.parametrize("year", [1999, 2000, 2026, 2030])
    def test_rejects_years_outside_coverage(self, year: int) -> None:
        with pytest.raises(ValueError, match="outside Hansen's coverage"):
            loss_code_for_year(year)

    def test_error_names_the_version_bump_needed(self) -> None:
        # A future year is most often a sign the asset needs bumping, so the
        # message should say so rather than just refusing.
        with pytest.raises(ValueError, match="HANSEN_LAST_LOSS_YEAR"):
            loss_code_for_year(HANSEN_LAST_LOSS_YEAR + 1)


class TestYearForLossCode:
    @pytest.mark.parametrize(("code", "year"), [(1, 2001), (20, 2020), (25, 2025)])
    def test_known_conversions(self, code: int, year: int) -> None:
        assert year_for_loss_code(code) == year

    def test_zero_is_not_a_year(self) -> None:
        # 0 encodes "no loss recorded". Treating it as a year would turn every
        # undisturbed forest pixel into a loss event dated 2000.
        with pytest.raises(ValueError, match="does not correspond"):
            year_for_loss_code(0)

    @pytest.mark.parametrize("code", [-1, 26, 100])
    def test_rejects_out_of_range_codes(self, code: int) -> None:
        with pytest.raises(ValueError, match="does not correspond"):
            year_for_loss_code(code)


class TestRoundTrip:
    @pytest.mark.parametrize("year", range(HANSEN_BASE_YEAR + 1, HANSEN_LAST_LOSS_YEAR + 1))
    def test_every_covered_year_survives_a_round_trip(self, year: int) -> None:
        # Exhaustive across the full 25-year range: cheap, and the strongest
        # possible guard against an off-by-one at either boundary.
        assert year_for_loss_code(loss_code_for_year(year)) == year


class TestAssetConsistency:
    def test_last_loss_year_matches_the_pinned_asset(self) -> None:
        # The encoding is relative to the collection's own end year, so these
        # two must be bumped together. This test fails loudly if only one is.
        from vanachakshu import datasets

        assert str(HANSEN_LAST_LOSS_YEAR) in datasets.HANSEN_GFC, (
            f"HANSEN_LAST_LOSS_YEAR ({HANSEN_LAST_LOSS_YEAR}) disagrees with "
            f"the pinned asset ({datasets.HANSEN_GFC})"
        )
