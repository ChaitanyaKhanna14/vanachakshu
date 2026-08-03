"""Tests for detection scoring maths.

Entirely pure — no Earth Engine, no network.

These metrics are what the project will eventually publish, so the edge cases
matter as much as the ordinary ones. A metric that raises ZeroDivisionError on
an empty run would crash the scheduled job on a quiet week; one that silently
returns a flattering number would be worse.
"""

from __future__ import annotations

import pytest

from vanachakshu.score import ConfusionCounts


class TestPerfectAndEmptyCases:
    def test_perfect_detection_scores_one_everywhere(self) -> None:
        counts = ConfusionCounts(true_positive=100.0, false_positive=0.0, false_negative=0.0)
        assert counts.precision == 1.0
        assert counts.recall == 1.0
        assert counts.f1 == 1.0
        assert counts.iou == 1.0

    def test_detecting_nothing_scores_zero_without_crashing(self) -> None:
        # A quiet week is normal, not an error. The scheduled job must survive it.
        counts = ConfusionCounts(true_positive=0.0, false_positive=0.0, false_negative=50.0)
        assert counts.precision == 0.0
        assert counts.recall == 0.0
        assert counts.f1 == 0.0
        assert counts.iou == 0.0

    def test_no_reference_loss_scores_zero_recall_without_crashing(self) -> None:
        counts = ConfusionCounts(true_positive=0.0, false_positive=10.0, false_negative=0.0)
        assert counts.recall == 0.0
        assert counts.precision == 0.0

    def test_completely_empty_run_is_all_zeros(self) -> None:
        counts = ConfusionCounts(0.0, 0.0, 0.0)
        assert (counts.precision, counts.recall, counts.f1, counts.iou) == (0.0, 0.0, 0.0, 0.0)


class TestKnownValues:
    @pytest.fixture
    def counts(self) -> ConfusionCounts:
        # Flagged 100 ha, 75 of it real; missed 25 ha of real loss.
        return ConfusionCounts(true_positive=75.0, false_positive=25.0, false_negative=25.0)

    def test_precision(self, counts: ConfusionCounts) -> None:
        assert counts.precision == pytest.approx(0.75)

    def test_recall(self, counts: ConfusionCounts) -> None:
        assert counts.recall == pytest.approx(0.75)

    def test_f1(self, counts: ConfusionCounts) -> None:
        assert counts.f1 == pytest.approx(0.75)

    def test_iou_is_stricter_than_f1(self, counts: ConfusionCounts) -> None:
        # 75 / (75 + 25 + 25) = 0.6. IoU charges for both error types in one
        # denominator, so it is always the harsher number.
        assert counts.iou == pytest.approx(0.60)
        assert counts.iou < counts.f1

    def test_totals(self, counts: ConfusionCounts) -> None:
        assert counts.detected == 100.0
        assert counts.reference == 100.0


class TestF1CannotBeGamed:
    def test_flagging_everything_gives_perfect_recall_but_poor_f1(self) -> None:
        # The failure mode F1 exists to punish: catch everything by flagging
        # the whole forest. Recall is perfect and the detector is useless.
        counts = ConfusionCounts(true_positive=100.0, false_positive=9_900.0, false_negative=0.0)
        assert counts.recall == 1.0
        assert counts.precision == pytest.approx(0.01)
        assert counts.f1 < 0.02

    def test_flagging_almost_nothing_gives_perfect_precision_but_poor_f1(self) -> None:
        # The opposite gaming strategy: flag only the single most obvious
        # clearing. Precision is perfect and almost everything is missed.
        counts = ConfusionCounts(true_positive=1.0, false_positive=0.0, false_negative=99.0)
        assert counts.precision == 1.0
        assert counts.recall == pytest.approx(0.01)
        assert counts.f1 < 0.02


class TestPrecisionRecallTradeoff:
    def test_precision_reflects_wasted_field_visits(self) -> None:
        # Precision 0.4 means three of five site visits find nothing. This is
        # the number that decides whether a forest officer keeps opening the
        # alerts, which is why the design rules optimise for it.
        counts = ConfusionCounts(true_positive=40.0, false_positive=60.0, false_negative=10.0)
        assert counts.precision == pytest.approx(0.40)
        assert counts.recall > counts.precision  # caught most, but noisily


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"true_positive": -1.0, "false_positive": 0.0, "false_negative": 0.0},
            {"true_positive": 0.0, "false_positive": -1.0, "false_negative": 0.0},
            {"true_positive": 0.0, "false_positive": 0.0, "false_negative": -1.0},
        ],
    )
    def test_rejects_negative_areas(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            ConfusionCounts(**kwargs)

    def test_is_immutable(self) -> None:
        counts = ConfusionCounts(1.0, 1.0, 1.0)
        with pytest.raises(AttributeError):
            counts.true_positive = 5.0  # type: ignore[misc]


class TestReporting:
    def test_row_contains_every_metric_and_count(self) -> None:
        row = ConfusionCounts(75.0, 25.0, 25.0).as_row()
        for token in ("P=", "R=", "F1=", "IoU=", "TP=", "FP=", "FN="):
            assert token in row
