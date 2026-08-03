"""Scoring detections against Hansen's published forest-loss record.

This module answers the only question that matters yet: **is the detector any
good?**

Four numbers, all built from three quantities:

* **True positive (TP)** — we said loss, Hansen agrees.
* **False positive (FP)** — we said loss, Hansen does not. A wasted field trip.
* **False negative (FN)** — Hansen recorded loss, we missed it.

True negatives are deliberately never computed. Undisturbed forest is >99% of
the area, so including it would make *accuracy* look superb no matter what: a
detector that reports nothing at all would score above 99%. Every metric here
ignores the easy majority on purpose.

One honest caveat that must accompany any number produced here: **Hansen is
itself a model, not ground truth.** Agreeing with it means agreeing with
another algorithm. Real validation is Phase 4, against high-resolution imagery.
Numbers from this module are a development signal, not a published result.
"""

from __future__ import annotations

from dataclasses import dataclass

import ee

__all__ = [
    "ConfusionCounts",
    "score_detection",
]


@dataclass(frozen=True)
class ConfusionCounts:
    """Overlap between detections and reference labels, in hectares.

    Hectares rather than pixel counts, so the numbers stay meaningful when the
    comparison scale changes and so they can be quoted directly.
    """

    true_positive: float
    false_positive: float
    false_negative: float

    def __post_init__(self) -> None:
        for name in ("true_positive", "false_positive", "false_negative"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def detected(self) -> float:
        """Total area we flagged (TP + FP)."""
        return self.true_positive + self.false_positive

    @property
    def reference(self) -> float:
        """Total area the reference records as loss (TP + FN)."""
        return self.true_positive + self.false_negative

    @property
    def precision(self) -> float:
        """Of what we flagged, how much was real.

        This is the number a forest officer feels. Precision of 0.4 means three
        of every five site visits find nothing, which is how a system loses its
        users. The project's design rules optimise for this over recall.

        Returns 0.0 when nothing was flagged — vacuously "nothing correct".
        """
        return self.true_positive / self.detected if self.detected else 0.0

    @property
    def recall(self) -> float:
        """Of the real loss, how much we caught.

        Returns 0.0 when the reference records no loss at all, since there was
        nothing available to catch.
        """
        return self.true_positive / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall.

        Harmonic, not arithmetic, so it cannot be gamed by maximising one at the
        expense of the other: flagging everything gives recall 1.0 and precision
        near 0, and F1 stays near 0.
        """
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def iou(self) -> float:
        """Intersection over union — overlap of the two maps as a fraction.

        The standard measure for segmentation, and stricter than F1: it charges
        for false positives and false negatives in a single denominator.
        """
        union = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / union if union else 0.0

    def as_row(self) -> str:
        """One-line summary for sweep tables."""
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} "
            f"F1={self.f1:.3f} IoU={self.iou:.3f}  "
            f"(TP={self.true_positive:6.1f} FP={self.false_positive:6.1f} "
            f"FN={self.false_negative:6.1f} ha)"
        )


def score_detection(
    detected: ee.Image,
    reference: ee.Image,
    geometry: ee.Geometry,
    scale: float = 30.0,
) -> ConfusionCounts:
    """Compare a detection mask against a reference mask over ``geometry``.

    ``scale`` defaults to 30 m, Hansen's native resolution. Scoring 10 m
    detections at 10 m would upsample the reference and invent a precision the
    comparison cannot support — the reference genuinely does not know where
    loss falls within its own 30 m cell. Comparing at the coarser of the two
    resolutions is the honest choice.

    Both inputs are unmasked to 0 first. Masked pixels are not zero in Earth
    Engine, and letting a mask through here would silently drop pixels from the
    comparison — the same class of bug that once emptied the forest mask.
    """
    predicted = detected.unmask(0).gt(0)
    truth = reference.unmask(0).gt(0)

    hectares = ee.Image.pixelArea().divide(10_000)
    overlap = (
        hectares.updateMask(predicted.And(truth))
        .rename("tp")
        .addBands(hectares.updateMask(predicted.And(truth.Not())).rename("fp"))
        .addBands(hectares.updateMask(predicted.Not().And(truth)).rename("fn"))
    )

    totals: dict[str, float | None] = (
        overlap.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=scale,
            maxPixels=int(1e10),
        ).getInfo()
        or {}
    )

    return ConfusionCounts(
        true_positive=float(totals.get("tp") or 0.0),
        false_positive=float(totals.get("fp") or 0.0),
        false_negative=float(totals.get("fn") or 0.0),
    )
