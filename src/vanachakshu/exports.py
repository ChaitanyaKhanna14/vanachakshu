"""Batch exports, for computations interactive calls cannot finish.

``getInfo`` runs synchronously against a shared pool and gives up after a couple
of minutes. That ceiling was hit six times building this project — sampling
training points, vectorising detections, scoring at native resolution — and each
time the workaround was to shrink the question. The accumulated cost was that
**the detector could not be measured at its own 10 m resolution at all**, which
is a capability gap rather than an inconvenience: tuning against a coarser
measurement tunes against the wrong thing.

Batch tasks have far higher limits and run asynchronously, which is the actual
answer.

The pattern that matters here is **materialise once, query many times**. The
expensive part of a threshold sweep is not the threshold — it is recomputing 128
bands of embedding difference for every candidate value. Exporting that once to
an asset turns a sweep from N impossible computations into N cheap ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import ee

__all__ = [
    "ExportResult",
    "asset_exists",
    "export_image",
    "wait_for",
]

# Earth Engine reports task state through these strings.
_TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


@dataclass(frozen=True)
class ExportResult:
    """Outcome of a batch export."""

    asset_id: str
    state: str
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == "COMPLETED"


def asset_exists(asset_id: str) -> bool:
    """True if the asset is already present.

    Checked before every export so a repeated sweep reuses the materialised
    result instead of recomputing it — which is the entire point of exporting.
    """
    try:
        ee.data.getAsset(asset_id)
    except ee.EEException:
        return False
    return True


def export_image(
    image: ee.Image,
    asset_id: str,
    region: ee.Geometry,
    scale: float,
    description: str = "vanachakshu_export",
    max_pixels: int = int(1e10),
) -> ee.batch.Task:
    """Start a batch export of ``image`` to an Earth Engine asset.

    Returns immediately; the task runs on Google's infrastructure. Use
    :func:`wait_for` to block until it finishes.

    ``pyramidingPolicy`` is set to mean so that overviews of a continuous band
    are averaged rather than sampled. The default for a float band would pick a
    single representative pixel, which for sparse detections means most
    overviews would show nothing — the same class of error as scoring 10 m
    detections at 30 m.
    """
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        region=region,
        scale=scale,
        maxPixels=max_pixels,
        pyramidingPolicy={".default": "mean"},
    )
    task.start()
    return task


def wait_for(
    task: ee.batch.Task,
    asset_id: str,
    poll_seconds: float = 20.0,
    timeout_seconds: float = 3600.0,
) -> ExportResult:
    """Block until a batch task reaches a terminal state.

    Polls rather than streams because Earth Engine offers no completion
    callback. The default 20 s interval is a compromise: often enough that a
    two-minute export is not padded much, rare enough not to hammer the API
    across an hour-long one.
    """
    started = time.monotonic()

    while True:
        status: dict[str, Any] = task.status()
        state = str(status.get("state", "UNKNOWN"))
        elapsed = time.monotonic() - started

        if state in _TERMINAL:
            return ExportResult(
                asset_id=asset_id,
                state=state,
                seconds=elapsed,
                error=status.get("error_message"),
            )

        if elapsed > timeout_seconds:
            # Deliberately not cancelled: a long-running export is usually
            # still progressing, and killing it discards the work. The caller
            # can check back later, or cancel explicitly.
            return ExportResult(
                asset_id=asset_id,
                state=state,
                seconds=elapsed,
                error=f"still {state} after {timeout_seconds:.0f}s; not cancelled",
            )

        time.sleep(poll_seconds)
