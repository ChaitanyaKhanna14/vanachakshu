"""Tune the embedding detector at its native 10 m and report honest intervals.

Run with ``python scripts/tune_detector.py``. Results are checkpointed, so an
interrupted run resumes where it stopped; delete the checkpoint to start over.

Why sampling rather than a whole-AOI reduction
----------------------------------------------
A 10 m ``reduceRegion`` over this AOI times out — 128 bands of embedding
difference across 14.6M pixels exceeds the interactive budget — and the batch
``Export`` route needs an asset root the Cloud project does not have yet. But
the sweep never needed either. It needs the *same* pixels evaluated under many
parameter settings, and a stratified sample delivers that far more cheaply.

The trick that makes the sweep nearly free is sampling ``connectedPixelCount``
at each candidate threshold as its own band. Patch size is the one parameter
that cannot be swept client-side from point values — it is a spatial operation —
so it is computed server-side once per candidate, in the same request. After
that both parameters sweep in memory.

Why the sample must be stratified, and weighted
------------------------------------------------
Recorded loss is 10.7 ha in 106,543 ha of forest: one pixel in 9,939. A uniform
sample of any affordable size contains almost no positives. Stratifying fixes
that, but it means raw sample counts are **not** population counts — precision
computed from them directly would be inflated by orders of magnitude. Every
count here is weighted by (stratum area / stratum sample size), computed per
tile, because each tile has its own stratum sizes and its own achieved sample.

Why precision carries a confidence interval and recall does not
----------------------------------------------------------------
The loss stratum holds ~1,070 pixels and the sample contains ~1,071 loss points,
so essentially every recorded-loss pixel in the taluk is in the sample. Recall
is measured, not extrapolated. Precision depends on false-positive area
extrapolated from stable points each standing for up to 3.85 ha — so when two
points fire, precision rests on n=2, and reporting it bare would be dishonest.

This matters concretely: every configuration above threshold 0.45 reports
precision 1.000 purely because no false positive survived into the sample. That
is a resolution floor, not a result, and the interval says so.

Three earlier failures shaped the structure
--------------------------------------------
Asking for more rare-class points than exist makes Earth Engine scan the whole
AOI hunting for pixels that were never there, so counts are sized to the
stratum. A single tile timing out used to discard every tile before it, so
results are checkpointed and a failed tile subdivides rather than aborts.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import ee

from vanachakshu import embeddings, hansen, sentinel2
from vanachakshu.config import WESTERN_GHATS_CLEAR_SEASON as SEASON
from vanachakshu.config import (
    YELLAPUR_TALUK,
    BoundingBox,
    OpticalDetectionConfig,
)
from vanachakshu.gee import initialize

BASE, TARGET = 2024, 2025
SCALE = 10
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55)
MIN_PIXELS = (5, 10, 20, 40)
N_LOSS, N_STABLE = 400, 1000
TILE_DIVISIONS = 6
MAX_SPLIT_DEPTH = 3
CHECKPOINT = Path(__file__).with_name("tune_detector.checkpoint.json")

Row = dict[str, Any]


def subdivide(b: BoundingBox, n: int) -> list[BoundingBox]:
    dx, dy = (b.east - b.west) / n, (b.north - b.south) / n
    return [
        BoundingBox(
            west=b.west + i * dx,
            south=b.south + j * dy,
            east=b.west + (i + 1) * dx,
            north=b.south + (j + 1) * dy,
        )
        for i in range(n)
        for j in range(n)
    ]


def poisson_interval(k: int) -> tuple[float, float]:
    """95% interval on a Poisson count, correct at k=0.

    A normal approximation gives a zero-width interval at k=0, which is the
    single most misleading thing it could say here — k=0 is exactly the case
    that produces the spurious precision 1.000 rows.
    """

    def chi2_inv(p: float, df: int) -> float:
        if df <= 0:
            return 0.0
        z = statistics.NormalDist().inv_cdf(p)
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3

    lo = 0.0 if k == 0 else 0.5 * chi2_inv(0.025, 2 * k)
    return lo, 0.5 * chi2_inv(0.975, 2 * k + 2)


def build_stack() -> tuple[ee.Image, ee.Image, ee.Image]:
    """Return (sampling stack, forest mask, truth mask)."""
    ocfg = OpticalDetectionConfig()
    geom = ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())

    distance = embeddings.euclidean_distance(
        embeddings.annual(geom, BASE), embeddings.annual(geom, TARGET)
    ).rename("dist")
    forest = hansen.forest_mask(BASE, ocfg)
    base_ndvi = sentinel2.seasonal_composite(geom, SEASON, BASE, ocfg).select("NDVI")
    target_ndvi = sentinel2.seasonal_composite(geom, SEASON, TARGET, ocfg).select("NDVI")
    greener = base_ndvi.subtract(target_ndvi).lte(0).rename("greener")
    truth = hansen.loss_mask(BASE, TARGET).unmask(0).gt(0).rename("loss")

    # Only forest pixels can ever be detected, so the population of interest is
    # forest. Sampling outside it spends points on a question never asked.
    stack = distance.addBands(greener).addBands(truth).updateMask(forest)

    for t in THRESHOLDS:
        # unmask(0) because the count is masked wherever the candidate is false,
        # and a masked band silently drops the whole sample row rather than
        # recording a zero — which would quietly delete every negative from the
        # sample and make precision meaningless.
        #
        # maxSize sits just above the largest min-patch tested; counting higher
        # costs real time per request and answers nothing the sweep asks.
        stack = stack.addBands(
            distance.gte(t)
            .selfMask()
            .connectedPixelCount(maxSize=max(MIN_PIXELS) + 24, eightConnected=True)
            .unmask(0)
            .rename(f"pc{int(t * 100):03d}")
        )
    return stack, forest, truth


def sample_tile(
    box: BoundingBox, stack: ee.Image, forest: ee.Image, truth: ee.Image, depth: int = 0
) -> list[Row]:
    """Sample one tile, subdividing on timeout rather than giving up.

    A timeout means the request was too big, and the only lever that reliably
    shrinks it is area. Four smaller requests cost the same in total but each
    fits inside the interactive budget. Depth is capped because past three
    levels the failure is no longer about size and retrying only burns quota.
    """
    pad = "  " * (depth + 1)
    # The tile restricts only the sampling REGION — the image is never clipped,
    # so connectedPixelCount still sees whole patches across tile borders
    # instead of truncating them at the seam.
    tile = ee.Geometry.Rectangle(box.as_ee_coords())
    hect = ee.Image.pixelArea().divide(10_000)

    try:
        areas = (
            hect.updateMask(forest)
            .rename("forest")
            .addBands(hect.updateMask(forest.And(truth)).rename("loss"))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=tile,
                crs="EPSG:32643",
                scale=SCALE,
                maxPixels=int(1e10),
                tileScale=16,
            )
            .getInfo()
        )
        forest_ha, loss_ha = areas["forest"], areas["loss"]
        if forest_ha <= 0:
            print(f"{pad}no forest, skipped", flush=True)
            return []

        # Ask for more loss points than the tile can hold; EE returns what
        # exists. Sizing this to the stratum is what fixed the original timeout.
        sample = stack.stratifiedSample(
            numPoints=0,
            classBand="loss",
            classValues=[1, 0],
            classPoints=[N_LOSS, N_STABLE],
            region=tile,
            scale=SCALE,
            projection="EPSG:32643",
            seed=7,
            geometries=False,
            tileScale=16,
        ).getInfo()
    except ee.ee_exception.EEException as exc:
        if depth >= MAX_SPLIT_DEPTH:
            print(f"{pad}GAVE UP: {str(exc)[:60]}", flush=True)
            return []
        print(f"{pad}{str(exc)[:40]} — splitting", flush=True)
        out: list[Row] = []
        for sub in subdivide(box, 2):
            out.extend(sample_tile(sub, stack, forest, truth, depth + 1))
        return out

    rows = [f["properties"] for f in sample["features"]]
    loss_rows = [r for r in rows if r["loss"] == 1]
    stable_rows = [r for r in rows if r["loss"] == 0]
    for r in loss_rows:
        r["w"] = loss_ha / len(loss_rows)
    for r in stable_rows:
        r["w"] = (forest_ha - loss_ha) / len(stable_rows)

    print(
        f"{pad}forest {forest_ha:8,.0f} ha, loss {loss_ha:6.2f} ha, "
        f"sampled {len(loss_rows):4d}/{len(stable_rows):5d}",
        flush=True,
    )
    return loss_rows + stable_rows


def collect() -> list[Row]:
    """Sample every tile, resuming from the checkpoint if one exists."""
    stack, forest, truth = build_stack()
    boxes = subdivide(YELLAPUR_TALUK.bbox, TILE_DIVISIONS)
    area = YELLAPUR_TALUK.bbox.area_sq_km

    print(f"AOI {area:,.0f} km², {BASE} vs {TARGET}, sampled at {SCALE} m")
    print(f"{len(boxes)} tiles of ~{area / len(boxes):.0f} km²\n")

    done: dict[str, list[Row]] = {}
    if CHECKPOINT.exists():
        done = json.loads(CHECKPOINT.read_text())
        print(f"resuming: {len(done)}/{len(boxes)} tiles already sampled\n")

    for n, box in enumerate(boxes, 1):
        if str(n) in done:
            continue
        print(f"tile {n:2d}/{len(boxes)}:", flush=True)
        done[str(n)] = sample_tile(box, stack, forest, truth)
        CHECKPOINT.write_text(json.dumps(done))

    return [r for tile_rows in done.values() for r in tile_rows]


def report(rows: list[Row]) -> None:
    loss_rows = [r for r in rows if r["loss"] == 1]
    stable_rows = [r for r in rows if r["loss"] == 0]
    if not loss_rows:
        sys.exit("no loss points sampled — nothing to tune against")

    loss_ha = sum(float(r["w"]) for r in loss_rows)
    stable_ha = sum(float(r["w"]) for r in stable_rows)
    weights = sorted(float(r["w"]) for r in stable_rows)

    print(f"\nforest {loss_ha + stable_ha:,.0f} ha, of which recorded loss {loss_ha:,.1f} ha")
    print(f"base rate: 1 loss pixel per {stable_ha / max(loss_ha, 1e-9):,.0f} stable")
    print(f"sample: {len(loss_rows):,} loss points, {len(stable_rows):,} stable points")
    print(f"each stable point stands for {weights[0]:.2f} to {weights[-1]:.2f} ha\n")

    header = f"{'thr':>5} {'min ha':>7} {'gate':>5} {'detected':>10} {'prec':>7}"
    print(f"{header} {'n_fp':>5} {'precision 95% CI':>18} {'recall':>7} {'F1':>7}")
    ranked: list[tuple[float, str]] = []

    for gate in (False, True):
        for t in THRESHOLDS:
            key = f"pc{int(t * 100):03d}"
            for k in MIN_PIXELS:

                def fires(r: Row, key: str = key, k: int = k, gate: bool = gate) -> bool:
                    if float(r[key]) < k:
                        return False
                    return not (gate and r["greener"] == 1)

                tp_ha = sum(float(r["w"]) for r in loss_rows if fires(r))
                fp = [float(r["w"]) for r in stable_rows if fires(r)]
                fp_ha, n = sum(fp), len(fp)
                mean_w = fp_ha / n if n else sum(weights) / len(weights)

                lo_k, hi_k = poisson_interval(n)
                # Convert the interval on COUNT into one on precision, holding
                # the true-positive estimate fixed: it rests on the whole
                # stratum and carries almost no sampling error of its own.
                p_hi = tp_ha / (tp_ha + lo_k * mean_w) if tp_ha else 0.0
                p_lo = tp_ha / (tp_ha + hi_k * mean_w) if tp_ha else 0.0
                p = tp_ha / (tp_ha + fp_ha) if (tp_ha + fp_ha) else 0.0
                rc = tp_ha / loss_ha
                f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0

                flag = "  <- no false positives sampled; not a measurement" if n == 0 else ""
                line = (
                    f"{t:5.2f} {k / 100:7.2f} {'yes' if gate else 'no':>5} "
                    f"{tp_ha + fp_ha:9,.1f}h {p:7.3f} {n:5d} "
                    f"{f'[{p_lo:.3f}, {p_hi:.3f}]':>18} {rc:7.3f} {f1:7.3f}"
                )
                print(line + flag)
                if n > 0:
                    ranked.append((f1, line))

    print("\nbest measurable configurations by F1 (rows with n_fp = 0 excluded,")
    print("because their precision is a resolution floor rather than a result):")
    for _, line in sorted(ranked, reverse=True)[:5]:
        print(f"  {line}")


if __name__ == "__main__":
    initialize()
    report(collect())
