"""Can radar work here at all? Measure separability and decide.

Run with ``python scripts/radar_separability.py``.

**Why this decides radar's fate.** The embedding detector is accurate but
annual, published months in arrears — a *record* of last year's clearing, not an
alert. Sentinel-1 is the only free path to sub-annual latency, which is the
project's actual stated goal. So radar is not a nice-to-have; it is the
difference between the system doing what it says on the front page and not.

But radar has never had its detection accuracy measured, only its terrain
correction (88% of the slope-backscatter correlation removed). Building a full
radar detector and *then* discovering it cannot work would waste weeks.

**Separability answers it cheaply.** Recorded loss is 10.7 ha in 106,543 ha of
forest — one pixel in 9,939. At that base rate the binding constraint is not
threshold choice, it is whether the two populations overlap at all. Cohen's d
over the same AOI already told us:

    NDVI difference        d = 1.36   ->  unusable, precision 0.012-0.32
    Embedding L2 distance  d = 2.20   ->  usable,   precision 0.80

That is the calibration. A radar d near 1.4 means radar cannot carry a detector
here no matter how it is tuned, and the honest move is to say so and drop it. A
d near or above 2.2 means it can, and the latency goal is reachable.

Measured on VH backscatter, which responds to canopy volume scattering and is
the polarisation RADD uses. Both a dry-season window (to avoid soil moisture
confounding) and a full-year window are reported, because if radar only
separates in the dry season that is itself the answer: the monsoon months are
exactly when optical goes blind and radar was supposed to cover.
"""

from __future__ import annotations

import math

import ee

from vanachakshu import hansen, sentinel1
from vanachakshu.config import (
    YELLAPUR_TALUK,
    BoundingBox,
    OpticalDetectionConfig,
    RadarDetectionConfig,
)
from vanachakshu.gee import initialize

BASE, TARGET = 2024, 2025
N_LOSS, N_STABLE = 300, 800
TILE_DIVISIONS = 3

WINDOWS = {
    # Dry season only: December-March, when the ground is not saturated.
    "dry season (Dec-Mar)": (f"{BASE}-12-01", f"{BASE + 1}-03-31"),
    # Everything, monsoon included - the window radar exists to cover.
    "full year": (f"{TARGET}-01-01", f"{TARGET}-12-31"),
}
BASELINE = {
    "dry season (Dec-Mar)": (f"{BASE - 1}-12-01", f"{BASE}-03-31"),
    "full year": (f"{BASE}-01-01", f"{BASE}-12-31"),
}


def cohens_d(a_mean: float, a_var: float, b_mean: float, b_var: float) -> float:
    """Standardised difference between two group means.

    Pooled rather than raw because the groups have wildly different sizes and
    typically different spread; the raw difference in dB would not be comparable
    to the NDVI and embedding figures this is being judged against.
    """
    pooled = math.sqrt((a_var + b_var) / 2.0) if (a_var + b_var) > 0 else 0.0
    return abs(a_mean - b_mean) / pooled if pooled else 0.0


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


def main() -> None:
    initialize()
    ocfg = OpticalDetectionConfig()
    rcfg = RadarDetectionConfig()
    geom = ee.Geometry.Rectangle(YELLAPUR_TALUK.bbox.as_ee_coords())

    forest = hansen.forest_mask(BASE, ocfg)
    truth = hansen.loss_mask(BASE, TARGET).unmask(0).gt(0).rename("loss")

    print(f"AOI {YELLAPUR_TALUK.bbox.area_sq_km:,.0f} km2, {BASE} vs {TARGET}, VH backscatter")
    print(f"orbit {rcfg.orbit_pass}\n")
    print("Calibration from the same AOI: NDVI d=1.36 (unusable), embeddings d=2.20 (usable)\n")

    for label, (start, end) in WINDOWS.items():
        b_start, b_end = BASELINE[label]
        try:
            base = sentinel1.collection(geom, b_start, b_end, rcfg).select("VH")
            recent = sentinel1.collection(geom, start, end, rcfg).select("VH")
            n_base = base.size().getInfo()
            n_recent = recent.size().getInfo()
        except ee.ee_exception.EEException as exc:
            print(f"{label}: FAILED building collection - {str(exc)[:70]}")
            continue

        if not n_base or not n_recent:
            print(f"{label}: no scenes ({n_base} baseline, {n_recent} recent)")
            continue

        # Median over the window, not a single pass: one date's backscatter is
        # dominated by that day's soil moisture, which is the whole reason
        # single-date radar change detection produces false positives.
        drop = base.median().subtract(recent.median()).rename("drop").updateMask(forest)
        stack = drop.addBands(truth)

        loss_vals: list[float] = []
        stable_vals: list[float] = []
        for box in subdivide(YELLAPUR_TALUK.bbox, TILE_DIVISIONS):
            try:
                sample = stack.stratifiedSample(
                    numPoints=0,
                    classBand="loss",
                    classValues=[1, 0],
                    classPoints=[N_LOSS, N_STABLE],
                    region=ee.Geometry.Rectangle(box.as_ee_coords()),
                    scale=10,
                    projection="EPSG:32643",
                    seed=7,
                    geometries=False,
                    tileScale=16,
                ).getInfo()
            except ee.ee_exception.EEException as exc:
                print(f"  tile failed: {str(exc)[:50]}", flush=True)
                continue
            for f in sample["features"]:
                p = f["properties"]
                if p.get("drop") is None:
                    continue
                (loss_vals if p["loss"] == 1 else stable_vals).append(float(p["drop"]))

        if len(loss_vals) < 10 or len(stable_vals) < 10:
            print(f"{label}: too few samples ({len(loss_vals)} loss, {len(stable_vals)} stable)")
            continue

        def stats(v: list[float]) -> tuple[float, float]:
            m = sum(v) / len(v)
            return m, sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)

        lm, lv = stats(loss_vals)
        sm, sv = stats(stable_vals)
        d = cohens_d(lm, lv, sm, sv)

        verdict = (
            "usable" if d >= 2.0 else "marginal" if d >= 1.6 else "NOT usable at this base rate"
        )
        print(f"{label}:  {n_base} + {n_recent} scenes")
        print(f"  loss   n={len(loss_vals):4d}  mean drop {lm:+.3f} dB  sd {math.sqrt(lv):.3f}")
        print(f"  stable n={len(stable_vals):4d}  mean drop {sm:+.3f} dB  sd {math.sqrt(sv):.3f}")
        print(f"  Cohen's d = {d:.2f}  ->  {verdict}\n", flush=True)


if __name__ == "__main__":
    main()
