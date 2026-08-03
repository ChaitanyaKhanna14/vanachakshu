# Phase 1 finding: the optical baseline is weak, and that is the point

**Date:** 2026-08-03
**AOI:** Yellapur Taluk bounding box, Uttara Kannada, Karnataka (1,463 km²)
**Season:** January–March (clear post-monsoon window), same months every year
**Reference:** Hansen Global Forest Change v1.13, `lossyear`
**Detector:** bi-temporal Sentinel-2 NDVI differencing

---

## Headline

At the chosen operating point the optical baseline scores roughly
**precision 0.38, recall 0.34** against Hansen, with a 60 m geolocation tolerance.

That is not deployment-grade. It is a **measuring stick** — the number every later
method must beat, and the reason the project moves to Sentinel-1 radar in Phase 3.

Reporting it honestly is the point. A baseline tuned to look good is worthless as a
comparison.

---

## 1. Threshold sweep

2020 vs 2025, pixel-level, no minimum patch size, exact pixel overlap.
Hansen records **233.8 ha** of loss over the period.

| NDVI drop threshold | Detected | True positive | Precision | Recall | IoU |
|---|---|---|---|---|---|
| 0.15 | 4,064.1 ha | 36.3 ha | 0.009 | 0.155 | 0.009 |
| 0.25 | 81.3 ha | 21.6 ha | 0.266 | 0.092 | 0.074 |
| 0.35 | 21.6 ha | 11.9 ha | 0.553 | 0.051 | 0.049 |
| 0.45 | 8.8 ha | 6.6 ha | 0.758 | 0.028 | 0.028 |

A textbook precision/recall trade-off, and an unusually steep one. At 0.15 the
detector flags **4,064 ha** — seventeen times more than all recorded loss — which is
what "detecting everything" looks like.

## 2. Does a shorter gap between years help?

Hypothesis: in a region receiving 3,000+ mm of rain, clearings regrow, so a long gap
should miss them. Threshold 0.25, exact overlap.

| Pair | Gap | Precision | Recall | IoU |
|---|---|---|---|---|
| 2021→2022 | 1 y | 0.105 | 0.155 | 0.067 |
| 2022→2023 | 1 y | 0.043 | 0.045 | 0.022 |
| 2023→2024 | 1 y | 0.176 | 0.095 | 0.066 |
| 2024→2025 | 1 y | 0.249 | 0.127 | 0.092 |
| 2021→2025 | 4 y | 0.274 | 0.088 | 0.071 |

**Hypothesis rejected.** One-year gaps do not outperform four-year gaps. Regrowth is
not the dominant limitation; the method is simply weak in this landscape.

## 3. How much is geolocation mismatch?

Our detections are 10 m Sentinel-2. Hansen is 30 m Landsat with its own geolocation
error. Scoring pixel-exact across that gap penalises correct detections that land one
pixel off. A spatial tolerance is standard practice in change-detection validation.

2021 vs 2025:

| Threshold | Tolerance | Precision | Recall |
|---|---|---|---|
| 0.25 | 0 m | 0.274 | 0.088 |
| 0.25 | **60 m** | **0.383** | **0.340** |
| 0.35 | 0 m | 0.448 | 0.045 |
| 0.35 | 60 m | 0.539 | 0.206 |

Recall nearly **quadruples** with a 60 m tolerance. A large share of the apparent
failure was alignment, not detection. Any number quoted from this pipeline must state
its tolerance, or it is not comparable to anything.

## 4. Patch-level view

2020 vs 2025 with the full pipeline (0.5 ha minimum patch size): **142 patches,
153.8 ha**. Sampling Hansen inside each:

| Hansen says | Patches |
|---|---|
| No loss recorded | 131 (92%) |
| Loss in 2021–2025 | 11 (8%) |

---

## Chosen operating point

`ndvi_drop_threshold = 0.25`, giving the best IoU in the sweep and the best F1 under
tolerance (≈0.36 versus ≈0.30 at 0.35).

The project's design rules prefer precision over recall. 0.35 would deliver precision
0.539 instead of 0.383 — but recall collapses to 0.206, and at that point the system
misses four clearings in five. **Neither point is good enough to send to a forest
officer.** That is the honest reading, and it is why the optical detector is scoped as
a baseline rather than a product.

---

## Why the score is low: candidate explanations

Not yet separated. Listed in rough order of suspected contribution.

1. **Geolocation mismatch** — measured, and large. See section 3.
2. **Hansen is not ground truth.** It is a 30 m model with its own detection floor;
   small clearings may be absent from it entirely. Some of our "false positives" may be
   real clearings Hansen missed. Only Phase 4 validation against <5 m NICFI imagery can
   separate this.
3. **Plantation cycles.** Areca, rubber and cashew harvest-and-replant looks
   spectrally like clearing, but Hansen does not record it as forest loss. In Uttara
   Kannada this is a large confounder.
4. **Partial and selective clearing.** Removing scattered trees barely moves NDVI once
   the canopy closes over.
5. **Residual terrain effects** on steep slopes, beyond what shadow masking removes.

## What this justifies

- **Radar (Phase 3) is not a nice-to-have.** The baseline is too weak to deploy, which
  is precisely the evidence needed to justify the harder approach.
- **Phase 4 validation against high-resolution imagery is mandatory**, because we
  currently cannot tell our errors from Hansen's.
- **Every future number must state its tolerance and its reference.**

## Reproducing

Composites via `sentinel2.seasonal_composite`, masks via `hansen.forest_mask` /
`hansen.loss_mask`, scoring via `score.score_detection`. All sweeps ran at
`crs="EPSG:32643"`, `scale=20`.
