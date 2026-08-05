# Phase 3 finding: terrain correction works, the detector does not yet

**Date:** 2026-08-05
**AOI:** central quarter of Yellapur Taluk, 366 km² (reduced to fit Earth Engine's memory ceiling)
**Sensor:** Sentinel-1 GRD, IW, descending passes only
**Reference:** Hansen Global Forest Change v1.13, `lossyear`

---

## Headline

Two results, and they point in opposite directions.

**The terrain correction works and is verified.** It removes 88% of the terrain
distortion in VV and 78% in VH — measured, not asserted.

**The change detector does not.** At its most productive setting it produces
50.2 ha of spatially coherent detections in a quarter, of which **0.0 ha
coincide with Hansen's record of forest loss.** Precision, as measured here, is
zero.

Whether that means the detector is wrong or the reference is incomplete is
**not resolvable with the data used here**, for reasons set out below.

---

## 1. Terrain correction: verified

Correlation of backscatter with the *signed* slope in the radar look direction.
That correlation is the distortion, so collapsing it is the goal.

| Polarisation | Uncorrected | Corrected | Reduction |
|---|---|---|---|
| VV | +0.4556 | −0.0550 | **88%** |
| VH | +0.4537 | −0.0980 | **78%** |

Pinned by a credentialed test that asserts the distortion **exists before**
correction as well as vanishing after — a correction applied to flat terrain
would otherwise "pass" while proving nothing.

## 2. The threshold was inside the noise floor

Stage-by-stage at the originally shipped 2.5 dB, over a full year:

| Stage | Area |
|---|---|
| All pixels in AOI | 36,449 ha |
| Dropped ≥2.5 dB on **any** pass | 35,964 ha — **98.7%** |
| Dropped on ≥2 passes (any) | 34,267 ha |
| Dropped on the **last** 2 passes | 965 ha |
| + Hansen says forest | 743 ha |
| + connected patch ≥0.5 ha | **0.0 ha** |

2.5 dB is exceeded by ordinary speckle and soil moisture almost everywhere. What
survived to the final stage was scattered single pixels, which is exactly why
the patch-size filter removed all of it. **That filter was not the problem — it
was the only thing preventing 743 ha of noise being reported as deforestation.**

Raising the threshold did not rescue it: 743 ha at 2.5 dB → 9 ha at 3.0 → 0 at
4.0. A falloff that steep is the signature of a noise distribution; real
sustained clearing would leave a tail at higher values. Heavier speckle
filtering (90 m) was strictly worse, smoothing genuine drops away too.

## 3. A semantic bug: "disturbed now" vs "disturbed during"

Persistence was implemented as *the last N passes of the window all dropped*.
That asks **"is this pixel disturbed now?"** — correct for a live alert, wrong
for scoring against an annual label, which asks **"did a disturbance occur
during 2025?"** A clearing in March only counted if its backscatter was still
suppressed the following December, nine months and a monsoon later.

Replaced with a sliding window: any N consecutive passes anywhere in the window.
Window *length* then supplies recency, so a live monitor watching three weeks
gets "disturbed now" from the same code.

**The fix demonstrably helped coherence.** Under the old rule the detector
produced 0.0 ha of coherent patches at every threshold. Under the sliding
window, at 2.5 dB it produces 50.2 ha.

## 4. Sliding-window sweep

90-day monitoring window (Jan–Mar 2025), 180-day baseline, 366 km².

| Drop (dB) | Any-run | Coherent patches | Overlap with Hansen | Precision |
|---|---|---|---|---|
| 2.5 | 686.4 ha | **50.2 ha** | 0.0 ha | 0.000 |
| 3.5 | 33.4 ha | 0.0 ha | 0.0 ha | 0.000 |
| 4.5 | 2.4 ha | 0.0 ha | 0.0 ha | 0.000 |
| 6.0 | 0.0 ha | 0.0 ha | 0.0 ha | 0.000 |

The consecutive-run requirement is doing real work: it cuts 98.7% of the
landscape down to 1.9%. But the surviving patches do not coincide with Hansen.

## 5. Why this measurement cannot settle the question

Hansen records **10.5 ha** of 2025 loss in this 366 km² AOI — a base rate of
**0.03%**. Detecting 0 of 10.5 ha is consistent with several very different
explanations, and this experiment cannot separate them:

1. **The detections are false positives.** Plausible.
2. **They are real disturbances Hansen missed.** Hansen is a 30 m annual
   product with its own detection floor; sub-hectare and selective clearing can
   be absent from it entirely.
3. **They are plantation cycles.** Areca, rubber and cashew harvest look like
   clearing to radar but are not recorded by Hansen as *forest* loss. In Uttara
   Kannada this is a large confounder.
4. **Hansen's 10.5 ha sits where radar cannot see** — layover or shadow.
5. **The AOI is simply too small.** With 10.5 ha of truth, a handful of pixels
   either way swings precision between 0 and something respectable.

**Any tuning done against a reference this sparse would be fitting noise.**

## What this justifies

- **Scoring must move to the full AOI**, which requires Earth Engine batch
  `Export` rather than `getInfo`. The memory ceiling was hit four times during
  this work; shrinking the question to fit it has reached the point of
  destroying the measurement.
- **Phase 4 validation is now the blocking item, not an optional polish step.**
  Until several hundred detections are checked by eye against <5 m imagery,
  there is no way to tell explanation 1 from explanations 2 and 3 — and
  therefore no defensible way to choose a threshold.
- **A fixed decibel threshold is probably the wrong instrument.** RADD uses
  probabilistic updating against each pixel's own backscatter distribution
  rather than one global cutoff. A per-pixel statistical threshold
  (drop relative to that pixel's baseline variability) is the obvious next
  design, and is what the evidence here points toward.

## Honest summary

The infrastructure is sound and the physics step is verified. The detector is
not yet usable, and the reference data available cannot currently tell us how
far off it is. Both facts are reported as they stand.
