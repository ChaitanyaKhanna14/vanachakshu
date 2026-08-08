# Phase 5 finding: the base rate was always the problem, and scale was hiding it

**Date:** 2026-08-09
**AOI:** Yellapur Taluk bounding box, Uttara Kannada, Karnataka (1,463 km²)
**Years:** 2024 vs 2025
**Reference:** Hansen Global Forest Change v1.13, `lossyear`
**Detector:** AlphaEarth embedding L2 distance + Hansen forest mask + NDVI direction gate

---

## Headline

Tuned at the pipeline's own 10 m over the full AOI:

**precision 0.803, 95% CI [0.531, 0.973] — recall 0.320**

up from 0.309 precision at the previous settings. The entire gain came from one
parameter: minimum connected patch size, 0.05 ha → 0.20 ha.

Two figures previously reported in this repository were wrong and are retracted
below. Both were wrong the same way.

---

## 1. The base rate, measured rather than estimated

| | |
|---|---|
| Forest in AOI | 106,543 ha |
| Recorded loss, 2024→2025 | 10.7 ha |
| Ratio | **1 loss pixel per 9,939 stable** |

Earlier work assumed 1 in 5,000, extrapolated at 30 m. The true figure is twice
as harsh.

This single number explains almost everything else about the project. A detector
that is wrong on 0.1% of stable forest produces 106 ha of false positives
against 10.7 ha of real loss — a 10:1 garbage ratio from a method that is
99.9% correct. **Separability, not threshold choice, is the binding constraint.**

It is also why precision is bought with recall throughout. At this base rate
precision is the property that makes output usable at all; recall is the
currency it is paid for in.

---

## 2. The sweep

Weighted stratified sample, 10 m, full AOI, direction gate off.

| threshold | min patch | precision | 95% CI | recall | n_fp |
|---|---|---|---|---|---|
| 0.35 | 0.05 ha | 0.012 | [0.011, 0.014] | 0.648 | 193 |
| 0.40 | 0.05 ha | 0.168 | [0.101, 0.288] | 0.504 | 11 |
| 0.45 | 0.05 ha | 0.319 | [0.154, 0.635] | 0.347 | 4 |
| **0.45** | **0.20 ha** | **0.803** | **[0.531, 0.973]** | **0.320** | **2** |
| 0.45 | 0.40 ha | 0.956 | [0.797, 0.999] | 0.196 | 1 |
| 0.50 | 0.05 ha | *1.000* | *[0.203, 1.000]* | 0.238 | **0** |

**The `n_fp` column is the most important one here.** It is the number of
false-positive sample points each precision estimate rests on.

Every configuration above threshold 0.45 reports precision 1.000 — not because
the detector is perfect, but because *zero* false positives survived into the
sample. There is nothing to divide by. The interval on those rows runs down to
0.20. Without that column those rows would have been read as the best settings
found and quoted as a perfect score.

`scripts/tune_detector.py` now labels them `not a measurement` and excludes them
from ranking.

### Why patch size and not threshold

Real clearings are contiguous; embedding noise is speckle. Requiring 20
connected pixels removes the second and keeps the first. At threshold 0.45,
detected area falls 11.7 ha → 4.3 ha while recall falls only 0.347 → 0.320 —
**63% of the detected area was discarded and it cost 8% of the recall**, because
almost all of it was noise.

RADD, the global standard system this project is modelled on, independently
reports 0.2 ha as its own accuracy cliff. Arriving at the same figure by a
different method on a different continent is weak corroboration, but it is not
nothing.

### Recall and precision do not rest on the same footing

The loss stratum holds ~1,070 pixels; the sample contains 1,071 loss points.
Essentially every recorded-loss pixel in the taluk is in the sample, so **recall
is measured, not extrapolated**, and carries almost no sampling error.

Precision extrapolates false-positive *area* from stable points each standing
for up to 3.85 ha. That is why its intervals are wide wherever detections are
sparse, and why they must be quoted.

---

## 3. Two retractions, both caused by scoring at the wrong scale

**Retracted: precision 0.773.** From a 30 m run. The median detection is
0.116 ha — roughly a single 30 m pixel — so 30 m sampling discards most of what
this detector emits. What survives is disproportionately correct, which flatters
precision and penalises recall.

**Retracted: "at 0.2 ha every embedding configuration collapses to zero
detections."** Also a 30 m artifact. At 10 m, 0.2 ha turned out to be the single
largest precision lever available. This one was worse than a wrong number: it
was a wrong number that closed off the correct answer for weeks.

**Also corrected: the direction gate is not free.** A 30 m measurement showed it
raising precision 0.583 → 1.000 with recall unchanged at 0.205, and that was
reported as a filter that removes only errors. At 10 m:

| threshold | precision without gate | with gate | recall without | with |
|---|---|---|---|---|
| 0.35 | 0.012 | 0.026 | 0.648 | 0.542 |
| 0.40 | 0.168 | 0.229 | 0.504 | 0.422 |

It roughly doubles precision and costs about **16% of recall**. Still worth its
price under a precision-first operating point — but it has a price.

**The lesson, stated plainly:** score at the resolution the system actually
outputs. Scoring against a 30 m reference is the obvious thing to do when the
reference is 30 m, and it was wrong three separate times in three different
directions.

---

## 4. What it took to measure this at all

Interactive Earth Engine calls could not complete a 10 m scoring run over this
AOI — six timeouts, each previously "fixed" by shrinking the question, which is
how the detector ended up tuned against 30 m numbers in the first place.

What worked was sampling rather than reducing, with one trick that made the
sweep nearly free: **sample `connectedPixelCount` at each candidate threshold as
its own band.** Patch size is the only parameter that cannot be swept
client-side from point values, so it is computed server-side once per candidate
in the same request. After that both parameters sweep in memory.

Two failures shaped the final script:

- Asking for 6,000 loss points from a stratum containing ~1,070 made Earth
  Engine scan all 14.6M pixels hunting for points that were never there. Sample
  counts must be sized to the stratum.
- One tile timing out discarded eight tiles of finished work. Results are now
  checkpointed per tile, and a tile that times out subdivides rather than
  aborting.

---

## 5. What this does and does not establish

**Does:** at these settings, on this taluk, for this year-pair, roughly four in
five detections correspond to Hansen-recorded loss, and roughly one in three
recorded losses is found.

**Does not:**

- Hansen is a model, not ground truth. Agreeing with it means agreeing with
  another algorithm. The independent check is the human validation against
  sub-metre imagery (precision 0.53 [0.32–0.73], n=19) — a lower figure on an
  untuned configuration, and the one to trust where they disagree.
- One taluk, one year-pair. Thresholds encode this landscape's base rate and
  vegetation. Nothing here should be assumed to transfer.
- Precision at the *deployed* configuration (gate on) is unmeasurable by this
  sample — no false positive survived. It is at least the 0.803 measured without
  the gate, since the gate only removes detections, but the sample cannot say
  more.

---

## 6. The limit this does not fix

AlphaEarth is annual and published months after the year it covers. **The
freshest comparable pair today is 2024 vs 2025.**

A system reporting last year's clearing is a record, not an alert. The project's
stated goal is near-real-time, and this detector does not meet it at any
accuracy. Sentinel-1 remains the only free path to sub-annual latency, which is
why `scripts/radar_separability.py` exists: it measures whether radar has enough
separability to work here *before* a detector is built on it, using the
d = 1.36 (unusable) / d = 2.20 (usable) calibration established on this same AOI.
