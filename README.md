# Vanachakshu

**Near-real-time forest disturbance alerts for the Western Ghats, India.**

> ⚠️ **Early development.** Nothing here is validated yet. Do not treat any output as
> evidence of illegal activity. See [Responsible use](#responsible-use).

---

## Why this exists

India currently has no active near-real-time deforestation alert system.

- **RADD**, the global standard for radar-based forest alerts, covers the Amazon, Congo
  Basin and insular Southeast Asia. India is not covered.
- **Anavaran**, the Forest Survey of India's AI-based deforestation alert portal, issued
  12,351 alerts between January 2024 and October 2025, then stopped. It has not been
  updated since November 2025 and monitoring ended in January 2026. It is under review,
  not formally cancelled.
- Only *Van Agni*, the forest **fire** alert portal, remains operational.

This project ports a published, validated method (Sentinel-1 radar backscatter change
detection with multi-pass confirmation) to a region that no operational system covers.

## What it does

Ingests Sentinel-2 optical and Sentinel-1 radar imagery over a monitored area, detects
sustained canopy loss, and delivers actionable alerts — location with nearest landmark,
area in hectares, date window, confidence, and a before/after image chip.

Radar matters because the Western Ghats is under monsoon cloud for months at a time. An
optical-only system goes blind exactly when clearing activity peaks.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, config, CI, Earth Engine access | ✅ complete |
| 1 | Sentinel-2 optical baseline, scored against Hansen | ✅ complete — superseded |
| 2 | Scheduled service: cron, alerts, reports | ✅ running unattended |
| 3 | Sentinel-1 radar detector | 🟡 terrain correction verified (88%); detection accuracy unmeasured |
| 4 | Human validation against sub-metre imagery | ✅ two rounds, n=13 and n=19 |
| 5 | AlphaEarth embedding detector | ✅ live — replaced the optical baseline |
| 6 | Public map, write-up | ⬜ not started |

## How it detects

The detector compares **AlphaEarth satellite embeddings** — Google's free
annual dataset where each 10 m pixel carries 64 numbers summarising a whole
year of fused Sentinel-1 radar and Sentinel-2 optical observation. A pixel that
changed land cover moves a long way in that space; one that merely had a dry
year does not.

This replaced NDVI differencing after measurement showed why NDVI could not
work here. Hansen records 28.6 ha of loss in 146,300 ha — **one loss pixel per
five thousand stable ones**. At that base rate, separability is the only lever
that matters, and NDVI's (Cohen's d = 1.36) is not enough for any threshold to
find a usable operating point. Embeddings score 2.20.

Two guards sit on top:

- **Was it forest?** Hansen's forest mask, or every land-cover change qualifies.
- **Did it get less green?** Embedding distance measures *how far* a pixel
  moved, not *which way* — so forest growing back looks identical to forest
  being cleared. NDVI supplies only the sign. Human validation found five of
  nine false positives were regrowth.

## Human-validated accuracy

**Precision 0.53, 95% CI [0.32–0.73], n=19** (2 unclear) — every detection
judged by eye against sub-metre imagery, 2024 vs 2025.

| Detector | Precision | 95% CI | n |
|---|---|---|---|
| NDVI differencing | 0.38 | [0.18–0.64] | 13 |
| **AlphaEarth embeddings** | **0.53** | **[0.32–0.73]** | **19** |

The interval matters more than the point estimate. At n=19 the true precision
could plausibly sit anywhere from 0.32 to 0.73, and that range is reported
rather than hidden because a single figure from nineteen samples would imply a
confidence the sample cannot support.

This figure is **independent of Hansen** — a human looking at pictures, not one
algorithm agreeing with another. The direction gate was added *after* this
validation, in response to it, and its measured effect is reported separately.

Reproduce with `vanachakshu validate-sample --seed 7`, then `validate-chips`,
then `validate-report`. The seed is in the filename so the sample can be redrawn
and audited.

## Known limits

**Detections lag by about a year.** AlphaEarth is published annually and the
newest year appears months after it ends, so today the freshest comparable pair
is 2024 vs 2025. A system reporting last year's clearing is a record, not an
alert — the "near-real-time" goal is not met by this detector, and the radar
path exists partly to address that.

**Accuracy depends on the scale it is scored at.** The median detection is
0.116 ha, roughly one 30 m pixel, so scoring against Hansen at 30 m discards
most of the output and flatters precision. Against Hansen the same detector
scores 0.583 at 30 m and 0.343 at 10 m. Any number from this project should
state its scale and its reference.

**Recall is low and stays low.** At the tuned operating point the detector finds
about a third of recorded loss. That is a deliberate trade, not an oversight —
see below.

## Tuned accuracy against Hansen, at 10 m

Parameters were swept over the full 1,463 km² AOI, 2024 vs 2025, using a
weighted stratified sample at the pipeline's own 10 m resolution.

| threshold | min patch | precision | 95% CI | recall |
|---|---|---|---|---|
| 0.35 | 0.05 ha | 0.012 | [0.011, 0.014] | 0.648 |
| 0.40 | 0.05 ha | 0.168 | [0.101, 0.288] | 0.504 |
| 0.45 | 0.05 ha | 0.319 | [0.154, 0.635] | 0.347 |
| **0.45** | **0.20 ha** | **0.803** | **[0.531, 0.973]** | **0.320** |
| 0.50 | 0.05 ha | *1.000* | *[0.203, 1.000]* | 0.238 |

**The 1.000 in that last row is a measurement failure, not a result.** Every
configuration above threshold 0.45 reports perfect precision because *zero*
false-positive samples survived — there is nothing to divide by. Its interval
runs down to 0.20. No number from this project quotes precision 1.000.

The table above is measured with the direction gate **off**. The deployed
configuration — threshold 0.45, 0.20 ha, gate **on** — falls into exactly the
unmeasurable region: no false positive survived into the sample, so its
precision cannot be resolved beyond `[0.238, 1.000]`. What can be said is that
the gate only ever *removes* detections, so deployed precision is at least the
0.803 measured without it, at a recall of 0.291 instead of 0.320.

Recall and precision rest on very different footing. The sample holds 1,071
loss points against a stratum of roughly 1,070 pixels, so **recall is measured,
not extrapolated**. Precision extrapolates false-positive area from stable
points each standing for up to 3.85 ha, which is why its intervals are wide.

**The direction gate roughly doubles precision and costs about 16% of recall**
— 0.012 → 0.026 at threshold 0.35, 0.168 → 0.229 at 0.40, the two levels with
enough false positives to measure. An earlier 30 m measurement showed the same
gain at *zero* recall cost; that was a scale artifact and is retracted. The
gate is not free, and is kept because precision is worth more here.

**Why precision is bought at recall's expense.** Recorded loss is 10.7 ha in
106,543 ha of forest — one loss pixel per 9,939 stable. At that base rate a
detector wrong on 0.1% of stable forest still buries every true detection ten
to one. Precision is the only property that makes the output usable; recall is
what it is bought with.

## Earlier accuracy against Hansen

**Optical baseline: precision 0.38, recall 0.34** against Hansen v1.13 over Yellapur
Taluk, 2021 vs 2025, with a 60 m geolocation tolerance.

That is **not deployment-grade**, and it is reported as-is on purpose. The optical
detector is a measuring stick — the number the radar detector must beat — not a
product. Full method, threshold sweep and caveats:
[docs/findings/2026-08-phase1-optical-baseline.md](docs/findings/2026-08-phase1-optical-baseline.md).

Two things that finding establishes:

- A **shorter gap between compared years does not help** (one-year gaps score no better
  than four-year), so regrowth is not the dominant limitation.
- **Roughly a quarter of the apparent failure is geolocation mismatch** between 10 m
  Sentinel-2 and 30 m Hansen. Recall nearly quadruples with a 60 m tolerance. Any
  accuracy number from this project must state its tolerance and its reference.

## Target accuracy

RADD reports 97.6% user's / 95.0% producer's accuracy for disturbances ≥0.2 ha — from a
funded team, over years, in **flat** humid tropics.

**This project targets 70–85% precision on clearings ≥0.5 ha** from the radar detector,
and expects worse below that. The Western Ghats is harder: steep terrain distorts radar
backscatter, clearings are small and fragmented, and 2000–7000 mm of monsoon rain drives
soil-moisture false positives. Those numbers will be published honestly, stratified by
clearing size, once Phase 4 validation is done.

Note that Hansen is itself a model, not ground truth. Agreeing with it means agreeing
with another algorithm — which is why Phase 4 validates against imagery instead.

## Architecture

The package is deliberately split so that **pure logic never imports Earth Engine**:

```
src/vanachakshu/
  config.py     validated settings — no I/O, no network
  geometry.py   area and projection maths — pure functions
  gee.py        the only module that talks to Earth Engine
```

This is not stylistic. Earth Engine requires Google credentials, which CI does not have
and should not have. Keeping detection thresholds, area arithmetic, date-window logic and
alert deduplication in credential-free modules means the majority of the test suite runs
on every push, with no secrets and no quota burn.

Tests that genuinely need Earth Engine are marked `@pytest.mark.ee` and excluded from CI.

## Setup

Requires Python 3.11+.

```bash
git clone <your-repo-url> && cd vanachakshu
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Optional heavy extras for local raster work (pulls GDAL-adjacent wheels — slow, and not
needed for the scheduled job):

```bash
pip install -e ".[analysis]"
```

### Earth Engine access

Earth Engine is free for noncommercial use, but **metered since 27 April 2026**.

1. Register at [earthengine.google.com/noncommercial](https://earthengine.google.com/noncommercial/)
   and create a Google Cloud project.
2. **Choose the Contributor tier, not Community.** Community gives 150 EECU-hours/month;
   Contributor gives **1,000 EECU-hours/month and is equally free** — it only asks you to
   attach a billing account for verification. You are not charged for noncommercial use.
   Dense radar time series burn through 150 hours quickly.
3. Authenticate and point the project at your config:

```bash
earthengine authenticate
echo "VANACHAKSHU_EE_PROJECT=your-project-id" > .env
```

`.env` is gitignored. Never commit credentials — an Earth Engine service-account key is
plain JSON and looks harmless, but grants anyone your full Google Cloud quota.

## Development

```bash
pytest                  # full suite
pytest -m "not ee"      # what CI runs — no credentials needed
ruff check .            # lint
ruff format .           # format
mypy                    # strict type checking
```

## Responsible use

These design rules are binding, not aspirational.

1. **Output is always framed as "possible forest disturbance — requires ground
   verification."** Never "illegal logging detected." A model output is not evidence of a
   crime, and presenting it as such can cause real harm to real people.
2. **Precision is optimised over recall.** A forest officer who receives 40 useless alerts
   out of 50 stops opening them permanently. Trust is the scarce resource.
3. **Lawful traditional land use is not deforestation.** In Northeast India, *jhum*
   (shifting cultivation) is legal, cyclical, and spectrally near-identical to clearing.
   Flagging it would be factually wrong and harmful to tribal communities. This is a
   primary reason the initial AOI is the Western Ghats rather than the Northeast — and
   even there, multi-year history is used to separate cyclical regrowth from permanent
   conversion.
4. **Known blind spots are published, not hidden.** Radar layover and shadow zones in
   steep terrain are permanent gaps. The map states where the system cannot see, rather
   than silently reporting no alerts there.

## Data sources

All free and openly licensed.

| Source | Use |
|---|---|
| Sentinel-1 GRD (IW) | Primary detector — cloud-penetrating radar, 6-day revisit |
| Sentinel-2 L2A | Optical baseline, before/after chips |
| [Vollrath et al. 2020 slope correction](https://github.com/ESA-PhiLab/radiometric-slope-correction) | Radiometric terrain correction, Alps-validated |
| Hansen Global Forest Change v1.13 | Training labels and forest mask |
| ESA WorldCover 10 m | Land cover mask |
| NICFI Planet basemaps (<5 m) | Independent manual validation |
| CHIRPS | Rainfall, for separating drought from degradation |

## Licence

MIT. See `LICENSE`.
