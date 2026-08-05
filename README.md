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
| 1 | Sentinel-2 optical baseline, scored against Hansen | ✅ complete |
| 2 | Scheduled service: cron, alerts, reports | ✅ running unattended |
| 3 | Sentinel-1 radar detector | 🟡 terrain correction verified; detector not yet usable |
| 4 | Manual validation against <5 m NICFI imagery | ⬜ **now the blocking item** |
| 5 | Documentation, runbook, write-up | ⬜ not started |

## Known limitations, stated up front

**Neither detector currently produces usable output.** Both failures are
understood and documented rather than hidden.

- **Optical** was tuned on a four-year gap (precision 0.38 / recall 0.34) but the
  scheduled monitor compares one year to the next, where the same settings detect
  **0.0 ha**. Re-tuning on the deployed regime is outstanding.
- **Radar**: the terrain correction is verified (88% of distortion removed), but
  the change detector's threshold sat inside the noise floor. See
  [the Phase 3 finding](docs/findings/2026-08-phase3-radar-detector.md).
- **The reference data cannot currently arbitrate.** Hansen records 10.5 ha of
  loss in the 366 km² test area — a 0.03% base rate — and is a 30 m annual
  product that misses sub-hectare and selective clearing. Tuning against a
  reference that sparse would be fitting noise, which is why Phase 4 validation
  against <5 m imagery is now the blocking item rather than a polish step.

The infrastructure around the detectors is sound: unattended operation,
notify-once guarantees, an audit trail, and a verified physics step. The
detection quality is the open problem.

## Measured accuracy so far

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
