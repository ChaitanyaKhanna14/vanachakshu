# Runbook

For whoever keeps this alive — including future you, who will not remember any
of it. Written on the assumption that the person reading it did not build it.

The system is designed to need nothing. It runs monthly on GitHub Actions, costs
nothing, and fails loudly rather than silently. Most entries below are things
that have actually gone wrong, not hypotheticals.

---

## What runs, and when

| What | Where | Cadence |
|---|---|---|
| Alert job | `.github/workflows/monitor.yml` | monthly cron, 90 min timeout |
| Tests, lint, types | `.github/workflows/ci.yml` | every push |

The alert job detects disturbance, deduplicates against
`data/alerts/yellapur-taluk.json`, writes a report, and commits both back.
**The alert store is the state.** Losing it means every past alert is re-sent as
new; it is committed to the repo for exactly that reason.

## Health check, in one command

```bash
vanachakshu doctor
```

Initialises Earth Engine and forces a real server round-trip. `ee.Initialize`
succeeding proves almost nothing — credentials and compute permission are
separate checks — so this deliberately evaluates something trivial server-side.

Other commands: `vanachakshu run` (one detection cycle), `validate-sample` →
`validate-chips` → `validate-report` (the human validation loop), `version`.

---

## Failures that have actually happened

### "CERTIFICATE_VERIFY_FAILED" — but the browser works fine

**Cause:** something is intercepting HTTPS. Consumer antivirus (AVG Web/Mail
Shield, on the development machine), a campus or corporate proxy, or a VPN. It
re-signs every connection with its own root and installs that root in the OS
trust store. Python's `ssl` module reads that store and is satisfied — but
`requests` and `httplib2`, which Earth Engine's client actually uses, trust only
the static CA list inside `certifi`, where the interceptor is absent.

**Fix:** already applied. `gee.use_system_certificates()` runs before every
`initialize()` and points those libraries at the OS trust store. Verification
stays fully on.

**If it still happens:** confirm the interceptor by reading the issuer back.

```python
import ssl, socket

ctx = ssl.create_default_context()
with socket.create_connection(("oauth2.googleapis.com", 443)) as s:
    cert = ctx.wrap_socket(s, server_hostname="oauth2.googleapis.com").getpeercert()
    print(dict(x[0] for x in cert["issuer"]))
```

An issuer that is not a real CA confirms it. Check `truststore` is installed.

**Never fix this with `verify=False`.** That accepts any certificate from
anyone and defeats the point of HTTPS. `tests/test_gee.py` guards against it.

### `git push` fails the same way

Git carries its own CA bundle and needs the same treatment:

```bash
git config --global http.sslBackend schannel   # Windows only
```

### "Computation timed out"

The single most common Earth Engine failure here, hit at least seven times
during development. It always means the request was too big, and the remedy is
always to make it smaller — never to retry unchanged.

In order of what has actually worked:

1. **Split the region.** Tile the AOI and issue one request per tile. Fixed
   vectorising, training-sample extraction, and the 10 m parameter sweep.
2. **Raise `tileScale`** (8 or 16). Trades speed for memory headroom.
3. **Shrink the question.** Reduce band count, or sample instead of reducing
   over every pixel — see `scripts/tune_detector.py`.
4. **Batch export.** Materialise once to an asset, then query it many times.
   `exports.py` exists for this. Requires an asset root (below).

### "Asset 'projects/<id>/assets' does not exist"

The Cloud project has no Earth Engine asset root. Open
[code.earthengine.google.com](https://code.earthengine.google.com), click
**Assets**, and add the project if it is not listed. One-time, browser-only —
there is no API path to it.

Batch exports fail without this; nothing else does.

### Quota exhausted

Contributor tier gives 1,000 EECU-hours/month, resetting on the 1st. The monthly
alert job uses a small fraction. If quota is gone, something is looping or a
parameter sweep was run repeatedly — check for the checkpoint file first, since
`scripts/tune_detector.py` resumes rather than recomputing.

### The job ran but sent nothing

Usually correct. Detections are genuinely rare — about 10 ha of loss per year
across 106,543 ha of forest — and the detector reports roughly a third of it.
Several consecutive empty months are expected, not a fault.

Confirm with `data/output/*.txt`, which records what each run considered.

Genuinely wrong if: the run reports *zero candidate pixels before* filtering.
That means the input, not the threshold. Check that the embedding year exists —
AlphaEarth is annual and published months in arrears, so a January run may find
no new year to compare against.

---

## Changing the detector

Parameters live in `EmbeddingDetectionConfig` in `config.py`, with the measured
sweep in its docstring. **Do not adjust them by eye.** Re-run:

```bash
python scripts/tune_detector.py
```

It checkpoints per tile and resumes, so an interrupted run is cheap. Delete
`scripts/tune_detector.checkpoint.json` to force a fresh measurement.

Two things that measurement taught, which are easy to lose:

- **Score at 10 m, never 30 m.** The median detection is 0.116 ha, roughly one
  30 m pixel. Scoring at 30 m discards most of the output and flatters
  precision — it produced two wrong numbers that reached the README before
  being caught.
- **Precision 1.000 is not a result.** Above threshold 0.45 no false positive
  survives into the sample, so precision has nothing to divide by. The script
  labels those rows and excludes them from ranking. Never quote them.

## Moving to a new area

`config.py` defines the AOI as a bounding box. A new one needs:

1. A `BoundingBox` and `AreaOfInterest` entry.
2. **Re-tuning.** Thresholds are not portable — they encode this landscape's
   base rate and vegetation. Assume nothing transfers.
3. **A think about land use before publishing anything.** In Northeast India,
   *jhum* shifting cultivation is legal, cyclical, and spectrally almost
   identical to deforestation. Flagging it would be both factually wrong and
   harmful to tribal communities. This is why the AOI is the Western Ghats.

## Handing it over

The repo is the handover: no external state, no database, no server. Someone
needs a Google account, an Earth Engine project on the Contributor tier, and the
`VANACHAKSHU_EE_PROJECT` secret set on the fork.

If it is abandoned, it stops quietly and harms nobody — which is the intended
failure mode for a system that people might otherwise come to rely on.
