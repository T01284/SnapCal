# SnapCal (IEEE Embedded Systems Letters companion)

Code, firmware, and tabulated results for:

> Tick-Aware Snapshot Scheduling and Reference-Injection Calibration
> for Native Baseband Capture on ESP32 Under FreeRTOS

Public repository: https://github.com/T01284/SnapCal

## What is in this archive

- `firmware/snapcal_fw/` — SnapCal fork of ESP-SDR (GPL-3.0-or-later). Upstream pin in `UPSTREAM_COMMIT`; local changes in `SNAPCAL_CHANGES.md`. PIO build trees and esp-dsp demos are omitted.
- `host/snapcal/` — host library (RIC/SIP/CFAR, ESP/Lime/Pluto/HackRF helpers).
- `host/scripts/` — experiment scripts that produce the letter's tables/figures.
- `host/tests/` — unit tests for estimators.
- `host/wsl/` — Lime/HackRF helpers used on the conducted bench.
- `paper/data/*.json` — tabulated numbers used by the letter / `gen_figures.py`.
- `paper/figures/*.pdf` — figure PDFs regenerated from data (not the manuscript).
- `results/` — rollups and conducted JSON/CSV that feed the letter (no raw IQ, no Phase-0 OTA dumps).

## What is not included

- The manuscript (`snapcal.tex` / PDF), cover letter, abstract PDF, and LaTeX class files.
- `third_party/` clones (recreate from `third_party/README.md` in the full repo if needed).
- PlatformIO `.pio/` build products.
- `results/_tmp*`, Phase-0 over-the-air bring-up dumps.
- Board passwords, serial-port private configs, large capture dumps.

## Reproduce paper figures/tables

```powershell
python -m pip install numpy matplotlib
python paper/scripts/gen_figures.py
```

Table numbers live in `paper/data/*.json`. Aggregates are under `results/*.json` and `results/sip/`, `results/cbm/`, `results/conducted/`.
Conducted start-phase lock `R` for S3-01 is in `results/conducted/s3-wroom1u-01/start_phase.json` (see also `paper/data/q_start_phase_s3-01.json`).

Firmware build (ESP32-S3 / C5) uses PlatformIO with `firmware/snapcal_fw/platformio.ini` (ESP-IDF 5.5.x via espressif32@6.13.0).
