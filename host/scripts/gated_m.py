"""Gated-reference check of the Q15 interval on a warm board.

The SPDT stays on only for the reference snapshots. A dark snapshot follows
each one, with GPIO4 low, to show the work window does not contain the tone.
The spacing is the wall time in which a cold board would drift 0.5 ppm
(q15_m_from_drift.json). On a board that has already been on for hours this
checks the steady regime: adjacent references should move far less than 0.5 ppm.

Example:
  python host/scripts/gated_m.py --port COM14 --minutes 10
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr
from snapcal.rfe import RfeGenerator
from conducted_tone import measure

FS = 80_000_000
N = 16380
F_BB = 10e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--gen", default="COM16")
    ap.add_argument("--board", default="s3-wroom1u-01")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--gain", type=int, default=44)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--interval", type=float, default=27.2,
                    help="seconds between reference groups; s3-01 warmup time for 0.5 ppm")
    ap.add_argument("--count", type=int, default=5, help="reference snapshots per point")
    ap.add_argument("--pin", type=int, default=4)
    ap.add_argument("--tag", default="", help="filename suffix, so a cold run does not overwrite the warm one")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    f_rf = args.esp_freq * 1e6 + F_BB
    dest = ROOT / "results" / "q15"
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / f"gated_{args.board}{suffix}.csv"
    rows = []
    dev = EspSdr(args.port)
    gen = RfeGenerator(args.gen)
    try:
        print(dev.identity, flush=True)
        dev.set_freq(args.esp_freq)
        dev.set_gain(args.gain)
        dev.gpio(args.pin, "0")
        gen.cw(f_rf / 1e6, level=args.level, settle=0.4)
        print(f"generator {gen.estimated_dbm():.1f} dBm, interval {args.interval} s", flush=True)
        t0 = time.monotonic()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = None
            while time.monotonic() - t0 < args.minutes * 60:
                t = time.monotonic() - t0
                dev.gpio(args.pin, "1")
                ref = measure(dev, F_BB, args.count, FS, N)
                dev.gpio(args.pin, "0")
                dark = measure(dev, F_BB, 1, FS, N)
                if ref["power_dbfs"] > -3:
                    raise SystemExit(f"reference {ref['power_dbfs']} dBFS is too hot")
                row = {
                    "t_s": round(t, 3),
                    "ppm": (ref["freq_hz"] - F_BB) / f_rf * 1e6,
                    "power_dbfs": ref["power_dbfs"],
                    "freq_std_hz": ref["freq_std_hz"],
                    "snr_db": ref["snr_db"],
                    "irr_db": ref["irr_db"],
                    "dark_dbfs": dark["power_dbfs"],
                }
                rows.append(row)
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(row))
                    w.writeheader()
                w.writerow(row)
                fh.flush()
                print(f"  {t / 60:5.2f} min  {row['ppm']:+.4f} ppm  ref {row['power_dbfs']:6.1f}"
                      f"  dark {row['dark_dbfs']:6.1f} dBFS  std {row['freq_std_hz']:.0f} Hz",
                      flush=True)
                remain = args.interval - (time.monotonic() - t0 - t)
                # A quiet USB serial port gets suspended and the next write fails.
                deadline = time.monotonic() + max(0.0, remain)
                while time.monotonic() < deadline and time.monotonic() - t0 < args.minutes * 60:
                    time.sleep(min(2.0, deadline - time.monotonic()))
                    dev.command("TS?")
    finally:
        try:
            dev.gpio(args.pin, "0")
        except Exception:
            pass
        gen.close()
        dev.close()

    ppm = np.array([r["ppm"] for r in rows])
    dark = np.array([r["dark_dbfs"] for r in rows])
    ref_p = np.array([r["power_dbfs"] for r in rows])
    dt = np.diff([r["t_s"] for r in rows])
    dp = np.diff(ppm)
    step = {
        "n_steps": int(dp.size),
        "median_abs_ppm": float(np.median(np.abs(dp))) if dp.size else None,
        "max_abs_ppm": float(np.max(np.abs(dp))) if dp.size else None,
        "median_dt_s": float(np.median(dt)) if dt.size else None,
    }
    summary = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "board": args.board,
        "esp_freq_mhz": args.esp_freq,
        "gain": args.gain,
        "generator_level": args.level,
        "interval_s": args.interval,
        "minutes": args.minutes,
        "tag": args.tag,
        "note": ("The board had been off. A tone check ran after USB was plugged, then the "
                 "serial port was busy once, so the first point is not the instant of power-on. "
                 "interval_s is this board's cold-start time for 0.5 ppm."
                 if args.tag == "cold" else
                 "Board was already warm. interval_s is the s3-01 cold-start time for 0.5 ppm, "
                 "not a claim that this run started from power-on."),
        "ppm_first": float(ppm[0]),
        "ppm_last": float(ppm[-1]),
        "ppm_span": float(ppm.max() - ppm.min()),
        "power_span_db": float(ref_p.max() - ref_p.min()),
        "dark_median_dbfs": float(np.median(dark)),
        "ref_median_dbfs": float(np.median(ref_p)),
        "blanking_db": float(np.median(ref_p) - np.median(dark)),
        "adjacent": step,
        "rows": rows,
    }
    path = dest / f"gated_{args.board}{suffix}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot([r["t_s"] / 60 for r in rows], ppm, ".-", ms=4)
    ax[0].set_xlabel("time (min)")
    ax[0].set_ylabel("LO error (ppm)")
    ax[0].set_title("gated reference")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot([r["t_s"] / 60 for r in rows], ref_p, ".-", ms=4, label="switch on")
    ax[1].plot([r["t_s"] / 60 for r in rows], dark, ".-", ms=4, label="switch off")
    ax[1].set_xlabel("time (min)")
    ax[1].set_ylabel("tone (dBFS)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest / f"gated_{args.board}{suffix}.png", dpi=120)
    plt.close(fig)
    print(f"span {summary['ppm_span']:.4f} ppm over {rows[-1]['t_s'] / 60:.1f} min, "
          f"adjacent median |Δ| {step['median_abs_ppm']:.4f} ppm")
    print(f"blanking {summary['blanking_db']:.1f} dB  (ref {summary['ref_median_dbfs']:.1f}, "
          f"dark {summary['dark_median_dbfs']:.1f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
