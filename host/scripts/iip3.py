"""Q8 conducted two-tone IIP3: RFE6GEN + LimeSDR -> combiner -> att -> ESP.

Tones at +f1 / +f2 MHz (default 10 / 12); IM3 at 2*f1-f2 and 2*f2-f1 (8 / 14).
Lime digital offset stays inside TX Nyquist (default rate 8 MS/s → offset 2 MHz):
  lime_lo = esp + f2 - offset, so RF lands on esp+f2.

Pin for the RFE tone is gen_est_dbm - --atten-db (Q13 chain including combiner arm if known).
A dBFS↔dBm scale from RFE-only points converts Lime and IM3 to the same Pin axis.
IIP3 = Pin_avg + (P_avg - P_IM3)/2  (dBm), using the stronger IM3 product.

Example:
  python host/scripts/iip3.py --port COM15 --board s3-com15 --atten-db 34.5 --baud 1000000
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.dsp import clip_fraction
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.limetx import LimeTx
from snapcal.ric import estimate_tone
from snapcal.rfe import LEVELS, RfeGenerator, discover

FS = 80_000_000
N = 16380
MIN_ATTEN_DB = 30.0


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_esp(port, baud):
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"), flush=True)
        dev.ser.baudrate = baud
        return dev


def capture_retry(dev, retries=4):
    last = None
    for _ in range(retries):
        try:
            return dev.capture(N, FS, "iq8")
        except EspSdrError as e:
            last = e
            try:
                dev.ser.reset_input_buffer()
            except Exception:
                pass
    raise last


def measure_tone(dev, f_bb, count):
    es, clips = [], []
    for _ in range(count):
        snap = capture_retry(dev)
        clips.append(clip_fraction(snap.iq))
        es.append(estimate_tone(snap.iq, FS, f_bb))
    p = np.array([e.power_dbfs for e in es])
    return {
        "power_dbfs": float(np.median(p)),
        "snr_db": float(np.median([e.snr_db for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
        "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es])),
        "clip_fraction_max": float(np.max(clips)),
    }


def match_lime_gain(dev, f_lime, target_dbfs, gains, count):
    """Pick Lime gain whose tone power is closest to target_dbfs."""
    best = None
    for g in gains:
        yield_g = g
        # caller sets gain externally; this helper only measures — see sweep
        _ = yield_g
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--board", required=True)
    ap.add_argument("--atten-db", type=float, required=True,
                    help="RFE→ESP loss incl. combiner arm + att + pigtail (dB)")
    ap.add_argument("--atten-note", default="combiner arm + att30_a + ipex pigtail")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--f1-mhz", type=float, default=10.0, help="RFE baseband offset")
    ap.add_argument("--f2-mhz", type=float, default=12.0, help="Lime RF offset from ESP LO")
    ap.add_argument("--lime-offset-mhz", type=float, default=2.0)
    ap.add_argument("--lime-gains", type=float, nargs="+",
                    default=[20, 25, 30, 35, 40, 45, 50])
    ap.add_argument("--rfe-levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-lime-gain", type=float, default=55.0)
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--balance-db", type=float, default=3.0,
                    help="max |P1-P2| after Lime gain match")
    ap.add_argument("--im3-snr-min", type=float, default=6.0,
                    help="IM3 must exceed this SNR to count for IIP3 fit")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen", default=None)
    a = ap.parse_args()
    if a.atten_db < MIN_ATTEN_DB:
        raise SystemExit(f"--atten-db {a.atten_db} < {MIN_ATTEN_DB}")

    f1 = a.f1_mhz * 1e6
    f2 = a.f2_mhz * 1e6
    im3_lo = 2 * f1 - f2
    im3_hi = 2 * f2 - f1
    rfe_mhz = a.esp_freq + a.f1_mhz
    lime_offset = a.lime_offset_mhz * 1e6
    lime_lo = a.esp_freq * 1e6 + f2 - lime_offset

    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    rd = ROOT / "results" / "conducted" / a.board
    rd.mkdir(parents=True, exist_ok=True)
    out = {
        "utc": utc(), "board": a.board, "cmd": "iip3",
        "args": vars(a),
        "plan": {
            "rfe_mhz": rfe_mhz, "lime_tone_mhz": a.esp_freq + a.f2_mhz,
            "lime_lo_hz": lime_lo, "lime_offset_hz": lime_offset,
            "f1_mhz": a.f1_mhz, "f2_mhz": a.f2_mhz,
            "im3_bb_mhz": [im3_lo / 1e6, im3_hi / 1e6],
        },
        "setup": (
            f"conducted: RFE6GEN + Lime -> combiner -> {a.atten_db:g} dB ({a.atten_note}) "
            f"-> IPEX -> ESP"
        ),
        "rows": [],
    }
    print(f"IIP3  ESP {a.esp_freq}  RFE @{rfe_mhz:.1f} (+{a.f1_mhz})  "
          f"Lime @{a.esp_freq + a.f2_mhz:.1f} (+{a.f2_mhz})  "
          f"IM3 {[im3_lo/1e6, im3_hi/1e6]}", flush=True)

    with RfeGenerator(gp) as gen, \
            LimeTx(lime_lo, lime_offset, a.lime_gains[0], max_gain_db=a.max_lime_gain) as lime, \
            open_esp(a.port, a.baud) as dev:
        out["generator"] = gen.info()
        out["lime_ready"] = lime.ready
        out["firmware"] = dev.info()["version"]
        out["identity"] = dev.identity
        dev.set_freq(a.esp_freq)
        dev.set_bandwidth(0)
        dev.set_gain(a.esp_gain)

        # RFE-only calibration of dBFS -> Pin
        print("A. RFE-only scale", flush=True)
        lime.off()
        cal = []
        for level in a.rfe_levels:
            if level >= len(LEVELS):
                continue
            gen.cw(rfe_mhz, level=level)
            time.sleep(0.3)
            r = measure_tone(dev, f1, a.count)
            pin = gen.estimated_dbm() - a.atten_db
            r |= {"gen_level": level, "gen_est_dbm": gen.estimated_dbm(), "pin_dbm": pin,
                  "scale": r["power_dbfs"] - pin}
            cal.append(r)
            print(f"  L{level} Pin {pin:6.1f}  {r['power_dbfs']:6.1f} dBFS  "
                  f"SNR {r['snr_db']:5.1f}  scale {r['scale']:.1f}", flush=True)
        gen.off()
        # scale from non-clip mid points
        usable = [r for r in cal if r["clip_fraction_max"] < 1e-3 and r["snr_db"] > 25]
        if not usable:
            usable = [r for r in cal if r["snr_db"] > 20]
        scale = float(np.median([r["scale"] for r in usable])) if usable else float("nan")
        out["rfe_cal"] = {"rows": cal, "scale_dbfs_per_dbm": scale}
        print(f"  scale = {scale:.2f} dBFS/dBm ({len(usable)} pts)", flush=True)

        print("B. dual-tone sweep (match Lime to RFE)", flush=True)
        for level in a.rfe_levels:
            if level >= len(LEVELS):
                continue
            gen.cw(rfe_mhz, level=level)
            time.sleep(0.3)
            p1_ref = measure_tone(dev, f1, max(a.count // 2, 6))["power_dbfs"]
            # scan Lime gains, pick closest |P2-P1|
            best = None
            for g in a.lime_gains:
                if g > a.max_lime_gain:
                    continue
                lime.gain(g)
                lime.cw()
                time.sleep(0.25)
                p2 = measure_tone(dev, f2, max(a.count // 2, 6))["power_dbfs"]
                err = abs(p2 - p1_ref)
                if best is None or err < best["err"]:
                    best = {"gain": g, "p2": p2, "err": err}
            lime.gain(best["gain"])
            lime.cw()
            time.sleep(0.35)
            t1 = measure_tone(dev, f1, a.count)
            t2 = measure_tone(dev, f2, a.count)
            lo = measure_tone(dev, im3_lo, a.count)
            hi = measure_tone(dev, im3_hi, a.count)
            gen_est = gen.estimated_dbm()
            pin_rfe = gen_est - a.atten_db
            pin1 = t1["power_dbfs"] - scale
            pin2 = t2["power_dbfs"] - scale
            pin_avg = 0.5 * (pin1 + pin2)
            p_avg = 0.5 * (t1["power_dbfs"] + t2["power_dbfs"])
            # stronger IM3 (higher power) dominates
            im3 = lo if lo["power_dbfs"] >= hi["power_dbfs"] else hi
            im3_which = "lo" if im3 is lo else "hi"
            delta = p_avg - im3["power_dbfs"]
            iip3 = pin_avg + 0.5 * delta
            row = {
                "gen_level": level, "gen_est_dbm": gen_est, "pin_rfe_dbm": pin_rfe,
                "lime_gain": best["gain"], "balance_err_db": best["err"],
                "tone1": t1, "tone2": t2, "im3_lo": lo, "im3_hi": hi,
                "im3_used": im3_which,
                "pin1_dbm": pin1, "pin2_dbm": pin2, "pin_avg_dbm": pin_avg,
                "delta_fund_im3_db": delta,
                "iip3_dbm": iip3,
                "im3_snr_ok": im3["snr_db"] >= a.im3_snr_min,
                "clip_ok": max(t1["clip_fraction_max"], t2["clip_fraction_max"]) < 1e-2,
                "balanced": best["err"] <= a.balance_db,
            }
            out["rows"].append(row)
            flag = []
            if not row["im3_snr_ok"]:
                flag.append("IM3-weak")
            if not row["clip_ok"]:
                flag.append("CLIP")
            if not row["balanced"]:
                flag.append("unbal")
            print(f"  L{level}/g{best['gain']:.0f}: P1 {t1['power_dbfs']:6.1f} P2 {t2['power_dbfs']:6.1f} "
                  f"IM3[{im3_which}] {im3['power_dbfs']:6.1f} (SNR {im3['snr_db']:4.1f})  "
                  f"Δ {delta:5.1f}  IIP3 {iip3:6.1f} dBm  "
                  f"{' '.join(flag) if flag else 'ok'}", flush=True)

        gen.off()
        lime.off()

    fit_rows = [r for r in out["rows"] if r["im3_snr_ok"] and r["clip_ok"] and r["balanced"]]
    if fit_rows:
        iip3s = [r["iip3_dbm"] for r in fit_rows]
        out["iip3"] = {
            "median_dbm": float(np.median(iip3s)),
            "mean_dbm": float(np.mean(iip3s)),
            "std_dbm": float(np.std(iip3s)),
            "n": len(fit_rows),
            "definition": (
                "IIP3 = Pin_avg + (P_avg - P_IM3)/2; Pin from dBFS via RFE-only scale; "
                "P_IM3 = stronger of the two IM3 products; equal-tone within --balance-db"
            ),
            "atten_db": a.atten_db,
            "atten_note": a.atten_note,
        }
        print(f"IIP3 median {out['iip3']['median_dbm']:.1f} dBm  "
              f"(n={out['iip3']['n']}, std {out['iip3']['std_dbm']:.1f})", flush=True)
    else:
        out["iip3"] = None
        print("IIP3: no rows passed IM3-SNR / clip / balance cuts", flush=True)

    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pins = [r["pin_avg_dbm"] for r in out["rows"]]
    ax[0].plot(pins, [r["tone1"]["power_dbfs"] for r in out["rows"]], "o-", label="P1")
    ax[0].plot(pins, [r["tone2"]["power_dbfs"] for r in out["rows"]], "s-", label="P2")
    ax[0].plot(pins, [r["im3_lo"]["power_dbfs"] for r in out["rows"]], "v-", label="IM3 lo")
    ax[0].plot(pins, [r["im3_hi"]["power_dbfs"] for r in out["rows"]], "^-", label="IM3 hi")
    ax[0].set_xlabel("Pin_avg (dBm)"); ax[0].set_ylabel("dBFS")
    ax[0].set_title("Two-tone + IM3"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].plot(pins, [r["iip3_dbm"] for r in out["rows"]], "o-")
    if out["iip3"]:
        ax[1].axhline(out["iip3"]["median_dbm"], color="C3", ls="--",
                      label=f"median {out['iip3']['median_dbm']:.1f} dBm")
    ax[1].set_xlabel("Pin_avg (dBm)"); ax[1].set_ylabel("IIP3 (dBm)")
    ax[1].set_title(f"{a.board} IIP3"); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()

    stem = f"iip3_{a.esp_freq}" + (f"_{a.tag}" if a.tag else "")
    (rd / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig.savefig(rd / f"{stem}.png", dpi=130)
    print("wrote", rd / f"{stem}.json")

    summary_path = ROOT / "results" / "rf_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": []}
    summary.setdefault("runs", []).append({
        "utc": out["utc"], "board": a.board, "file": str(rd / f"{stem}.json"),
        "iip3": out.get("iip3"),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
