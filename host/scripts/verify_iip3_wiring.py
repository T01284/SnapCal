"""Smoke-check the Q8 dual-tone (IIP3) conducted wiring: RFE + Lime -> combiner -> att -> ESP.

Sequence: both off, RFE only, Lime only, both on. Reports tone power at each baseband offset.
Exit 0 if both tones appear when their source is on and the inactive tone stays near the floor.

Example:
  python host/scripts/verify_iip3_wiring.py --port COM15 --baud 1000000 --lime-gain 40
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.limetx import LimeTx
from snapcal.ric import estimate_tone
from snapcal.rfe import RfeGenerator, discover

FS = 80_000_000
N = 16380


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
            return dev.capture(N, FS, "iq8").iq
        except EspSdrError as e:
            last = e
            try:
                dev.ser.reset_input_buffer()
            except Exception:
                pass
    raise last


def measure(dev, f_bb, count=12):
    es = [estimate_tone(capture_retry(dev), FS, f_bb) for _ in range(count)]
    p = np.array([e.power_dbfs for e in es])
    return {
        "power_dbfs": float(np.median(p)),
        "snr_db": float(np.median([e.snr_db for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
        "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM15")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--rfe-bb-mhz", type=float, default=10.0, help="RFE tone baseband offset")
    ap.add_argument("--lime-bb-mhz", type=float, default=12.0, help="Lime tone baseband offset")
    ap.add_argument("--lime-offset-mhz", type=float, default=2.0,
                    help="Lime digital offset; must be << rate/2 (default rate 8 MS/s → keep <=2 MHz). "
                         "Lime LO is set to esp+lime_bb - lime_offset so the RF tone lands on lime_bb.")
    ap.add_argument("--rfe-level", type=int, default=3, help="RFE level 0..7 (3≈L3; avoid clip)")
    ap.add_argument("--lime-gain", type=float, default=40.0)
    ap.add_argument("--max-lime-gain", type=float, default=50.0,
                    help="conducted with attenuators; raise only if chain has >=30 dB")
    ap.add_argument("--gen", default=None)
    ap.add_argument("--margin-db", type=float, default=8.0,
                    help="tone must exceed RF-off same-bin by this many dB")
    a = ap.parse_args()

    f_rfe = a.rfe_bb_mhz * 1e6
    f_lime = a.lime_bb_mhz * 1e6
    # Digital offset must be inside the Lime TX Nyquist (rate/2). Default 8 MS/s → ≤2 MHz.
    lime_offset = a.lime_offset_mhz * 1e6
    lime_lo = a.esp_freq * 1e6 + f_lime - lime_offset
    rfe_mhz = a.esp_freq + a.rfe_bb_mhz

    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    out = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "args": vars(a),
        "plan": {
            "esp_mhz": a.esp_freq,
            "rfe_mhz": rfe_mhz,
            "lime_tone_mhz": a.esp_freq + a.lime_bb_mhz,
            "im3_bb_mhz": [2 * a.rfe_bb_mhz - a.lime_bb_mhz, 2 * a.lime_bb_mhz - a.rfe_bb_mhz],
        },
        "steps": {},
    }
    print(f"ESP {a.esp_freq}  RFE @{rfe_mhz:.1f} (+{a.rfe_bb_mhz} MHz)  "
          f"Lime @{a.esp_freq + a.lime_bb_mhz:.1f} (+{a.lime_bb_mhz} MHz)  "
          f"IM3 bb {out['plan']['im3_bb_mhz']} MHz", flush=True)

    with RfeGenerator(gp) as gen, \
            LimeTx(lime_lo, lime_offset, a.lime_gain, max_gain_db=a.max_lime_gain) as lime, \
            open_esp(a.port, a.baud) as dev:
        out["generator"] = gen.info()
        out["lime"] = lime.ready
        out["firmware"] = dev.info()["version"]
        dev.set_freq(a.esp_freq)
        dev.set_bandwidth(0)
        dev.set_gain(a.esp_gain)

        print("1. both off", flush=True)
        gen.off()
        lime.off()
        time.sleep(0.3)
        off_r = measure(dev, f_rfe)
        off_l = measure(dev, f_lime)
        out["steps"]["off"] = {"rfe_bin": off_r, "lime_bin": off_l}
        print(f"   RFE-bin {off_r['power_dbfs']:.1f} dBFS  Lime-bin {off_l['power_dbfs']:.1f} dBFS", flush=True)

        print(f"2. RFE only L{a.rfe_level} @ {rfe_mhz:.1f} MHz", flush=True)
        lime.off()
        gen.cw(rfe_mhz, level=a.rfe_level)
        time.sleep(0.4)
        rfe_only_r = measure(dev, f_rfe)
        rfe_only_l = measure(dev, f_lime)
        out["steps"]["rfe_only"] = {"rfe_bin": rfe_only_r, "lime_bin": rfe_only_l,
                                    "gen_est_dbm": gen.estimated_dbm()}
        print(f"   RFE-bin {rfe_only_r['power_dbfs']:.1f} (SNR {rfe_only_r['snr_db']:.1f})  "
              f"Lime-bin {rfe_only_l['power_dbfs']:.1f}  gen {gen.estimated_dbm():.1f} dBm", flush=True)

        print(f"3. Lime only gain {a.lime_gain} @ +{a.lime_bb_mhz} MHz", flush=True)
        gen.off()
        lime.cw()
        time.sleep(0.5)
        lime_only_r = measure(dev, f_rfe)
        lime_only_l = measure(dev, f_lime)
        out["steps"]["lime_only"] = {"rfe_bin": lime_only_r, "lime_bin": lime_only_l}
        print(f"   RFE-bin {lime_only_r['power_dbfs']:.1f}  "
              f"Lime-bin {lime_only_l['power_dbfs']:.1f} (SNR {lime_only_l['snr_db']:.1f})", flush=True)

        print("4. both on", flush=True)
        gen.cw(rfe_mhz, level=a.rfe_level)
        lime.cw()
        time.sleep(0.5)
        both_r = measure(dev, f_rfe)
        both_l = measure(dev, f_lime)
        out["steps"]["both"] = {"rfe_bin": both_r, "lime_bin": both_l,
                                "gen_est_dbm": gen.estimated_dbm()}
        print(f"   RFE-bin {both_r['power_dbfs']:.1f} (SNR {both_r['snr_db']:.1f})  "
              f"Lime-bin {both_l['power_dbfs']:.1f} (SNR {both_l['snr_db']:.1f})", flush=True)

        gen.off()
        lime.off()

    m = a.margin_db
    checks = {
        "rfe_appears_alone": rfe_only_r["power_dbfs"] >= off_r["power_dbfs"] + m,
        "lime_appears_alone": lime_only_l["power_dbfs"] >= off_l["power_dbfs"] + m,
        "rfe_present_both": both_r["power_dbfs"] >= off_r["power_dbfs"] + m,
        "lime_present_both": both_l["power_dbfs"] >= off_l["power_dbfs"] + m,
        "rfe_bin_quiet_when_lime_only": lime_only_r["power_dbfs"] < rfe_only_r["power_dbfs"] - 3,
        "lime_bin_quiet_when_rfe_only": rfe_only_l["power_dbfs"] < lime_only_l["power_dbfs"] - 3,
    }
    out["checks"] = checks
    out["ok"] = all(checks.values())
    balance = abs(both_r["power_dbfs"] - both_l["power_dbfs"])
    out["tone_balance_db"] = float(balance)
    print("checks:", json.dumps(checks, indent=2), flush=True)
    print(f"tone balance (both on): {balance:.1f} dB  "
          f"({'ok-ish' if balance < 6 else 'unbalanced — trim Lime gain or RFE level'})", flush=True)
    print("PASS" if out["ok"] else "FAIL — see which check is false", flush=True)

    rd = ROOT / "results" / "conducted" / "s3-com15"
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / "iip3_wiring_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
