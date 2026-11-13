"""Q21 conducted combiner smoke test (一分四 + SPDT + RFE + Lime TX -> ESP).

Expected wiring:
  RFE6GEN -- SPDT(RF) -------- arm1 -+
               ^ GPIO4                 |
  Lime TX -- att30(+pigtail) -- arm2 --+-- 一分四 common -- IPEX -- ESP antenna
  arm3/arm4 -- 50 ohm

Checks:
  0. floor (both off, SPDT=1)
  1. RFE-only (SPDT=1)
  2. Lime-only
  3. both
  4. SPDT=0 with RFE on (RFE should drop)

Example:
  python host/scripts/q21_combiner_smoke.py --port COM14 --board s3-com14
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.espsdr import EspSdr
from snapcal.limetx import LimeTx
from snapcal.rfe import RfeGenerator, discover
from snapcal.ric import estimate_tone

N = 16380
FS = 80_000_000


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measure(dev: EspSdr, f_bb: float, count: int = 8) -> dict:
    es = [estimate_tone(dev.capture(N, FS, "iq8").iq, FS, f_bb) for _ in range(count)]
    return {
        "power_dbfs": float(np.median([e.power_dbfs for e in es])),
        "snr_db": float(np.median([e.snr_db for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--board", default="s3-com14")
    ap.add_argument("--lo-mhz", type=float, default=2442.0)
    ap.add_argument("--f1-mhz", type=float, default=10.0, help="RFE baseband offset")
    ap.add_argument("--f2-mhz", type=float, default=2.0, help="Lime RF offset from LO")
    ap.add_argument("--lime-offset-mhz", type=float, default=2.0)
    ap.add_argument("--lime-gain", type=float, default=40.0)
    ap.add_argument("--max-lime-gain", type=float, default=50.0)
    ap.add_argument("--rfe-level", type=int, default=3)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--spdt-pin", type=int, default=4)
    ap.add_argument("--margin-db", type=float, default=8.0)
    ap.add_argument("--gen", default=None)
    a = ap.parse_args()

    f1, f2 = a.f1_mhz * 1e6, a.f2_mhz * 1e6
    rfe_mhz = a.lo_mhz + a.f1_mhz
    lime_offset = a.lime_offset_mhz * 1e6
    lime_lo = a.lo_mhz * 1e6 + f2 - lime_offset
    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    out = {
        "utc": utc(),
        "cmd": "q21_combiner_smoke",
        "board": a.board,
        "args": vars(a),
        "plan": {
            "rfe_mhz": rfe_mhz,
            "lime_tone_mhz": a.lo_mhz + a.f2_mhz,
            "lime_lo_hz": lime_lo,
            "f1_mhz": a.f1_mhz,
            "f2_mhz": a.f2_mhz,
        },
        "setup": (
            "RFE->SPDT->comb4 arm1; LimeTX->att30->arm2; arm3/4=50ohm; "
            "comb4 common->IPEX->ESP; SPDT=GPIO4"
        ),
        "checks": {},
    }
    print(f"Q21 smoke  LO={a.lo_mhz}  RFE +{a.f1_mhz}  Lime +{a.f2_mhz}", flush=True)

    with RfeGenerator(gp) as gen, \
            LimeTx(lime_lo, lime_offset, min(a.lime_gain, a.max_lime_gain),
                   max_gain_db=a.max_lime_gain) as lime, \
            EspSdr(a.port) as esp:
        out["generator"] = gen.info()
        out["lime_ready"] = lime.ready
        out["esp"] = esp.identity
        print("READY", esp.identity, flush=True)
        print("READY lime", lime.ready, flush=True)

        esp.set_freq(int(a.lo_mhz))
        esp.set_gain(a.esp_gain)
        if a.spdt_pin:
            esp.gpio(a.spdt_pin, "1")
            print(f"SPDT GPIO{a.spdt_pin}=1", flush=True)

        print("0. floor", flush=True)
        lime.off(); gen.off(); time.sleep(0.4)
        floor1, floor2 = measure(esp, f1), measure(esp, f2)
        print(f"  f1={floor1['power_dbfs']:.1f}  f2={floor2['power_dbfs']:.1f}", flush=True)

        print("1. RFE-only (SPDT=1)", flush=True)
        lime.off(); gen.cw(rfe_mhz, level=a.rfe_level); time.sleep(0.4)
        rfe_only = measure(esp, f1)
        print(f"  {rfe_only['power_dbfs']:.1f} dBFS  SNR {rfe_only['snr_db']:.1f}", flush=True)

        print("2. Lime-only", flush=True)
        gen.off(); lime.gain(a.lime_gain); lime.cw(); time.sleep(0.7)
        lime_only = measure(esp, f2)
        print(f"  {lime_only['power_dbfs']:.1f} dBFS  SNR {lime_only['snr_db']:.1f}", flush=True)

        print("3. both", flush=True)
        gen.cw(rfe_mhz, level=a.rfe_level); lime.cw(); time.sleep(0.5)
        both1, both2 = measure(esp, f1), measure(esp, f2)
        print(f"  f1={both1['power_dbfs']:.1f}  f2={both2['power_dbfs']:.1f}", flush=True)

        print("4. SPDT=0 with RFE on", flush=True)
        lime.off()
        if a.spdt_pin:
            esp.gpio(a.spdt_pin, "0")
        gen.cw(rfe_mhz, level=a.rfe_level); time.sleep(0.4)
        gated = measure(esp, f1)
        print(f"  {gated['power_dbfs']:.1f} dBFS  SNR {gated['snr_db']:.1f}", flush=True)
        if a.spdt_pin:
            esp.gpio(a.spdt_pin, "1")

        gen.off(); lime.off()

        m = a.margin_db
        checks = {
            "rfe": rfe_only["power_dbfs"] >= floor1["power_dbfs"] + m,
            "lime": lime_only["power_dbfs"] >= floor2["power_dbfs"] + m,
            "both_rfe": both1["power_dbfs"] >= floor1["power_dbfs"] + m,
            "both_lime": both2["power_dbfs"] >= floor2["power_dbfs"] + m,
            "spdt_blocks": gated["power_dbfs"] <= rfe_only["power_dbfs"] - m,
        }
        out["rows"] = {
            "floor": {"f1": floor1, "f2": floor2},
            "rfe_only": rfe_only,
            "lime_only": lime_only,
            "both": {"f1": both1, "f2": both2},
            "spdt0_rfe": gated,
        }
        out["checks"] = checks
        out["ok"] = all(checks.values())
        print("CHECKS", {k: ("PASS" if v else "FAIL") for k, v in checks.items()}, flush=True)
        print("SMOKE", "PASS" if out["ok"] else "FAIL", flush=True)

    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()
    rd = ROOT / "results" / "cbm" / a.board
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / "combiner_smoke.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    if not out["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
