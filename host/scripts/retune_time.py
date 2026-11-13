"""Time from a FREQ command until the new centre shows a settled conducted tone.

The generator stays on one frequency. The ESP hops between two centres that put
that tone at +10 MHz and -20 MHz, both inside the 80 MS/s band. Each hop is
captured as soon as FREQ returns. A hop counts as settled when the new offset
has SNR >= 20 dB, |error| < 200 kHz, and power within 3 dB of the rested level.

Example:
  python host/scripts/retune_time.py --port COM20 --board s3-wroom1u-03
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.ric import estimate_tone
from snapcal.rfe import RfeGenerator

FS = 80_000_000
N = 4096
HOPS = (2442, 10e6), (2472, -20e6)


def snap(dev, f_bb):
    t0 = time.perf_counter()
    iq = dev.capture(N, FS, "iq10").iq
    t1 = time.perf_counter()
    e = estimate_tone(iq, FS, f_bb)
    return e, t1 - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--board", default="s3-wroom1u-03")
    ap.add_argument("--gen", default="COM16")
    ap.add_argument("--hops", type=int, default=30)
    ap.add_argument("--gain", type=int, default=44)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--gate-pin", type=int, default=None,
                    help="drive this GPIO high for the run (SPDT control, 1 = pass)")
    a = ap.parse_args()

    dev = EspSdr(a.port, baud=a.baud)
    gen = RfeGenerator(a.gen)
    rows = []
    try:
        print(dev.identity, flush=True)
        if a.gate_pin is not None:
            dev.gpio(a.gate_pin, "1")
        dev.set_gain(a.gain)
        gen.cw(2452.0, level=a.level, settle=0.4)
        rested = {}
        for mhz, f_bb in HOPS:
            dev.set_freq(mhz)
            time.sleep(0.02)
            es = [snap(dev, f_bb)[0] for _ in range(8)]
            rested[mhz] = float(np.median([e.power_dbfs for e in es]))
            print(f"rest {mhz}  {rested[mhz]:.1f} dBFS", flush=True)
        cur = HOPS[0]
        for i in range(a.hops):
            nxt = HOPS[(i + 1) % 2]
            t0 = time.perf_counter()
            dev.set_freq(nxt[0])
            cmd_s = time.perf_counter() - t0
            e, cap_s = snap(dev, nxt[1])
            ok = (e.snr_db >= 20 and abs(e.freq_hz - nxt[1]) < 200e3
                  and abs(e.power_dbfs - rested[nxt[0]]) < 3)
            row = {
                "to_mhz": nxt[0],
                "cmd_ms": round(cmd_s * 1e3, 3),
                "cap_ms": round(cap_s * 1e3, 3),
                "power_dbfs": round(e.power_dbfs, 2),
                "snr_db": round(e.snr_db, 2),
                "error_hz": round(e.freq_hz - nxt[1], 1),
                "settled": bool(ok),
            }
            rows.append(row)
            print(f"{nxt[0]}  cmd {row['cmd_ms']:7.2f} ms  {e.power_dbfs:7.1f} dBFS  "
                  f"SNR {e.snr_db:5.1f}  err {row['error_hz']:+8.0f} Hz  "
                  f"{'ok' if ok else 'NOT settled'}", flush=True)
            cur = nxt
    finally:
        if a.gate_pin is not None:
            try:
                dev.gpio(a.gate_pin, "0")
            except EspSdrError:
                pass
        gen.close()
        dev.close()

    cmd = np.array([r["cmd_ms"] for r in rows])
    by_mhz = {}
    for mhz, _ in HOPS:
        c = np.array([r["cmd_ms"] for r in rows if r["to_mhz"] == mhz])
        by_mhz[str(mhz)] = {
            "n": int(c.size),
            "cmd_ms_median": round(float(np.median(c)), 3),
            "cmd_ms_min": round(float(c.min()), 3),
            "cmd_ms_max": round(float(c.max()), 3),
        }
    out = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "board": a.board,
        "port": a.port,
        "gate_pin": a.gate_pin,
        "generator_mhz": 2452,
        "hops": "2442 (+10 MHz) <-> 2472 (-20 MHz)",
        "n": N,
        "rested_dbfs": rested,
        "n_hops": len(rows),
        "n_settled": sum(r["settled"] for r in rows),
        "cmd_ms_median": round(float(np.median(cmd)), 3),
        "by_mhz": by_mhz,
        "note": "A hop is settled when the capture started as FREQ returns already has "
                "SNR >= 20 dB, |error| < 200 kHz, and power within 3 dB of the rested tone. "
                "prepare_rx recalibrates when the centre changes. The S3 path then waits "
                "3 ms; the C5 path does not. Repeating the centre just calibrated skips "
                "recalibration.",
        "rows": rows,
    }
    dest = ROOT / "results" / "snapshot_timing" / f"retune_{a.board}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"settled {out['n_settled']}/{out['n_hops']}")
    for mhz, s in by_mhz.items():
        print(f"  -> {mhz}  {s['cmd_ms_median']} ms ({s['cmd_ms_min']}–{s['cmd_ms_max']})")
    print("wrote", dest)


if __name__ == "__main__":
    main()
