"""Host-side c / T0 for on-chip energy-detect (CAPE*): miss vs hit.

Miss replies are a short text line (no IQ payload); hit uploads like CAP*.
Fits T_cycle ≈ T_w + c·T_w + T0 from host timestamps (upper bound vs GPIO LA).

Examples:
  python host/scripts/energy_timing.py --port COM25 --baud 1000000 --thresh 200
  python host/scripts/energy_timing.py --port COM14 --thresh 0 --tag hit
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.espsdr import EspSdr, EspSdrError

SIZES = [1024, 2048, 4096, 8192, 12288, 16380]


def open_esp(port, baud):
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"), flush=True)
        dev.ser.baudrate = baud
        return dev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--freq", type=int, default=2442)
    ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--thresh", type=int, default=200, help="0=always hit; high=miss on quiet")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--rates", type=int, nargs="+", default=[80_000_000, 40_000_000])
    ap.add_argument("--formats", nargs="+", default=["iq8"])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--fast-poll", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="snapshot_timing")
    a = ap.parse_args()

    rows = []
    with open_esp(a.port, a.baud) as dev:
        info = dev.info()
        if "ENERGY" not in info["caps"]:
            raise SystemExit("firmware CAPS lacks ENERGY; flash CAPE* build first")
        dev.set_freq(a.freq)
        if a.gain is not None:
            dev.set_gain(a.gain)
        try:
            dev.set_fast_poll(a.fast_poll)
        except EspSdrError:
            pass
        for fmt in a.formats:
            for fs in a.rates:
                tw, tc, cu, hits, codes = [], [], [], [], []
                for n in a.sizes:
                    dev.capture_energy(n, fs, fmt, a.thresh)  # warm-up
                    for _ in range(a.repeats):
                        r = dev.capture_energy(n, fs, fmt, a.thresh)
                        tw.append(n / fs)
                        tc.append(r.t_host_end - r.t_host_start)
                        cu.append(r.capture_us * 1e-6)
                        hits.append(int(r.hit))
                        codes.append(r.code)
                tw, tc, cu = map(np.array, (tw, tc, cu))
                hits = np.array(hits)
                tx = tc - tw
                c, t0 = np.polyfit(tw, tx, 1)
                hit_frac = float(np.mean(hits))
                med = {n: float(np.median(tc[np.isclose(tw, n / fs)])) for n in a.sizes}
                rows.append({
                    "format": fmt, "rate_hz": fs, "thresh": a.thresh,
                    "hit_frac": round(hit_frac, 4),
                    "code_p50": int(np.median(codes)),
                    "c": round(float(c), 2), "T0_ms": round(float(t0 * 1e3), 3),
                    "fw_capture_overhead_us": round(float(np.median(cu - tw) * 1e6), 1),
                    "T_cycle_med_s": {str(n): round(med[n], 6) for n in a.sizes},
                })
                print(
                    f"{fmt} {fs/1e6:.0f} MS/s thresh={a.thresh} "
                    f"hit={hit_frac:.0%} c={c:.2f} T0={t0*1e3:.3f} ms "
                    f"code_p50={int(np.median(codes))}",
                    flush=True,
                )

    board = info["info"].split()[0]
    tag = f"_{a.tag}" if a.tag else ""
    out = ROOT / "results" / a.out / f"energy_timing_{board.lower()}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "when": datetime.now(timezone.utc).isoformat(),
        "port": a.port, "baud": a.baud, "freq_mhz": a.freq, "gain": a.gain,
        "thresh": a.thresh, "fast_poll": a.fast_poll, "repeats": a.repeats,
        "board": info, "host": envinfo.host(), "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
