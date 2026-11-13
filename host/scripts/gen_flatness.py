"""Generator-plus-attenuator flatness on the 6G Combo, for subtracting from Q7 passbands.

Wiring: RFE6GEN -> att30_a -> Combo PLUS right port (240 MHz-6.1 GHz). No ESP32, no IPEX
pigtail. One analyzer span per centre keeps the RBW fixed, so the relative shape is what
gets subtracted. Frequencies are the same baseband grid as conducted_tone.py passband.
Generator level is L3, matching those sweeps.

  python host/scripts/gen_flatness.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.rfe import RfeAnalyzer, RfeGenerator, discover

# Centre MHz -> sample rates whose passband grid is measured.
CENTRES = {
    2412: (80e6, 40e6, 16e6),
    2442: (80e6, 40e6, 20e6, 16e6),
    2472: (80e6, 40e6, 16e6),
    5745: (80e6, 40e6, 20e6),
    5800: (80e6, 40e6, 20e6),
    5850: (80e6, 40e6, 20e6),
}
SPAN_MHZ = 80.0
LEVEL = 3
POINTS = 25
OUT = ROOT / "results" / "conducted" / "generator_flatness.json"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def grids(rates):
    """bb_hz -> sample rates that use it. Same rounding as conducted_tone.py passband."""
    used = {}
    for fs in rates:
        for frac in np.linspace(-0.47, 0.47, POINTS):
            f_bb = float(round(frac * fs / 1e3) * 1e3)
            if abs(f_bb) < 0.3e6:
                continue
            used.setdefault(f_bb, []).append(int(fs))
    return used


def main():
    found = discover()
    gp = found.get("generator", {}).get("port")
    sp = found.get("analyzer", {}).get("port")
    if not (gp and sp):
        raise SystemExit(f"need generator and analyzer, found {list(found)}")
    rows = []
    with RfeGenerator(gp) as gen, RfeAnalyzer(sp) as sa:
        sa.set_input_stage("direct")
        for centre, rates in CENTRES.items():
            sa.configure(centre - SPAN_MHZ / 2, centre + SPAN_MHZ / 2)
            gen.off()
            floor = sa.tone(centre, n=5, window_mhz=3.0)
            print(f"{centre} MHz floor {floor['dbm']:.2f} dBm  "
                  f"RBW {floor['rbw_khz']:.0f} kHz  span {floor['span_mhz']} MHz", flush=True)
            for f_bb, rate_list in grids(rates).items():
                f_mhz = centre + f_bb / 1e6
                gen.cw(f_mhz, level=LEVEL, settle=0.3)
                r = sa.tone(f_mhz, n=5, window_mhz=3.0)
                row = {
                    "centre_mhz": centre,
                    "bb_hz": f_bb,
                    "f_mhz": f_mhz,
                    "rates_hz": rate_list,
                    "dbm": r["dbm"],
                    "std_db": r["std_db"],
                    "peak_mhz": r["peak_mhz"],
                    "rbw_khz": r["rbw_khz"],
                    "span_mhz": r["span_mhz"],
                    "floor_dbm": floor["dbm"],
                    "gen_est_dbm": gen.estimated_dbm(),
                    "above_floor_db": r["dbm"] - floor["dbm"],
                }
                rows.append(row)
                print(f"  {f_mhz:8.3f} MHz {r['dbm']:7.2f} dBm  "
                      f"above floor {row['above_floor_db']:5.1f} dB", flush=True)
                if row["above_floor_db"] < 10:
                    gen.off()
                    raise SystemExit(
                        "tone is under 10 dB above the floor. Check the right-hand "
                        "analyzer port and that att30_a is the only thing in between.")
        gen.off()
        info = {"generator": gen.info(), "analyzer": sa.info()}
    out = {
        "utc": utc(),
        "setup": "RFE6GEN L3 -> att30_a -> 6G Combo PLUS right port. No IPEX pigtail, no ESP32.",
        "level": LEVEL,
        "span_mhz": SPAN_MHZ,
        "input_stage": "direct",
        **info,
        "rows": rows,
        "toolchain": envinfo.toolchain(),
        "host": envinfo.host(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for centre in CENTRES:
        pts = [r for r in rows if r["centre_mhz"] == centre and 80_000_000 in r["rates_hz"]]
        if not pts:
            continue
        y = np.array([r["dbm"] for r in pts])
        ax.plot([r["bb_hz"] / 1e6 for r in pts], y - np.max(y), "o-", ms=3, label=f"{centre} MHz")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("baseband offset (MHz)")
    ax.set_ylabel("analyzer, relative to peak (dB)")
    ax.set_title("RFE6GEN L3 + att30_a, 80 MS/s grid")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    png = OUT.with_suffix(".png")
    fig.savefig(png, dpi=120)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
