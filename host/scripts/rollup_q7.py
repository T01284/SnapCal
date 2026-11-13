"""Q7: peak-to-peak and -3 dB bandwidth from conducted passband sweeps.

Reads results/conducted/<board>/passband_*.json and generator_flatness.json.
Does not touch the boards. Writes results/passband.json.

The -3 dB edges are the outermost crossings of -3 dB relative to the
generator-corrected peak in that sweep. A ripple dip between the edges does
not move them. An edge is omitted when the swept end (about +/-0.47 fs) is
still above -3 dB; bw_3db_mhz is then null.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from _path import ROOT

CONDUCTED = ROOT / "results" / "conducted"
OUT = ROOT / "results" / "passband.json"
MATCH_HZ = 1.5e6
THR_DB = -3.0


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def load_flatness():
    rows = json.loads((CONDUCTED / "generator_flatness.json").read_text(encoding="utf-8"))["rows"]
    flat = {}
    for r in rows:
        flat.setdefault(int(r["centre_mhz"]), []).append((float(r["bb_hz"]), float(r["dbm"])))
    return flat


def correct(flat, centre, bb, pwr):
    cands = flat[centre]
    corr = np.empty(len(bb))
    for i, f in enumerate(bb):
        j = min(range(len(cands)), key=lambda k: abs(cands[k][0] - f))
        df = abs(cands[j][0] - f)
        if df >= MATCH_HZ:
            raise SystemExit(f"no generator flatness within {MATCH_HZ / 1e6} MHz of {centre} MHz, bb {f}")
        corr[i] = cands[j][1]
    return pwr - corr


def _cross(f0, y0, f1, y1):
    if y1 == y0:
        return float(f1)
    t = (THR_DB - y0) / (y1 - y0)
    return float(f0 + t * (f1 - f0))


def edges(bb, y):
    """Outermost -3 dB crossings. None means that swept end is still above -3 dB."""
    lo = hi = None
    for i, v in enumerate(y):
        if v >= THR_DB:
            lo = None if i == 0 else _cross(bb[i - 1], y[i - 1], bb[i], y[i])
            break
    for i in range(len(y) - 1, -1, -1):
        if y[i] >= THR_DB:
            hi = None if i == len(y) - 1 else _cross(bb[i], y[i], bb[i + 1], y[i + 1])
            break
    return lo, hi


def rate_row(flat, centre, rows):
    bb = np.array([p["bb_hz"] for p in rows], dtype=float)
    pwr = np.array([p["power_dbfs"] for p in rows], dtype=float)
    order = np.argsort(bb)
    bb, pwr = bb[order], pwr[order]
    adj = correct(flat, centre, bb, pwr)
    y = adj - float(adj.max())
    lo, hi = edges(bb, y)
    inside = lo is not None and hi is not None
    if inside:
        between = (bb > lo) & (bb < hi)
        dips = int(np.sum(y[between] < THR_DB))
        bw = (hi - lo) / 1e6
    else:
        dips = int(np.sum(y < THR_DB))
        bw = None
    return {
        "n_points": int(len(bb)),
        "span_mhz": round(float(bb[-1] - bb[0]) / 1e6, 2),
        "peak_to_peak_db": round(float(pwr.max() - pwr.min()), 2),
        "peak_to_peak_corrected_db": round(float(adj.max() - adj.min()), 2),
        "bw_3db_mhz": None if bw is None else round(bw, 2),
        "f_low_mhz": None if lo is None else round(lo / 1e6, 2),
        "f_high_mhz": None if hi is None else round(hi / 1e6, 2),
        "edges_inside": inside,
        "samples_below_3db_between_edges": dips,
    }


def main():
    flat = load_flatness()
    boards = {}
    for path in sorted(CONDUCTED.glob("*/passband_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        centre = int(d["args"]["esp_freq"])
        board = d["board"]
        rates = {rate: rate_row(flat, centre, rows) for rate, rows in d["rates"].items()}
        top = rates["80000000"]
        boards.setdefault(board, []).append({
            "source": rel(path),
            "esp_freq_mhz": centre,
            "n_points": top["n_points"],
            "peak_to_peak_db": top["peak_to_peak_db"],
            "peak_to_peak_corrected_db": top["peak_to_peak_corrected_db"],
            "bw_3db_mhz": top["bw_3db_mhz"],
            "f_low_mhz": top["f_low_mhz"],
            "f_high_mhz": top["f_high_mhz"],
            "edges_inside": top["edges_inside"],
            "rates": rates,
        })
    out = {
        "utc": utc(),
        "note": (
            "80 MS/s fields are repeated at the top of each entry. "
            "corrected subtracts generator_flatness at the matching centre and the nearest bb "
            "when |delta bb| < 1.5 MHz. "
            "bw_3db_mhz is the width between the outermost -3 dB crossings relative to the "
            "generator-corrected peak. It is null when a swept end (about +/-0.47 fs) is still "
            "above -3 dB. A ripple sample between the edges can sit below -3 dB."
        ),
        "boards": boards,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
