"""Q6 spur list from existing floor captures.

Reads results/conducted/*/floor_spurs_*.json. Does not touch the boards.
Writes results/floor_spurs.json.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

CONDUCTED = ROOT / "results" / "conducted"
OUT = ROOT / "results" / "floor_spurs.json"
CH12_MHZ = 2467.0
CH12_HALF_MHZ = 10.0


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def compact_line(ln):
    row = {
        "f_rf_mhz": ln["f_rf_mhz"],
        "excess_db": ln["excess_db"],
        "snapshot_fraction": ln["snapshot_fraction"],
        "level_dbfs_bin": ln["level_dbfs_bin"],
    }
    if "f_rf_fine_mhz" in ln:
        row["f_rf_fine_mhz"] = ln["f_rf_fine_mhz"]
    if ln.get("harmonic"):
        row["harmonic"] = ln["harmonic"]
    return row


def channel12(freq_mhz, lines):
    if abs(freq_mhz - CH12_MHZ) > 40:
        return None
    near = [ln for ln in lines if abs(ln["f_rf_mhz"] - CH12_MHZ) <= CH12_HALF_MHZ]
    if not near:
        return None
    return compact_line(max(near, key=lambda ln: ln["excess_db"]))


def main():
    runs = []
    for path in sorted(CONDUCTED.glob("*/floor_spurs_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for r in d["runs"]:
            intermittent = r.get("intermittent_lines") or []
            persistent = r.get("persistent_spurs") or []
            top = sorted(intermittent, key=lambda ln: ln["excess_db"], reverse=True)[:5]
            runs.append({
                "source": rel(path),
                "utc": d.get("utc"),
                "board": d["board"],
                "tag": d["tag"],
                "same_day_20261025": "20261025" in d["tag"],
                "termination": d.get("termination"),
                "freq_mhz": r["freq_mhz"],
                "gain_index": r["gain_index"],
                "count": r["count"],
                "floor_dbfs_per_hz": r["floor_dbfs_per_hz"],
                "rms_median_dbfs": r["rms_dbfs"]["median"],
                "rms_max_dbfs": r["rms_dbfs"]["max"],
                "outlier_fraction_3db": r["rms_dbfs"]["outlier_fraction_3db"],
                "n_persistent": len(persistent),
                "persistent_spurs": [compact_line(ln) for ln in persistent],
                "n_intermittent": len(intermittent),
                "intermittent_strongest": [compact_line(ln) for ln in top],
                "channel12_strongest": channel12(r["freq_mhz"], intermittent),
            })
    hotspot60 = []
    for r in runs:
        if r["tag"] != "hotspot-20261025" or r["freq_mhz"] != 2442 or r["gain_index"] != 60:
            continue
        line = r["channel12_strongest"]
        hotspot60.append({
            "board": r["board"],
            "outlier_fraction_3db": r["outlier_fraction_3db"],
            "floor_dbfs_per_hz": r["floor_dbfs_per_hz"],
            "channel12_mhz": None if line is None else line["f_rf_mhz"],
            "channel12_excess_db": None if line is None else line["excess_db"],
            "channel12_level_dbfs_bin": None if line is None else line["level_dbfs_bin"],
        })
    above = [r for r in hotspot60 if (r["channel12_excess_db"] or 0) >= 10]
    out = {
        "utc": utc(),
        "note": (
            "Spur list rolled up from floor_spurs_*.json. Oct 12 files (tag box, box-load) "
            "are a different day and are not the 2026-10-25 baseline. "
            "channel12_strongest is the intermittent line within 10 MHz of 2467 MHz "
            "when that frequency is inside the 80 MS/s window. "
            "P_fa is not estimated."
        ),
        "acceptance": {
            "snapshots": 1000,
            "spatial_coupling_10db_below_floor": len(above) == 0,
            "hotspot_2442_gain60": hotspot60,
            "reason": (
                "False when any 2026-10-25 closed-box gain-60 capture at 2442 MHz "
                "has an intermittent line inside 2457–2477 MHz at least 10 dB above the floor. "
                f"{len(above)} of {len(hotspot60)} boards do. "
                "The median floor did not rise; the leakage is bursts. P_fa is not in this file."
            ),
        },
        "runs": runs,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT, "runs", len(runs))
    print("coupling 10 dB below floor:", out["acceptance"]["spatial_coupling_10db_below_floor"])
    for r in hotspot60:
        print(f"  {r['board']}: excess {r['channel12_excess_db']} dB at {r['channel12_mhz']} MHz, "
              f"outliers {100 * r['outlier_fraction_3db']:.1f} %")


if __name__ == "__main__":
    main()
