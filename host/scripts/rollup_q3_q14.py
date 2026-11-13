"""Roll conducted gain sweeps and lock / IRR stats into results/nbc_modes.json and ric.json.

Does not touch the board. Reads results/conducted/<board>/levels_*.json,
lock.json (+ edges), and stats_*_l*.json.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

CONDUCTED = ROOT / "results" / "conducted"
OUT = ROOT / "results"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gain_map(levels_path: Path):
    d = json.loads(levels_path.read_text(encoding="utf-8"))
    rows = d.get("B_esp_gain_sweep") or []
    if not rows:
        return None
    # Relative dB vs the lowest usable index that still has SNR >= 30 dB.
    usable = [r for r in rows if r.get("snr_db", 0) >= 30 and r["power_dbfs"] < -1]
    if len(usable) < 2:
        usable = [r for r in rows if r["power_dbfs"] < -1]
    if not usable:
        return None
    base = usable[0]["power_dbfs"]
    base_idx = usable[0]["esp_gain_index"]
    last = usable[-1]
    return {
        "source": str(levels_path.relative_to(ROOT)).replace("\\", "/"),
        "esp_freq_mhz": d.get("args", {}).get("esp_freq"),
        "generator_level": d.get("args", {}).get("level"),
        "reference_gain_index": base_idx,
        "reference_power_dbfs": base,
        "usable_span_db": last["power_dbfs"] - base,
        "usable_gain_index_first": base_idx,
        "usable_gain_index_last": last["esp_gain_index"],
        "rows": [
            {
                "gain_index": r["esp_gain_index"],
                "power_dbfs": r["power_dbfs"],
                "relative_db": r["power_dbfs"] - base,
                "snr_db": r.get("snr_db"),
            }
            for r in rows
        ],
    }


def lock_band(board_dir: Path):
    points = []
    for name in ("lock.json", "lock_edges.json", "lock_edges2.json",
                 "lock_5g_edges.json", "lock_l7_holes.json"):
        p = board_dir / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for pt in d.get("points", []):
            points.append(pt)
    if not points:
        return None
    locked = sorted({pt["mhz"] for pt in points if pt.get("verdict") == "locked"})
    no_tone = sorted({pt["mhz"] for pt in points if pt.get("verdict") == "no-tone"})
    rejected = []
    lock = board_dir / "lock.json"
    if lock.exists():
        rejected = json.loads(lock.read_text(encoding="utf-8")).get("rejected", [])
    return {
        "locked_mhz": locked,
        "no_tone_mhz": no_tone,
        "rejected": rejected,
        "criterion": "snr >= 20 dB and |baseband error| < 200 kHz",
    }


def ric_entry(stats_path: Path):
    d = json.loads(stats_path.read_text(encoding="utf-8"))
    s = d.get("n16380") or {}
    irr = s.get("irr_db_median") or {}
    raw = irr.get("raw")
    glob = irr.get("global")
    if raw is None or glob is None:
        return None
    return {
        "source": str(stats_path.relative_to(ROOT)).replace("\\", "/"),
        "board": d.get("board"),
        "esp_freq_mhz": d.get("args", {}).get("esp_freq"),
        "generator_level": d.get("args", {}).get("level"),
        "tag": d.get("args", {}).get("tag"),
        "snapshots": s.get("snapshots"),
        "power_dbfs_median": (s.get("power_dbfs") or {}).get("median"),
        "snr_db_median": s.get("snr_db_median"),
        "irr_raw_db": raw,
        "irr_global_db": glob,
        "irr_lift_db": glob - raw,
        "power_robust_std_db": (s.get("power_dbfs") or {}).get("robust_std_db"),
    }


def main():
    boards = sorted(p.name for p in CONDUCTED.iterdir() if p.is_dir())
    nbc = {
        "utc": utc(),
        "note": "Conducted rollup. Relative gain is tone power vs the first SNR>=30 dB "
                "index on that board's 2442 MHz L3 sweep. Lock uses the conducted tone "
                "criterion, not mere FREQ acceptance. Phase 0 OTA nbc_modes.json is kept "
                "as nbc_modes_phase0_ota.json.",
        "boards": {},
    }
    ric = {"utc": utc(), "boards": []}

    phase0 = OUT / "nbc_modes.json"
    if phase0.exists() and not (OUT / "nbc_modes_phase0_ota.json").exists():
        phase0.rename(OUT / "nbc_modes_phase0_ota.json")

    for board in boards:
        bdir = CONDUCTED / board
        entry = {"board": board}
        levels = bdir / "levels_2442.json"
        if levels.exists():
            entry["gain_index_to_db"] = gain_map(levels)
        lock = lock_band(bdir)
        if lock:
            entry["lock"] = lock
        if "gain_index_to_db" in entry or "lock" in entry:
            nbc["boards"][board] = entry

        for stats in sorted(bdir.glob("stats_*_l*.json")):
            e = ric_entry(stats)
            if e:
                ric["boards"].append(e)

    (OUT / "nbc_modes.json").write_text(json.dumps(nbc, indent=2), encoding="utf-8")
    (OUT / "ric.json").write_text(json.dumps(ric, indent=2), encoding="utf-8")
    print("wrote", OUT / "nbc_modes.json")
    print("wrote", OUT / "ric.json")
    print("boards in nbc:", list(nbc["boards"]))
    print("ric entries:", len(ric["boards"]))


if __name__ == "__main__":
    main()
