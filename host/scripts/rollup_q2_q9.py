"""Q2 tone check and Q9 snapshot-parameter classes from conducted stats.

Reads results/conducted/*/stats_*.json. Does not touch the boards.
Writes results/q2_tone_conducted.json and results/snapshot_params.json.

Frequency matches the setting when the baseband offset is inside 200 kHz,
the same limit as the conducted lock scan. Amplitude stays in dBFS.
The 30.7 dB figure stored on those runs is the nominal attenuator label,
not a per-frequency loss to the antenna port.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

CONDUCTED = ROOT / "results" / "conducted"
OUT = ROOT / "results"
FREQ_LIMIT_HZ = 200e3
PHASE_PER_SNAPSHOT = 0.2
POWER_STABLE_DB = 0.1


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def load_stats():
    rows = []
    for path in sorted(CONDUCTED.glob("*/stats_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        s = d["n16380"]
        a = d["args"]
        rows.append((path, d, s, a))
    return rows


def q2(rows):
    runs = []
    for path, d, s, a in rows:
        offset = s["freq_hz"]["offset_from_nominal"]
        f_rf_mhz = a["esp_freq"] + a["bb_mhz"]
        runs.append({
            "source": rel(path),
            "board": d["board"],
            "esp_freq_mhz": a["esp_freq"],
            "bb_mhz": a["bb_mhz"],
            "rf_mhz": f_rf_mhz,
            "generator_level": a.get("level"),
            "esp_gain": a.get("esp_gain"),
            "tag": a.get("tag") or "",
            "atten_db_nominal": a.get("atten_db"),
            "snapshots": s["snapshots"],
            "offset_hz": offset,
            "offset_ppm": offset / (f_rf_mhz * 1e6) * 1e6,
            "power_dbfs_median": s["power_dbfs"]["median"],
            "snr_db_median": s["snr_db_median"],
            "freq_within_200khz": abs(offset) < FREQ_LIMIT_HZ,
            "tone_present": s["snr_db_median"] >= 20,
        })
    return {
        "utc": utc(),
        "note": (
            "Conducted RFE6GEN tones from stats_*.json, 1000 snapshots, iq10, 80 MS/s. "
            "freq_within_200khz uses the lock-scan limit. power_dbfs_median is the tone "
            "at the ESP32, not an antenna-port dBm. atten_db_nominal is the label stored "
            "with the run (30.7 dB), not the later per-frequency chain."
        ),
        "freq_limit_hz": FREQ_LIMIT_HZ,
        "all_within_200khz": all(r["freq_within_200khz"] for r in runs),
        "all_tone_present": all(r["tone_present"] for r in runs),
        "runs": runs,
    }


def q9(rows):
    runs = []
    for path, d, s, a in rows:
        phase_len = s["phase"]["mean_resultant_length"]
        power_std = s["power_dbfs"]["robust_std_db"]
        irr = s["irr_db_median"]
        runs.append({
            "source": rel(path),
            "board": d["board"],
            "esp_freq_mhz": a["esp_freq"],
            "tag": a.get("tag") or "",
            "esp_gain": a.get("esp_gain"),
            "generator_level": a.get("level"),
            "snapshots": s["snapshots"],
            "phase_mean_resultant_length": phase_len,
            "phase_class": "per_snapshot" if phase_len < PHASE_PER_SNAPSHOT else "stable_within_capture",
            "power_robust_std_db": power_std,
            "power_class": "stable_within_capture" if power_std < POWER_STABLE_DB else "varies_within_capture",
            "freq_robust_std_hz": s["freq_hz"]["robust_std"],
            "freq_offset_hz": s["freq_hz"]["offset_from_nominal"],
            "freq_class": "stable_within_capture",
            "irr_raw_db": irr["raw"],
            "irr_prev_db": irr["prev"],
            "irr_global_db": irr["global"],
            "image_abs_db": s["image_ratio"]["abs_db"],
            "image_angle_robust_std_deg": s["image_ratio"]["angle_robust_std_deg"],
            "image_class": "global_median",
        })
    untagged = [r for r in runs if r["tag"] == ""]
    return {
        "utc": utc(),
        "note": (
            "Classes from the same stats_*.json as Q2. "
            "Phase is per_snapshot when the mean resultant length is below 0.2 "
            "(0 would be a new random phase on every snapshot). "
            "Power is stable_within_capture when the robust standard deviation is below 0.1 dB. "
            "Frequency is stable inside one 1000-snapshot capture; the median offset still "
            "drifts after power-on, which is Q10. "
            "The image coefficient used for IRR is the global median. "
            "A large image_angle_robust_std_deg means a per-snapshot coefficient does not hold."
        ),
        "rules": {
            "phase_per_snapshot_below": PHASE_PER_SNAPSHOT,
            "power_stable_below_db": POWER_STABLE_DB,
        },
        "untagged_power_robust_std_db": {
            "min": min(r["power_robust_std_db"] for r in untagged),
            "max": max(r["power_robust_std_db"] for r in untagged),
        },
        "runs": runs,
    }


def main():
    rows = load_stats()
    tone = q2(rows)
    params = q9(rows)
    (OUT / "q2_tone_conducted.json").write_text(json.dumps(tone, indent=2), encoding="utf-8")
    (OUT / "snapshot_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print("wrote", OUT / "q2_tone_conducted.json", "runs", len(tone["runs"]),
          "within_200khz", tone["all_within_200khz"], "tone_present", tone["all_tone_present"])
    worst = max(tone["runs"], key=lambda r: abs(r["offset_hz"]))
    print("largest |offset|", worst["board"], worst["esp_freq_mhz"], worst["tag"], f"{worst['offset_hz']:.0f} Hz")
    phases = [r for r in params["runs"] if r["tag"] == "" and r["esp_freq_mhz"] == 2442 and r["board"].startswith("s3")]
    print("S3 2442 phase", [(r["board"], round(r["phase_mean_resultant_length"], 3)) for r in phases])
    print("phase classes", sorted({r["phase_class"] for r in params["runs"]}))
    print("power classes", sorted({r["power_class"] for r in params["runs"]}))
    print("untagged power std", params["untagged_power_robust_std_db"])


if __name__ == "__main__":
    main()
