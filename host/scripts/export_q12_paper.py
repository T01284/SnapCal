"""Export Q12 Logic fits into paper/data/q12_snapshot_modes.json for table 2.

Reads results/snapshot_timing.json (+ LA files for duty at n=16380).
Does not invent numbers; re-run after new LA captures.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

ST = ROOT / "results" / "snapshot_timing"
ROLLUP = ROOT / "results" / "snapshot_timing.json"
OUT = ROOT / "paper" / "data" / "q12_snapshot_modes.json"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def duty_at(name: str, fmt: str, rate: int, n: int = 16380):
    d = json.loads((ST / name).read_text(encoding="utf-8"))
    for r in d["rows"]:
        if r.get("format") == fmt and r["rate_hz"] == rate:
            b = r["by_n"][str(n)]
            tw = b["Tw_mark_us_median"] * 1e-6
            tc = b["T_cycle_ms_median"] * 1e-3
            return {
                "duty_pct": round(100.0 * tw / tc, 3),
                "Tw_mark_us_median": b["Tw_mark_us_median"],
                "T_cycle_ms_median": b["T_cycle_ms_median"],
                "Tw_nominal_us": b["Tw_nominal_us"],
            }
    return None


def main():
    roll = json.loads(ROLLUP.read_text(encoding="utf-8"))
    h = roll["headline"]

    def row(mode, board, link, c_key, duty_file, note=""):
        fit = h[c_key]
        dty = duty_at(duty_file, "iq8", 80_000_000) if duty_file else None
        return {
            "mode": mode,
            "board": board,
            "link": link,
            "format": "iq8",
            "rate_hz": 80_000_000,
            "n": 16380,
            "c": fit["c"],
            "T0_ms": fit["T0_ms"],
            "duty_pct_at_nmax": None if not dty else dty["duty_pct"],
            "Tw_mark_us_median": None if not dty else dty["Tw_mark_us_median"],
            "T_cycle_ms_median": None if not dty else dty["T_cycle_ms_median"],
            "Tw_nominal_us": 204.75,
            "source": fit["source"],
            "method": fit.get("method", "logic"),
            "note": note,
        }

    modes = [
        row("raw_iq", "s3-wroom1u-01", "USB",
            "raw_iq8_80Ms_s3_01_logic_fastpoll", "la_timing_fastpoll.json",
            "POLL 1; SIP baseline upload"),
        row("energy_miss", "s3-wroom1u-01", "USB",
            "energy_miss_iq8_80Ms_s3_01_logic", "la_timing_s3-01-energy-miss.json",
            "CAPE* miss path; paper on-chip detect when hits are rare"),
        row("energy_hit", "s3-wroom1u-01", "USB",
            "energy_hit_iq8_80Ms_s3_01_logic", "la_timing_s3-01-energy-hit.json",
            "CAPE* hit path; ~raw IQ upload cost"),
        row("raw_iq", "c5-03", "UART_1M",
            "raw_iq8_80Ms_c5_03_logic_fastpoll", "la_timing_c5-03-fastpoll.json",
            "POLL 1; UART shift time dominates c"),
        row("energy_miss", "c5-03", "UART_1M",
            "energy_miss_iq8_80Ms_c5_03_logic", "la_timing_c5-03-energy-miss.json",
            "CAPE* miss; mark width includes C5 stock_capture overhead"),
        row("energy_hit", "c5-03", "UART_1M",
            "energy_hit_iq8_80Ms_c5_03_logic", "la_timing_c5-03-energy-hit.json",
            "CAPE* hit ≈ raw IQ UART"),
    ]

    spec = []
    for s in roll["spec_duty"]:
        if s["board"] not in ("s3-wroom1u-01", "c5-01"):
            continue
        for fs, block in s["summary"].items():
            spec.append({
                "board": s["board"],
                "link": s["transport"],
                "rate_hz": int(fs),
                "duty_pct_range": block["duty_pct_range"],
                "duty_pct_by_nfft": block["duty_pct_by_nfft"],
                "source": s["source"],
                "note": "SPEC has no c/T0 fit; duty from on-chip FFT stream",
            })

    Tw_max_us = {
        "80000000": 16380 / 80e6 * 1e6,
        "40000000": 16380 / 40e6 * 1e6,
        "16000000": 16380 / 16e6 * 1e6,
        "20000000": 16380 / 20e6 * 1e6,
    }

    payload = {
        "utc": utc(),
        "question": "Q12",
        "rollup": "results/snapshot_timing.json",
        "Tw_max_us_nominal": {k: round(v, 2) for k, v in Tw_max_us.items()},
        "IQ_WORDS": 16380,
        "modes": modes,
        "spec_duty": spec,
        "retune_to_2442_ms_median": h["retune_to_2442_ms_median"],
        "table2_primary": {
            "comment": "Sensors Letters table 2: S3 USB IQ8 @80 MS/s, busy-poll; detect = CAPE miss",
            "raw_iq8": {"c": h["raw_iq8_80Ms_s3_01_logic_fastpoll"]["c"],
                        "T0_ms": h["raw_iq8_80Ms_s3_01_logic_fastpoll"]["T0_ms"],
                        "duty_pct": modes[0]["duty_pct_at_nmax"]},
            "spec_80Ms": {"c": None, "T0_ms": None, "duty_pct_range": [7.78, 11.32],
                          "source": "results/snapshot_timing/spec_timing_s3-wroom1u-01.json"},
            "energy_miss": {"c": h["energy_miss_iq8_80Ms_s3_01_logic"]["c"],
                            "T0_ms": h["energy_miss_iq8_80Ms_s3_01_logic"]["T0_ms"],
                            "duty_pct": modes[1]["duty_pct_at_nmax"]},
        },
        "note": h.get("note"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    t = payload["table2_primary"]
    print("table2 S3:", t)


if __name__ == "__main__":
    main()
