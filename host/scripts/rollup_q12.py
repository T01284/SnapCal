"""Roll Q12 timing artefacts into results/snapshot_timing.json.

Reads LA fits, SPEC duty, retune medians, and CAPE* host fits under
results/snapshot_timing/. Does not touch the board.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT

DIR = ROOT / "results" / "snapshot_timing"
OUT = ROOT / "results" / "snapshot_timing.json"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def load(name: str):
    p = DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def la_rows(name: str, board: str, method: str, transport: str):
    d = load(name)
    if not d:
        return []
    out = []
    for r in d.get("rows", []):
        fit = r.get("fit") or {}
        out.append({
            "board": board,
            "source": rel(DIR / name),
            "method": method,
            "transport": transport,
            "fast_poll": bool(d.get("fast_poll")),
            "format": r.get("format"),
            "rate_hz": r.get("rate_hz"),
            "c": fit.get("c"),
            "T0_ms": fit.get("T0_ms"),
        })
    return out


def energy_rows(name: str, board: str, transport: str, method: str = "host"):
    d = load(name)
    if not d:
        return []
    out = []
    for r in d.get("rows", []):
        fit = r.get("fit") or {}
        hit_frac = r.get("hit_frac")
        if hit_frac is None and d.get("energy_thresh") is not None:
            # Older host-only files put hit_frac on each row; LA files too after energy mode.
            hit_frac = r.get("hit_frac")
        path = "hit" if (hit_frac or 0) >= 0.5 else "miss"
        c = fit.get("c") if fit else r.get("c")
        t0 = fit.get("T0_ms") if fit else r.get("T0_ms")
        out.append({
            "board": board,
            "source": rel(DIR / name),
            "method": method,
            "transport": transport,
            "mode": "energy",
            "path": path,
            "thresh": r.get("energy_thresh", r.get("thresh", d.get("energy_thresh"))),
            "hit_frac": hit_frac,
            "format": r.get("format"),
            "rate_hz": r.get("rate_hz"),
            "c": c,
            "T0_ms": t0,
            "code_p50": r.get("energy_code_p50", r.get("code_p50")),
            "fast_poll": bool(d.get("fast_poll")),
        })
    return out


def retune_entry(name: str, board: str):
    d = load(name)
    if not d:
        return None
    by = d.get("by_mhz") or {}
    return {
        "board": board,
        "source": rel(DIR / name),
        "gate_pin": d.get("gate_pin"),
        "to_2442_ms_median": (by.get("2442") or {}).get("cmd_ms_median"),
        "to_2472_ms_median": (by.get("2472") or {}).get("cmd_ms_median"),
        "by_mhz": {
            k: {
                "n": v.get("n"),
                "cmd_ms_median": v.get("cmd_ms_median"),
                "cmd_ms_min": v.get("cmd_ms_min"),
                "cmd_ms_max": v.get("cmd_ms_max"),
            }
            for k, v in by.items()
        },
    }


def spec_duty(name: str, board: str, transport: str):
    d = load(name)
    if not d:
        return None
    by_fs = {}
    for p in d.get("profiles", []):
        fs = int(p["fs"])
        nfft = int(p["nfft"])
        by_fs.setdefault(fs, {})[nfft] = round(float(p["duty_from_ffts"]) * 100.0, 2)
    # Compact 80/40(/16) × 256..2048 for the usual paper quote.
    summary = {}
    for fs in (80_000_000, 40_000_000, 16_000_000, 20_000_000):
        if fs not in by_fs:
            continue
        vals = [by_fs[fs][n] for n in (256, 512, 1024, 2048) if n in by_fs[fs]]
        if vals:
            summary[str(fs)] = {
                "duty_pct_by_nfft": {str(n): by_fs[fs][n] for n in (256, 512, 1024, 2048) if n in by_fs[fs]},
                "duty_pct_range": [min(vals), max(vals)],
            }
    return {
        "board": board,
        "source": rel(DIR / name),
        "transport": transport,
        "summary": summary,
    }


def main():
    raw_iq = []
    raw_iq += la_rows("la_timing.json", "s3-wroom1u-01", "logic", "USB")
    raw_iq += la_rows("la_timing_fastpoll.json", "s3-wroom1u-01", "logic", "USB")
    raw_iq += la_rows("la_timing_c5-03.json", "c5-03", "logic", "UART_1M")
    raw_iq += la_rows("la_timing_c5-03-fastpoll.json", "c5-03", "logic", "UART_1M")

    energy = []
    energy += energy_rows("energy_timing_c5sdr_miss.json", "c5-03", "UART_1M")
    energy += energy_rows("energy_timing_c5sdr_hit.json", "c5-03", "UART_1M")
    energy += energy_rows("energy_timing_s3sdr_s3-03-miss.json", "s3-wroom1u-03", "USB")
    energy += energy_rows("energy_timing_s3sdr_s3-03-hit.json", "s3-wroom1u-03", "USB")
    energy += energy_rows("la_timing_s3-01-energy-miss.json", "s3-wroom1u-01", "USB", method="logic")
    energy += energy_rows("la_timing_s3-01-energy-hit.json", "s3-wroom1u-01", "USB", method="logic")
    energy += energy_rows("la_timing_c5-03-energy-miss.json", "c5-03", "UART_1M", method="logic")
    energy += energy_rows("la_timing_c5-03-energy-hit.json", "c5-03", "UART_1M", method="logic")

    retune = [
        e for e in (
            retune_entry("retune_s3-wroom1u-01.json", "s3-wroom1u-01"),
            retune_entry("retune_s3-wroom1u-02.json", "s3-wroom1u-02"),
            retune_entry("retune_s3-wroom1u-03.json", "s3-wroom1u-03"),
            retune_entry("retune_c5-01.json", "c5-01"),
            retune_entry("retune_c5-02.json", "c5-02"),
            retune_entry("retune_c5-03.json", "c5-03"),
        ) if e
    ]

    spec = [
        e for e in (
            spec_duty("spec_timing_s3-wroom1u-01.json", "s3-wroom1u-01", "USB"),
            spec_duty("spec_timing_s3-wroom1u-02.json", "s3-wroom1u-02", "USB"),
            spec_duty("spec_timing_s3-wroom1u-03.json", "s3-wroom1u-03", "USB"),
            spec_duty("spec_timing_c5-01.json", "c5-01", "UART_1M"),
            spec_duty("spec_timing_c5-02.json", "c5-02", "UART_1M"),
            spec_duty("spec_timing_c5-03.json", "c5-03", "UART_1M"),
        ) if e
    ]

    # Headline numbers quoted in STAGE-TODO (IQ8 @ 80 MS/s, fast-poll where available).
    def pick(rows, board, fmt, rate, fast=None, path=None, method=None):
        for r in rows:
            if r.get("board") != board or r.get("format") != fmt or r.get("rate_hz") != rate:
                continue
            if fast is not None and r.get("fast_poll") != fast:
                continue
            if path is not None and r.get("path") != path:
                continue
            if method is not None and r.get("method") != method:
                continue
            return {"c": r.get("c"), "T0_ms": r.get("T0_ms"), "source": r.get("source"),
                    "method": r.get("method")}
        return None

    headline = {
        "raw_iq8_80Ms_s3_01_logic_fastpoll": pick(raw_iq, "s3-wroom1u-01", "iq8", 80_000_000, True),
        "raw_iq8_80Ms_c5_03_logic_fastpoll": pick(raw_iq, "c5-03", "iq8", 80_000_000, True),
        "energy_miss_iq8_80Ms_s3_01_logic": pick(
            energy, "s3-wroom1u-01", "iq8", 80_000_000, path="miss", method="logic"),
        "energy_hit_iq8_80Ms_s3_01_logic": pick(
            energy, "s3-wroom1u-01", "iq8", 80_000_000, path="hit", method="logic"),
        "energy_miss_iq8_80Ms_s3_03_host": pick(
            energy, "s3-wroom1u-03", "iq8", 80_000_000, path="miss", method="host"),
        "energy_hit_iq8_80Ms_s3_03_host": pick(
            energy, "s3-wroom1u-03", "iq8", 80_000_000, path="hit", method="host"),
        "energy_miss_iq8_80Ms_c5_03_logic": pick(
            energy, "c5-03", "iq8", 80_000_000, path="miss", method="logic"),
        "energy_hit_iq8_80Ms_c5_03_logic": pick(
            energy, "c5-03", "iq8", 80_000_000, path="hit", method="logic"),
        "energy_miss_iq8_80Ms_c5_03_host": pick(
            energy, "c5-03", "iq8", 80_000_000, path="miss", method="host"),
        "energy_hit_iq8_80Ms_c5_03_host": pick(
            energy, "c5-03", "iq8", 80_000_000, path="hit", method="host"),
        "retune_to_2442_ms_median": {
            e["board"]: e["to_2442_ms_median"] for e in retune if e.get("to_2442_ms_median") is not None
        },
        "note": (
            "Raw IQ and CAPE* c/T0 for table 2: prefer Logic GPIO fits (S3-01 and C5-03). "
            "S3 USB and C5 UART c are not directly comparable."
        ),
    }

    payload = {
        "utc": utc(),
        "question": "Q12",
        "headline": headline,
        "raw_iq": raw_iq,
        "energy_detect": energy,
        "retune": retune,
        "spec_duty": spec,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    for k, v in headline.items():
        if k == "note":
            continue
        print(k, v)


if __name__ == "__main__":
    main()
