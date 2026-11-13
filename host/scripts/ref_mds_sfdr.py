"""Q8 reference MDS + SFDR on LimeSDR Mini or HackRF One RX (same conducted chain as ESP).

Chain: RFE6GEN -> combiner/att/pigtail -> DUT RX. Reports analytic MDS and single-tone SFDR
into results/conducted/<board>/.

Example:
  python host/scripts/ref_mds_sfdr.py --dut lime --board lime-mini --atten-db 37.7 --rx-gain 40
  python host/scripts/ref_mds_sfdr.py --dut hackrf --board hackrf-one --atten-db 37.7 --rx-gain 40
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.hackrfrx import HackRfRx
from snapcal.limerx import LimeRx
from snapcal.plutorx import PlutoRx
from snapcal.ric import estimate_tone
from snapcal.rfe import LEVELS, RfeGenerator, discover
import rf_metrics as rm

MIN_ATTEN_DB = 20.0  # reference SDRs tolerate less front-end pad than ESP


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measure(rx, f_bb, count):
    es = [estimate_tone(rx.capture(rm.N), rx.rate_hz, f_bb) for _ in range(count)]
    p = np.array([e.power_dbfs for e in es])
    return {
        "power_dbfs": float(np.median(p)),
        "snr_db": float(np.median([e.snr_db for e in es])),
        "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
        "enbw_hz": float(1.5 * rx.rate_hz / rm.N),
    }


def averaged_psd(rx, count):
    n = rm.N
    w = np.hanning(n)
    acc = np.zeros(n)
    for _ in range(count):
        x = rx.capture(n).astype(np.complex128)
        x = x - np.mean(x)
        # resample note: capture length n at rx.rate_hz
        if len(x) < n:
            raise RuntimeError(f"short capture {len(x)}")
        x = x[:n]
        acc += np.abs(np.fft.fft(x * w)) ** 2
    P = np.fft.fftshift(acc / count) / np.sum(w) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / rx.rate_hz))
    return f, P


def open_rx(a):
    lo = a.esp_freq * 1e6
    if a.dut == "lime":
        return LimeRx(lo, a.rx_gain, a.rx_rate or 30e6, antenna=a.antenna or "Auto")
    if a.dut == "hackrf":
        # 10 MS/s is stable under usbipd; keep |bb| << rate/2 (use --bb-mhz 2)
        return HackRfRx(lo, a.rx_gain, a.rx_rate or 10e6, antenna=a.antenna or "TX/RX")
    if a.dut == "pluto":
        return PlutoRx(lo, a.rx_gain, a.rx_rate or 20e6, uri=a.pluto_uri)
    raise SystemExit(f"unknown --dut {a.dut}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dut", choices=["lime", "hackrf", "pluto"], default="lime")
    ap.add_argument("--board", default="")
    ap.add_argument("--atten-db", type=float, required=True)
    ap.add_argument("--atten-note", default="combiner + att30_a + ipex (same as ESP IIP3 chain)")
    ap.add_argument("--esp-freq", type=int, default=2442, help="RX LO MHz")
    ap.add_argument("--bb-mhz", type=float, default=10.0)
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--lime-gain", type=float, default=None, help="alias for --rx-gain")
    ap.add_argument("--rx-rate", type=float, default=0.0, help="0 = dut default")
    ap.add_argument("--lime-rate", type=float, default=0.0, help="alias for --rx-rate")
    ap.add_argument("--antenna", default="")
    ap.add_argument("--pluto-uri", default="ip:192.168.20.100")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--sfdr-count", type=int, default=20)
    ap.add_argument("--snr-thresh", type=float, default=10.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen", default=None)
    a = ap.parse_args()
    if a.lime_gain is not None:
        a.rx_gain = a.lime_gain
    if a.lime_rate:
        a.rx_rate = a.lime_rate
    if not a.rx_rate:
        a.rx_rate = 0.0
    if not a.board:
        a.board = {"lime": "lime-mini", "hackrf": "hackrf-one", "pluto": "pluto"}[a.dut]
    if a.atten_db < MIN_ATTEN_DB:
        raise SystemExit(f"--atten-db {a.atten_db} < {MIN_ATTEN_DB}")

    f_bb = a.bb_mhz * 1e6
    f_rf = a.esp_freq * 1e6 + f_bb
    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    rd = ROOT / "results" / "conducted" / a.board
    rd.mkdir(parents=True, exist_ok=True)
    dut_name = {"lime": "LimeSDR Mini RX", "hackrf": "HackRF One RX", "pluto": "PlutoSDR RX"}[a.dut]
    out = {
        "utc": utc(), "board": a.board, "cmd": "ref_mds_sfdr", "dut": dut_name,
        "args": vars(a),
        "setup": f"RFE6GEN -> {a.atten_db:g} dB ({a.atten_note}) -> {dut_name}",
    }

    with RfeGenerator(gp) as gen, open_rx(a) as rx:
        out["generator"] = gen.info()
        out["rx_ready"] = rx.ready
        out["rx_rate_hz"] = rx.rate_hz
        print("READY", rx.ready, flush=True)

        print("A. RF-off", flush=True)
        gen.off()
        off = measure(rx, f_bb, a.count)
        out["rf_off"] = off
        print(f"  N0 {off['noise_dbfs_hz']:.1f} dBFS/Hz  tone-bin {off['power_dbfs']:.1f}", flush=True)

        print("B. level sweep", flush=True)
        rows = []
        for level in range(len(LEVELS)):
            gen.cw(f_rf / 1e6, level=level)
            r = measure(rx, f_bb, a.count)
            r["gen_level"] = level
            r["gen_est_dbm"] = gen.estimated_dbm()
            r["pin_dbm"] = r["gen_est_dbm"] - a.atten_db
            rows.append(r)
            print(f"  L{level} Pin {r['pin_dbm']:6.1f}  {r['power_dbfs']:6.1f} dBFS  "
                  f"SNR {r['snr_db']:5.1f}", flush=True)
        out["mds_sweep"] = rows
        mds = rm.interpolate_mds(rows, a.snr_thresh)
        cal = [r for r in rows if 20 <= r["snr_db"] <= 65]
        analytic = None
        if cal:
            scale = float(np.median([r["power_dbfs"] - r["pin_dbm"] for r in cal]))
            n0, enbw = off["noise_dbfs_hz"], off["enbw_hz"]
            mds_dbfs = n0 + 10 * np.log10(enbw) + a.snr_thresh
            analytic = {
                "scale_dbfs_per_dbm": scale, "cal_points": len(cal),
                "mds_dbfs": float(mds_dbfs), "mds_pin_dbm": float(mds_dbfs - scale),
                "snr_thresh_db": a.snr_thresh, "enbw_hz": enbw,
            }
        out["mds"] = {"snr_crossing": mds, "analytic_from_floor": analytic,
                      "atten_db": a.atten_db, "atten_note": a.atten_note}
        if analytic:
            print(f"  analytic MDS Pin ≈ {analytic['mds_pin_dbm']:.1f} dBm", flush=True)

        print("C. SFDR", flush=True)
        # strongest without absurd level — prefer L3-ish
        sfdr_row = max(rows[:4], key=lambda r: r["power_dbfs"]) if len(rows) >= 4 else rows[-1]
        gen.cw(f_rf / 1e6, level=sfdr_row["gen_level"])
        f, P = averaged_psd(rx, a.sfdr_count)
        tone = measure(rx, f_bb, max(a.count // 2, 5))
        sfdr = rm.find_sfdr(f, P, tone["freq_hz"], dc_hz=200e3)
        sfdr_ex = rm.find_sfdr(f, P, tone["freq_hz"], dc_hz=1e6)
        out["sfdr"] = {
            "gen_level": sfdr_row["gen_level"], "pin_dbm": sfdr_row["pin_dbm"],
            "tone_power_dbfs": tone["power_dbfs"], **sfdr,
            "exclude_lo_1mhz": sfdr_ex,
        }
        print(f"  SFDR {sfdr.get('sfdr_db')} dB  excl-LO {sfdr_ex.get('sfdr_db')} dB", flush=True)
        gen.off()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r["pin_dbm"] for r in rows], [r["snr_db"] for r in rows], "o-")
    ax[0].axhline(a.snr_thresh, color="grey", ls=":")
    if analytic:
        ax[0].axvline(analytic["mds_pin_dbm"], color="C2", ls="--",
                      label=f"analytic {analytic['mds_pin_dbm']:.1f}")
    ax[0].set_xlabel("Pin (dBm)"); ax[0].set_ylabel("SNR (dB)")
    ax[0].set_title(f"{a.board} MDS"); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)
    ax[1].plot(f / 1e6, 10 * np.log10(P + 1e-30), lw=0.6)
    if sfdr.get("spur_f_bb_hz") is not None:
        ax[1].plot(sfdr["carrier_f_bb_hz"] / 1e6, sfdr["carrier_dbfs_bin"], "g^")
        ax[1].plot(sfdr["spur_f_bb_hz"] / 1e6, sfdr["spur_dbfs_bin"], "rv")
    ax[1].set_xlabel("baseband (MHz)"); ax[1].set_title(f"SFDR {sfdr.get('sfdr_db')} dB")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()

    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()
    stem = f"mds_sfdr_{a.esp_freq}" + (f"_{a.tag}" if a.tag else "")
    (rd / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig.savefig(rd / f"{stem}.png", dpi=130)
    print("wrote", rd / f"{stem}.json")

    summary_path = ROOT / "results" / "rf_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": []}
    summary.setdefault("runs", []).append({
        "utc": out["utc"], "board": a.board, "file": str(rd / f"{stem}.json"),
        "mds": out.get("mds"), "sfdr": {"sfdr_db": out["sfdr"].get("sfdr_db")},
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
