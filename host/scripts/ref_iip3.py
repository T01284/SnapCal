"""Q8 reference two-tone IIP3: RFE + (Lime|Pluto) TX -> combiner -> att -> DUT RX.

Same definition as host/scripts/iip3.py (ESP). Default tones +5/+7 MHz.
--tx lime  (default for pluto/hackrf DUT): Lime Mini TX, digital offset 2 MHz
--tx pluto (default for lime DUT):        Pluto TX at BB = f2 (same LO as RX plan)

Example:
  python host/scripts/ref_iip3.py --dut pluto --atten-db 37.7 --rx-gain 10
  python host/scripts/ref_iip3.py --dut lime --tx pluto --atten-db 37.7 --rx-gain 40
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
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
from snapcal.limetx import LimeTx
from snapcal.plutorx import PlutoRx
from snapcal.plutotx import PlutoTx
from snapcal.ric import estimate_tone
from snapcal.rfe import RfeGenerator, discover

N = 16384
MIN_ATTEN_DB = 20.0
LIME_TX_GAINS = [20, 25, 30, 35, 40, 45, 50]
PLUTO_TX_GAINS = [-50, -40, -30, -20, -10, -5, 0]


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def measure_tone(rx, f_bb, count):
    es = []
    for _ in range(count):
        x = rx.capture(N)
        es.append(estimate_tone(x, rx.rate_hz, f_bb))
    p = np.array([e.power_dbfs for e in es])
    return {
        "power_dbfs": float(np.median(p)),
        "snr_db": float(np.median([e.snr_db for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
        "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es])),
        "clip_fraction_max": float(np.mean(np.abs(rx.capture(N)) > 0.95)),
    }


def open_rx(a):
    lo = a.lo_mhz * 1e6
    if a.dut == "pluto":
        return PlutoRx(lo, a.rx_gain, a.rx_rate or 20e6, uri=a.pluto_uri)
    if a.dut == "lime":
        return LimeRx(lo, a.rx_gain, a.rx_rate or 30e6, antenna=a.antenna or "Auto")
    if a.dut == "hackrf":
        return HackRfRx(lo, a.rx_gain, a.rx_rate or 10e6, antenna=a.antenna or "TX/RX")
    raise SystemExit(a.dut)


def open_tx(a, tone2_lo_hz, tone2_offset_hz):
    if a.tx == "lime":
        return LimeTx(tone2_lo_hz, tone2_offset_hz, a.tx_gains[0], max_gain_db=a.max_tx_gain)
    if a.tx == "pluto":
        return PlutoTx(
            tone2_lo_hz, tone2_offset_hz, a.tx_gains[0],
            uri=a.pluto_uri, max_gain_db=a.max_tx_gain,
        )
    raise SystemExit(a.tx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dut", choices=["pluto", "lime", "hackrf"], default="pluto")
    ap.add_argument("--tx", choices=["lime", "pluto"], default=None,
                    help="Second-tone source. Default: pluto when DUT=lime, else lime.")
    ap.add_argument("--board", default="")
    ap.add_argument("--atten-db", type=float, required=True)
    ap.add_argument("--atten-note", default="combiner arm + att30_a + ipex")
    ap.add_argument("--lo-mhz", type=float, default=2442.0)
    ap.add_argument("--f1-mhz", type=float, default=5.0, help="RFE baseband offset")
    ap.add_argument("--f2-mhz", type=float, default=7.0, help="tone2 RF offset from RX LO")
    ap.add_argument("--lime-offset-mhz", type=float, default=2.0,
                    help="Lime TX digital offset (ignored for Pluto TX)")
    ap.add_argument("--rx-gain", type=float, default=10.0)
    ap.add_argument("--rx-rate", type=float, default=0.0)
    ap.add_argument("--antenna", default="")
    ap.add_argument("--pluto-uri", default="ip:192.168.20.100")
    ap.add_argument("--tx-gains", type=float, nargs="+", default=None)
    ap.add_argument("--lime-gains", type=float, nargs="+", default=None,
                    help="alias of --tx-gains (legacy)")
    ap.add_argument("--rfe-levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--max-tx-gain", type=float, default=None)
    ap.add_argument("--max-lime-gain", type=float, default=None, help="alias of --max-tx-gain")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--balance-db", type=float, default=3.0)
    ap.add_argument("--im3-snr-min", type=float, default=6.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen", default=None)
    a = ap.parse_args()

    if a.tx is None:
        a.tx = "pluto" if a.dut == "lime" else "lime"
    if a.dut == a.tx:
        raise SystemExit(f"DUT and tone2 TX cannot be the same device ({a.dut})")
    if not a.board:
        a.board = a.dut if a.dut != "lime" else "lime-mini"
    if a.tx_gains is None:
        a.tx_gains = a.lime_gains if a.lime_gains is not None else (
            PLUTO_TX_GAINS if a.tx == "pluto" else LIME_TX_GAINS)
    if a.max_tx_gain is None:
        a.max_tx_gain = a.max_lime_gain if a.max_lime_gain is not None else (
            0.0 if a.tx == "pluto" else 55.0)
    if a.dut == "lime" and a.rx_gain == 10.0:
        a.rx_gain = 40.0
    if a.dut == "hackrf" and a.f2_mhz >= 5:
        print("note: HackRF default 10 MS/s — keep |f1|,|f2| << 5 MHz", flush=True)
    if a.atten_db < MIN_ATTEN_DB:
        raise SystemExit(f"--atten-db {a.atten_db} < {MIN_ATTEN_DB}")

    f1, f2 = a.f1_mhz * 1e6, a.f2_mhz * 1e6
    im3_lo, im3_hi = 2 * f1 - f2, 2 * f2 - f1
    rfe_mhz = a.lo_mhz + a.f1_mhz
    if a.tx == "lime":
        tone2_offset = a.lime_offset_mhz * 1e6
        tone2_lo = a.lo_mhz * 1e6 + f2 - tone2_offset
    else:
        # Pluto DDS: same LO as RX plan, CW at +f2 (verified on bench).
        tone2_offset = f2
        tone2_lo = a.lo_mhz * 1e6
    probe_gain = -10.0 if a.tx == "pluto" else 50.0

    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    rd = ROOT / "results" / "conducted" / a.board
    rd.mkdir(parents=True, exist_ok=True)
    out = {
        "utc": utc(), "board": a.board, "cmd": "ref_iip3", "dut": a.dut, "tx": a.tx,
        "args": vars(a),
        "plan": {
            "rfe_mhz": rfe_mhz, "tone2_mhz": a.lo_mhz + a.f2_mhz,
            "tone2_lo_hz": tone2_lo, "tone2_offset_hz": tone2_offset,
            "f1_mhz": a.f1_mhz, "f2_mhz": a.f2_mhz,
            "im3_bb_mhz": [im3_lo / 1e6, im3_hi / 1e6],
        },
        "setup": f"RFE + {a.tx} TX -> combiner -> {a.atten_db:g} dB -> {a.dut} RX",
        "rows": [],
    }
    print(f"IIP3 {a.dut}  TX={a.tx}  LO {a.lo_mhz}  RFE +{a.f1_mhz}  tone2 +{a.f2_mhz}  "
          f"IM3 {[im3_lo/1e6, im3_hi/1e6]}", flush=True)

    with ExitStack() as stack:
        gen = stack.enter_context(RfeGenerator(gp))
        tx = stack.enter_context(open_tx(a, tone2_lo, tone2_offset))
        rx = stack.enter_context(open_rx(a))
        out["generator"] = gen.info()
        out["tx_ready"] = tx.ready
        out["rx_ready"] = rx.ready
        out["rx_rate_hz"] = rx.rate_hz
        print("READY tx", tx.ready, flush=True)
        print("READY rx", rx.ready, flush=True)

        print("0. wiring check", flush=True)
        tx.off(); gen.off(); time.sleep(0.3)
        off1, off2 = measure_tone(rx, f1, 6), measure_tone(rx, f2, 6)
        gen.cw(rfe_mhz, level=3); tx.off(); time.sleep(0.4)
        rfe_only = measure_tone(rx, f1, 8)
        gen.off(); tx.gain(probe_gain); tx.cw(); time.sleep(0.9)
        tx_only = measure_tone(rx, f2, 8)
        gen.cw(rfe_mhz, level=3); tx.cw(); time.sleep(0.4)
        both1, both2 = measure_tone(rx, f1, 8), measure_tone(rx, f2, 8)
        checks = {
            "rfe": rfe_only["power_dbfs"] >= off1["power_dbfs"] + 8,
            "tx": tx_only["power_dbfs"] >= off2["power_dbfs"] + 8,
            "both_rfe": both1["power_dbfs"] >= off1["power_dbfs"] + 8,
            "both_tx": both2["power_dbfs"] >= off2["power_dbfs"] + 8,
        }
        out["wiring_check"] = {
            "off": {"f1": off1, "f2": off2}, "rfe_only": rfe_only, "tx_only": tx_only,
            "both": {"f1": both1, "f2": both2}, "checks": checks, "ok": all(checks.values()),
        }
        print(f"  RFE-only {rfe_only['power_dbfs']:.1f}  {a.tx}-only {tx_only['power_dbfs']:.1f}  "
              f"both {both1['power_dbfs']:.1f}/{both2['power_dbfs']:.1f}  "
              f"{'PASS' if out['wiring_check']['ok'] else 'FAIL'}", flush=True)
        if not out["wiring_check"]["ok"]:
            raise SystemExit("wiring check failed — see JSON / reconnect")

        print("A. RFE-only scale", flush=True)
        tx.off()
        cal = []
        for level in a.rfe_levels:
            gen.cw(rfe_mhz, level=level); time.sleep(0.3)
            r = measure_tone(rx, f1, a.count)
            pin = gen.estimated_dbm() - a.atten_db
            r |= {"gen_level": level, "gen_est_dbm": gen.estimated_dbm(), "pin_dbm": pin,
                  "scale": r["power_dbfs"] - pin}
            cal.append(r)
            print(f"  L{level} Pin {pin:6.1f}  {r['power_dbfs']:6.1f} dBFS  scale {r['scale']:.1f}", flush=True)
        gen.off()
        usable = [r for r in cal if r["snr_db"] > 20 and r["power_dbfs"] < 5]
        if not usable:
            usable = cal
        scale = float(np.median([r["scale"] for r in usable]))
        out["rfe_cal"] = {"rows": cal, "scale_dbfs_per_dbm": scale}
        print(f"  scale = {scale:.2f}", flush=True)

        print("B. dual-tone sweep", flush=True)
        for level in a.rfe_levels:
            gen.cw(rfe_mhz, level=level); time.sleep(0.3)
            p1_ref = measure_tone(rx, f1, max(a.count // 2, 6))["power_dbfs"]
            best = None
            for g in a.tx_gains:
                if g > a.max_tx_gain:
                    continue
                tx.gain(g); tx.cw(); time.sleep(0.25)
                p2 = measure_tone(rx, f2, max(a.count // 2, 6))["power_dbfs"]
                err = abs(p2 - p1_ref)
                if best is None or err < best["err"]:
                    best = {"gain": g, "p2": p2, "err": err}
            tx.gain(best["gain"]); tx.cw(); time.sleep(0.35)
            t1 = measure_tone(rx, f1, a.count)
            t2 = measure_tone(rx, f2, a.count)
            lo = measure_tone(rx, im3_lo, a.count)
            hi = measure_tone(rx, im3_hi, a.count)
            pin1, pin2 = t1["power_dbfs"] - scale, t2["power_dbfs"] - scale
            pin_avg = 0.5 * (pin1 + pin2)
            p_avg = 0.5 * (t1["power_dbfs"] + t2["power_dbfs"])
            im3 = lo if lo["power_dbfs"] >= hi["power_dbfs"] else hi
            which = "lo" if im3 is lo else "hi"
            delta = p_avg - im3["power_dbfs"]
            iip3 = pin_avg + 0.5 * delta
            row = {
                "gen_level": level, "gen_est_dbm": gen.estimated_dbm(),
                "tx_gain": best["gain"], "balance_err_db": best["err"],
                "tone1": t1, "tone2": t2, "im3_lo": lo, "im3_hi": hi, "im3_used": which,
                "pin1_dbm": pin1, "pin2_dbm": pin2, "pin_avg_dbm": pin_avg,
                "delta_fund_im3_db": delta, "iip3_dbm": iip3,
                "im3_snr_ok": im3["snr_db"] >= a.im3_snr_min,
                "clip_ok": t1["power_dbfs"] < 5 and t2["power_dbfs"] < 5,
                "balanced": best["err"] <= a.balance_db,
            }
            out["rows"].append(row)
            flags = [k for k in ("im3_snr_ok", "clip_ok", "balanced") if not row[k]]
            print(f"  L{level}/g{best['gain']:.0f}: P1 {t1['power_dbfs']:6.1f} P2 {t2['power_dbfs']:6.1f} "
                  f"IM3[{which}] {im3['power_dbfs']:6.1f}  IIP3 {iip3:6.1f} dBm  "
                  f"{flags or 'ok'}", flush=True)

        gen.off(); tx.off()

    fit = [r for r in out["rows"] if r["im3_snr_ok"] and r["clip_ok"] and r["balanced"]]
    if fit:
        vals = [r["iip3_dbm"] for r in fit]
        out["iip3"] = {
            "median_dbm": float(np.median(vals)), "mean_dbm": float(np.mean(vals)),
            "std_dbm": float(np.std(vals)), "n": len(fit),
            "definition": "IIP3 = Pin_avg + (P_avg - P_IM3)/2; Pin via RFE-only scale",
            "atten_db": a.atten_db, "atten_note": a.atten_note,
        }
        print(f"IIP3 median {out['iip3']['median_dbm']:.1f} dBm (n={len(fit)})", flush=True)
    else:
        out["iip3"] = None
        print("IIP3: no rows passed cuts", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pins = [r["pin_avg_dbm"] for r in out["rows"]]
    ax[0].plot(pins, [r["tone1"]["power_dbfs"] for r in out["rows"]], "o-", label="P1")
    ax[0].plot(pins, [r["tone2"]["power_dbfs"] for r in out["rows"]], "s-", label="P2")
    ax[0].plot(pins, [r["im3_lo"]["power_dbfs"] for r in out["rows"]], "v-", label="IM3 lo")
    ax[0].plot(pins, [r["im3_hi"]["power_dbfs"] for r in out["rows"]], "^-", label="IM3 hi")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3); ax[0].set_title(f"{a.board} tones")
    ax[1].plot(pins, [r["iip3_dbm"] for r in out["rows"]], "o-")
    if out["iip3"]:
        ax[1].axhline(out["iip3"]["median_dbm"], color="C3", ls="--",
                      label=f"median {out['iip3']['median_dbm']:.1f}")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.3); ax[1].set_title("IIP3")
    fig.tight_layout()

    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()
    stem = f"iip3_{int(a.lo_mhz)}" + (f"_{a.tag}" if a.tag else "")
    (rd / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig.savefig(rd / f"{stem}.png", dpi=130)
    print("wrote", rd / f"{stem}.json")

    summary_path = ROOT / "results" / "rf_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": []}
    summary.setdefault("runs", []).append({
        "utc": out["utc"], "board": a.board, "tx": a.tx,
        "file": str(rd / f"{stem}.json"), "iip3": out.get("iip3"),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
