"""Q19: bang-bang window choice with on-chip energy detect (CAPE* miss path).

CAPE miss has small c (~23 on S3 USB), so tau* rises to ~20 us. Bursts shorter than
tau* should favour the longest window by intercept rate; longer bursts favour the
shortest. Lime TX keys conducted bursts; ESP uses CAPE8 and counts HIT replies.

Example:
  python -u host/scripts/energy_intercept.py --port COM14 --conducted --lime-gain 64
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import sip
from snapcal.espsdr import EspSdr
from snapcal.limetx import LimeTx

FS = 80_000_000


def busy_wait(dt):
    end = time.perf_counter() + dt
    while time.perf_counter() < end:
        pass


def run_energy(dev, n, thresh, seconds, jitter_host, fw_jitter_us):
    hits, starts = 0, []
    t_end = time.perf_counter() + seconds
    if fw_jitter_us:
        dev.set_jitter(fw_jitter_us)
    while time.perf_counter() < t_end:
        if jitter_host > 0:
            busy_wait(random.uniform(0, jitter_host))
        starts.append(time.perf_counter())
        r = dev.capture_energy(n, FS, "iq8", thresh)
        if r.hit:
            hits += 1
    if fw_jitter_us:
        dev.set_jitter(0)
    count = len(starts)
    dt = np.diff(starts) if count > 1 else np.array([np.nan])
    return {
        "snapshots": count,
        "hits": hits,
        "p_hit": hits / count if count else 0.0,
        "t_p_s": float(np.mean(dt)),
        "t_p_std_s": float(np.std(dt)) if count > 2 else 0.0,
        "rate_hz": hits / (starts[-1] - starts[0]) if count > 1 else 0.0,
    }


def codes_sample(dev, n, thresh, count):
    vals = []
    for _ in range(count):
        r = dev.capture_energy(n, FS, "iq8", thresh)
        vals.append(r.code)
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--board", default="s3-wroom1u-01")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--lime-lo", type=float, default=2450e6)
    ap.add_argument("--lime-offset", type=float, default=2e6)
    ap.add_argument("--lime-gain", type=float, default=64)
    ap.add_argument("--period", type=float, default=10.37e-3)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 500e-6])
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 2048, 16380])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fw-jitter-us", type=int, default=1000)
    ap.add_argument("--spdt-pin", type=int, default=4)
    ap.add_argument("--tau-min", type=float, default=2e-6)
    ap.add_argument("--c-miss", type=float, default=23.3,
                    help="CAPE miss c from Q12 (S3 USB)")
    ap.add_argument("--t0", type=float, default=0.54e-3, help="T0 seconds from Q12")
    ap.add_argument("--conducted", action="store_true")
    ap.add_argument("--thresh", type=int, default=0,
                    help="CAPE energy code threshold; 0 = auto (FA-constrained, not CW-floor)")
    ap.add_argument("--fa-target", type=float, default=0.01,
                    help="Auto-thresh: max P_fa on the longest window")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    t_star = float(sip.tau_star(a.tau_min, a.c_miss, a.t0))
    out = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "q": "Q19",
        "board": a.board,
        "args": vars(a),
        "setup": ("conducted Lime TX -> att -> SPDT/coupler -> ESP IPEX"
                  if a.conducted else "OTA"),
        "tau_star_s": t_star,
        "tau_star_us": t_star * 1e6,
        "theory_best": {},
    }
    for tau in a.taus:
        lo, hi = min(a.windows) / FS, max(a.windows) / FS
        best = sip.best_window(tau, lo, hi, a.tau_min, a.c_miss, a.t0)
        out["theory_best"][str(tau)] = {
            "tau_us": tau * 1e6,
            "best_tw_us": best * 1e6,
            "side": "lt_tau_star" if tau < t_star else "gt_tau_star",
        }

    print(f"tau* = {t_star*1e6:.2f} us (c={a.c_miss}, T0={a.t0*1e3:.2f} ms)", flush=True)
    with LimeTx(a.lime_lo, a.lime_offset, a.lime_gain) as tx, EspSdr(a.port) as dev:
        print("Lime:", tx.ready, flush=True)
        if "ENERGY" not in dev.info()["caps"]:
            raise SystemExit("firmware lacks ENERGY")
        dev.set_freq(a.esp_freq)
        dev.set_gain(a.esp_gain)
        if a.spdt_pin:
            dev.gpio(a.spdt_pin, "1")

        # Calibrate energy codes at longest window.
        # Mean-power CAPE dilutes short bursts by ~tau/Tw, so a CW-floor thresh
        # (on_p10-2) rejects tau < Tw. Prefer FA-constrained thresh instead.
        n_cal = max(a.windows)
        tx.off()
        time.sleep(0.2)
        # thresh=0 always HIT so we can read codes
        off_codes = codes_sample(dev, n_cal, 0, 60)
        tx.cw()
        time.sleep(0.2)
        on_codes = codes_sample(dev, n_cal, 0, 40)
        off_p90 = float(np.percentile(off_codes, 90))
        off_p99 = float(np.percentile(off_codes, 99))
        on_p10 = float(np.percentile(on_codes, 10))
        if on_p10 <= off_p90 + 2:
            raise SystemExit(
                f"CW energy not above quiet: off_p90={off_p90:.0f} on_p10={on_p10:.0f}; "
                "raise --lime-gain or lower attenuator"
            )
        if a.thresh > 0:
            thr = int(a.thresh)
            thr_mode = "manual"
        else:
            # Short bursts are diluted by ~tau/Tw under mean-power CAPE, so a
            # CW-floor thresh rejects tau < Tw. Quiet also has rare full-scale
            # spikes (beacon/leakage) that pin FA-sweeps near CW. Use quiet p90
            # + margin, capped below CW p10, and judge bang-bang by excess P.
            margin = 8
            thr = int(np.clip(round(off_p90 + margin), 1, max(1, int(on_p10) - 5)))
            thr_mode = "off_p90_margin"
        out["energy_cal"] = {
            "n": n_cal,
            "off_code_p50": float(np.median(off_codes)),
            "off_code_p90": off_p90,
            "off_code_p99": off_p99,
            "on_code_p50": float(np.median(on_codes)),
            "on_code_p10": on_p10,
            "thresh": thr,
            "thresh_mode": thr_mode,
            "fa_target": a.fa_target,
        }
        print(
            f"energy codes quiet p50/p90={out['energy_cal']['off_code_p50']:.0f}/"
            f"{off_p90:.0f}  CW p10/p50={on_p10:.0f}/{out['energy_cal']['on_code_p50']:.0f}  "
            f"thresh={thr} ({thr_mode})",
            flush=True,
        )

        # Verify FA / CW hit with chosen thresh
        tx.off()
        fa = {}
        for n in a.windows:
            r = run_energy(dev, n, thr, max(a.seconds / 3, 5), 0, a.fw_jitter_us)
            fa[str(n)] = r
            print(f"TX off n={n:5d}: P_fa={r['p_hit']:.4f} rate={r['rate_hz']:.2f}/s "
                  f"over {r['snapshots']}", flush=True)
        out["false_alarm"] = fa
        tx.cw()
        cw = {}
        for n in a.windows:
            r = run_energy(dev, n, thr, 3, 0, 0)
            cw[str(n)] = r
            print(f"CW     n={n:5d}: P_hit={r['p_hit']:.4f}", flush=True)
        out["cw_hit"] = cw

        rows = []
        for tau in a.taus:
            tx.burst(tau, a.period)
            time.sleep(0.15)
            side = "lt_tau_star" if tau < t_star else "gt_tau_star"
            for n in a.windows:
                r = run_energy(dev, n, thr, a.seconds, 0, a.fw_jitter_us)
                t_w = n / FS
                p_th = float(sip.p_int(t_w, tau, a.tau_min, a.c_miss, a.t0))
                # expected rate ≈ p_int / T_cycle, T_cycle ≈ (1+c)T_w + T0 for miss-dominated
                t_cycle = (1 + a.c_miss) * t_w + a.t0
                rate_th = p_th / t_cycle
                row = r | {
                    "tau_s": tau,
                    "n": n,
                    "t_w_s": t_w,
                    "side": side,
                    "p_theory": p_th,
                    "rate_theory_hz": rate_th,
                    "mode": "fwjitter",
                }
                rows.append(row)
                print(
                    f"tau {tau*1e6:6.1f} us n={n:5d} {side}: "
                    f"P={r['p_hit']:.4f} rate={r['rate_hz']:6.2f}/s "
                    f"(th P={p_th:.4f} rate={rate_th:.2f}/s)  "
                    f"T_p={r['t_p_s']*1e3:.2f} ms",
                    flush=True,
                )
        out["bursts"] = rows
        tx.off()
        if a.spdt_pin:
            try:
                dev.gpio(a.spdt_pin, "0")
            except Exception:
                pass

    # Per-tau: extreme window with higher FA-subtracted intercept rate vs theory.
    # (Script docstring / SIP bang-bang compare rate; raw P is FA-noise dominated
    # when mean-power CAPE barely clears short bursts.)
    verdict = []
    for tau in a.taus:
        sub = [r for r in rows if r["tau_s"] == tau]
        extremes = [r for r in sub if r["n"] in (min(a.windows), max(a.windows))]

        def excess_rate(r):
            fa_r = float(fa.get(str(r["n"]), {}).get("rate_hz", 0.0))
            return r["rate_hz"] - fa_r

        meas_best = max(extremes, key=excess_rate)
        th_n = int(round(sip.best_window(
            tau, min(a.windows) / FS, max(a.windows) / FS, a.tau_min, a.c_miss, a.t0
        ) * FS))
        th_best_n = min(a.windows, key=lambda n: abs(n - th_n))
        verdict.append({
            "tau_us": tau * 1e6,
            "side": "lt_tau_star" if tau < t_star else "gt_tau_star",
            "meas_best_n": meas_best["n"],
            "theory_best_n": th_best_n,
            "agree": meas_best["n"] == th_best_n,
            "metric": "excess_rate_extremes",
            "meas_p_hit": {str(r["n"]): r["p_hit"] for r in sub},
            "meas_rates": {str(r["n"]): r["rate_hz"] for r in sub},
            "meas_excess_rate": {str(r["n"]): excess_rate(r) for r in sub},
        })
        print(
            f"tau={tau*1e6:.0f} us: meas best n={meas_best['n']} (by excess rate)  "
            f"theory best n={th_best_n}  agree={meas_best['n']==th_best_n}  "
            f"exR={{{', '.join(f'{r['n']}:{excess_rate(r):+.2f}/s' for r in extremes)}}}",
            flush=True,
        )
    out["verdict"] = verdict
    out["pass_sides"] = {
        "lt_tau_star": all(v["agree"] for v in verdict if v["side"] == "lt_tau_star"),
        "gt_tau_star": all(v["agree"] for v in verdict if v["side"] == "gt_tau_star"),
    }
    out["pass"] = bool(out["pass_sides"]["lt_tau_star"] and out["pass_sides"]["gt_tau_star"])

    rd = ROOT / "results" / "sip"
    rd.mkdir(parents=True, exist_ok=True)
    stem = f"energy_intercept_{a.board}" + (f"_{a.tag}" if a.tag else "")
    path = rd / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (rd / "q19_energy.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    for n, mk in zip(a.windows, ["o", "s", "^"]):
        R = [r for r in rows if r["n"] == n]
        ax.plot([r["tau_s"] * 1e6 for r in R], [r["rate_hz"] for r in R],
                marker=mk, label=f"n={n}")
    ax.axvline(t_star * 1e6, color="C3", ls="--", label=f"tau*={t_star*1e6:.1f} us")
    ax.set_xscale("log")
    ax.set_xlabel("tau (us)")
    ax.set_ylabel("intercept rate (1/s)")
    ax.set_title("Q19 CAPE miss-path bang-bang (fwjitter)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / f"{stem}.png", dpi=130)
    print("wrote", path, "pass", out["pass"], flush=True)


if __name__ == "__main__":
    main()
