"""SIP / sync-trap: LimeSDR keys a CW tone into periodic bursts (tau every T_b);
the ESP takes host-paced snapshots of length T_w.

Use --conducted for the cable bench (Lime TX -> attenuator -> coupler/SPDT ->
ESP IPEX). Over-the-air Phase-0 previews stay under results/phase0_bringup/;
conducted Q18–Q20 runs write results/sip/.

Scheduling modes
  periodic - back-to-back snapshots, stock firmware; the idle loop polls once per 1 ms
             FreeRTOS tick, so captures start on a 1 ms grid and with T_b = 10 ms
             the snapshot phase locks to the bursts (sync trap, Q20)
  random   - host-side uniform delay in [0, jitter] (re-quantised by the tick)
  fwjitter - on-chip JITTER: uniform delay in [0, fw_jitter] before each capture (ISS)
  fastpoll - on-chip POLL 1: busy polling, no tick grid

Detector: FFT band-pass of +-half_bw around the tone, a sample counts when the tone
envelope exceeds half the CW amplitude and is >= adj_ratio x the mean envelope of two
equal-width bands at +-1 MHz (wideband Wi-Fi fills those, the tone does not).
A hit needs >= tau_ov of such samples.

Geometry: P_snap = min(1, (T_w + tau - 2 tau_ov) / T_b) for uniformly random phase;
the plan's tau_min corresponds to 2 tau_ov here. Rate = P_snap / T_p (measured T_p).

Example:
  python -u host/scripts/burst_intercept.py --port COM31
"""
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
from snapcal.espsdr import EspSdr
from snapcal.limetx import LimeTx

FS = 80_000_000


class Detector:
    def __init__(self, f_bb, half_bw, adj_hz, adj_ratio, thr=None, min_len=80):
        self.f_bb, self.half_bw, self.adj_hz, self.adj_ratio = f_bb, half_bw, adj_hz, adj_ratio
        self.thr, self.min_len = thr, min_len

    def envelopes(self, x):
        X = np.fft.fft(x - np.mean(x))
        f = np.fft.fftfreq(len(x), 1 / FS)

        def band(fc):
            return np.abs(np.fft.ifft(np.where(np.abs(f - fc) <= self.half_bw, X, 0)))
        adj = 0.5 * (band(self.f_bb - self.adj_hz) + band(self.f_bb + self.adj_hz))
        return band(self.f_bb), adj

    def overlap_samples(self, x):
        e, adj = self.envelopes(x)
        return int(np.count_nonzero((e > self.thr) & (e >= self.adj_ratio * adj)))


def busy_wait(dt):
    end = time.perf_counter() + dt
    while time.perf_counter() < end:
        pass


def run(dev, det, n, fmt, seconds, jitter):
    hits, overlaps, starts = 0, [], []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        if jitter > 0:
            busy_wait(random.uniform(0, jitter))
        starts.append(time.perf_counter())
        k = det.overlap_samples(dev.capture(n, FS, fmt).iq)
        if k >= det.min_len:
            hits += 1
            overlaps.append(k / FS)
    dt = np.diff(starts)
    count = len(starts)
    return {"snapshots": count, "hits": hits, "p_hit": hits / count,
            "t_p_s": float(np.mean(dt)), "t_p_std_s": float(np.std(dt)),
            "overlap_us": [o * 1e6 for o in overlaps]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--lime-lo", type=float, default=2450e6)
    ap.add_argument("--lime-offset", type=float, default=2e6)
    ap.add_argument("--lime-gain", type=float, default=50)
    ap.add_argument("--period", type=float, default=10e-3)
    ap.add_argument("--taus", type=float, nargs="+", default=[20e-6, 100e-6, 500e-6, 2e-3])
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 2048, 16380])
    ap.add_argument("--modes", nargs="+", default=["periodic", "random"])
    ap.add_argument("--jitter", type=float, default=1e-3, help="host-side delay for mode 'random' (s)")
    ap.add_argument("--fw-jitter-us", type=int, default=1000, help="on-chip delay for mode 'fwjitter'")
    ap.add_argument("--fmt", default="iq8")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--half-bw", type=float, default=300e3)
    ap.add_argument("--adj-hz", type=float, default=1e6)
    ap.add_argument("--adj-ratio", type=float, default=4.0)
    ap.add_argument("--tau-ov", type=float, default=1e-6)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--conducted", action="store_true",
                    help="cable bench; write results/sip/ instead of phase0_bringup/")
    ap.add_argument("--board", default="", help="board id for filenames, e.g. s3-wroom1u-01")
    ap.add_argument("--spdt-pin", type=int, default=4,
                    help="drive this GPIO high for SPDT pass (0 disables)")
    ap.add_argument("--fa-only", action="store_true",
                    help="only CW calibrate + Lime-off P_fa (smoke for conducted ambient)")
    ap.add_argument("--max-lime-gain", type=float, default=64.0)
    a = ap.parse_args()
    f_bb = a.lime_lo + a.lime_offset - a.esp_freq * 1e6
    det = Detector(f_bb, a.half_bw, a.adj_hz, a.adj_ratio, min_len=int(round(a.tau_ov * FS)))

    setup = ("conducted: LimeSDR Mini TX -> att -> SPDT/coupler -> ESP IPEX; OUT terminated"
             if a.conducted else
             "over the air, LimeSDR Mini keyed CW -> ESP32-S3 PCB antenna")
    out = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "args": vars(a),
           "setup": setup, "q": "Q18-Q20" if a.conducted else "phase0"}
    print("starting Lime TX", flush=True)
    with LimeTx(a.lime_lo, a.lime_offset, a.lime_gain, max_gain_db=a.max_lime_gain) as tx, \
            EspSdr(a.port) as dev:
        print("Lime:", tx.ready, flush=True)
        dev.set_freq(a.esp_freq)
        dev.set_bandwidth(0)
        dev.set_gain(a.esp_gain)
        if a.spdt_pin:
            try:
                dev.gpio(a.spdt_pin, "1")
                print(f"SPDT GPIO{a.spdt_pin}=1 (pass)", flush=True)
            except Exception as e:
                print(f"SPDT GPIO warn: {e}", flush=True)
        out["firmware"] = dev.info()["version"]

        tx.cw()
        a_cw = float(np.median([np.median(det.envelopes(dev.capture(16380, FS, a.fmt).iq)[0]) for _ in range(50)]))
        det.thr = 0.5 * a_cw
        out["cw_envelope"], out["threshold"] = a_cw, det.thr
        cw_hit = [run(dev, det, n, a.fmt, 2, 0)["p_hit"] for n in a.windows]
        out["cw_p_hit"] = dict(zip(map(str, a.windows), cw_hit))
        print(f"CW envelope {a_cw:.4f}, threshold {det.thr:.4f}, CW P(hit) {cw_hit}", flush=True)

        def false_alarm(tag):
            tx.off()
            res = {}
            for n in a.windows:
                r = run(dev, det, n, a.fmt, a.seconds / 2, a.jitter)
                res[str(n)] = {k: v for k, v in r.items() if k != "overlap_us"}
                print(f"TX off ({tag}) n={n:5d}: P_fa {r['p_hit']:.4f} over {r['snapshots']} snapshots", flush=True)
            return res
        out["false_alarm_before"] = false_alarm("before")

        rows = []
        if not a.fa_only:
            for tau in a.taus:
                tx.burst(tau, a.period)
                for n in a.windows:
                    for mode in a.modes:
                        if mode in ("fwjitter", "fastpoll"):
                            dev.set_jitter(a.fw_jitter_us if mode == "fwjitter" else 0)
                            dev.set_fast_poll(mode == "fastpoll")
                        r = run(dev, det, n, a.fmt, a.seconds, a.jitter if mode == "random" else 0)
                        if mode in ("fwjitter", "fastpoll"):
                            dev.set_jitter(0)
                            dev.set_fast_poll(False)
                        t_w = n / FS
                        p_th = min(1.0, (t_w + tau - 2 * a.tau_ov) / a.period)
                        se = float(np.sqrt(max(r["p_hit"] * (1 - r["p_hit"]), 1e-12) / r["snapshots"]))
                        row = r | {"mode": mode, "tau_s": tau, "n": n, "t_w_s": t_w, "p_theory": p_th, "p_hit_se": se,
                                   "rate_meas_hz": r["p_hit"] / r["t_p_s"], "rate_pred_hz": p_th / r["t_p_s"]}
                        rows.append(row)
                        print(f"tau {tau*1e6:5.0f} us n={n:5d} {mode:8s}: P {r['p_hit']:.4f}+-{se:.4f} vs {p_th:.4f}  "
                              f"T_p {r['t_p_s']*1e3:6.2f}+-{r['t_p_std_s']*1e3:5.2f} ms  "
                              f"rate {row['rate_meas_hz']:6.2f}/s vs {row['rate_pred_hz']:6.2f}/s", flush=True)
            out["bursts"] = rows
            out["false_alarm_after"] = false_alarm("after")
        else:
            out["bursts"] = []
            print("fa-only: skipping burst sweep", flush=True)
        if a.spdt_pin:
            try:
                dev.gpio(a.spdt_pin, "0")
            except Exception:
                pass
    out["lime_log_tail"] = tx.log[-3:]

    rd = ROOT / "results" / ("sip" if a.conducted else "phase0_bringup")
    rd.mkdir(parents=True, exist_ok=True)
    slim = dict(out)
    slim["bursts"] = [{k: v for k, v in r.items() if k != "overlap_us"} for r in rows]
    parts = ["burst_intercept"]
    if a.board:
        parts.append(a.board)
    if a.tag:
        parts.append(a.tag)
    elif a.fa_only:
        parts.append("fa_smoke")
    stem = "_".join(parts)
    (rd / f"{stem}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    if a.conducted and not a.fa_only:
        (rd / "sip_sweep.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    if rows:
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        colors = dict(zip(a.windows, ["C0", "C1", "C2", "C3"]))
        for n in a.windows:
            drawn = False
            for mode, mk, fill in (("periodic", "s", "none"), ("random", "o", None),
                                   ("fwjitter", "^", None), ("fastpoll", "D", "none")):
                R = [r for r in rows if r["n"] == n and r["mode"] == mode]
                if not R:
                    continue
                tau = np.array([r["tau_s"] for r in R]) * 1e6
                kw = dict(color=colors[n], marker=mk, ls="none", ms=6)
                if fill:
                    kw["mfc"] = fill
                ax[0].errorbar(tau, [r["p_hit"] for r in R], yerr=[2 * r["p_hit_se"] for r in R],
                               label=f"T_w={n / FS * 1e6:.1f} us {mode}", **kw)
                ax[1].plot(tau, [r["rate_meas_hz"] for r in R], label=f"T_w={n / FS * 1e6:.1f} us {mode}", **kw)
                if not drawn:
                    ax[0].plot(tau, [r["p_theory"] for r in R], "--", color=colors[n], lw=1)
                    ax[1].plot(tau, [r["rate_pred_hz"] for r in R], "--", color=colors[n], lw=1)
                    drawn = True
        for ax_ in ax[:2]:
            ax_.set_xscale("log"); ax_.set_yscale("log"); ax_.set_xlabel("burst length tau (us)"); ax_.legend(fontsize=7)
        ax[0].set_ylabel("P(hit) per snapshot")
        ax[0].set_title(f"per-snapshot intercept, T_b={a.period*1e3:.0f} ms (dashed: geometry)")
        ax[1].set_ylabel("intercepts per second"); ax[1].set_title("intercept rate (dashed: P_theory / measured T_p)")
        for r in [r for r in rows if r["n"] == max(a.windows) and r["mode"] == a.modes[-1] and r["overlap_us"]]:
            ax[2].hist(r["overlap_us"], bins=np.linspace(0, max(a.windows) / FS * 1e6 + 5, 42), histtype="step",
                       label=f"tau={r['tau_s']*1e6:.0f} us")
        ax[2].set_xlabel("measured overlap (us)")
        ax[2].set_title(f"overlap distribution, T_w={max(a.windows)/FS*1e6:.1f} us, {a.modes[-1]}")
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(rd / f"{stem}.png", dpi=130)
    print("wrote", rd / f"{stem}.json", flush=True)
    print("lime:", out["lime_log_tail"], flush=True)


if __name__ == "__main__":
    main()
