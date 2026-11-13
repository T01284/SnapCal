"""Q21 E2W-CBM conducted scorecard on the 一分四 combiner chain.

Wiring (same as combiner smoke):
  RFE -> SPDT(GPIO4) -> comb4 arm1
  Lime TX -> att30 -> comb4 arm2
  arm3/4 = 50 ohm; common -> IPEX -> ESP

Scenes (S1–S6), schedules (periodic / fwjitter / fastpoll), RIC cal on/off.
Each cell runs ``--seconds`` (default 20) so snapshot count >> 20.

Metrics per cell:
  Pd     — hit rate on burst scenes (S2–S6); blank on S1
  Pfa    — hit rate with Lime off (measured once per cal mode, reused)
  occup_err_pp — (measured duty − tau/Tb) in percentage points
                 measured duty = mean(overlap_samples / N) over ALL snapshots
                 (zeros on misses; unbiased estimate of RF on-fraction)

Example:
  python -u host/scripts/q21_cbm.py --port COM14 --board s3-com14
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.espsdr import EspSdr
from snapcal.limetx import LimeTx
from snapcal.rfe import RfeGenerator, discover
from snapcal.ric import estimate_tone

FS = 80_000_000

# Fixed Tw for the scorecard (Q19 middle window).
N_WIN = 2048
T_W = N_WIN / FS

SCENES = {
    "S1": {"tau_s": 0.0, "period_s": 10e-3, "rfe": "off", "note": "quiet (Pfa cell)"},
    "S2": {"tau_s": 20e-6, "period_s": 10e-3, "rfe": "off", "note": "short rare burst"},
    "S3": {"tau_s": 100e-6, "period_s": 10e-3, "rfe": "off", "note": "medium burst"},
    "S4": {"tau_s": 500e-6, "period_s": 10e-3, "rfe": "off", "note": "long / sync-trap"},
    "S5": {"tau_s": 2e-3, "period_s": 10e-3, "rfe": "off", "note": "dense / high duty"},
    "S6": {"tau_s": 500e-6, "period_s": 10e-3, "rfe": "cw", "note": "burst + RFE interferer"},
}
MODES = ("periodic", "fwjitter", "fastpoll")
CALS = ("off", "on")


class Detector:
    def __init__(self, f_bb, half_bw=300e3, adj_hz=1e6, adj_ratio=4.0, thr=None, min_len=80):
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


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def apply_mode(dev, mode, fw_jitter_us):
    if mode == "fwjitter":
        dev.set_jitter(fw_jitter_us)
        dev.set_fast_poll(False)
    elif mode == "fastpoll":
        dev.set_jitter(0)
        dev.set_fast_poll(True)
    else:
        dev.set_jitter(0)
        dev.set_fast_poll(False)


def clear_mode(dev):
    try:
        dev.set_jitter(0)
        dev.set_fast_poll(False)
    except Exception:
        pass


def run_cell(dev, det, seconds, mode, fw_jitter_us, cal_on, ric_every, rfe_mhz, gen, spdt_pin):
    hits, ov_fracs, starts = 0, [], []
    ric_snaps = 0
    apply_mode(dev, mode, fw_jitter_us)
    t_end = time.perf_counter() + seconds
    k = 0
    while time.perf_counter() < t_end:
        if cal_on and ric_every > 0 and k > 0 and (k % ric_every) == 0 and gen is not None:
            # Brief gated REF; not scored as a Lime hit.
            if spdt_pin:
                dev.gpio(spdt_pin, "1")
            gen.cw(rfe_mhz, level=3)
            time.sleep(0.002)
            _ = dev.capture(N_WIN, FS, "iq8")
            gen.off()
            ric_snaps += 1
            k += 1
            continue
        starts.append(time.perf_counter())
        iq = dev.capture(N_WIN, FS, "iq8").iq
        ov = det.overlap_samples(iq)
        ov_fracs.append(ov / N_WIN)
        if ov >= det.min_len:
            hits += 1
        k += 1
    clear_mode(dev)
    count = len(starts)
    dt = np.diff(starts) if count > 1 else np.array([np.nan])
    occup = float(np.mean(ov_fracs)) if ov_fracs else 0.0
    return {
        "snapshots": count,
        "hits": hits,
        "p_hit": hits / count if count else 0.0,
        "t_p_s": float(np.mean(dt)),
        "t_p_std_s": float(np.std(dt)) if count > 2 else 0.0,
        "mean_overlap_s": occup * T_W,
        "occup_meas": occup,
        "ric_snaps": ric_snaps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--board", default="s3-com14")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--lime-offset", type=float, default=2e6)
    ap.add_argument("--lime-gain", type=float, default=60.0,
                    help="match Q18/Q19 conducted Lime level (smoke used 40)")
    ap.add_argument("--max-lime-gain", type=float, default=64.0)
    ap.add_argument("--rfe-bb-mhz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fw-jitter-us", type=int, default=1000)
    ap.add_argument("--spdt-pin", type=int, default=4)
    ap.add_argument("--tau-ov", type=float, default=1e-6)
    ap.add_argument("--ric-every", type=int, default=50, help="gated REF every N snaps when cal=on")
    ap.add_argument("--scenes", nargs="+", default=list(SCENES))
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--cals", nargs="+", default=list(CALS))
    ap.add_argument("--gen", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    # Lime LO = ESP LO; digital CW at +lime_offset (same plan as combiner smoke).
    lime_lo = a.esp_freq * 1e6
    f_bb = a.lime_offset
    rfe_mhz = a.esp_freq + a.rfe_bb_mhz
    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    det = Detector(f_bb, min_len=int(round(a.tau_ov * FS)))
    out = {
        "utc": utc(),
        "cmd": "q21_cbm",
        "q": "Q21",
        "board": a.board,
        "args": vars(a),
        "scenes_def": SCENES,
        "window_n": N_WIN,
        "t_w_us": T_W * 1e6,
        "setup": (
            "RFE->SPDT->comb4 arm1; LimeTX->att30->arm2; arm3/4=50ohm; "
            "common->IPEX->ESP"
        ),
        "rows": [],
        "pfa": {},
    }
    print(f"Q21 CBM  board={a.board}  Tw={T_W*1e6:.1f} us  seconds/cell={a.seconds}", flush=True)

    with RfeGenerator(gp) as gen, \
            LimeTx(lime_lo, a.lime_offset, a.lime_gain, max_gain_db=a.max_lime_gain) as lime, \
            EspSdr(a.port) as esp:
        out["generator"] = gen.info()
        out["lime_ready"] = lime.ready
        out["esp"] = esp.identity
        print("READY", esp.identity, flush=True)
        print("READY lime", lime.ready, flush=True)
        esp.set_freq(a.esp_freq)
        esp.set_gain(a.esp_gain)
        if a.spdt_pin:
            esp.gpio(a.spdt_pin, "1")

        # CW calibrate detector on Lime tone.
        gen.off()
        lime.cw()
        time.sleep(0.5)
        a_cw = float(np.median([
            np.median(det.envelopes(esp.capture(16380, FS, "iq8").iq)[0]) for _ in range(40)
        ]))
        det.thr = 0.5 * a_cw
        out["cw_envelope"] = a_cw
        out["threshold"] = det.thr
        print(f"CW env {a_cw:.4f}  thr {det.thr:.4f}", flush=True)

        # Optional RIC tone check (cal path alive).
        lime.off()
        gen.cw(rfe_mhz, level=3)
        time.sleep(0.3)
        ric = estimate_tone(esp.capture(16380, FS, "iq8").iq, FS, a.rfe_bb_mhz * 1e6)
        out["ric_probe"] = ric.as_dict()
        print(f"RIC probe {ric.power_dbfs:.1f} dBFS SNR {ric.snr_db:.1f}", flush=True)
        gen.off()

        # Pfa once per cal mode (Lime off, RFE off).
        for cal in a.cals:
            lime.off(); gen.off(); time.sleep(0.3)
            r = run_cell(esp, det, a.seconds, "periodic", a.fw_jitter_us,
                         cal == "on", a.ric_every, rfe_mhz, gen if cal == "on" else None,
                         a.spdt_pin)
            out["pfa"][cal] = r
            print(f"Pfa cal={cal}: {r['p_hit']:.4f} over {r['snapshots']} snaps", flush=True)

        for scene in a.scenes:
            sc = SCENES[scene]
            tau, tb = sc["tau_s"], sc["period_s"]
            occup_true = (tau / tb) if tb > 0 and tau > 0 else 0.0
            for cal in a.cals:
                for mode in a.modes:
                    # Configure RF.
                    if sc["rfe"] == "cw":
                        if a.spdt_pin:
                            esp.gpio(a.spdt_pin, "1")
                        gen.cw(rfe_mhz, level=3)
                    else:
                        gen.off()
                    if tau > 0:
                        lime.burst(tau, tb)
                    else:
                        lime.off()
                    time.sleep(0.35)

                    r = run_cell(
                        esp, det, a.seconds, mode, a.fw_jitter_us,
                        cal == "on" and sc["rfe"] != "cw",  # S6 already has CW RFE
                        a.ric_every, rfe_mhz,
                        gen if (cal == "on" and sc["rfe"] != "cw") else None,
                        a.spdt_pin,
                    )
                    pfa = out["pfa"][cal]["p_hit"]
                    row = {
                        "scene": scene,
                        "note": sc["note"],
                        "mode": mode,
                        "cal": cal,
                        "tau_us": tau * 1e6,
                        "period_ms": tb * 1e3,
                        "occup_true": occup_true,
                        "occup_meas": r["occup_meas"],
                        "occup_err_pp": (r["occup_meas"] - occup_true) * 100.0,
                        "Pd": None if tau <= 0 else r["p_hit"],
                        "Pfa": pfa,
                        "p_hit": r["p_hit"],
                        "snapshots": r["snapshots"],
                        "hits": r["hits"],
                        "t_p_ms": r["t_p_s"] * 1e3,
                        "ric_snaps": r["ric_snaps"],
                        "n_ok": r["snapshots"] >= 20,
                    }
                    out["rows"].append(row)
                    pd_s = "—" if row["Pd"] is None else f"{row['Pd']:.3f}"
                    print(
                        f"{scene} {mode:8s} cal={cal}: Pd={pd_s}  Pfa={pfa:.4f}  "
                        f"occup {r['occup_meas']*100:.2f}% (true {occup_true*100:.2f}%)  "
                        f"err {row['occup_err_pp']:+.2f} pp  n={r['snapshots']}",
                        flush=True,
                    )

        lime.off(); gen.off()
        if a.spdt_pin:
            try:
                esp.gpio(a.spdt_pin, "0")
            except Exception:
                pass

    # Scene summary (median over modes×cals for table).
    summary = []
    for scene in a.scenes:
        rs = [r for r in out["rows"] if r["scene"] == scene]
        pds = [r["Pd"] for r in rs if r["Pd"] is not None]
        summary.append({
            "scene": scene,
            "note": SCENES[scene]["note"],
            "Pd_median": float(np.median(pds)) if pds else None,
            "Pfa_median": float(np.median([r["Pfa"] for r in rs])),
            "occup_err_pp_median": float(np.median([r["occup_err_pp"] for r in rs])),
            "n_cells": len(rs),
        })
    out["summary"] = summary
    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()

    rd = ROOT / "results" / "cbm" / a.board
    rd.mkdir(parents=True, exist_ok=True)
    stem = "q21_cbm" + (f"_{a.tag}" if a.tag else "")
    path = rd / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Mirror for paper table.
    paper = ROOT / "paper" / "data"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "q21_e2w.json").write_text(json.dumps({
        "utc": out["utc"],
        "board": a.board,
        "window_n": N_WIN,
        "t_w_us": T_W * 1e6,
        "seconds_per_cell": a.seconds,
        "summary": summary,
        "source": str(path),
    }, indent=2) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)
    print("summary:")
    for s in summary:
        print(f"  {s['scene']}: Pd={s['Pd_median']}  Pfa={s['Pfa_median']:.4f}  "
              f"occup_err={s['occup_err_pp_median']:+.2f} pp", flush=True)


if __name__ == "__main__":
    main()
