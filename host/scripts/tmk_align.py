"""Q17: TMK alignment — gated REF pulses into LimeSDR continuous RX.

Chain (conducted):
  RFE6GEN -> SPDT (GPIO4, 1=pass) -> coupler IN
  CPL -> (optional att) -> IPEX -> ESP32 antenna
  OUT -> att -> LimeSDR Mini RX

Each ESP snapshot arms GATE so GPIO4 rises a known delay after sample 0; after
the capture the host pulls GPIO4 low. Lime sees one RF burst per snapshot.
Acceptance: every pulse found; after a constant time offset, |residual| < 1% of Tw.

Example:
  python host/scripts/tmk_align.py --port COM20 --board s3-wroom1u-03
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr
from snapcal.rfe import RfeGenerator
from snapcal.ric import estimate_tone

FS_ESP = 80_000_000
N_ESP = 16380
F_BB = 10e6


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def envelope_tone(iq: np.ndarray, rate: float, offset_hz: float, lp_hz: float = 50e3) -> np.ndarray:
    """Baseband power of a CW at LO+offset, zero-phase lowpass."""
    t = np.arange(iq.size, dtype=np.float64) / rate
    bb = iq * np.exp(-2j * np.pi * offset_hz * t)
    # Moving-average lowpass (rect); length ~ 1/(2*lp)
    k = max(8, int(round(rate / (2 * lp_hz))))
    ker = np.ones(k, dtype=np.float64) / k
    # |bb|^2 smoothed
    p = np.convolve(np.abs(bb) ** 2, ker, mode="same")
    return p


def _midpoint_crossing(power: np.ndarray, i_rise: int, rate: float, win_s: float = 0.0004) -> float:
    """Sample index (float) of 50% power crossing near a coarse rising edge."""
    w = max(4, int(round(win_s * rate)))
    a = max(0, i_rise - w)
    b = min(power.size, i_rise + w)
    seg = power[a:b]
    if seg.size < 4:
        return float(i_rise)
    lo = float(np.percentile(seg, 10))
    hi = float(np.percentile(seg, 90))
    mid = 0.5 * (lo + hi)
    for j in range(1, seg.size):
        if seg[j - 1] < mid <= seg[j]:
            frac = (mid - seg[j - 1]) / (seg[j] - seg[j - 1] + 1e-30)
            return a + (j - 1) + frac
    return float(i_rise)


def find_rising_edges(power: np.ndarray, rate: float, n_expect: int, min_gap_s: float = 0.05):
    """Hysteresis + min-gap rising edges. Default gap 50 ms (snapshots are ~70 ms apart).

    Returns more candidates than n_expect when present; caller should run
    select_periodic_run() to pick the TMK burst train.
    """
    med = float(np.median(power))
    mad = float(np.median(np.abs(power - med))) + 1e-20
    peak = float(np.percentile(power, 99.5))
    thr_hi = med + max(8 * mad, 0.25 * (peak - med))
    thr_lo = med + max(4 * mad, 0.08 * (peak - med))
    if thr_lo >= thr_hi:
        thr_lo = 0.5 * (med + thr_hi)
    min_gap = int(round(min_gap_s * rate))
    min_high = max(4, int(round(0.00005 * rate)))  # ≥50 µs above thr_hi
    kept = []
    i = 1
    n = power.size
    while i < n:
        if power[i - 1] < thr_hi <= power[i]:
            j = min(n, i + min_high)
            if float(np.min(power[i:j])) >= thr_lo:
                if not kept or i - kept[-1] >= min_gap:
                    kept.append(int(i))
                while i < n and power[i] > thr_lo:
                    i += 1
                continue
        i += 1
    # Keep extras for cadence selection (interference before/after the train).
    cap = max(n_expect * 3, n_expect + 8)
    if len(kept) > cap:
        heights = [float(power[i: min(n, i + int(0.0005 * rate))].max()) for i in kept]
        order = np.argsort(heights)[::-1][:cap]
        kept = sorted(int(kept[j]) for j in order)
    edges = [_midpoint_crossing(power, i, rate) for i in kept]
    return edges, thr_hi, med, peak


def select_periodic_run(edges, rate: float, n_expect: int,
                        period_s: float = 0.072, tol_s: float = 0.015):
    """Contiguous edge run whose spacings match period±tol; length n_expect."""
    if len(edges) < n_expect:
        return list(edges)
    t = np.asarray(edges, dtype=np.float64) / rate
    best = None
    best_score = -1e99
    for i in range(len(t) - n_expect + 1):
        seg = t[i: i + n_expect]
        dt = np.diff(seg)
        ok = np.abs(dt - period_s) <= tol_s
        score = float(ok.sum()) - float(np.mean(np.abs(dt - period_s)) / period_s)
        if score > best_score:
            best_score = score
            best = edges[i: i + n_expect]
    return list(best)


def select_by_esp_intervals(edges, rate: float, snap_start_us: list):
    """Pick n contiguous Lime edges whose intervals best match ESP snap_start spacings."""
    n = len(snap_start_us)
    if len(edges) < n or n < 2:
        return list(edges[:n])
    t_esp = (np.asarray(snap_start_us, dtype=np.float64) - snap_start_us[0]) * 1e-6
    esp_dt = np.diff(t_esp)
    t = np.asarray(edges, dtype=np.float64) / rate
    best_i = 0
    best_key = None
    for i in range(len(t) - n + 1):
        lime_dt = np.diff(t[i: i + n])
        err = float(np.max(np.abs(lime_dt - esp_dt)))
        rms = float(np.sqrt(np.mean((lime_dt - esp_dt) ** 2)))
        key = (err, rms)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return list(edges[best_i: best_i + n])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="COM20")
    ap.add_argument("--gen", default="COM16")
    ap.add_argument("--board", default="s3-wroom1u-03")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--esp-gain", type=int, default=36)
    ap.add_argument("--lime-lo", type=float, default=2450e6)
    ap.add_argument("--lime-rate", type=float, default=8e6)
    ap.add_argument("--lime-gain", type=float, default=50)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--gate-delay-us", type=int, default=20)
    ap.add_argument("--pin", type=int, default=4)
    ap.add_argument("--record-s", type=float, default=8.0)
    ap.add_argument("--settle-pre-s", type=float, default=1.2,
                    help="wait after Lime READY before first snapshot")
    ap.add_argument("--warmup", type=int, default=1,
                    help="throwaway snapshots before the counted set (Lime AGC)")
    args = ap.parse_args()

    f_rf = args.esp_freq * 1e6 + F_BB
    tone_offset = f_rf - args.lime_lo
    tw_s = N_ESP / FS_ESP
    dest = ROOT / "results" / "tmk"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    iq_path = dest / f"align_{args.board}_{stamp}.npy"
    script = ROOT / "host" / "wsl" / "lime_rx_record.py"

    # CRLF-safe launch
    subprocess.run(
        ["wsl", "-e", "bash", "-lc", f"sed -i 's/\\r$//' {_wsl_path(script)}"],
        check=False,
    )

    rec_cmd = [
        "wsl", "-e", "bash", "-lc",
        (
            f"timeout {int(args.record_s) + 30} python3 {_wsl_path(script)} "
            f"--out {_wsl_path(iq_path)} --lo {args.lime_lo:.0f} "
            f"--rate {args.lime_rate:.0f} --gain {args.lime_gain} "
            f"--seconds {args.record_s}"
        ),
    ]
    proc = subprocess.Popen(
        rec_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def wait_ready(timeout=60.0) -> list[str]:
        lines = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                line = line.rstrip()
                lines.append(line)
                print(f"[lime] {line}", flush=True)
                if line.startswith("READY"):
                    return lines
        raise RuntimeError("Lime recorder did not print READY:\n" + "\n".join(lines))

    log = wait_ready()
    t_ready = time.time()
    time.sleep(args.settle_pre_s)

    dev = EspSdr(args.port, baud=2_000_000)
    gen = RfeGenerator(args.gen)
    snaps = []
    try:
        print(dev.identity, flush=True)
        dev.set_freq(args.esp_freq)
        dev.set_gain(args.esp_gain)
        # Sample-0 marker on GPIO6 (optional); GATE needs xfer pin set for the 1 us pulse.
        try:
            dev.set_marks(5, 6)
        except Exception:
            pass
        # Keep REF off until Lime has settled so early RFI/leakage is not recorded as TMK.
        dev.gpio(args.pin, "0")
        try:
            gen.off()
        except Exception:
            pass
        time.sleep(0.3)
        gen.cw(f_rf / 1e6, level=args.level, settle=0.4)
        print(f"generator {gen.estimated_dbm():.1f} dBm  tone_offset {tone_offset/1e6:.3f} MHz",
              flush=True)

        total = args.warmup + args.count
        for k in range(total):
            warmup = k < args.warmup
            i = k - args.warmup
            dev.gpio(args.pin, "0")
            time.sleep(0.002)
            dev.command(f"GATE {args.pin} {args.gate_delay_us} 1")
            snap = dev.capture(N_ESP, FS_ESP, "iq10")
            # Firmware TS after capture
            ts_line = ""
            try:
                ts_line = dev.command("TS?")
            except Exception:
                pass
            snap_start_us = None
            snap_cmd_us = None
            if ts_line.startswith("TS"):
                parts = ts_line.split()
                if len(parts) >= 3:
                    snap_cmd_us = int(parts[1])
                    snap_start_us = int(parts[2])
            # Pull REF off so the next burst has a quiet gap.
            dev.gpio(args.pin, "0")
            tone = estimate_tone(snap.iq, FS_ESP, F_BB)
            row = {
                "i": i,
                "warmup": warmup,
                "t_host_start": snap.t_host_start,
                "t_host_end": snap.t_host_end,
                "capture_us": snap.capture_us,
                "snap_cmd_us": snap_cmd_us,
                "snap_start_us": snap_start_us,
                "power_dbfs": float(tone.power_dbfs),
                "snr_db": float(tone.snr_db),
                "freq_hz": float(tone.freq_hz),
            }
            tag = "warm" if warmup else f"{i:02d}"
            print(
                f"  snap {tag:>4s}  {row['power_dbfs']:6.1f} dBFS  SNR {row['snr_db']:5.1f}  "
                f"cap {row['capture_us']} us  start_us {snap_start_us}",
                flush=True,
            )
            if row["power_dbfs"] > -3:
                raise SystemExit(f"ESP tone {row['power_dbfs']} dBFS too hot; lower --level or --esp-gain")
            if not warmup:
                snaps.append(row)
    finally:
        try:
            dev.command("GATE OFF")
            dev.command("MARK OFF")
            dev.gpio(args.pin, "0")
        except Exception:
            pass
        try:
            gen.off()
        except Exception:
            pass
        gen.close()
        dev.close()

    # Drain recorder
    while proc.poll() is None:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            line = line.rstrip()
            log.append(line)
            print(f"[lime] {line}", flush=True)
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.05)
    rc = proc.wait(timeout=30)
    if rc not in (0, None) and not iq_path.exists():
        raise SystemExit(f"Lime recorder failed rc={rc}\n" + "\n".join(log[-30:]))

    meta = json.loads(iq_path.with_suffix(".json").read_text(encoding="utf-8"))
    iq = np.load(iq_path)
    rate = float(meta["rate_hz"])
    power = envelope_tone(iq, rate, tone_offset)
    n_pulse_expect = args.warmup + args.count
    edges_all, thr, med, peak = find_rising_edges(power, rate, n_pulse_expect)
    if len(snaps) >= 2 and all(s["snap_start_us"] is not None for s in snaps):
        period_s = float(np.median(np.diff([s["snap_start_us"] for s in snaps]))) * 1e-6
        starts = [s["snap_start_us"] for s in snaps]
        edges_use = select_by_esp_intervals(edges_all, rate, starts)
        edge_select = "esp_intervals"
        # Fallback: older simple min-gap detector (works when hysteresis over-rejects).
        t_probe = np.asarray(edges_use, dtype=np.float64) / rate
        t_esp_probe = (np.asarray(starts, dtype=np.float64) - starts[0]) * 1e-6
        if len(edges_use) == args.count and len(t_probe) >= 2:
            dt_err = float(np.max(np.abs(np.diff(t_probe) - np.diff(t_esp_probe))))
        else:
            dt_err = 1e9
        if dt_err > 0.005 or len(edges_use) != args.count:
            med = float(np.median(power))
            mad = float(np.median(np.abs(power - med))) + 1e-20
            peak = float(np.percentile(power, 99.5))
            thr_s = med + max(8 * mad, 0.2 * (peak - med))
            above = power > thr_s
            rise = np.flatnonzero(above[1:] & ~above[:-1]) + 1
            min_gap = int(round(0.002 * rate))
            kept, last = [], -10**12
            for i in rise:
                if i - last >= min_gap:
                    kept.append(int(i))
                    last = i
            if len(kept) > n_pulse_expect:
                heights = [float(power[i: i + int(0.0005 * rate)].max()) for i in kept]
                order = np.argsort(heights)[::-1][:n_pulse_expect]
                kept = sorted(int(kept[j]) for j in order)
            if len(kept) >= n_pulse_expect:
                edges_use = kept[args.warmup: args.warmup + args.count]
                edge_select = "simple_mingap2ms_drop_warmup"
                edges_all = kept
    else:
        period_s = 0.072
        edges = select_periodic_run(edges_all, rate, n_pulse_expect, period_s=period_s, tol_s=0.02)
        edges_use = edges[args.warmup: args.warmup + args.count] if len(edges) >= n_pulse_expect else edges[-args.count:]
        edge_select = "periodic_run"
    t_edges = np.asarray(edges_use, dtype=np.float64) / rate  # seconds from Lime t0

    # ESP timeline from firmware snap_start_us (preferred) or host start.
    if all(s["snap_start_us"] is not None for s in snaps):
        t_esp = (np.array([s["snap_start_us"] for s in snaps], dtype=np.float64)
                 - snaps[0]["snap_start_us"]) * 1e-6
        esp_clock = "firmware_snap_start_us"
    else:
        t_esp = np.array([s["t_host_start"] for s in snaps]) - snaps[0]["t_host_start"]
        esp_clock = "host_t_host_start"

    n_found = len(edges_use)
    found_all = n_found == args.count
    residuals_us = []
    residuals_offset_only_us = []
    max_abs_us = None
    rms_us = None
    max_abs_offset_only_us = None
    interval_err_us = []
    clock_ppm = None
    if found_all and args.count >= 2:
        t_lime = t_edges - t_edges[0]
        # Offset-only (shows relative clock drift over the run).
        offset = float(np.mean(t_lime - t_esp))
        resid_o = t_lime - t_esp - offset
        residuals_offset_only_us = [round(float(x) * 1e6, 3) for x in resid_o]
        max_abs_offset_only_us = float(np.max(np.abs(resid_o)) * 1e6)
        # Affine fit absorbs TCXO vs ESP crystal rate; residual is TMK placement error.
        A = np.vstack([np.ones_like(t_esp), t_esp]).T
        coef, _, _, _ = np.linalg.lstsq(A, t_lime, rcond=None)
        resid = t_lime - (A @ coef)
        residuals_us = [round(float(x) * 1e6, 3) for x in resid]
        max_abs_us = float(np.max(np.abs(resid)) * 1e6)
        rms_us = float(np.sqrt(np.mean(resid ** 2)) * 1e6)
        clock_ppm = float((coef[1] - 1.0) * 1e6)
        interval_err_us = [round(float(x) * 1e6, 3)
                           for x in (np.diff(t_lime) - np.diff(t_esp))]

    limit_us = 0.01 * tw_s * 1e6
    ok = bool(found_all and max_abs_us is not None and max_abs_us < limit_us)

    # Figure: power + edge markers
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
    t_axis = np.arange(power.size) / rate
    ax[0].plot(t_axis, 10 * np.log10(power + 1e-20), lw=0.6, color="C0")
    ax[0].axhline(10 * np.log10(thr + 1e-20), color="C3", ls="--", lw=0.8, label="thr")
    for e in edges_use:
        ax[0].axvline(float(e) / rate, color="C2", alpha=0.5, lw=0.7)
    ax[0].set_ylabel("tone power (dB)")
    ax[0].set_title(f"TMK Lime RX  found {n_found}/{args.count}")
    ax[0].legend(loc="upper right", fontsize=8)
    if residuals_us:
        ax[1].plot(range(len(residuals_us)), residuals_us, "o-", ms=4)
        ax[1].axhline(limit_us, color="C3", ls="--", lw=0.8)
        ax[1].axhline(-limit_us, color="C3", ls="--", lw=0.8)
        ax[1].set_xlabel("snapshot index")
        ax[1].set_ylabel("align residual (µs)")
        ax[1].set_title(f"max |r|={max_abs_us:.2f} µs  limit={limit_us:.2f} µs (1% of Tw)")
    else:
        ax[1].text(0.5, 0.5, "alignment not computed", ha="center", va="center")
        ax[1].set_axis_off()
    fig.tight_layout()
    png = dest / f"align_{args.board}_{stamp}.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    result = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "q": "Q17",
        "board": args.board,
        "port": args.port,
        "gen_port": args.gen,
        "wiring": (
            "RFE6GEN->SPDT(GPIO4)->coupler IN; CPL->ESP IPEX; OUT->att->Lime RX; "
            "GATE rises REF mid-capture, host clears GPIO4 after CAP"
        ),
        "esp_mhz": args.esp_freq,
        "bb_hz": F_BB,
        "rf_hz": f_rf,
        "gen_level": args.level,
        "esp_gain": args.esp_gain,
        "gate_delay_us": args.gate_delay_us,
        "n_esp": N_ESP,
        "fs_esp": FS_ESP,
        "tw_us": tw_s * 1e6,
        "limit_us_1pct_tw": limit_us,
        "count": args.count,
        "lime": {
            "iq_path": str(iq_path.relative_to(ROOT)).replace("\\", "/"),
            "meta": meta,
            "tone_offset_hz": tone_offset,
            "gain_db": args.lime_gain,
            "threshold": thr,
            "power_median": med,
            "power_p99_5": peak,
        },
        "snapshots": snaps,
        "esp_clock": esp_clock,
        "t_ready_host": t_ready,
        "warmup_snapshots": args.warmup,
        "edge_select": edge_select,
        "edges_samples_all": edges_all,
        "period_s_nominal": period_s,
        "edges_samples": edges_use,
        "edges_s": [round(float(x), 9) for x in t_edges],
        "n_found": n_found,
        "found_all": found_all,
        "fit": "affine t_lime = a + b t_esp (b absorbs relative clock)",
        "clock_ppm_lime_minus_esp": clock_ppm,
        "residuals_us": residuals_us,
        "max_abs_residual_us": max_abs_us,
        "rms_residual_us": rms_us,
        "residuals_offset_only_us": residuals_offset_only_us,
        "max_abs_residual_offset_only_us": max_abs_offset_only_us,
        "interval_err_us": interval_err_us,
        "pass": ok,
        "png": str(png.relative_to(ROOT)).replace("\\", "/"),
        "lime_log_tail": log[-20:],
    }
    out_json = dest / "tmk_align.json"
    # also keep stamped copy
    stamped = dest / f"align_{args.board}_{stamp}.json"
    text = json.dumps(result, indent=2)
    out_json.write_text(text, encoding="utf-8")
    stamped.write_text(text, encoding="utf-8")
    # paper/data mirror
    paper = ROOT / "paper" / "data"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "tmk_align.json").write_text(text, encoding="utf-8")

    print(
        f"found {n_found}/{args.count}  max|r|={max_abs_us} us  "
        f"limit={limit_us:.2f} us  pass={ok}",
        flush=True,
    )
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
