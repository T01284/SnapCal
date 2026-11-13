"""RF switch edge of the SPDT on an ESP32-S3, timed against the IQ envelope.

Wiring, common ground:
  RFE6GEN -> >=30 dB -> SPDT -> IPEX -> antenna port
  SPDT control <- GPIO4 (1 passes, 0 blocks)
  Logic digital 0 <- GPIO4, digital 1 <- GPIO5 SNAP_MARK, digital 2 <- GPIO6

GATE <pin> <delay_us> <0|1> fires once during the next capture, after the
dump has started. GPIO6 rises for 1 us at the dump-start write; that edge is
sample 0. XFER_DONE is the later GPIO6 pulse, after SNAP_MARK falls.

Logic 8: pass --digital-hz 16000000 and do not set a threshold.
"""
import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr
from snapcal.rfe import RfeGenerator
from snapcal.saleae_bin import high_pulses, parse_digital

N = 16380
FS = 80_000_000
F_BB = 10e6


def envelope(iq: np.ndarray) -> np.ndarray:
    """Zero-phase baseband amplitude. A 1.5 MHz lowpass sets the fastest edge we can claim."""
    t = np.arange(iq.size) / FS
    bb = iq * np.exp(-2j * np.pi * F_BB * t)
    sos = signal.butter(4, 1.5e6, btype="low", fs=FS, output="sos")
    return np.abs(signal.sosfiltfilt(sos, bb))


def filter_floor_us() -> float:
    """10–90% width the lowpass itself gives a perfect step. Faster edges cannot be claimed."""
    t = np.arange(N) / FS
    step = np.zeros(N, dtype=np.complex64)
    step[N // 2:] = np.exp(2j * np.pi * F_BB * t[N // 2:])
    return edge_points(envelope(step))["rise_10_90_us"]


def edge_points(env: np.ndarray) -> dict:
    n = env.size
    a = float(np.median(env[: n // 8]))
    b = float(np.median(env[-(n // 8):]))
    low, high = (a, b) if b >= a else (b, a)
    span = high - low
    t10, t50, t90 = low + 0.1 * span, low + 0.5 * span, low + 0.9 * span
    d = np.diff(env)
    peak = int(np.argmax(d) if b >= a else np.argmin(d))
    # Noise in the off state can cross the 10% level long before the edge.
    guard = int(3e-6 * FS)
    lo_lim, hi_lim = max(0, peak - guard), min(n - 1, peak + guard)
    i10 = peak
    while i10 > lo_lim and ((env[i10] > t10) if b >= a else (env[i10] < t90)):
        i10 -= 1
    i90 = peak
    while i90 < hi_lim and ((env[i90] < t90) if b >= a else (env[i90] > t10)):
        i90 += 1
    i50 = peak
    while i50 > lo_lim and ((env[i50] > t50) if b >= a else (env[i50] < t50)):
        i50 -= 1
    return {
        "rising": bool(b >= a),
        "off_dbfs": round(20 * np.log10(low + 1e-20), 2),
        "on_dbfs": round(20 * np.log10(high + 1e-20), 2),
        "i10": int(i10),
        "i50": int(i50),
        "i90": int(i90),
        "rise_10_90_us": round(abs(i90 - i10) / FS * 1e6, 3),
        "t50_us": round(i50 / FS * 1e6, 3),
    }


def logic_marks(tmp: Path) -> tuple[list, list, list]:
    chunks = parse_digital(str(tmp / "digital_0.bin"))
    edges = []
    for chunk in chunks:
        state = int(chunk.initial_state)
        for t in chunk.transition_times:
            t = float(t)
            edges.append({"t": t, "to": 1 - state})
            state = 1 - state
    marks = high_pulses(parse_digital(str(tmp / "digital_1.bin")), 50e-6)
    stamps = high_pulses(parse_digital(str(tmp / "digital_2.bin")), 0.2e-6)
    return edges, marks, stamps


def align(edges, mark, stamps) -> dict:
    inside = [e for e in edges if mark.rise <= e["t"] <= mark.fall]
    stamp = [p for p in stamps if mark.rise <= p.rise <= mark.fall]
    xfer = [p for p in stamps if p.rise > mark.fall and p.rise - mark.fall < 5e-3]
    out = {
        "mark_us": round(mark.width * 1e6, 2),
        "ctrl_edges": len(inside),
        "stamp": bool(stamp),
    }
    if inside:
        out["ctrl_from_mark_us"] = round((inside[0]["t"] - mark.rise) * 1e6, 3)
        out["ctrl_to"] = inside[0]["to"]
    if stamp:
        out["stamp_from_mark_us"] = round((stamp[0].rise - mark.rise) * 1e6, 3)
        if inside:
            out["ctrl_from_sample0_us"] = round((inside[0]["t"] - stamp[0].rise) * 1e6, 3)
    if xfer:
        out["xfer_after_mark_us"] = round((xfer[0].rise - mark.fall) * 1e6, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM14")
    ap.add_argument("--gen", default="COM16")
    ap.add_argument("--board", default="s3-wroom1u-01")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--digital-hz", type=int, default=16_000_000)
    ap.add_argument("--logic-port", type=int, default=10430)
    args = ap.parse_args()

    import saleae.automation as automation

    try:
        manager = automation.Manager.connect(port=args.logic_port, connect_timeout_seconds=2)
    except Exception:
        manager = automation.Manager.launch(port=args.logic_port, connect_timeout_seconds=60)

    plan = [("rise", 80, 1)] * 4 + [("fall", 80, 0)] * 4 + [("rise", 40, 1), ("rise", 120, 1)]
    floor = filter_floor_us()
    trials = []
    with manager:
        dev = EspSdr(args.port)
        gen = RfeGenerator(args.gen)
        try:
            print(dev.identity, flush=True)
            dev.set_freq(2442)
            dev.set_gain(44)
            dev.gpio(4, "0")
            dev.set_marks(5, 6)
            dev.capture(1024, FS, "iq10")
            gen.cw(2452.0, level=args.level, settle=0.4)
            print(f"generator {gen.estimated_dbm():.1f} dBm", flush=True)
            devices = [d for d in manager.get_devices() if not d.is_simulation]
            if not devices:
                raise SystemExit("no Logic hardware")
            cfg = automation.LogicDeviceConfiguration(
                enabled_digital_channels=[0, 1, 2],
                digital_sample_rate=args.digital_hz,
            )
            capcfg = automation.CaptureConfiguration(
                buffer_size_megabytes=32,
                capture_mode=automation.ManualCaptureMode(),
            )
            # One Logic capture per snapshot. A single long capture on this Logic 8
            # kept only the first mark.
            for name, delay, level in plan:
                dev.gpio(4, "0" if level == 1 else "1")
                time.sleep(0.008)
                with manager.start_capture(
                        device_id=devices[0].device_id,
                        device_configuration=cfg,
                        capture_configuration=capcfg) as cap:
                    time.sleep(0.02)
                    dev.command(f"GATE 4 {delay} {level}")
                    snap = dev.capture(N, FS, "iq10")
                    time.sleep(0.02)
                    cap.stop()
                    with tempfile.TemporaryDirectory() as tmp:
                        cap.export_raw_data_binary(directory=tmp, digital_channels=[0, 1, 2])
                        edges, marks, stamps = logic_marks(Path(tmp))
                env = envelope(snap.iq)
                pts = edge_points(env)
                if pts["on_dbfs"] > -3:
                    raise SystemExit(f"tone {pts['on_dbfs']} dBFS is too hot")
                pts.update({
                    "name": name,
                    "delay_us": delay,
                    "level": level,
                    "capture_us": snap.capture_us,
                    "env_dbfs": [round(float(20 * np.log10(v + 1e-20)), 2) for v in env[::40]],
                })
                if len(marks) == 1:
                    info = align(edges, marks[0], stamps)
                    pts.update(info)
                    if "ctrl_from_sample0_us" in info:
                        pts["pin_to_50_us"] = round(pts["t50_us"] - info["ctrl_from_sample0_us"], 3)
                else:
                    pts["ctrl_edges"] = len(edges)
                    pts["marks"] = len(marks)
                pts.pop("i10"), pts.pop("i50"), pts.pop("i90")
                trials.append(pts)
                print(f"{name:4s} delay {delay:3d}  10-90 {pts['rise_10_90_us']:7.3f} us"
                      f"  t50 {pts['t50_us']:7.2f}  pin→50 {pts.get('pin_to_50_us')} us"
                      f"  on {pts['on_dbfs']:6.1f}  off {pts['off_dbfs']:6.1f}",
                      flush=True)
        finally:
            try:
                dev.command("GATE OFF")
                dev.command("MARK OFF")
                dev.gpio(4, "0")
            except Exception:
                pass
            gen.close()
            dev.close()

    def med(key, pred):
        vals = [t[key] for t in trials if pred(t) and key in t]
        return round(float(np.median(vals)), 3) if vals else None

    summary = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "board": args.board,
        "port": args.port,
        "wiring": "GPIO4 SPDT (1=pass), Logic 0/1/2 = GPIO4/5/6, GPIO6 rise at dump start = sample 0",
        "generator_level": args.level,
        "digital_hz": args.digital_hz,
        "filter_10_90_us": floor,
        "rise_10_90_us": med("rise_10_90_us", lambda t: t["name"] == "rise" and t["delay_us"] == 80),
        "fall_10_90_us": med("rise_10_90_us", lambda t: t["name"] == "fall" and t["delay_us"] == 80),
        "rise_pin_to_50_us": med("pin_to_50_us", lambda t: t["name"] == "rise" and t["delay_us"] == 80),
        "fall_pin_to_50_us": med("pin_to_50_us", lambda t: t["name"] == "fall" and t["delay_us"] == 80),
        "trials": trials,
    }
    dest = ROOT / "results" / "spdt"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{args.board}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary, dest / f"{args.board}.png")
    print(f"10-90 rise/fall {summary['rise_10_90_us']} / {summary['fall_10_90_us']} us"
          f"  (filter floor {floor} us)")
    print(f"pin to 50% rise/fall {summary['rise_pin_to_50_us']} / {summary['fall_pin_to_50_us']} us")
    print(f"wrote {path}")


def plot(summary: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, name, title in ((axes[0], "rise", "GPIO4 0→1"), (axes[1], "fall", "GPIO4 1→0")):
        rows = [t for t in summary["trials"] if t["name"] == name and t["delay_us"] == 80]
        for t in rows:
            us = np.arange(len(t["env_dbfs"])) * 40 / FS * 1e6
            ax.plot(us, t["env_dbfs"], color="C0", alpha=0.45, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("sample time (µs)")
        ax.set_xlim(40, 130)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("baseband amplitude (dBFS)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
