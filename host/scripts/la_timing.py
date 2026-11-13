"""Measure snapshot timing with a Saleae Logic analyser (Logic 2 automation API).

Wiring, common ground:
  S3: digital 0 <- GPIO5 SNAP_MARK, digital 1 <- GPIO6 XFER_DONE
  C5: digital 0 <- GPIO4 SNAP_MARK, digital 1 <- GPIO5 XFER_DONE

Logic 2.4 or newer must be installed. This script connects to the automation
server on port 10430, or launches Logic 2 if it is not already serving.
Logic 8 has a fixed threshold: do not pass --threshold. Logic Pro: --threshold 3.3.
The digital sample rate has to be one the device offers in the Logic 2 UI.

SNAP_MARK brackets the firmware capture, including the register writes and the
done-bit poll, so its width tracks ``capture_us`` rather than n/sample_rate alone.
XFER_DONE is not host-receipt; T_cycle is mark rise to the next mark rise.

Examples:
  python host/scripts/la_timing.py --list
  python host/scripts/la_timing.py --port COM14
  python host/scripts/la_timing.py --port COM14 --fast-poll --tag fastpoll
  python host/scripts/la_timing.py --port COM22 --snap-pin 4 --xfer-pin 5 --digital-hz 16000000 --rates 80000000 40000000 20000000
  python host/scripts/la_timing.py --port COM14 --energy-thresh 200 --formats iq8 --tag energy-miss --digital-hz 16000000 --fast-poll
  python host/scripts/la_timing.py --port COM14 --energy-thresh 0 --formats iq8 --tag energy-hit --digital-hz 16000000 --fast-poll
"""
import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.saleae_bin import high_pulses, pair_snapshots, parse_digital, summarize_group

SIZES = [1024, 2048, 4096, 8192, 12288, 16380]


def open_esp(port, baud):
    """C5's CH340 bridge stays at the last BAUD until reboot."""
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"), flush=True)
        dev.ser.baudrate = baud
        return dev


def open_manager(port: int, logic_path: str | None):
    import saleae.automation as automation
    try:
        return automation.Manager.connect(port=port, connect_timeout_seconds=2)
    except Exception:
        return automation.Manager.launch(
            application_path=logic_path, port=port, connect_timeout_seconds=60)


def list_devices(manager) -> None:
    info = manager.get_app_info()
    print(f"Logic {info.app_version}  api {info.api_version.major}.{info.api_version.minor}.{info.api_version.patch}")
    devices = manager.get_devices()
    if not devices:
        print("no Logic hardware found")
        return
    for d in devices:
        kind = "simulation" if d.is_simulation else d.device_type.name
        print(f"{d.device_id}  {kind}")


def capture_group(manager, dev, args, fmt: str, rate_hz: int) -> tuple[list[dict], list[dict]]:
    import saleae.automation as automation
    cfg = {
        "enabled_digital_channels": [args.snap_ch, args.xfer_ch],
        "digital_sample_rate": args.digital_hz,
    }
    if args.threshold is not None:
        cfg["digital_threshold_volts"] = args.threshold
    device_configuration = automation.LogicDeviceConfiguration(**cfg)
    capture_configuration = automation.CaptureConfiguration(
        buffer_size_megabytes=args.buffer_mb,
        capture_mode=automation.ManualCaptureMode(),
    )
    def one_shot(n):
        if args.energy_thresh is None:
            s = dev.capture(n, rate_hz, fmt)
            return {
                "n": n,
                "capture_us": s.capture_us,
                "host_cycle_s": s.t_host_end - s.t_host_start,
            }
        r = dev.capture_energy(n, rate_hz, fmt, args.energy_thresh)
        return {
            "n": n,
            "capture_us": r.capture_us,
            "host_cycle_s": r.t_host_end - r.t_host_start,
            "hit": r.hit,
            "energy_code": r.code,
        }

    one_shot(args.sizes[0])
    host: list[dict] = []
    with manager.start_capture(
            device_id=args.device_id,
            device_configuration=device_configuration,
            capture_configuration=capture_configuration) as capture:
        time.sleep(0.05)
        for n in args.sizes:
            for _ in range(args.repeats):
                host.append(one_shot(n))
        time.sleep(0.02)
        capture.stop()
        with tempfile.TemporaryDirectory() as tmp:
            capture.export_raw_data_binary(directory=tmp, digital_channels=[args.snap_ch, args.xfer_ch])
            snap = high_pulses(parse_digital(str(Path(tmp) / f"digital_{args.snap_ch}.bin")),
                               args.min_mark_us * 1e-6)
            xfer = high_pulses(parse_digital(str(Path(tmp) / f"digital_{args.xfer_ch}.bin")),
                               args.min_xfer_us * 1e-6)
            if args.save_capture:
                dest = ROOT / "results" / args.out
                dest.mkdir(parents=True, exist_ok=True)
                capture.save_capture(filepath=str(dest / f"la_{fmt}_{rate_hz}.sal"))
    return host, pair_snapshots(snap, xfer)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print Logic 2 version and attached devices, then exit")
    ap.add_argument("--port", help="ESP32 serial port")
    ap.add_argument("--baud", type=int, default=2_000_000,
                    help="UART baud after connect; 1000000 for a CH340 bridge")
    ap.add_argument("--logic-port", type=int, default=10430)
    ap.add_argument("--logic-path", help="Logic.exe if it is not on the default install path")
    ap.add_argument("--device-id", help="Saleae serial number; default is the first real device")
    ap.add_argument("--snap-ch", type=int, default=0)
    ap.add_argument("--xfer-ch", type=int, default=1)
    ap.add_argument("--snap-pin", type=int, default=5)
    ap.add_argument("--xfer-pin", type=int, default=6)
    ap.add_argument("--digital-hz", type=int, default=50_000_000)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Logic Pro only: 1.2, 1.8, or 3.3. Omit for Logic 8")
    ap.add_argument("--buffer-mb", type=int, default=256)
    ap.add_argument("--min-mark-us", type=float, default=0.4, help="drop shorter SNAP_MARK pulses")
    ap.add_argument("--min-xfer-us", type=float, default=0.2, help="drop shorter XFER_DONE pulses")
    ap.add_argument("--freq", type=int, default=2442)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--rates", type=int, nargs="+", default=[80_000_000, 40_000_000, 16_000_000])
    ap.add_argument("--formats", nargs="+", default=["iq8", "iq10", "raw"])
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--fast-poll", action="store_true")
    ap.add_argument("--energy-thresh", type=int, default=None,
                    help="use CAPE* instead of CAP*; 0=always hit, high=miss on quiet RF")
    ap.add_argument("--save-capture", action="store_true", help="also write a .sal next to the json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="snapshot_timing", help="results subdirectory; json name is la_timing")
    args = ap.parse_args()
    if args.energy_thresh is not None and not 0 <= args.energy_thresh <= 255:
        raise SystemExit("--energy-thresh must be 0..255")

    try:
        manager = open_manager(args.logic_port, args.logic_path)
    except Exception as exc:
        raise SystemExit(
            "could not connect to or launch Logic 2. Install Logic 2.4+ from saleae.com, "
            f"or pass --logic-path. ({exc})") from exc
    with manager:
        if args.list:
            list_devices(manager)
            return
        if not args.port:
            raise SystemExit("--port is required unless --list")
        rows = []
        with open_esp(args.port, args.baud) as dev:
            info = dev.info()
            if args.energy_thresh is not None and "ENERGY" not in info["caps"]:
                raise SystemExit("firmware CAPS lacks ENERGY; flash a CAPE* build first")
            dev.set_freq(args.freq)
            dev.set_fast_poll(args.fast_poll)
            dev.set_marks(args.snap_pin, args.xfer_pin)
            try:
                for fmt in args.formats:
                    for rate in args.rates:
                        host, paired = capture_group(manager, dev, args, fmt, rate)
                        summary = summarize_group(host, paired, rate)
                        summary["format"] = fmt
                        summary["rate_hz"] = rate
                        summary["la_resolution_us"] = round(1e6 / args.digital_hz, 4)
                        if args.energy_thresh is not None:
                            hits = [1 if h.get("hit") else 0 for h in host]
                            summary["energy_thresh"] = args.energy_thresh
                            summary["hit_frac"] = round(sum(hits) / len(hits), 4) if hits else None
                            summary["energy_code_p50"] = int(sorted(h["energy_code"] for h in host)[len(host) // 2])
                        rows.append(summary)
                        fit = summary.get("fit") or {}
                        extra = ""
                        if args.energy_thresh is not None:
                            extra = f"  hit={summary['hit_frac']:.0%}  thresh={args.energy_thresh}"
                        print(f"{fmt:4s} {rate/1e6:4.0f} MS/s  marks {summary['n_marks']}/{summary['n_host']}"
                              + (f"  c={fit['c']}  T0={fit['T0_ms']} ms" if fit else "  (no fit)")
                              + extra)
                        for n, b in summary.get("by_n", {}).items():
                            print(f"    n={n:>5}  mark {b['Tw_mark_us_median']:8.2f} us"
                                  f"  nominal {b['Tw_nominal_us']:8.2f} us"
                                  f"  err {b['Tw_error_pct']:+6.2f} %"
                                  f"  fw {b['fw_capture_us_median']:.0f} us"
                                  f"  queue {b['T_queue_us_median']} us")
            finally:
                dev.set_marks(None)
                dev.set_fast_poll(False)

    out = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Saleae Logic 2 digital export; SNAP_MARK width, XFER_DONE after USB queue, "
                  "cycle from mark rise to next mark rise"
                  + ("; CAPE* energy gate" if args.energy_thresh is not None else ""),
        "mode": "energy" if args.energy_thresh is not None else "raw_iq",
        "energy_thresh": args.energy_thresh,
        "wiring": {
            "snap_mark": f"GPIO{args.snap_pin} -> digital {args.snap_ch}",
            "xfer_done": f"GPIO{args.xfer_pin} -> digital {args.xfer_ch}",
        },
        "digital_hz": args.digital_hz,
        "digital_threshold_volts": args.threshold,
        "fast_poll": args.fast_poll,
        "repeats": args.repeats,
        "sizes": args.sizes,
        "board_info": info.get("info"),
        "firmware": info["version"],
        "toolchain": envinfo.toolchain(),
        "host": envinfo.host(),
        "rows": rows,
    }
    name = "la_timing" + (f"_{args.tag}" if args.tag else "") + ".json"
    path = ROOT / "results" / args.out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
