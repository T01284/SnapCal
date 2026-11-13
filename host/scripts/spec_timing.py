"""D7 / Q12 preview: on-chip FFT (SPEC) mode throughput on the ESP32-S3.

For each SPECINFO? profile, runs SPEC for a few seconds with statistics enabled and
records frames, FFTs, the chip-reported analysed fraction (coverage, the effective
duty cycle of the spectrum mode) and FFTs per second. Compare with the raw IQ
snapshot duty cycle (host_timing.json, ~0.2-0.4 %).

Frame layout: ESP-SDR docs/spectrum.md (SPC1 frames, SPS1 statistics, SPECEND).

Example:
  python -u host/scripts/spec_timing.py --port COM31
"""
import argparse
import json
import struct
import zlib
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal.espsdr import EspSdr

SPC = struct.Struct("<4sIQIHBBHBB")
SPS = struct.Struct("<4sHHHHIIIIHHI")
END_FIELDS = ["status", "detail", "units", "pairs", "elapsed_us", "late_max", "work_max_cycles",
              "frames", "drops", "abandoned", "ffts", "stopped_by_host"]


def run_spec(dev, ms, stride, upf, rate, nfft, detector=1):
    ser = dev.ser
    ser.write(f"SPEC {ms} {stride} {upf} {detector} {rate} {nfft} 1\n".encode())
    head = dev._readline()
    if not head.startswith("SPEC "):
        raise RuntimeError(f"SPEC refused: {head}")
    _, n_s, fs_s, _, mhz = head.split()
    frames, stats, ffts, spectra, bad = [], [], 0, [], 0
    end = None
    while end is None:
        first = dev._read_exact(4)
        if first == b"SPC1":
            blob = first + dev._read_exact(SPC.size - 4 + int(n_s) + 4)
            body, crc = blob[:-4], struct.unpack("<I", blob[-4:])[0]
            if zlib.crc32(body) != crc:
                bad += 1
                continue
            h = SPC.unpack(body[:SPC.size])
            frames.append(h[1:])
            ffts += h[4]
            spectra.append(np.frombuffer(body[SPC.size:], np.uint8).copy())
        elif first == b"SPS1":
            blob = first + dev._read_exact(SPS.size - 4 + 4)
            body, crc = blob[:-4], struct.unpack("<I", blob[-4:])[0]
            if zlib.crc32(body) == crc:
                stats.append(SPS.unpack(body))
        else:
            line = (first + dev.ser.readline()).decode("ascii", "replace").strip()
            if line.startswith("SPECEND"):
                end = dict(zip(END_FIELDS, map(int, line.split()[1:])))
            else:
                raise RuntimeError(f"unexpected: {line[:80]!r}")
    return {"nfft": int(n_s), "fs": int(fs_s), "lo_mhz": int(mhz), "frames": frames, "stats": stats,
            "ffts": ffts, "crc_bad": bad, "end": end, "spectra": spectra}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--board", default="")
    ap.add_argument("--ms", type=int, default=3000)
    ap.add_argument("--freq", type=int, default=2442)
    ap.add_argument("--gain", type=int, default=44)
    ap.add_argument("--out", default="phase0_bringup", help="results subdirectory")
    a = ap.parse_args()

    out = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "args": vars(a), "profiles": []}
    with EspSdr(a.port, baud=a.baud) as dev:
        out["firmware"] = dev.info()["version"]
        out["identity"] = dev.identity
        try:
            out["dual"] = dev.command("DUAL?")
        except Exception as exc:
            out["dual"] = str(exc)
        info = json.loads(dev.command("SPECINFO?").split(" ", 1)[1])
        out["specinfo"] = info
        dev.set_freq(a.freq)
        dev.set_gain(a.gain)
        for row in info["profiles"]:
            fs, rate, nfft, stride, upf = row[:5]
            try:
                r = run_spec(dev, a.ms, stride, upf, rate, nfft)
            except Exception as exc:
                print(f"{fs/1e6:4.0f} MS/s nfft {nfft}: FAILED {exc}", flush=True)
                out["profiles"].append({"fs": fs, "nfft": nfft, "error": str(exc)})
                continue
            e = r["end"]
            el = e["elapsed_us"] / 1e6
            cov = [s[3] / 1000 for s in r["stats"]]
            fps_chip = [s[11] for s in r["stats"]]
            duty_ffts = r["ffts"] * nfft / (el * fs) if el > 0 else float("nan")
            rec = {"fs": fs, "rate_code": rate, "nfft": nfft, "stride": stride, "upf": upf,
                   "elapsed_s": el, "frames": len(r["frames"]), "frame_rate_hz": len(r["frames"]) / el,
                   "ffts": r["ffts"], "ffts_per_s": r["ffts"] / el,
                   "duty_from_ffts": duty_ffts,
                   "coverage_chip_median": float(np.median(cov)) if cov else None,
                   "ffts_per_s_chip_median": float(np.median(fps_chip)) if fps_chip else None,
                   "load_core0_permille": float(np.median([s[1] for s in r["stats"]])) if r["stats"] else None,
                   "load_core1_permille": float(np.median([s[2] for s in r["stats"]])) if r["stats"] else None,
                   "drops": e["drops"], "abandoned": e["abandoned"], "status": e["status"],
                   "crc_bad": r["crc_bad"],
                   "bytes_per_frame": SPC.size + nfft + 4}
            rec["throughput_kBps"] = rec["frame_rate_hz"] * rec["bytes_per_frame"] / 1e3
            out["profiles"].append(rec)
            print(f"{fs/1e6:4.0f} MS/s nfft {nfft:4d}: {rec['frames']:5d} frames {rec['frame_rate_hz']:6.1f}/s, "
                  f"{rec['ffts_per_s']:8.0f} FFT/s, duty {duty_ffts*100:6.2f} %, chip coverage "
                  f"{(rec['coverage_chip_median'] or 0)*100:6.2f} %, drops {e['drops']}, status {e['status']}",
                  flush=True)
    rd = ROOT / "results" / a.out
    rd.mkdir(parents=True, exist_ok=True)
    name = f"spec_timing_{a.board}.json" if a.board else "spec_timing.json"
    (rd / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", rd / name)


if __name__ == "__main__":
    main()
