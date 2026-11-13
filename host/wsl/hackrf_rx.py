#!/usr/bin/env python3
"""Persistent HackRF One RX (WSL / SoapySDR). Captures write a .npy for the host.

stdin: freq <lo_hz> | gain <dB> | capture <n> <out.npy> | quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=2442e6)
    ap.add_argument("--rate", type=float, default=20e6)
    ap.add_argument("--gain", type=float, default=40.0)
    ap.add_argument("--antenna", default="TX/RX")
    a = ap.parse_args()

    try:
        sdr = SoapySDR.Device(dict(driver="hackrf"))
    except Exception as e:
        print(f"ERR open {e}", flush=True)
        return 1

    sdr.setSampleRate(SOAPY_SDR_RX, 0, a.rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, a.lo)
    ants = list(sdr.listAntennas(SOAPY_SDR_RX, 0))
    for name in (a.antenna, "TX/RX", "RX", "Antenna"):
        if name in ants:
            try:
                sdr.setAntenna(SOAPY_SDR_RX, 0, name)
                break
            except Exception:
                continue
    # HackRF named gains: LNA 0..40, VGA 0..62, AMP 0/14. Keep AMP off for conducted.
    lna = float(min(40.0, max(0.0, a.gain)))
    vga = float(min(62.0, max(0.0, a.gain)))  # mirror requested gain onto both stages
    for gname, gdb in (("AMP", 0.0), ("LNA", lna), ("VGA", vga)):
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, gname, gdb)
        except Exception:
            pass
    try:
        sdr.setBandwidth(SOAPY_SDR_RX, 0, min(a.rate, 20e6))
    except Exception:
        pass

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    buf = np.zeros(16384, np.complex64)
    for _ in range(8):
        sdr.readStream(stream, [buf], len(buf), timeoutUs=200000)

    gains = list(sdr.listGains(SOAPY_SDR_RX, 0))
    rng = sdr.getGainRange(SOAPY_SDR_RX, 0)
    print(
        f"READY lo={sdr.getFrequency(SOAPY_SDR_RX, 0):.0f} rate={sdr.getSampleRate(SOAPY_SDR_RX, 0):.0f} "
        f"gain={sdr.getGain(SOAPY_SDR_RX, 0):.1f} range=[{rng.minimum()},{rng.maximum()}] "
        f"antenna={sdr.getAntenna(SOAPY_SDR_RX, 0)} ants={','.join(ants)} gains={','.join(gains)}",
        flush=True,
    )

    for line in sys.stdin:
        p = line.split()
        if not p:
            continue
        try:
            if p[0] == "freq":
                sdr.setFrequency(SOAPY_SDR_RX, 0, float(p[1]))
                print(f"OK freq {sdr.getFrequency(SOAPY_SDR_RX, 0):.0f}", flush=True)
            elif p[0] == "gain":
                g = float(p[1])
                sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 0.0)
                sdr.setGain(SOAPY_SDR_RX, 0, "LNA", min(40.0, g))
                sdr.setGain(SOAPY_SDR_RX, 0, "VGA", min(62.0, g))
                print(f"OK gain {sdr.getGain(SOAPY_SDR_RX, 0):.1f}", flush=True)
            elif p[0] == "capture":
                n = int(p[1])
                out = Path(p[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                iq = np.empty(n, np.complex64)
                got = 0
                chunk = 16384
                idle = 0
                while got < n:
                    need = min(chunk, n - got)
                    sr = sdr.readStream(stream, [iq[got: got + need]], need, timeoutUs=1000000)
                    if sr.ret > 0:
                        got += sr.ret
                        idle = 0
                    elif sr.ret < 0:
                        # -4 = overflow under usbipd; flush and keep going
                        idle += 1
                        if idle > 200:
                            print(f"ERR readStream {sr.ret} got={got}/{n}", flush=True)
                            break
                        continue
                    else:
                        idle += 1
                        if idle > 50:
                            print(f"ERR readStream timeout got={got}/{n}", flush=True)
                            break
                np.save(out, iq[:got])
                print(f"CAPOK {got} {out}", flush=True)
            elif p[0] == "quit":
                print("OK quit", flush=True)
                break
            else:
                print("ERR command", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {e}", flush=True)

    sdr.deactivateStream(stream)
    sdr.closeStream(stream)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
