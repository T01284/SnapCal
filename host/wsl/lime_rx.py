#!/usr/bin/env python3
"""Persistent LimeSDR Mini RX (WSL). Captures write a .npy; host reads it via /mnt.

stdin: freq <lo_hz> | gain <dB> | antenna <name> | capture <n> <out.npy> | quit
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
    ap.add_argument("--rate", type=float, default=30e6)
    ap.add_argument("--gain", type=float, default=40.0)
    ap.add_argument("--antenna", default="Auto",
                    help="Lime Mini RX: NONE/LNAH/LNAL_NC/LNAW/Auto — use Auto or LNAW at 2.4 GHz")
    ap.add_argument("--serial", default="1D40ED03BF7C98")
    a = ap.parse_args()

    # Prefer kwargs dict; some Soapy builds mangle comma-args after usbipd reattach.
    try:
        sdr = SoapySDR.Device(dict(driver="lime", serial=a.serial)) if a.serial else SoapySDR.Device(dict(driver="lime"))
    except Exception:
        sdr = SoapySDR.Device(dict(driver="lime"))

    sdr.setSampleRate(SOAPY_SDR_RX, 0, a.rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, a.lo)
    ants = list(sdr.listAntennas(SOAPY_SDR_RX, 0))
    for name in (a.antenna, "Auto", "LNAW", "LNAH"):
        if name in ants:
            try:
                sdr.setAntenna(SOAPY_SDR_RX, 0, name)
                break
            except Exception:
                continue
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, a.gain)
    except Exception:
        pass
    try:
        sdr.setBandwidth(SOAPY_SDR_RX, 0, min(a.rate, 30e6))
    except Exception:
        pass

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    buf = np.zeros(16384, np.complex64)
    for _ in range(8):
        sdr.readStream(stream, [buf], len(buf), timeoutUs=200000)

    rng = sdr.getGainRange(SOAPY_SDR_RX, 0)
    print(
        f"READY lo={sdr.getFrequency(SOAPY_SDR_RX, 0):.0f} rate={sdr.getSampleRate(SOAPY_SDR_RX, 0):.0f} "
        f"gain={sdr.getGain(SOAPY_SDR_RX, 0):.1f} range=[{rng.minimum()},{rng.maximum()}] "
        f"antenna={sdr.getAntenna(SOAPY_SDR_RX, 0)} ants={','.join(ants)}",
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
                sdr.setGain(SOAPY_SDR_RX, 0, float(p[1]))
                print(f"OK gain {sdr.getGain(SOAPY_SDR_RX, 0):.1f}", flush=True)
            elif p[0] == "antenna":
                sdr.setAntenna(SOAPY_SDR_RX, 0, p[1])
                print(f"OK antenna {sdr.getAntenna(SOAPY_SDR_RX, 0)}", flush=True)
            elif p[0] == "capture":
                n = int(p[1])
                out = Path(p[2])
                out.parent.mkdir(parents=True, exist_ok=True)
                iq = np.empty(n, np.complex64)
                got = 0
                chunk = 16384
                while got < n:
                    need = min(chunk, n - got)
                    sr = sdr.readStream(stream, [iq[got: got + need]], need, timeoutUs=1000000)
                    if sr.ret > 0:
                        got += sr.ret
                    elif sr.ret < 0:
                        print(f"ERR readStream {sr.ret}", flush=True)
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
