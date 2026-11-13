#!/usr/bin/env python3
"""Continuous LimeSDR Mini RX into a complex64 .npy (runs inside WSL).

Prints READY when the stream is live, then records --seconds of IQ and writes
<meta>.json next to the .npy. Use driver=lime only (bare --find can hang).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path to .npy")
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--rate", type=float, default=8e6)
    ap.add_argument("--gain", type=float, default=45)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--bw", type=float, default=0.0, help="0 = min(rate, 30e6)")
    ap.add_argument("--serial", default="1D40ED03BF7C98")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bw = args.bw if args.bw > 0 else min(args.rate, 30e6)

    # Prefer a single Soapy args string; bare dict(driver=…) occasionally opens '' after usbipd churn.
    arg_str = f"driver=lime,serial={args.serial}" if args.serial else "driver=lime"
    try:
        dev = SoapySDR.Device(arg_str)
    except Exception as first:
        try:
            dev = SoapySDR.Device(dict(driver="lime"))
        except Exception:
            raise first
    dev.setSampleRate(SOAPY_SDR_RX, 0, args.rate)
    dev.setFrequency(SOAPY_SDR_RX, 0, args.lo)
    try:
        dev.setGain(SOAPY_SDR_RX, 0, args.gain)
    except Exception:
        pass
    try:
        dev.setBandwidth(SOAPY_SDR_RX, 0, bw)
    except Exception:
        pass

    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    dev.activateStream(stream)

    chunk = 16384
    buf = np.zeros(chunk, np.complex64)
    # Flush a little before declaring READY.
    t_flush = time.time() + 0.05
    while time.time() < t_flush:
        dev.readStream(stream, [buf], chunk, timeoutUs=100000)

    n_total = int(round(args.rate * args.seconds))
    iq = np.empty(n_total, np.complex64)
    got = 0
    t0 = time.time()
    print("READY", flush=True)
    while got < n_total:
        need = min(chunk, n_total - got)
        sr = dev.readStream(stream, [iq[got:got + need]], need, timeoutUs=1000000)
        if sr.ret > 0:
            got += sr.ret
        elif sr.ret < 0:
            print(f"ERR readStream {sr.ret}", flush=True)
            break
    t1 = time.time()

    dev.deactivateStream(stream)
    dev.closeStream(stream)

    iq = iq[:got]
    np.save(out, iq)
    meta = {
        "path": str(out),
        "n": int(iq.size),
        "rate_hz": args.rate,
        "lo_hz": args.lo,
        "gain_db": args.gain,
        "bw_hz": bw,
        "seconds_requested": args.seconds,
        "seconds_wall": t1 - t0,
        "t_host_start": t0,
        "t_host_end": t1,
        "serial": "1D40ED03BF7C98",
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE n={iq.size} path={out}", flush=True)
    return 0 if got >= n_total * 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
