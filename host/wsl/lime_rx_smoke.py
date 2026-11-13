#!/usr/bin/env python3
"""One-shot LimeSDR Mini RX peak at a CW offset. Prints READY then one JSON line."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=2452e6)
    ap.add_argument("--rate", type=float, default=8e6)
    ap.add_argument("--gain", type=float, default=30)
    ap.add_argument("--seconds", type=float, default=0.05)
    ap.add_argument("--tone-offset", type=float, default=0.0,
                    help="expected tone Hz relative to LO (0 = LO)")
    args = ap.parse_args()

    args_dev = dict(driver="lime")
    dev = SoapySDR.Device(args_dev)
    dev.setSampleRate(SOAPY_SDR_RX, 0, args.rate)
    dev.setFrequency(SOAPY_SDR_RX, 0, args.lo)
    try:
        dev.setGain(SOAPY_SDR_RX, 0, args.gain)
    except Exception:
        pass
    try:
        dev.setBandwidth(SOAPY_SDR_RX, 0, min(args.rate, 30e6))
    except Exception:
        pass

    n = max(4096, int(args.rate * args.seconds))
    n = 1 << int(np.ceil(np.log2(n)))
    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    dev.activateStream(stream)
    buf = np.zeros(n, np.complex64)
    # flush
    t_end = time.time() + 0.05
    while time.time() < t_end:
        dev.readStream(stream, [buf], len(buf), timeoutUs=100000)

    got = 0
    while got < n:
        sr = dev.readStream(stream, [buf[got:]], n - got, timeoutUs=1000000)
        if sr.ret > 0:
            got += sr.ret
        elif sr.ret < 0:
            print(f"ERR readStream {sr.ret}", file=sys.stderr)
            break

    dev.deactivateStream(stream)
    dev.closeStream(stream)

    iq = buf[:got]
    win = np.hanning(iq.size)
    spec = np.fft.fftshift(np.fft.fft(iq * win))
    psd = 20 * np.log10(np.maximum(np.abs(spec) / iq.size, 1e-12))
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.size, 1.0 / args.rate))
    peak_i = int(np.argmax(psd))
    # bin nearest expected offset
    want = args.tone_offset
    near_i = int(np.argmin(np.abs(freqs - want)))
    noise = float(np.median(psd))
    out = {
        "n": int(iq.size),
        "lo_hz": args.lo,
        "rate_hz": args.rate,
        "gain_db": args.gain,
        "peak_hz": float(freqs[peak_i]),
        "peak_db": float(psd[peak_i]),
        "tone_offset_hz": want,
        "tone_bin_hz": float(freqs[near_i]),
        "tone_db": float(psd[near_i]),
        "noise_median_db": noise,
        "tone_snr_db": float(psd[near_i] - noise),
    }
    print("READY", flush=True)
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
