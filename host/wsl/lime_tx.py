#!/usr/bin/env python3
"""LimeSDR Mini TX source for over-the-air bring-up (runs inside WSL, SoapySDR).

Streams a CW tone at lo + offset, optionally keyed into bursts of `tau` seconds every
`period` seconds. Controlled by newline commands on stdin, replies on stdout:
  freq <lo_hz> | gain <dB> | burst <tau_s> <period_s> | cw | off | status | quit
Keep the gain low: this radiates through an antenna in the 2.4 GHz ISM band.
"""
import argparse
import sys
import threading
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_TX


class State:
    def __init__(self, gain):
        self.gain = gain
        self.mode = "cw"
        self.tau = 0.0
        self.period = 0.0
        self.dirty = True
        self.quit = False
        self.lock = threading.Lock()


def reader(st, sdr):
    for line in sys.stdin:
        p = line.split()
        if not p:
            continue
        with st.lock:
            try:
                if p[0] == "freq":
                    sdr.setFrequency(SOAPY_SDR_TX, 0, float(p[1]))
                    print(f"OK freq {sdr.getFrequency(SOAPY_SDR_TX, 0):.0f}", flush=True)
                elif p[0] == "gain":
                    st.gain = float(p[1])
                    sdr.setGain(SOAPY_SDR_TX, 0, st.gain)
                    print(f"OK gain {sdr.getGain(SOAPY_SDR_TX, 0):.1f}", flush=True)
                elif p[0] == "burst":
                    st.mode, st.tau, st.period, st.dirty = "burst", float(p[1]), float(p[2]), True
                    print(f"OK burst {st.tau} {st.period}", flush=True)
                elif p[0] in ("cw", "off"):
                    st.mode, st.dirty = p[0], True
                    print(f"OK {p[0]}", flush=True)
                elif p[0] == "status":
                    print(f"STATUS mode={st.mode} gain={sdr.getGain(SOAPY_SDR_TX, 0):.1f} "
                          f"tau={st.tau} period={st.period}", flush=True)
                elif p[0] == "quit":
                    st.quit = True
                    print("OK quit", flush=True)
                    return
                else:
                    print("ERR command", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"ERR {e}", flush=True)
    st.quit = True


def waveform(st, rate, offset, amp):
    if st.mode == "off":
        return np.zeros(8192, np.complex64)
    if st.mode == "cw":
        # integer number of tone cycles so the buffer loops without phase jumps
        n = 8192
        k = round(offset * n / rate)
        return (amp * np.exp(2j * np.pi * k * np.arange(n) / n)).astype(np.complex64)
    n = int(round(st.period * rate))
    on = int(round(st.tau * rate))
    x = np.zeros(n, np.complex64)
    t = np.arange(on)
    x[:on] = amp * np.exp(2j * np.pi * offset * t / rate)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lo", type=float, default=2450e6)
    ap.add_argument("--offset", type=float, default=2e6)
    ap.add_argument("--rate", type=float, default=8e6)
    ap.add_argument("--gain", type=float, default=20.0)
    ap.add_argument("--amp", type=float, default=0.7)
    ap.add_argument("--antenna", default="BAND2")
    a = ap.parse_args()

    sdr = SoapySDR.Device(dict(driver="lime"))
    sdr.setSampleRate(SOAPY_SDR_TX, 0, a.rate)
    sdr.setFrequency(SOAPY_SDR_TX, 0, a.lo)
    sdr.setAntenna(SOAPY_SDR_TX, 0, a.antenna)
    sdr.setGain(SOAPY_SDR_TX, 0, a.gain)
    rng = sdr.getGainRange(SOAPY_SDR_TX, 0)
    st = State(a.gain)
    stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    sdr.activateStream(stream)
    print(f"READY lo={sdr.getFrequency(SOAPY_SDR_TX, 0):.0f} rate={sdr.getSampleRate(SOAPY_SDR_TX, 0):.0f} "
          f"gain={sdr.getGain(SOAPY_SDR_TX, 0):.1f} range=[{rng.minimum()},{rng.maximum()}] "
          f"antenna={sdr.getAntenna(SOAPY_SDR_TX, 0)} tone={a.lo + a.offset:.0f}", flush=True)
    threading.Thread(target=reader, args=(st, sdr), daemon=True).start()

    buf = None
    pos = 0
    chunk = 16384
    underflows = 0
    while not st.quit:
        with st.lock:
            if st.dirty:
                buf = waveform(st, a.rate, a.offset, a.amp)
                pos = 0
                st.dirty = False
        idx = (pos + np.arange(chunk)) % len(buf)
        sr = sdr.writeStream(stream, [buf[idx]], chunk, timeoutUs=100000)
        if sr.ret > 0:
            pos = (pos + sr.ret) % len(buf)
        else:
            underflows += 1
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)
    print(f"DONE underflows={underflows}", flush=True)


if __name__ == "__main__":
    main()
