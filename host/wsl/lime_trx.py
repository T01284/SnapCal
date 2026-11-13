#!/usr/bin/env python3
"""LimeSDR Mini full-duplex TX+RX in one Soapy process (WSL / usbipd).

Needed for Q8 Lime IIP3: same Mini TX feeds the combiner while RX is the DUT.
stdin commands (newline-terminated), replies on stdout:
  txfreq <lo_hz> | txgain <dB> | cw | off | txstatus
  freq <lo_hz> | gain <dB> | antenna <name> | capture <n> <out.npy>
  quit
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TX


class TxState:
    def __init__(self, gain: float, offset: float, amp: float, rate: float):
        self.gain = gain
        self.offset = offset
        self.amp = amp
        self.rate = rate
        self.mode = "off"
        self.dirty = True
        self.quit = False
        self.lock = threading.Lock()


def waveform(st: TxState) -> np.ndarray:
    if st.mode == "off":
        return np.zeros(8192, np.complex64)
    n = 8192
    k = round(st.offset * n / st.rate)
    return (st.amp * np.exp(2j * np.pi * k * np.arange(n) / n)).astype(np.complex64)


def tx_loop(sdr, stream, st: TxState) -> None:
    buf = waveform(st)
    pos = 0
    chunk = 16384
    while not st.quit:
        with st.lock:
            if st.dirty:
                buf = waveform(st)
                pos = 0
                st.dirty = False
        idx = (pos + np.arange(chunk)) % len(buf)
        sr = sdr.writeStream(stream, [buf[idx]], chunk, timeoutUs=100000)
        if sr.ret > 0:
            pos = (pos + sr.ret) % len(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tx-lo", type=float, required=True)
    ap.add_argument("--rx-lo", type=float, required=True)
    ap.add_argument("--offset", type=float, default=2e6)
    ap.add_argument("--tx-rate", type=float, default=8e6)
    ap.add_argument("--rx-rate", type=float, default=30e6)
    ap.add_argument("--tx-gain", type=float, default=20.0)
    ap.add_argument("--rx-gain", type=float, default=40.0)
    ap.add_argument("--amp", type=float, default=0.7)
    ap.add_argument("--tx-antenna", default="BAND2")
    ap.add_argument("--rx-antenna", default="Auto")
    ap.add_argument("--serial", default="1D40ED03BF7C98")
    a = ap.parse_args()

    try:
        sdr = SoapySDR.Device(dict(driver="lime", serial=a.serial)) if a.serial else SoapySDR.Device(dict(driver="lime"))
    except Exception:
        sdr = SoapySDR.Device(dict(driver="lime"))

    # Lime Mini often locks TX/RX sample rates together — set RX first, then force TX
    # to the same rate and build the CW waveform from the *actual* TX rate.
    sdr.setSampleRate(SOAPY_SDR_RX, 0, a.rx_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, a.rx_lo)
    ants = list(sdr.listAntennas(SOAPY_SDR_RX, 0))
    for name in (a.rx_antenna, "Auto", "LNAW", "LNAH"):
        if name in ants:
            try:
                sdr.setAntenna(SOAPY_SDR_RX, 0, name)
                break
            except Exception:
                continue
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, a.rx_gain)
    except Exception:
        pass
    try:
        sdr.setBandwidth(SOAPY_SDR_RX, 0, min(a.rx_rate, 30e6))
    except Exception:
        pass

    tx_rate_req = a.tx_rate if a.tx_rate > 0 else a.rx_rate
    sdr.setSampleRate(SOAPY_SDR_TX, 0, tx_rate_req)
    # Re-assert after RX; some Lime builds silently retarget TX to RX rate.
    sdr.setSampleRate(SOAPY_SDR_TX, 0, sdr.getSampleRate(SOAPY_SDR_RX, 0))
    sdr.setFrequency(SOAPY_SDR_TX, 0, a.tx_lo)
    sdr.setAntenna(SOAPY_SDR_TX, 0, a.tx_antenna)
    sdr.setGain(SOAPY_SDR_TX, 0, a.tx_gain)

    tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(tx_stream)
    sdr.activateStream(rx_stream)

    warm = np.zeros(16384, np.complex64)
    for _ in range(8):
        sdr.readStream(rx_stream, [warm], len(warm), timeoutUs=200000)

    tx_rate_actual = float(sdr.getSampleRate(SOAPY_SDR_TX, 0))
    st = TxState(a.tx_gain, a.offset, a.amp, tx_rate_actual)
    threading.Thread(target=tx_loop, args=(sdr, tx_stream, st), daemon=True).start()

    tx_rng = sdr.getGainRange(SOAPY_SDR_TX, 0)
    rx_rng = sdr.getGainRange(SOAPY_SDR_RX, 0)
    print(
        f"READY tx_lo={sdr.getFrequency(SOAPY_SDR_TX, 0):.0f} "
        f"rx_lo={sdr.getFrequency(SOAPY_SDR_RX, 0):.0f} "
        f"tx_rate={sdr.getSampleRate(SOAPY_SDR_TX, 0):.0f} "
        f"rx_rate={sdr.getSampleRate(SOAPY_SDR_RX, 0):.0f} "
        f"tx_gain={sdr.getGain(SOAPY_SDR_TX, 0):.1f} "
        f"rx_gain={sdr.getGain(SOAPY_SDR_RX, 0):.1f} "
        f"tx_range=[{tx_rng.minimum()},{tx_rng.maximum()}] "
        f"rx_range=[{rx_rng.minimum()},{rx_rng.maximum()}] "
        f"tx_ant={sdr.getAntenna(SOAPY_SDR_TX, 0)} "
        f"rx_ant={sdr.getAntenna(SOAPY_SDR_RX, 0)} "
        f"tone={a.tx_lo + a.offset:.0f}",
        flush=True,
    )

    for line in sys.stdin:
        p = line.split()
        if not p:
            continue
        try:
            if p[0] == "txfreq":
                sdr.setFrequency(SOAPY_SDR_TX, 0, float(p[1]))
                print(f"OK txfreq {sdr.getFrequency(SOAPY_SDR_TX, 0):.0f}", flush=True)
            elif p[0] == "txgain":
                with st.lock:
                    st.gain = float(p[1])
                sdr.setGain(SOAPY_SDR_TX, 0, st.gain)
                print(f"OK txgain {sdr.getGain(SOAPY_SDR_TX, 0):.1f}", flush=True)
            elif p[0] in ("cw", "off"):
                with st.lock:
                    st.mode, st.dirty = p[0], True
                print(f"OK {p[0]}", flush=True)
            elif p[0] == "txstatus":
                print(
                    f"STATUS mode={st.mode} gain={sdr.getGain(SOAPY_SDR_TX, 0):.1f}",
                    flush=True,
                )
            elif p[0] == "freq":
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
                    sr = sdr.readStream(rx_stream, [iq[got: got + need]], need, timeoutUs=1000000)
                    if sr.ret > 0:
                        got += sr.ret
                    elif sr.ret < 0:
                        print(f"ERR readStream {sr.ret}", flush=True)
                        break
                np.save(out, iq[:got])
                print(f"CAPOK {got} {out}", flush=True)
            elif p[0] == "quit":
                st.quit = True
                print("OK quit", flush=True)
                break
            else:
                print("ERR command", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {e}", flush=True)

    time.sleep(0.2)
    sdr.deactivateStream(tx_stream)
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(tx_stream)
    sdr.closeStream(rx_stream)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
