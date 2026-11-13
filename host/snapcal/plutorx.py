"""Windows-side PlutoSDR RX via pyadi-iio (USB Ethernet, default ip:192.168.20.100).

Requires libiio.dll on PATH (see third_party/libiio/win/Windows-VS-2022-x64).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_DLL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "libiio" / "win" / "Windows-VS-2022-x64"
_VENV_SCRIPTS = Path(__file__).resolve().parents[2] / ".venv" / "Scripts"


def _ensure_dll():
    for d in (_VENV_SCRIPTS, _DLL_DIR):
        if (d / "libiio.dll").exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("libiio.dll not found; unpack third_party/libiio Windows.zip")


class PlutoRx:
    def __init__(self, lo_hz: float, gain_db: float = 30.0, rate_hz: float = 20e6,
                 uri: str = "ip:192.168.20.100", rf_bw_hz: float | None = None):
        _ensure_dll()
        import adi  # noqa: WPS433 — after DLL path
        self.s = adi.Pluto(uri)
        self.uri = uri
        self.rate_hz = float(rate_hz)
        self.lo_hz = float(lo_hz)
        self.s.rx_lo = int(lo_hz)
        self.s.sample_rate = int(rate_hz)
        bw = int(rf_bw_hz if rf_bw_hz is not None else min(rate_hz * 0.9, 56e6))
        self.s.rx_rf_bandwidth = bw
        self.s.rx_buffer_size = 16384
        self.s.gain_control_mode_chan0 = "manual"
        self.s.rx_hardwaregain_chan0 = float(gain_db)
        self.ready = (
            f"READY uri={uri} lo={self.s.rx_lo} rate={self.s.sample_rate} "
            f"gain={self.s.rx_hardwaregain_chan0} bw={self.s.rx_rf_bandwidth}"
        )
        # flush one buffer
        _ = self.s.rx()

    def freq(self, lo_hz: float) -> str:
        self.lo_hz = float(lo_hz)
        self.s.rx_lo = int(lo_hz)
        return f"OK freq {self.s.rx_lo}"

    def gain(self, db: float) -> str:
        self.s.rx_hardwaregain_chan0 = float(db)
        return f"OK gain {self.s.rx_hardwaregain_chan0}"

    def capture(self, n: int = 16384) -> np.ndarray:
        if int(self.s.rx_buffer_size) != int(n):
            self.s.rx_buffer_size = int(n)
        x = self.s.rx()
        # adi returns complex64 already scaled ~FS
        return np.asarray(x, dtype=np.complex64)

    def close(self) -> None:
        try:
            self.s.rx_destroy_buffer()
        except Exception:
            pass
        try:
            del self.s
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
