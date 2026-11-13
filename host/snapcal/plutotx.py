"""Windows-side PlutoSDR TX via pyadi-iio DDS CW (not DMA buffer).

Gain is tx_hardwaregain: 0 = max power, more negative = quieter (−89.75 .. 0).
DDS single-tone is used because DMA cyclic TX was unreliable on this bench.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_DLL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "libiio" / "win" / "Windows-VS-2022-x64"
_VENV_SCRIPTS = Path(__file__).resolve().parents[2] / ".venv" / "Scripts"


def _ensure_dll():
    for d in (_VENV_SCRIPTS, _DLL_DIR):
        if (d / "libiio.dll").exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("libiio.dll not found; unpack third_party/libiio Windows.zip")


class PlutoTx:
    def __init__(
        self,
        lo_hz: float,
        offset_hz: float,
        gain_db: float = -20.0,
        uri: str = "ip:192.168.20.100",
        rate_hz: float = 20e6,
        amp: float = 0.5,
        max_gain_db: float = 0.0,
    ):
        """amp is DDS scale (0..1). offset_hz is the BB tone relative to tx_lo."""
        _ensure_dll()
        import adi  # noqa: WPS433

        self.max_gain_db = max_gain_db
        self._check_gain(gain_db)
        self.uri = uri
        self.rate_hz = float(rate_hz)
        self.offset_hz = float(offset_hz)
        self.amp = float(amp)
        self._gain = float(gain_db)
        self._on = False
        self.s = adi.Pluto(uri)
        self.s.sample_rate = int(rate_hz)
        self.s.tx_rf_bandwidth = int(min(rate_hz * 0.9, 56e6))
        self.s.tx_lo = int(lo_hz)
        self.s.tx_hardwaregain_chan0 = float(gain_db)
        self.s.disable_dds()
        self.tone_hz = lo_hz + offset_hz
        self.ready = (
            f"READY uri={uri} lo={self.s.tx_lo} rate={self.s.sample_rate} "
            f"gain={self.s.tx_hardwaregain_chan0} tone={self.tone_hz:.0f} via=dds"
        )
        time.sleep(0.2)

    def _check_gain(self, db: float) -> None:
        if db > self.max_gain_db:
            raise ValueError(
                f"Pluto TX gain {db} dB above the limit {self.max_gain_db} dB for this setup"
            )

    def freq(self, lo_hz: float, offset_hz: float | None = None) -> str:
        if offset_hz is not None:
            self.offset_hz = float(offset_hz)
        self.s.tx_lo = int(lo_hz)
        self.tone_hz = lo_hz + self.offset_hz
        if self._on:
            self.cw()
        return f"OK freq {self.s.tx_lo}"

    def gain(self, db: float) -> str:
        self._check_gain(db)
        self._gain = float(db)
        self.s.tx_hardwaregain_chan0 = self._gain
        return f"OK gain {self.s.tx_hardwaregain_chan0}"

    def cw(self) -> str:
        self.s.dds_single_tone(int(round(self.offset_hz)), float(self.amp))
        self._on = True
        time.sleep(0.2)
        return "OK cw"

    def off(self) -> str:
        self.s.disable_dds()
        self._on = False
        time.sleep(0.1)
        return "OK off"

    def close(self) -> None:
        try:
            self.off()
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
