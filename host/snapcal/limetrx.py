"""Windows handle for host/wsl/lime_trx.py — one Lime Mini, TX + RX together."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "wsl" / "lime_trx.py"
CAP_DIR = Path(__file__).resolve().parents[2] / "results" / "_tmp_lime_rx"


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


class LimeTrx:
    """TX API mirrors LimeTx; RX API mirrors LimeRx (same process)."""

    def __init__(
        self,
        tx_lo_hz: float,
        offset_hz: float,
        tx_gain_db: float,
        rx_lo_hz: float,
        rx_gain_db: float = 40.0,
        rx_rate_hz: float = 30e6,
        tx_rate_hz: float = 8e6,
        rx_antenna: str = "Auto",
        max_gain_db: float = 64.0,
        timeout: float = 90.0,
    ):
        self.max_gain_db = max_gain_db
        self._check_gain(tx_gain_db)
        CAP_DIR.mkdir(parents=True, exist_ok=True)
        # Full-duplex on Mini: keep TX rate == RX rate so CW offset lands correctly.
        tx_rate_hz = rx_rate_hz
        cmd = [
            "wsl", "-e", "python3", "-u", _wsl_path(SCRIPT),
            "--tx-lo", str(tx_lo_hz),
            "--rx-lo", str(rx_lo_hz),
            "--offset", str(offset_hz),
            "--tx-gain", str(tx_gain_db),
            "--rx-gain", str(rx_gain_db),
            "--tx-rate", str(tx_rate_hz),
            "--rx-rate", str(rx_rate_hz),
            "--rx-antenna", rx_antenna,
        ]
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.log: list[str] = []
        self.rate_hz = rx_rate_hz
        self.lo_hz = rx_lo_hz
        self.tone_hz = tx_lo_hz + offset_hz
        self._n = 0
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._expect("READY", timeout)
        for part in self.ready.split():
            if part.startswith("rx_rate="):
                self.rate_hz = float(part.split("=", 1)[1])
            if part.startswith("rx_lo="):
                self.lo_hz = float(part.split("=", 1)[1])
        time.sleep(0.4)

    def _pump(self):
        for line in self.p.stdout:
            line = line.strip()
            self.log.append(line)
            self.lines.put(line)

    def _expect(self, prefix: str, timeout: float = 15.0) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError("lime_trx exited:\n" + "\n".join(self.log[-30:]))
                continue
            if line.startswith(prefix):
                return line
            if line.startswith("ERR"):
                raise RuntimeError(line)
        raise TimeoutError(f"lime_trx: no {prefix!r}; log:\n" + "\n".join(self.log[-30:]))

    def send(self, text: str, settle: float = 0.25) -> str:
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()
        cmd0 = text.split()[0]
        expect = "STATUS" if cmd0 == "txstatus" else ("CAPOK" if cmd0 == "capture" else "OK")
        if cmd0 == "capture":
            r = self._expect("CAPOK", timeout=30.0)
        else:
            r = self._expect(expect)
        time.sleep(settle)
        return r

    def _check_gain(self, db: float) -> None:
        if db > self.max_gain_db:
            raise ValueError(f"Lime TX gain {db} dB above the limit {self.max_gain_db} dB for this setup")

    # --- TX ---
    def gain(self, db: float) -> str:
        self._check_gain(db)
        return self.send(f"txgain {db}")

    def cw(self) -> str:
        return self.send("cw")

    def off(self) -> str:
        return self.send("off")

    # --- RX ---
    def capture(self, n: int = 16384) -> np.ndarray:
        self._n += 1
        path = CAP_DIR / f"cap_{self._n}.npy"
        wsl_out = _wsl_path(path)
        self.p.stdin.write(f"capture {int(n)} {wsl_out}\n")
        self.p.stdin.flush()
        self._expect("CAPOK", timeout=30.0)
        return np.load(path)

    def close(self) -> None:
        try:
            self.send("quit", settle=0)
            self._expect("DONE", 10)
        except Exception:
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
