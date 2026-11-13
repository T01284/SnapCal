"""Windows-side handle for host/wsl/lime_rx.py (LimeSDR Mini RX via usbipd/WSL)."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "wsl" / "lime_rx.py"
CAP_DIR = Path(__file__).resolve().parents[2] / "results" / "_tmp_lime_rx"


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


class LimeRx:
    def __init__(self, lo_hz: float, gain_db: float = 40.0, rate_hz: float = 30e6,
                 antenna: str = "Auto", timeout: float = 60.0):
        CAP_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            "wsl", "-e", "python3", "-u", _wsl_path(SCRIPT),
            "--lo", str(lo_hz), "--gain", str(gain_db), "--rate", str(rate_hz),
            "--antenna", antenna,
        ]
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace",
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.log: list[str] = []
        self.rate_hz = rate_hz
        self.lo_hz = lo_hz
        self._n = 0
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._expect("READY", timeout)
        for part in self.ready.split():
            if part.startswith("rate="):
                self.rate_hz = float(part.split("=", 1)[1])
            if part.startswith("lo="):
                self.lo_hz = float(part.split("=", 1)[1])

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
                    raise RuntimeError("lime_rx exited:\n" + "\n".join(self.log[-20:]))
                continue
            if line.startswith(prefix):
                return line
            if line.startswith("ERR"):
                raise RuntimeError(line)
        raise TimeoutError(f"lime_rx: no {prefix!r}; log:\n" + "\n".join(self.log[-20:]))

    def send(self, text: str, settle: float = 0.2) -> str:
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()
        r = self._expect("OK")
        time.sleep(settle)
        return r

    def freq(self, lo_hz: float) -> str:
        self.lo_hz = lo_hz
        return self.send(f"freq {lo_hz}", settle=0.4)

    def gain(self, db: float) -> str:
        return self.send(f"gain {db}")

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
