"""Windows-side handle for host/wsl/lime_tx.py running inside WSL (LimeSDR via usbipd)."""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "wsl" / "lime_tx.py"


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


class LimeTx:
    def __init__(self, lo_hz: float, offset_hz: float, gain_db: float, rate_hz: float = 8e6,
                 amp: float = 0.7, timeout: float = 60.0, max_gain_db: float = 64.0):
        """max_gain_db guards conducted setups (TX port cabled to the ESP32 without attenuators)."""
        self.max_gain_db = max_gain_db
        self._check_gain(gain_db)
        cmd = ["wsl", "-e", "python3", "-u", _wsl_path(SCRIPT), "--lo", str(lo_hz),
               "--offset", str(offset_hz), "--gain", str(gain_db), "--rate", str(rate_hz), "--amp", str(amp)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  encoding="utf-8", errors="replace")
        self.lines: queue.Queue[str] = queue.Queue()
        self.log: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready = self._expect("READY", timeout)
        self.tone_hz = lo_hz + offset_hz
        time.sleep(0.5)

    def _pump(self):
        for line in self.p.stdout:
            line = line.strip()
            self.log.append(line)
            self.lines.put(line)

    def _expect(self, prefix: str, timeout: float = 10.0) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError("lime_tx exited:\n" + "\n".join(self.log[-20:]))
                continue
            if line.startswith(prefix):
                return line
            if line.startswith("ERR"):
                raise RuntimeError(line)
        raise TimeoutError(f"lime_tx: no {prefix!r}; log:\n" + "\n".join(self.log[-20:]))

    def send(self, text: str, settle: float = 0.3) -> str:
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()
        r = self._expect("OK" if text.split()[0] != "status" else "STATUS")
        time.sleep(settle)
        return r

    def freq(self, lo_hz: float, offset_hz: float) -> str:
        self.tone_hz = lo_hz + offset_hz
        return self.send(f"freq {lo_hz}", settle=0.5)

    def _check_gain(self, db: float) -> None:
        if db > self.max_gain_db:
            raise ValueError(f"Lime TX gain {db} dB above the limit {self.max_gain_db} dB for this setup")

    def gain(self, db: float) -> str:
        self._check_gain(db)
        return self.send(f"gain {db}")

    def cw(self) -> str:
        return self.send("cw")

    def off(self) -> str:
        return self.send("off")

    def burst(self, tau_s: float, period_s: float) -> str:
        return self.send(f"burst {tau_s} {period_s}")

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
