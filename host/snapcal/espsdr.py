"""Serial client for the SnapCal NBC firmware (ESP-SDR burst protocol 6).

Commands are newline-terminated ASCII; replies are text lines, and capture
replies (`DATA <n> <crc32> <capture_us>`) are followed by a binary payload.
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import serial

RATE_INDEX = {80_000_000: 0, 40_000_000: 1, 20_000_000: 2, 10_000_000: 3,
              8_000_000: 4, 4_000_000: 5, 16_000_000: 6}
# Wire words carry I in bits 0..9 and Q in bits 10..19. On the ESP32-S3 and the
# ESP32-C5 that spectrum is inverted (C5 checked at 2442 and 5800 MHz, 2026-10-18).
# Conjugating gives absolute frequency = fc + f. Re-check the 5/6 LO mode (<2210 MHz).
CONJUGATE = {"S3SDR": True, "C5SDR": True}

# fmt -> (command, bits per complex sample on the wire)
FORMATS = {"raw": ("CAP", 32), "iq10": ("CAP20", 20), "iq8": ("CAP16", 16)}
# Energy-detect uploads (paper mode B): same formats, CAPE* command family.
ENERGY_FORMATS = {"raw": ("CAPE", 32), "iq10": ("CAPE20", 20), "iq8": ("CAPE16", 16)}


class EspSdrError(RuntimeError):
    pass


@dataclass
class Snapshot:
    iq: np.ndarray            # complex64, full scale = 1.0, absolute frequency = fc + f
    n: int
    rate_hz: int
    fmt: str
    capture_us: int           # firmware-measured capture time
    t_host_start: float       # time.time() when the command was sent
    t_host_end: float         # time.time() when the payload was fully read
    crc32: int
    meta: dict = field(default_factory=dict)


@dataclass
class EnergyResult:
    """CAPE* reply: miss is text-only; hit carries the same payload as CAP*."""
    hit: bool
    code: int                 # mean-power wire code (~0.5 dB/step)
    capture_us: int
    t_host_start: float
    t_host_end: float
    snapshot: Snapshot | None = None


def _sign_extend(v: np.ndarray, bits: int) -> np.ndarray:
    v = v.astype(np.int32)
    return (v ^ (1 << (bits - 1))) - (1 << (bits - 1))


def unpack_raw(payload: bytes, n: int) -> np.ndarray:
    w = np.frombuffer(payload, dtype="<u4", count=n)
    i = _sign_extend(w & 0x3FF, 10)
    q = _sign_extend((w >> 10) & 0x3FF, 10)
    return (i + 1j * q).astype(np.complex64) / 512.0


def unpack_iq10(payload: bytes, n: int) -> np.ndarray:
    """Two 20-bit samples per 5 bytes, little-endian; odd tail uses 3 bytes."""
    b = np.frombuffer(payload, dtype=np.uint8).astype(np.uint64)
    pairs = n // 2
    words = np.empty(n, dtype=np.uint64)
    if pairs:
        g = b[: pairs * 5].reshape(pairs, 5)
        v = g[:, 0] | (g[:, 1] << 8) | (g[:, 2] << 16) | (g[:, 3] << 24) | (g[:, 4] << 32)
        words[0::2][:pairs] = v & 0xFFFFF
        words[1::2][:pairs] = (v >> 20) & 0xFFFFF
    if n % 2:
        t = b[pairs * 5: pairs * 5 + 3]
        words[-1] = (t[0] | (t[1] << 8) | (t[2] << 16)) & 0xFFFFF
    i = _sign_extend(words & 0x3FF, 10)
    q = _sign_extend((words >> 10) & 0x3FF, 10)
    return (i + 1j * q).astype(np.complex64) / 512.0


def unpack_iq8(payload: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(payload, dtype=np.int8, count=2 * n).astype(np.float32)
    return (a[0::2] + 1j * a[1::2]).astype(np.complex64) / 128.0


def payload_size(n: int, bits: int) -> int:
    return (n * bits + 7) // 8


class EspSdr:
    """One client session. DTR/RTS stay low so opening the port does not reset the S3."""

    def __init__(self, port: str, baud: int = 2_000_000, timeout: float = 5.0):
        s = serial.Serial(port=None, baudrate=baud, timeout=timeout)
        s.dtr = False
        s.rts = False
        s.port = port
        s.open()
        self.ser = s
        self.port = port
        try:
            self.sync()
        except EspSdrError:
            s.close()
            raise
        self.identity = self.command("INFO")
        self.conjugate = CONJUGATE.get(self.identity.split()[0], False)

    def close(self) -> None:
        try:
            self.command("RELEASE")
        except Exception:
            pass
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- transport -------------------------------------------------------
    def _readline(self) -> str:
        line = self.ser.readline()
        if not line:
            raise EspSdrError("timeout waiting for reply")
        return line.decode("ascii", "replace").strip()

    def _read_exact(self, count: int) -> bytes:
        buf = bytearray()
        while len(buf) < count:
            chunk = self.ser.read(count - len(buf))
            if not chunk:
                raise EspSdrError(f"short payload {len(buf)}/{count}")
            buf.extend(chunk)
        return bytes(buf)

    def command(self, text: str) -> str:
        self.ser.write((text + "\n").encode("ascii"))
        reply = self._readline()
        if reply.startswith("ERR"):
            raise EspSdrError(f"{text!r} -> {reply}")
        return reply

    def sync(self, attempts: int = 20) -> None:
        old = self.ser.timeout
        self.ser.timeout = 0.2
        try:
            for k in range(attempts):
                nonce = int(time.time() * 1000) + k
                self.ser.reset_input_buffer()
                self.ser.write(f"\nSYNC {nonce}\n".encode())
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    if self.ser.readline().strip() == f"SYNC {nonce}".encode():
                        self.ser.reset_input_buffer()
                        return
            raise EspSdrError("could not synchronise with firmware")
        finally:
            self.ser.timeout = old

    # -- queries ---------------------------------------------------------
    def info(self) -> dict:
        out = {
            "info": self.command("INFO"),
            "caps": self.command("CAPS").split()[1:],
            "transport": self.command("TRANSPORT?"),
            "range": self.command("RANGE?"),
            "gain": self.command("GAIN?"),
            "lpf": self.command("LPF?"),
        }
        out["version"] = json.loads(self.command("VERSION?").split(" ", 1)[1])
        out["limits"] = json.loads(self.command("LIMITS?").split(" ", 1)[1])
        return out

    # -- receiver controls -----------------------------------------------
    def set_freq(self, mhz: int) -> None:
        self.command(f"FREQ {int(mhz)}")

    def set_bandwidth(self, mhz: int) -> None:
        self.command(f"BANDWIDTH {int(mhz)}")

    def set_gain(self, index: int | None) -> None:
        self.command("GAIN HARDWARE" if index is None else f"GAIN MANUAL {int(index)}")

    def gpio(self, pin: int, state: str) -> None:
        self.command(f"GPIO {int(pin)} {state}")

    # -- SnapCal scheduling hooks (firmware >= ffe0067) -------------------
    def set_jitter(self, max_us: int) -> None:
        """Uniform random delay in [0, max_us] before every capture, drawn on chip."""
        self.command(f"JITTER {int(max_us)}")

    def set_fast_poll(self, on: bool) -> None:
        self.command(f"POLL {int(bool(on))}")

    def timestamps(self) -> tuple[int, int]:
        """(command parsed, capture start) of the last capture, esp_timer microseconds."""
        _, cmd_us, start_us = self.command("TS?").split()
        return int(cmd_us), int(start_us)

    def set_marks(self, snap_pin: int | None, xfer_pin: int | None = None) -> None:
        if snap_pin is None:
            self.command("MARK OFF")
            return
        for p in (snap_pin, xfer_pin):
            self.gpio(p, "0")
        self.command(f"MARK {int(snap_pin)} {int(xfer_pin)}")

    # -- capture ---------------------------------------------------------
    def capture(self, n: int, rate_hz: int = 80_000_000, fmt: str = "iq10") -> Snapshot:
        cmd, bits = FORMATS[fmt]
        t0 = time.time()
        self.ser.write(f"{cmd} {int(n)} {RATE_INDEX[rate_hz]}\n".encode())
        head = self._readline()
        if not head.startswith("DATA"):
            raise EspSdrError(f"{cmd} -> {head}")
        _, n_s, crc_s, us_s = head.split()
        n_rx, crc = int(n_s), int(crc_s, 16)
        payload = self._read_exact(payload_size(n_rx, bits))
        t1 = time.time()
        if zlib.crc32(payload) != crc:
            raise EspSdrError("payload CRC mismatch")
        unpack = {32: unpack_raw, 20: unpack_iq10, 16: unpack_iq8}[bits]
        iq = unpack(payload, n_rx)
        if self.conjugate:
            iq = np.conj(iq)
        return Snapshot(iq, n_rx, rate_hz, fmt, int(us_s), t0, t1, crc,
                        meta={"conjugated": self.conjugate})

    def capture_energy(
        self,
        n: int,
        rate_hz: int = 80_000_000,
        fmt: str = "iq8",
        thresh: int = 0,
    ) -> EnergyResult:
        """On-chip mean-power gate. ``thresh`` 0 always hits; high values miss on quiet RF."""
        if not 0 <= int(thresh) <= 255:
            raise ValueError("thresh must be 0..255")
        cmd, bits = ENERGY_FORMATS[fmt]
        t0 = time.time()
        self.ser.write(f"{cmd} {int(n)} {RATE_INDEX[rate_hz]} {int(thresh)}\n".encode())
        head = self._readline()
        parts = head.split()
        if parts[0] == "MISS" and len(parts) == 3:
            t1 = time.time()
            return EnergyResult(False, int(parts[1]), int(parts[2]), t0, t1)
        if parts[0] == "HIT" and len(parts) == 5:
            n_rx, crc, us, code = int(parts[1]), int(parts[2], 16), int(parts[3]), int(parts[4])
            payload = self._read_exact(payload_size(n_rx, bits))
            t1 = time.time()
            if zlib.crc32(payload) != crc:
                raise EspSdrError("payload CRC mismatch")
            unpack = {32: unpack_raw, 20: unpack_iq10, 16: unpack_iq8}[bits]
            iq = unpack(payload, n_rx)
            if self.conjugate:
                iq = np.conj(iq)
            snap = Snapshot(
                iq, n_rx, rate_hz, fmt, us, t0, t1, crc,
                meta={"conjugated": self.conjugate, "energy_code": code, "thresh": int(thresh)},
            )
            return EnergyResult(True, code, us, t0, t1, snap)
        raise EspSdrError(f"{cmd} -> {head}")
