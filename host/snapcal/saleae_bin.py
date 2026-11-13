"""Saleae Logic 2 digital binary export (version 0 and 1) and snapshot edge timing.

SNAP_MARK is high for the capture window. XFER_DONE is a short pulse after the
payload has been queued to the USB FIFO, which is earlier than the host
finishing the read. See firmware ``MARK`` in ``receiver.c``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

IDENTIFIER = b"<SALEAE>"
TYPE_DIGITAL = 0


@dataclass
class DigitalChunk:
    initial_state: int
    begin_time: float
    end_time: float
    transition_times: np.ndarray
    sample_rate: float | None = None


@dataclass
class Pulse:
    rise: float
    fall: float

    @property
    def width(self) -> float:
        return self.fall - self.rise


def parse_digital(path: str) -> list[DigitalChunk]:
    with open(path, "rb") as f:
        ident = f.read(8)
        if ident != IDENTIFIER:
            raise ValueError(f"not a Saleae binary export: {ident!r}")
        version, datatype = struct.unpack("<ii", f.read(8))
        if datatype != TYPE_DIGITAL:
            raise ValueError(f"expected digital export, got type {datatype}")
        if version == 0:
            chunk_count = 1
        elif version == 1:
            chunk_count = struct.unpack("<Q", f.read(8))[0]
        else:
            raise ValueError(f"unsupported Saleae digital export version {version}")
        chunks = []
        for _ in range(chunk_count):
            initial = struct.unpack("<I", f.read(4))[0]
            sample_rate = struct.unpack("<d", f.read(8))[0] if version >= 1 else None
            begin, end, n_trans = struct.unpack("<ddQ", f.read(24))
            times = np.fromfile(f, dtype="<f8", count=n_trans)
            if times.size != n_trans:
                raise ValueError("truncated Saleae digital export")
            chunks.append(DigitalChunk(initial, begin, end, times, sample_rate))
    return chunks


def high_pulses(chunks: list[DigitalChunk], min_width_s: float = 0.0) -> list[Pulse]:
    """Complete high pulses. A level that is already high at the start of a chunk,
    or still high at the end, is not a pulse: one edge was outside the capture."""
    found: list[Pulse] = []
    for chunk in chunks:
        state = int(chunk.initial_state)
        rise: float | None = None
        for t in chunk.transition_times:
            t = float(t)
            if state == 0:
                rise = t
                state = 1
            else:
                if rise is not None and t - rise >= min_width_s:
                    found.append(Pulse(rise, t))
                rise = None
                state = 0
    return found


def pair_snapshots(snap: list[Pulse], xfer: list[Pulse]) -> list[dict]:
    """Match each SNAP_MARK pulse to the next XFER_DONE pulse before the following mark.

    ``t_queue_s`` is mark falling edge to XFER_DONE rising edge. ``t_cycle_s`` is
    mark rising edge to the next mark rising edge (absent on the last pulse).
    """
    out = []
    xi = 0
    for i, sp in enumerate(snap):
        nxt = snap[i + 1].rise if i + 1 < len(snap) else sp.fall + 1.0
        while xi < len(xfer) and xfer[xi].rise < sp.fall:
            xi += 1
        found = None
        if xi < len(xfer) and xfer[xi].rise < nxt:
            found = xfer[xi]
            xi += 1
        out.append({
            "tw_mark_s": sp.width,
            "t_queue_s": None if found is None else found.rise - sp.fall,
            "t_xfer_s": None if found is None else found.width,
            "t_cycle_s": None if i + 1 >= len(snap) else snap[i + 1].rise - sp.rise,
        })
    return out


def fit_overhead(tw_s: np.ndarray, t_cycle_s: np.ndarray) -> tuple[float, float]:
    """Fit T_cycle = (1+c) T_w + T_0. Returns (c, T_0 seconds)."""
    if tw_s.size < 2 or np.unique(tw_s).size < 2:
        raise ValueError("need at least two window lengths to fit c and T_0")
    c, t0 = np.polyfit(tw_s, t_cycle_s - tw_s, 1)
    return float(c), float(t0)


def _median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=float)))


def summarize_group(host: list[dict], paired: list[dict], sample_rate_hz: float) -> dict:
    """One (format, radio sample rate) group. ``host`` and ``paired`` are in capture order.

    ``host`` entries need ``n``, ``capture_us``, and ``host_cycle_s``.
    """
    if len(host) != len(paired):
        return {
            "n_host": len(host),
            "n_marks": len(paired),
            "error": "SNAP_MARK pulse count does not match the number of captures",
        }
    by_n: dict[str, dict] = {}
    tw_nom, t_cycle = [], []
    for h, p in zip(host, paired):
        n = int(h["n"])
        nominal = n / sample_rate_hz
        bucket = by_n.setdefault(str(n), {"tw_nominal_s": nominal, "mark": [], "queue": [],
                                          "cycle": [], "fw_us": [], "host_cycle_s": []})
        bucket["mark"].append(p["tw_mark_s"])
        bucket["fw_us"].append(h["capture_us"])
        bucket["host_cycle_s"].append(h["host_cycle_s"])
        if p["t_queue_s"] is not None:
            bucket["queue"].append(p["t_queue_s"])
        if p["t_cycle_s"] is not None:
            bucket["cycle"].append(p["t_cycle_s"])
            tw_nom.append(nominal)
            t_cycle.append(p["t_cycle_s"])
    rows = {}
    dt = 1.0 / 50_000_000  # replaced by caller when known; placeholder overwritten below
    del dt
    for n, b in by_n.items():
        mark = _median(b["mark"])
        nominal = b["tw_nominal_s"]
        err = (mark - nominal) / nominal * 100.0
        rows[n] = {
            "Tw_nominal_us": round(nominal * 1e6, 3),
            "Tw_mark_us_median": round(mark * 1e6, 3),
            "Tw_error_pct": round(err, 3),
            "within_1pct": abs(err) < 1.0,
            "fw_capture_us_median": round(_median(b["fw_us"]), 1),
            "T_queue_us_median": None if not b["queue"] else round(_median(b["queue"]) * 1e6, 3),
            "T_cycle_ms_median": None if not b["cycle"] else round(_median(b["cycle"]) * 1e3, 3),
            "host_cycle_ms_median": round(_median(b["host_cycle_s"]) * 1e3, 3),
            "n_marks": len(b["mark"]),
            "n_queued": len(b["queue"]),
        }
    fit = None
    if len(tw_nom) >= 2 and len(set(np.round(tw_nom, 12))) >= 2:
        c, t0 = fit_overhead(np.asarray(tw_nom), np.asarray(t_cycle))
        fit = {"c": round(c, 2), "T0_ms": round(t0 * 1e3, 3)}
    return {"n_host": len(host), "n_marks": len(paired), "by_n": rows, "fit": fit}
