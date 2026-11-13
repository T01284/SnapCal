"""Snapshot Intercept Probability (SIP) and schedule simulation.

Closed forms (times in seconds):
  p_int     - plan formula: chance that one burst with uniformly random arrival is
              caught by continuous snapshotting with window T_w and cycle (1+c)T_w+T_0
  p_snap    - chance that one snapshot overlaps a periodic burst train by >= tau_ov
              when its phase is uniform: (T_w + tau - 2 tau_ov) / T_b
  tau_star  - switch point of the bang-bang optimum, tau_min + T_0 / (1 + c)

The plan's tau_min equals 2 tau_ov when detection needs an overlap of tau_ov.

simulate() reproduces the firmware timing: a capture command is only parsed on a
polling grid (1 ms FreeRTOS tick in the stock firmware), optionally followed by an
on-chip uniform delay (JITTER), so with bursts commensurate with the grid the
snapshot phase can lock (sync trap).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def p_int(t_w, tau, tau_min, c, t0):
    t_w = np.asarray(t_w, dtype=float)
    return np.clip((t_w + tau - tau_min) / ((1 + c) * t_w + t0), 0.0, 1.0)


def p_snap(t_w, tau, t_b, tau_ov=0.0):
    t_w = np.asarray(t_w, dtype=float)
    p = (t_w + tau - 2 * tau_ov) / t_b
    ok = (tau_ov <= t_w) & (tau_ov <= tau)
    return np.where(ok, np.clip(p, 0.0, 1.0), 0.0)


def tau_star(tau_min, c, t0):
    return tau_min + t0 / (1 + c)


def best_window(tau, t_w_min, t_w_max, tau_min, c, t0):
    """Window (t_w_min or t_w_max) with the higher intercept probability per cycle."""
    return t_w_max if p_int(t_w_max, tau, tau_min, c, t0) >= p_int(t_w_min, tau, tau_min, c, t0) else t_w_min


def p_detect(p, m):
    """Independent snapshots: chance of at least one hit in m snapshots."""
    return 1 - (1 - p) ** m


@dataclass
class Schedule:
    t_w: float                 # capture window
    busy: float                # capture start -> next command parsed if polling were instant
    grid: float = 1e-3         # polling grid; 0 = command parsed immediately
    cmd_to_start: float = 65e-6
    jitter: float = 0.0        # on-chip uniform delay before capture
    host_sigma: float = 0.0    # Gaussian host latency jitter (re-quantised by the grid)


def snapshot_starts(s: Schedule, count: int, rng: np.random.Generator, t0: float = 0.0) -> np.ndarray:
    starts = np.empty(count)
    t = t0
    for k in range(count):
        arrive = t + max(s.busy + (rng.normal(0, s.host_sigma) if s.host_sigma else 0.0), 0.0)
        parsed = np.ceil(arrive / s.grid) * s.grid if s.grid > 0 else arrive
        t = parsed + s.cmd_to_start + (rng.uniform(0, s.jitter) if s.jitter > 0 else 0.0)
        starts[k] = t
    return starts


def overlaps(starts, t_w, tau, t_b, phase=0.0, ppm=0.0):
    """Overlap of each window [s, s+t_w] with bursts [m T_b' + phase, + tau], T_b' = T_b(1+ppm 1e-6)."""
    tb = t_b * (1 + ppm * 1e-6)
    rel = np.mod(np.asarray(starts) - phase, tb)          # window start inside the burst period
    ov_cur = np.clip(np.minimum(rel + t_w, tau) - rel, 0, None)
    ov_next = np.clip(rel + t_w - tb, 0, tau)              # window running into the next burst
    return ov_cur + ov_next


def simulate(s: Schedule, tau, t_b, count=10000, tau_ov=0.0, phase=None, ppm=0.0, seed=0):
    """Fraction of snapshots with overlap >= tau_ov (and > 0)."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, t_b) if phase is None else phase
    st = snapshot_starts(s, count, rng)
    ov = overlaps(st, s.t_w, tau, t_b, ph, ppm)
    hit = (ov >= max(tau_ov, 1e-12))
    return float(np.mean(hit)), st
