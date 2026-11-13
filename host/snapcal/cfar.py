"""Cell-averaging CFAR on square-law (power) data.

For exponentially distributed noise power and N reference cells, the CA-CFAR scale
alpha = N (Pfa^(-1/N) - 1) gives the design false-alarm probability exactly.
Hann-windowed spectrogram bins are correlated with their neighbours, which shrinks
the effective N: on such data the measured Pfa is ~1.7-3.6x the design value
(N = 16/8 per side, Pfa 1e-3/1e-4). Calibrate Pfa empirically on terminated-input
snapshots before using it in the paper.
"""
from __future__ import annotations

import numpy as np


def ca_alpha(n_ref: int, pfa: float) -> float:
    return n_ref * (pfa ** (-1.0 / n_ref) - 1.0)


def ca_cfar(power: np.ndarray, n_ref: int, n_guard: int, pfa: float, axis: int = -1,
            wrap: bool = False):
    """Return (detections, threshold) along `axis`. n_ref / n_guard are per side.
    Edges use whatever reference cells exist (scale adjusted to their count) unless
    wrap=True (circular, e.g. an FFT frequency axis)."""
    p = np.moveaxis(np.asarray(power, dtype=float), axis, -1)
    L = p.shape[-1]
    k = np.zeros(2 * (n_ref + n_guard) + 1)
    k[:n_ref] = 1
    k[-n_ref:] = 1
    if wrap:
        idx = (np.arange(L)[:, None] + np.arange(-(n_ref + n_guard), n_ref + n_guard + 1)[None, :]) % L
        sums = (p[..., idx] * k).sum(-1)
        counts = np.full(L, 2 * n_ref)
    else:
        pad = n_ref + n_guard
        pp = np.concatenate([np.zeros(p.shape[:-1] + (pad,)), p, np.zeros(p.shape[:-1] + (pad,))], -1)
        ones = np.concatenate([np.zeros(pad), np.ones(L), np.zeros(pad)])
        idx = np.arange(L)[:, None] + np.arange(2 * pad + 1)[None, :]
        sums = (pp[..., idx] * k).sum(-1)
        counts = (ones[idx] * k).sum(-1)
    counts = np.maximum(counts, 1)
    alpha = np.array([ca_alpha(int(n), pfa) for n in counts])
    thr = alpha * sums / counts
    det = p > thr
    return np.moveaxis(det, -1, axis), np.moveaxis(thr, -1, axis)


def spectrogram(x: np.ndarray, nfft: int, hop: int | None = None) -> np.ndarray:
    """|STFT|^2 with a Hann window, fftshifted frequency axis last: shape (frames, nfft)."""
    hop = hop or nfft
    x = np.asarray(x)
    nfr = 1 + (len(x) - nfft) // hop
    if nfr < 1:
        raise ValueError("snapshot shorter than nfft")
    w = np.hanning(nfft)
    frames = np.stack([x[i * hop: i * hop + nfft] * w for i in range(nfr)])
    return np.fft.fftshift(np.abs(np.fft.fft(frames, axis=-1)) ** 2, axes=-1)
