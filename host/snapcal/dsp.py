"""Spectral helpers shared by the capture and analysis scripts."""
from __future__ import annotations

import numpy as np

# The S3 NBC path saturates around +/-500..507 codes, never at +/-511/512.
NEAR_FULL_SCALE = 480.0 / 512.0


def psd_db(x: np.ndarray, fs: float, nfft: int = 1024, remove_dc: bool = False):
    """Welch PSD (Hann, 50 % overlap), fftshifted. Returns (freq_hz, dBFS per bin)."""
    if remove_dc:
        x = x - np.mean(x)
    w = np.hanning(nfft).astype(np.float32)
    step = nfft // 2
    segs = 1 + (len(x) - nfft) // step
    acc = np.zeros(nfft)
    for k in range(segs):
        seg = x[k * step: k * step + nfft] * w
        acc += np.abs(np.fft.fft(seg)) ** 2
    acc /= segs * np.sum(w) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return f, 10 * np.log10(np.fft.fftshift(acc) + 1e-20)


def clip_fraction(x: np.ndarray) -> float:
    return float(np.mean((np.abs(x.real) >= NEAR_FULL_SCALE) | (np.abs(x.imag) >= NEAR_FULL_SCALE)))


def rms_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20))


def write_cf32(path, snapshots) -> None:
    """Concatenate complex64 snapshots as interleaved float32 (GNU Radio / SDR++ .cf32)."""
    with open(path, "wb") as fp:
        for x in snapshots:
            np.asarray(x, dtype=np.complex64).tofile(fp)


def read_cf32(path, n_per_snapshot: int) -> np.ndarray:
    x = np.fromfile(path, dtype=np.complex64)
    return x.reshape(-1, n_per_snapshot)
