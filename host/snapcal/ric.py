"""Reference-Injection Calibration (RIC): per-snapshot estimation from a known tone.

Model: y_k[n] = g_k (alpha r + beta r*) + d_k + w,  r = x e^{j(2 pi df n/fs + phi_k)}.
The IQ imbalance sits in the analog baseband after the CFO, so for a tone at +f_r the
snapshot shows the main peak at +f_r + df, its image at -(f_r + df) with complex ratio
image / conj(main) = beta / alpha*, and DC d_k. (The draft formula in the plan applies
the rotation outside the imbalance, which would put the image at -f_r + df.)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ToneEstimate:
    freq_hz: float          # refined main-tone baseband frequency
    amp: complex            # complex amplitude of the main tone (full scale 1.0)
    image: complex          # complex amplitude at the mirrored frequency
    dc: complex             # mean of the snapshot
    noise_dbfs_hz: float    # median noise density away from tone / image / DC
    power_dbfs: float       # 20 log10 |amp|
    irr_db: float           # image rejection ratio 20 log10 |amp/image|
    snr_db: float           # tone power over noise in the Hann ENBW (1.5 fs / n)

    def as_dict(self):
        d = asdict(self)
        for k in ("amp", "image", "dc"):
            d[k] = [float(np.real(d[k])), float(np.imag(d[k]))]
        return d


def _dft(x: np.ndarray, f_norm: float, w: np.ndarray) -> complex:
    n = np.arange(len(x))
    return complex(np.sum(x * w * np.exp(-2j * np.pi * f_norm * n)) / np.sum(w))


def estimate_tone(x: np.ndarray, fs: float, f_expect: float, search_hz: float = 200e3,
                  pad: int = 4) -> ToneEstimate:
    """Find the tone near f_expect, refine by parabolic interpolation on a zero-padded
    Hann spectrum, then read complex amplitudes by single-frequency DFTs."""
    x = np.asarray(x, dtype=np.complex128)
    n = len(x)
    w = np.hanning(n)
    dc = complex(np.mean(x))
    xd = x - dc
    nfft = 1 << int(np.ceil(np.log2(n * pad)))
    X = np.fft.fft(xd * w, nfft)
    f = np.fft.fftfreq(nfft, 1 / fs)
    band = np.abs(f - f_expect) <= search_hz
    idx = np.flatnonzero(band)
    k = idx[np.argmax(np.abs(X[idx]))]
    a, b, c = np.abs(X[(k - 1) % nfft]), np.abs(X[k]), np.abs(X[(k + 1) % nfft])
    delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    f0 = (k + delta) * fs / nfft
    if f0 > fs / 2:
        f0 -= fs
    amp = _dft(xd, f0 / fs, w)
    image = _dft(xd, -f0 / fs, w)
    # noise: median power of bins away from tone, image and DC, Hann ENBW = 1.5 bins
    P = np.abs(X) ** 2 / np.sum(w) ** 2
    excl = (np.abs(f - f0) < 100e3) | (np.abs(f + f0) < 100e3) | (np.abs(f) < 200e3)
    enbw_hz = 1.5 * fs / n
    noise_bin = np.median(P[~excl]) / np.log(2)      # median of exp-distributed -> mean
    p_amp = abs(amp) ** 2
    return ToneEstimate(
        freq_hz=float(f0), amp=amp, image=image, dc=dc,
        noise_dbfs_hz=float(10 * np.log10(noise_bin / enbw_hz + 1e-30)),
        power_dbfs=float(10 * np.log10(p_amp + 1e-30)),
        irr_db=float(10 * np.log10(p_amp / (abs(image) ** 2 + 1e-30))),
        snr_db=float(10 * np.log10(p_amp / noise_bin + 1e-30)),
    )


def compensate(y: np.ndarray, est: ToneEstimate, fs: float, f_ref: float,
               ref_amp: complex = 1.0) -> np.ndarray:
    """Reverse order of the model: remove DC, undo IQ imbalance, de-rotate CFO and
    phase, normalise gain so the reference tone has amplitude ref_amp."""
    yt = np.asarray(y, dtype=np.complex128) - est.dc
    # main = g alpha A e^{j phi}, image = g beta conj(A) e^{-j phi}; with A real:
    # beta/alpha* ~ image / conj(main)
    r = est.image / np.conj(est.amp) if est.amp != 0 else 0.0
    xi = (yt - r * np.conj(yt)) / (1 - abs(r) ** 2)
    n = np.arange(len(y))
    rot = np.exp(-1j * (2 * np.pi * (est.freq_hz - f_ref) * n / fs + np.angle(est.amp)))
    return (xi * rot * abs(ref_amp) / abs(est.amp)).astype(np.complex64)
