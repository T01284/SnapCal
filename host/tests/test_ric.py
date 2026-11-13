import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapcal.ric import compensate, estimate_tone  # noqa: E402

FS = 80e6
N = 16380


def impaired(f_ref=10e6, df=3.1e3, g=0.05, phi=0.7, alpha=1.0, beta=0.03 - 0.02j, dc=0.01 + 0.004j,
             noise=1e-3, seed=0):
    # CFO and phase act before the baseband IQ imbalance, so the image mirrors f_ref + df
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    xr = np.exp(2j * np.pi * f_ref * n / FS) * np.exp(1j * (2 * np.pi * df * n / FS + phi))
    y = g * (alpha * xr + beta * np.conj(xr)) + dc
    return y + noise * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


class RicTest(unittest.TestCase):
    def test_estimates(self):
        y = impaired()
        e = estimate_tone(y, FS, 10e6)
        self.assertAlmostEqual(e.freq_hz, 10e6 + 3.1e3, delta=50)
        self.assertAlmostEqual(abs(e.amp), 0.05, delta=0.0005)
        self.assertAlmostEqual(e.irr_db, -20 * np.log10(abs(0.03 - 0.02j)), delta=0.5)
        self.assertAlmostEqual(e.dc.real, 0.01, delta=2e-4)

    def test_compensation_removes_image_dc_and_gain(self):
        y = impaired()
        e = estimate_tone(y, FS, 10e6)
        z = compensate(y, e, FS, 10e6)
        e2 = estimate_tone(z, FS, 10e6)
        self.assertGreater(e2.irr_db, e.irr_db + 20)
        self.assertAlmostEqual(abs(e2.amp), 1.0, delta=0.01)
        self.assertLess(abs(e2.dc), 1e-3)
        self.assertAlmostEqual(e2.freq_hz, 10e6, delta=50)


if __name__ == "__main__":
    unittest.main()
