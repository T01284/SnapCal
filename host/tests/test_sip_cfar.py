import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapcal import cfar, sip  # noqa: E402


class TestSip(unittest.TestCase):
    def test_p_snap_matches_random_phase(self):
        rng = np.random.default_rng(1)
        t_w, tau, t_b, tau_ov = 25.6e-6, 500e-6, 10e-3, 1e-6
        starts = rng.uniform(0, 1.0, 200000)
        hit = sip.overlaps(starts, t_w, tau, t_b) >= tau_ov
        self.assertAlmostEqual(hit.mean(), float(sip.p_snap(t_w, tau, t_b, tau_ov)), delta=0.002)

    def test_tau_star_switch(self):
        c, t0, tau_min = 280.0, 1e-3, 2e-6
        ts = sip.tau_star(tau_min, c, t0)
        lo, hi = 3.2e-6, 204.75e-6
        self.assertEqual(sip.best_window(ts * 0.5, lo, hi, tau_min, c, t0), hi)
        self.assertEqual(sip.best_window(ts * 2.0, lo, hi, tau_min, c, t0), lo)

    def test_sync_trap_and_on_chip_jitter(self):
        # stock firmware, 256-point window: one capture every two 1 ms ticks
        s = sip.Schedule(t_w=3.2e-6, busy=1.5e-3)
        theory = float(sip.p_snap(3.2e-6, 500e-6, 10e-3, 1e-6))
        locked = [sip.simulate(s, 500e-6, 10e-3, 5000, 1e-6, phase=ph)[0] for ph in np.linspace(0, 10e-3, 21)[:-1]]
        # each run is either all-miss or one-in-five phases hit, never the theory value
        self.assertTrue(all(p < 1e-3 or abs(p - 0.2) < 1e-3 for p in locked))
        self.assertGreater(sum(p < 1e-3 for p in locked), 10)
        s.jitter = 1e-3
        free = [sip.simulate(s, 500e-6, 10e-3, 20000, 1e-6, seed=k)[0] for k in range(5)]
        self.assertAlmostEqual(np.mean(free), theory, delta=0.006)

    def test_incommensurate_period_escapes_trap(self):
        s = sip.Schedule(t_w=3.2e-6, busy=1.5e-3)
        p, _ = sip.simulate(s, 500e-6, 10.37e-3, 20000, 1e-6, phase=0.0)
        self.assertAlmostEqual(p, float(sip.p_snap(3.2e-6, 500e-6, 10.37e-3, 1e-6)), delta=0.006)

    def test_p_detect(self):
        self.assertAlmostEqual(sip.p_detect(0.1, 10), 1 - 0.9 ** 10)


class TestCfar(unittest.TestCase):
    def test_false_alarm_rate(self):
        rng = np.random.default_rng(2)
        p = rng.exponential(1.0, (400, 512))
        det, _ = cfar.ca_cfar(p, n_ref=16, n_guard=2, pfa=1e-3, wrap=True)
        self.assertAlmostEqual(det.mean(), 1e-3, delta=3e-4)

    def test_detects_tone(self):
        rng = np.random.default_rng(3)
        n = 4096
        x = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
        x += 0.5 * np.exp(2j * np.pi * 52 / 256 * np.arange(n))
        S = cfar.spectrogram(x, 256)
        det, _ = cfar.ca_cfar(S, 8, 2, 1e-4, wrap=True)
        k = 128 + 52
        self.assertTrue(det[:, k].all())
        # 4096 cells; Hann-correlated bins run ~3.6x the design Pfa of 1e-4
        self.assertLessEqual(det.sum() - det[:, k - 1:k + 2].sum(), 8)


if __name__ == "__main__":
    unittest.main()
