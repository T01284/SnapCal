import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapcal.saleae_bin import (  # noqa: E402
    IDENTIFIER, DigitalChunk, fit_overhead, high_pulses, pair_snapshots, parse_digital,
    summarize_group,
)


def write_v0(path: Path, initial: int, begin: float, end: float, times: list[float]) -> None:
    with path.open("wb") as f:
        f.write(IDENTIFIER)
        f.write(struct.pack("<ii", 0, 0))
        f.write(struct.pack("<IddQ", initial, begin, end, len(times)))
        for t in times:
            f.write(struct.pack("<d", t))


class SaleaeBinTest(unittest.TestCase):
    def test_parse_v0_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "digital_0.bin"
            write_v0(path, 0, 0.0, 0.01, [1e-4, 2e-4, 5e-4])
            chunks = parse_digital(str(path))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].initial_state, 0)
        self.assertEqual(chunks[0].transition_times.tolist(), [1e-4, 2e-4, 5e-4])

    def test_high_pulses_drop_partials_and_glitches(self):
        chunk = DigitalChunk(0, 0.0, 1.0, np.array([0.001, 0.0012, 0.002, 0.0020001, 0.003]))
        pulses = high_pulses([chunk], min_width_s=1e-6)
        self.assertEqual(len(pulses), 1)
        self.assertAlmostEqual(pulses[0].width, 0.0002)
        # still high at the end: the 0.003 rise is not a pulse
        self.assertAlmostEqual(pulses[0].rise, 0.001)

    def test_pair_and_fit(self):
        snap = high_pulses([DigitalChunk(0, 0.0, 0.05, np.array([
            0.001, 0.0012, 0.010, 0.0104]))])
        xfer = high_pulses([DigitalChunk(0, 0.0, 0.05, np.array([
            0.0015, 0.001501, 0.0108, 0.010801]))])
        paired = pair_snapshots(snap, xfer)
        self.assertAlmostEqual(paired[0]["t_queue_s"], 0.0003)
        self.assertAlmostEqual(paired[0]["t_cycle_s"], 0.009)
        self.assertIsNone(paired[1]["t_cycle_s"])
        c, t0 = fit_overhead(np.array([1e-4, 2e-4]), np.array([1.1e-3, 1.2e-3]))
        # T_cycle - Tw = 0.001 and 0.001, so c = 0 and T0 = 1 ms
        self.assertAlmostEqual(c, 0.0, places=6)
        self.assertAlmostEqual(t0, 0.001)

    def test_summarize_flags_width_error(self):
        host = [
            {"n": 8000, "capture_us": 100.0, "host_cycle_s": 0.002},
            {"n": 16000, "capture_us": 200.0, "host_cycle_s": 0.003},
        ]
        paired = [
            {"tw_mark_s": 100e-6, "t_queue_s": 20e-6, "t_xfer_s": 1e-6, "t_cycle_s": 0.002},
            {"tw_mark_s": 250e-6, "t_queue_s": 20e-6, "t_xfer_s": 1e-6, "t_cycle_s": 0.003},
        ]
        s = summarize_group(host, paired, 80_000_000)
        self.assertTrue(s["by_n"]["8000"]["within_1pct"])
        self.assertFalse(s["by_n"]["16000"]["within_1pct"])
        # Tx = 1.9 ms and 2.8 ms at Tw = 100 us and 200 us -> c = 9, T0 = 1 ms
        self.assertAlmostEqual(s["fit"]["c"], 9.0)
        self.assertAlmostEqual(s["fit"]["T0_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
