"""Round-trip the firmware packers (main/targets/esp32s3/receiver.c) through the host unpackers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapcal.espsdr import payload_size, unpack_iq8, unpack_iq10, unpack_raw  # noqa: E402


def words(i, q):
    return ((np.asarray(i) & 0x3FF) | ((np.asarray(q) & 0x3FF) << 10)).astype(np.uint32)


def pack_iq10(w):
    out = bytearray()
    n = len(w)
    for j in range(0, n, 2):
        a = int(w[j]) & 0xFFFFF
        b = int(w[j + 1]) & 0xFFFFF if j + 1 < n else 0
        out += bytes([a & 255, (a >> 8) & 255, ((a >> 16) | (b << 4)) & 255])
        if j + 1 < n:
            out += bytes([(b >> 4) & 255, (b >> 12) & 255])
    return bytes(out)


def pack_iq8(w):
    out = bytearray()
    for x in w:
        x = int(x)
        out += bytes([(x >> 2) & 255, (x >> 12) & 255])
    return bytes(out)


class UnpackTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.i = rng.integers(-512, 512, 1001)
        self.q = rng.integers(-512, 512, 1001)
        self.w = words(self.i, self.q)
        self.ref = (self.i + 1j * self.q) / 512.0

    def test_raw(self):
        x = unpack_raw(self.w.tobytes(), len(self.w))
        np.testing.assert_allclose(x, self.ref, atol=1e-7)

    def test_iq10_odd_length(self):
        p = pack_iq10(self.w)
        self.assertEqual(len(p), payload_size(len(self.w), 20))
        np.testing.assert_allclose(unpack_iq10(p, len(self.w)), self.ref, atol=1e-7)

    def test_iq8_truncates_two_lsbs(self):
        p = pack_iq8(self.w)
        self.assertEqual(len(p), payload_size(len(self.w), 16))
        ref = ((self.i >> 2) + 1j * (self.q >> 2)) / 128.0
        np.testing.assert_allclose(unpack_iq8(p, len(self.w)), ref, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
