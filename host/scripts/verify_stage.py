"""Stage acceptance: unit tests, result files, and (with --port) a firmware smoke test.

Example:
  python host/scripts/verify_stage.py
  python host/scripts/verify_stage.py --port COM31
"""
import argparse
import json
import subprocess
import sys
import unittest

import _path  # noqa: F401
from _path import ROOT

PHASE0 = {
    "results/env.json": [],
    "results/nbc_modes.json": [],
    "results/phase0_bringup/ota_survey.json": [],
    "results/phase0_bringup/host_timing.json": ["rows"],
    "results/phase0_bringup/host_timing_tickpoll.json": ["rows"],
    "results/phase0_bringup/host_timing_fastpoll.json": ["rows"],
    "results/phase0_bringup/tone_test.json": ["A_lime_gain_sweep", "B_esp_gain_sweep", "C_passband", "D_freq_error"],
    "results/phase0_bringup/snapshot_stats.json": ["n16380", "n2048"],
    "results/phase0_bringup/burst_intercept_Tb10ms.json": ["bursts"],
    "results/phase0_bringup/burst_intercept_Tb10p37ms.json": ["bursts"],
    "results/phase0_bringup/burst_intercept_Tb10ms_fw.json": ["bursts"],
    "results/phase0_bringup/start_phase.json": ["runs"],
    "results/phase0_bringup/spec_timing.json": ["profiles"],
    "results/phase0_bringup/spec_intercept.json": ["runs"],
    "results/phase0_bringup/fa_probe.json": ["cases"],
}


def check_files():
    ok = True
    for rel, keys in PHASE0.items():
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            ok = False
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        miss = [k for k in keys if k not in d]
        print(f"  {'OK     ' if not miss else 'BADKEYS'}  {rel}" + (f"  missing {miss}" if miss else ""))
        ok &= not miss
    return ok


def check_tests():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "host" / "tests"))
    r = unittest.TextTestRunner(verbosity=0, stream=sys.stdout).run(suite)
    return r.wasSuccessful()


def check_firmware(port):
    from snapcal.espsdr import EspSdr
    ok = True

    def step(name, cond, detail=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'}  {name} {detail}")
        ok &= bool(cond)

    with EspSdr(port) as dev:
        info = dev.info()
        step("INFO", dev.identity.startswith("S3SDR"), dev.identity)
        step("CAPS has JITTER POLL TS MARK", all(c in info["caps"] for c in ("JITTER", "POLL", "TS", "MARK")))
        rev = str(info["version"].get("revision", ""))
        fw_head = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", "firmware/snapcal_fw", ":!*.md"],
                                 capture_output=True, text=True).stdout.strip()
        step("VERSION revision", bool(rev), rev)
        if rev and fw_head and subprocess.run(
                ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", fw_head, rev.removesuffix("-dirty")],
                capture_output=True).returncode != 0:
            print(f"  WARN  firmware {rev[:10]} predates last firmware commit {fw_head[:10]}: rebuild + flash")
        s = dev.capture(16380, 80_000_000, "iq10")
        step("CAP20 16380 CRC", s.n == 16380, f"capture_us={s.capture_us}")
        c0, st0 = dev.timestamps()
        s = dev.capture(256, 80_000_000, "iq8")
        c1, st1 = dev.timestamps()
        step("TS? advances", st1 > st0 and st1 >= c1, f"cmd->start {st1 - c1} us")
        dev.set_jitter(1000)
        step("JITTER?", dev.command("JITTER?") == "JITTER 1000")
        delays = []
        for _ in range(30):
            dev.capture(256, 80_000_000, "iq8")
            c, st = dev.timestamps()
            delays.append(st - c)
        step("JITTER spreads cmd->start", max(delays) - min(delays) > 500, f"range {min(delays)}..{max(delays)} us")
        dev.set_jitter(0)
        dev.set_marks(5, 6)
        s = dev.capture(2048, 80_000_000, "iq8")
        step("MARK 5 6 + capture", s.n == 2048)
        dev.set_marks(None)
        dev.gpio(5, "Z")
        dev.gpio(6, "Z")
        dev.set_fast_poll(True)
        s = dev.capture(256, 80_000_000, "iq8")
        dev.set_fast_poll(False)
        step("POLL 1 + capture", s.n == 256)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", help="also run the firmware smoke test on this port")
    a = ap.parse_args()
    print("unit tests:")
    t = check_tests()
    print("result files (Phase 0):")
    f = check_files()
    fw = True
    if a.port:
        print(f"firmware smoke test on {a.port}:")
        fw = check_firmware(a.port)
    ok = t and f and fw
    print("STAGE CHECK", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
