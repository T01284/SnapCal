import sys
import unittest
from pathlib import Path

HOST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(HOST / "scripts"))
from RFExplorer.RFE6GEN_CalibrationData import RFE6GEN_CalibrationData  # noqa: E402
from snapcal.rfe import LEVELS, cw_command, cw_command_expansion  # noqa: E402
import level_calib  # noqa: E402


class RfeTest(unittest.TestCase):
    def test_cw_command(self):
        self.assertEqual(cw_command(2442.0, 0), "C3-F:2442000,0,0")
        self.assertEqual(cw_command(5800.5, 7), "C3-F:5800500,1,3")
        self.assertEqual(cw_command_expansion(5800.0, -12.5), "C5-F:5800000,-12.5")

    def test_level_order_matches_vendor_table(self):
        cal = RFE6GEN_CalibrationData()
        n = len(cal.m_arrSignalGeneratorCalRanges_KHZ)
        cal.InitializeCal(n, "$q" + chr(n) + "".join(chr(20) for _ in range(n)))
        for f in (2442.0, 5800.0):
            arr = cal.GetEstimatedAmplitudeArray(f)
            est = [cal.GetEstimatedAmplitude(f, hp, pl) for hp, pl in LEVELS]
            self.assertEqual(est, arr)

    def test_summary_loss(self):
        def row(f, k, dbm, stage="direct", span=10.0, sd=0.1):
            return {"f_mhz": f, "setting": k, "dbm": dbm, "std_db": sd, "gen_est_dbm": -1.0,
                    "floor_dbm": -100.0, "above_floor": True, "input_stage": stage, "span_mhz": span}
        d = {"thru_runs": [{"rows": [row(2442.0, "L7", -2.0), row(2442.0, "L7", -1.0, "att30")]},
                           {"rows": [row(2442.0, "L7", -2.2), row(2442.0, "L7", -1.2, "att30")]}],
             "components": {"att30": [{"nominal_db": 30, "rows": [row(2442.0, "L7", -32.4),
                                                                  row(2442.0, "L7", -40.0, span=20.0)]},
                                      {"nominal_db": 30, "rows": [row(2442.0, "L7", -32.2)]}]}}
        s = level_calib.summarize(d)
        t = next(x for x in s["thru"] if x["input_stage"] == "direct")
        self.assertAlmostEqual(t["dbm"], -2.1)
        self.assertAlmostEqual(t["gen_minus_analyzer_db"], 1.1)
        self.assertAlmostEqual(s["stage_check"][0]["direct_minus_att30_db"], -1.0)
        rows = s["components"]["att30"]["rows"]
        self.assertEqual(len(rows), 1)  # the 20 MHz-span reading has no thru at that span
        c = rows[0]
        self.assertAlmostEqual(c["loss_db"], 30.2)
        self.assertAlmostEqual(c["minus_nominal_db"], 0.2)
        self.assertEqual(c["repeats"], 2)
        self.assertGreater(c["u_db"], 0.14)


if __name__ == "__main__":
    unittest.main()
