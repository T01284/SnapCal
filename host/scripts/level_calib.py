"""Q13 level calibration with the RF Explorer generator and analyzer -> results/level_calib.json.

Setup: RFE6GEN -> reference cable -> 6G Combo PLUS right connector (240-6100 MHz) is the
"thru". Thru and inserted-part readings are compared only at the same frequency, generator
setting, analyzer input stage and span (RBW follows span). A passive part (attenuator,
coupler port, splitter arm, combiner arm, IPEX pigtail) is measured by inserting it between
the reference cable and the analyzer, so its loss is a relative analyzer reading and the
analyzer's absolute error cancels. Each invocation appends to the JSON; `report` rebuilds
the summary.

  thru    all generator settings at every frequency; run it again after reconnecting the
          cables to get the connection repeatability
  dut     one part at one generator setting
  report  loss table with repeatability; --chain predicts the level at the end of a chain

Examples:
  python host/scripts/level_calib.py thru
  python host/scripts/level_calib.py --input-stage att30 thru
  python host/scripts/level_calib.py dut --name att30_a --nominal 30
  python host/scripts/level_calib.py dut --name coupler_cpl --nominal 20 --note "IN->CPL, OUT terminated"
  python host/scripts/level_calib.py report --chain att30_a coupler_cpl --setting L3
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.rfe import INPUT_STAGES, LEVELS, RfeAnalyzer, RfeGenerator, discover

OUT = ROOT / "results" / "level_calib.json"
FREQS = [2412.0, 2442.0, 2472.0, 5745.0, 5800.0, 5850.0]
MIN_SNR_DB = 10.0


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"thru_runs": [], "components": {}}


def save(d):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("wrote", OUT)


def setting_key(gen, level, dbm):
    return f"{dbm:+.1f}dBm" if gen.expansion else f"L{level}"


def condition(r):
    """Readings are comparable only at the same frequency, generator setting, analyzer input stage and span."""
    return r["f_mhz"], r["setting"], r["input_stage"], r["span_mhz"]


def open_pair(a):
    gp, sp = a.gen, a.sa
    if not (gp and sp):
        found = discover()
        gp = gp or found.get("generator", {}).get("port")
        sp = sp or found.get("analyzer", {}).get("port")
    if not (gp and sp):
        raise SystemExit("need both units: plug them in or pass --gen/--sa")
    return RfeGenerator(gp), RfeAnalyzer(sp)


def measure(gen, sa, f, settings, a):
    sa.configure(f - a.span / 2, f + a.span / 2)
    sa.set_input_stage(a.input_stage)
    gen.off()
    floor = sa.tone(f, n=a.n)
    rows = []
    for level, dbm in settings:
        gen.cw(f, level=level, dbm=dbm)
        r = sa.tone(f, n=a.n) | {"f_mhz": f, "setting": setting_key(gen, level, dbm),
                                 "gen_est_dbm": gen.estimated_dbm(), "floor_dbm": floor["dbm"]}
        r["above_floor"] = r["dbm"] - floor["dbm"] >= MIN_SNR_DB
        rows.append(r)
        flag = "" if r["above_floor"] else "  (< 10 dB above RF-off floor)"
        print(f"  {f:7.1f} MHz {r['setting']:>9} {r['input_stage']}: analyzer {r['dbm']:7.2f} dBm "
              f"(sd {r['std_db']:.2f}, RBW {r['rbw_khz']:.0f} kHz), "
              f"generator estimate {r['gen_est_dbm']:6.1f} dBm{flag}")
    gen.off()
    return rows


def cmd_thru(a):
    d = load()
    gen, sa = open_pair(a)
    with gen, sa:
        settings = [(None, x) for x in a.dbm] if gen.expansion else [(i, None) for i in range(len(LEVELS))]
        rows = [r for f in a.freqs for r in measure(gen, sa, f, settings, a)]
        d["thru_runs"].append({"utc": utc(), "note": a.note, "generator": gen.info(), "analyzer": sa.info(),
                               "rows": rows, "host": envinfo.host()})
    save(d)


def cmd_dut(a):
    d = load()
    gen, sa = open_pair(a)
    with gen, sa:
        settings = [(None, a.dbm[0])] if gen.expansion else [(a.level, None)]
        rows = [r for f in a.freqs for r in measure(gen, sa, f, settings, a)]
        d["components"].setdefault(a.name, []).append(
            {"utc": utc(), "nominal_db": a.nominal, "note": a.note, "analyzer": sa.info()["serial"], "rows": rows})
    save(d)


def summarize(d):
    thru = {}
    for run in d["thru_runs"]:
        for r in run["rows"]:
            thru.setdefault(condition(r), []).append(r)
    s = {"thru": [], "components": {}, "stage_check": []}
    for (f, k, stage, span), rs in sorted(thru.items()):
        v = np.array([r["dbm"] for r in rs])
        s["thru"].append({"f_mhz": f, "setting": k, "input_stage": stage, "span_mhz": span,
                          "dbm": float(v.mean()), "runs": len(v),
                          "reconnect_sd_db": float(v.std(ddof=1)) if len(v) > 1 else None,
                          "gen_minus_analyzer_db": float(np.mean([r["gen_est_dbm"] for r in rs]) - v.mean())})
    ref = {condition(t): t for t in s["thru"]}
    for t in s["thru"]:
        if t["input_stage"] == "direct":
            o = ref.get((t["f_mhz"], t["setting"], "att30", t["span_mhz"]))
            if o is not None:
                s["stage_check"].append({"f_mhz": t["f_mhz"], "setting": t["setting"], "span_mhz": t["span_mhz"],
                                         "direct_minus_att30_db": t["dbm"] - o["dbm"]})
    for name, meas in d["components"].items():
        by_c = {}
        for m in meas:
            for r in m["rows"]:
                t = ref.get(condition(r))
                if t is None or not r["above_floor"]:
                    continue
                by_c.setdefault((r["f_mhz"], r["input_stage"]), []).append(
                    (t["dbm"] - r["dbm"], r["std_db"], t["reconnect_sd_db"]))
        rows = []
        for (f, stage), xs in sorted(by_c.items()):
            loss = np.array([x[0] for x in xs])
            u2 = [x[1] ** 2 + (x[2] or 0.0) ** 2 for x in xs]
            if len(loss) > 1:
                u2.append(loss.var(ddof=1))
            rows.append({"f_mhz": f, "input_stage": stage, "loss_db": float(loss.mean()),
                         "u_db": float(np.sqrt(np.mean(u2))), "repeats": len(loss),
                         "minus_nominal_db": float(loss.mean() - meas[-1]["nominal_db"])})
        s["components"][name] = {"nominal_db": meas[-1]["nominal_db"], "rows": rows}
    s["uncertainty_note"] = ("u_db is repeatability only (sweep-to-sweep, reconnection, repeated inserts). "
                             "Analyzer linearity (vendor: about +/-3 dB at constant RBW) is checked with "
                             "fixed attenuators of known tolerance (minus_nominal_db) and by direct vs att30; "
                             "absolute accuracy of both handheld instruments is reported separately")
    return s


# Difference of two stored losses. The pigtail cannot be mated without the coupler
# barrel and the IPEX adapter, so this is an upper bound on pigtail+adapter, not
# the pigtail alone and not a term in the antenna-port chain.
DERIVED = [(
    "ipex_pigtail_plus_adapter",
    "ipex_pigtail_chain",
    "coupler_thru_pigtail_setup",
    "ipex_pigtail_chain minus coupler_thru_pigtail_setup. Upper bound on "
    "pigtail+adapter from one mating. Do not subtract ipex_adapter_direct. "
    "A negative value is not a loss and must not be chained.",
)]


def difference(s, plus, minus):
    a = {(r["f_mhz"], r["input_stage"]): r for r in s["components"][plus]["rows"]}
    b = {(r["f_mhz"], r["input_stage"]): r for r in s["components"][minus]["rows"]}
    rows = []
    for key in sorted(set(a) & set(b)):
        loss = a[key]["loss_db"] - b[key]["loss_db"]
        u = float(np.sqrt(a[key]["u_db"] ** 2 + b[key]["u_db"] ** 2))
        rows.append({"f_mhz": key[0], "input_stage": key[1], "loss_db": float(loss),
                     "u_db": u, "usable_loss": bool(loss > 0)})
    return rows


def cmd_report(a):
    d = load()
    s = summarize(d)
    d["summary"] = s | {"utc": utc()}
    print("thru (generator -> reference cable -> analyzer)")
    for t in s["thru"]:
        sd = "-" if t["reconnect_sd_db"] is None else f"{t['reconnect_sd_db']:.2f}"
        print(f"  {t['f_mhz']:7.1f} MHz {t['setting']:>9} {t['input_stage']:>6} {t['span_mhz']:g} MHz: "
              f"{t['dbm']:7.2f} dBm  reconnect sd {sd}  "
              f"generator - analyzer {t['gen_minus_analyzer_db']:+.2f} dB  ({t['runs']} runs)")
    if s["stage_check"]:
        print("direct - att30 (should not depend on level; a drop at high levels means direct compresses)")
        for c in s["stage_check"]:
            print(f"  {c['f_mhz']:7.1f} MHz {c['setting']:>9}: {c['direct_minus_att30_db']:+.2f} dB")
    for name, c in s["components"].items():
        print(f"{name} (nominal {c['nominal_db']} dB)")
        for r in c["rows"]:
            print(f"  {r['f_mhz']:7.1f} MHz {r['input_stage']:>6}: {r['loss_db']:6.2f} dB +/- {r['u_db']:.2f}  "
                  f"(measured - nominal {r['minus_nominal_db']:+.2f}, {r['repeats']}x)")
    if a.chain:
        losses = {n: {(r["f_mhz"], r["input_stage"]): r for r in s["components"][n]["rows"]} for n in a.chain}
        print(f"chain {' + '.join(a.chain)} at {a.setting}, {a.input_stage}")
        pred = []
        for t in s["thru"]:
            key = (t["f_mhz"], t["input_stage"])
            if (t["setting"], t["input_stage"]) != (a.setting, a.input_stage) or \
                    not all(key in losses[n] for n in a.chain):
                continue
            rs = [losses[n][key] for n in a.chain]
            lvl = t["dbm"] - sum(r["loss_db"] for r in rs)
            u = float(np.sqrt((t["reconnect_sd_db"] or 0.0) ** 2 + sum(r["u_db"] ** 2 for r in rs)))
            pred.append({"f_mhz": t["f_mhz"], "span_mhz": t["span_mhz"], "dbm": lvl, "u_db": u})
            print(f"  {t['f_mhz']:7.1f} MHz: {lvl:7.2f} dBm +/- {u:.2f}")
        d.setdefault("chains", {})[" + ".join(a.chain) + f" @ {a.setting} {a.input_stage}"] = pred
    derived = {}
    for name, plus, minus, note in DERIVED:
        if plus in s["components"] and minus in s["components"]:
            rows = difference(s, plus, minus)
            derived[name] = {"plus": plus, "minus": minus, "note": note, "rows": rows}
            print(f"{name} = {plus} - {minus}")
            for r in rows:
                flag = "" if r["usable_loss"] else "  (not a loss)"
                print(f"  {r['f_mhz']:7.1f} MHz {r['input_stage']:>6}: {r['loss_db']:6.2f} dB +/- {r['u_db']:.2f}{flag}")
    if derived:
        s_out = d["summary"]
        s_out["derived"] = derived
    save(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", help="generator COM port (default: auto-detect among CP210x ports)")
    ap.add_argument("--sa", help="analyzer COM port")
    ap.add_argument("--freqs", type=float, nargs="+", default=FREQS, help="MHz")
    ap.add_argument("--n", type=int, default=10, help="sweeps per reading")
    ap.add_argument("--span", type=float, default=10.0, help="analyzer span, MHz (RBW follows it; keep it fixed)")
    ap.add_argument("--input-stage", choices=list(INPUT_STAGES), default="direct",
                    help="analyzer input stage; run thru with direct and att30 to check compression")
    ap.add_argument("--note", default="")
    ap.add_argument("--dbm", type=float, nargs="+", default=[-30.0, -20.0, -10.0],
                    help="generator dBm settings when its expansion board is active")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("thru")
    p = sub.add_parser("dut")
    p.add_argument("--name", required=True)
    p.add_argument("--nominal", type=float, required=True, help="nominal loss, dB")
    p.add_argument("--level", type=int, default=7, help="generator level 0..7 (7 = high power H3)")
    p = sub.add_parser("report")
    p.add_argument("--chain", nargs="+")
    p.add_argument("--setting", default="L3")
    a = ap.parse_args()
    {"thru": cmd_thru, "dut": cmd_dut, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
