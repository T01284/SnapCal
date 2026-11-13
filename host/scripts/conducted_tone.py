"""Conducted single-tone characterisation with the RFE6GEN -> results/conducted/<board>/.

Chain: RFE6GEN -> fixed attenuators (>= 30 dB, protects the ESP32 input) -> IPEX pigtail ->
ESP32 antenna connector. No coupler or enclosure needed for these strong-tone tests; noise
floor, sensitivity and P_fa still belong in the shielded box (Q6/Q8).

  levels    Q2/Q3: spectrum orientation (+f_bb vs -f_bb), generator level sweep at a fixed
            ESP gain index, ESP gain-index sweep at a fixed generator level, absolute tone
            frequency error against the generator TCXO (+/-1 ppm)
  passband  Q7: tone stepped across the band at each sample rate
  stats     Q9: >= 1000 snapshots, per-snapshot gain / phase / image / DC (snapshot_stats.py)
  drift     Q10: tone frequency every --interval s for --minutes, from a cold start

Absolute levels at the ESP32 = generator level - chain loss, taken later from
results/level_calib.json (--chain names the parts); --atten-db is the nominal value.

Example:
  python host/scripts/conducted_tone.py --port COM40 --board s3-wroom1u-01 --atten-db 40 levels
  python host/scripts/conducted_tone.py --port COM41 --board c5-01 --esp-freq 5800 --atten-db 40 levels
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.ric import estimate_tone
from snapcal.rfe import LEVELS, RfeGenerator, discover
import snapshot_stats

MIN_ATTEN_DB = 30.0


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_esp(port, baud):
    """The C5 UART bridge stays at the last BAUD until reboot."""
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"))
        dev.ser.baudrate = baud
        return dev


def measure(dev, f_bb, count, fs, n=16380):
    es = [estimate_tone(dev.capture(n, fs, "iq10").iq, fs, f_bb) for _ in range(count)]
    p = np.array([e.power_dbfs for e in es])
    return {"power_dbfs": float(np.median(p)),
            "power_p10_p90": [float(np.percentile(p, 10)), float(np.percentile(p, 90))],
            "irr_db": float(np.median([e.irr_db for e in es])),
            "snr_db": float(np.median([e.snr_db for e in es])),
            "freq_hz": float(np.median([e.freq_hz for e in es])),
            "freq_std_hz": float(np.std([e.freq_hz for e in es])),
            "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es]))}


def orientation(dev, f_bb, fs, count=10):
    """Tone power at +f_bb and -f_bb after the client's conjugation; + must win."""
    pos = measure(dev, f_bb, count, fs)["power_dbfs"]
    neg = measure(dev, -f_bb, count, fs)["power_dbfs"]
    return {"plus_dbfs": pos, "minus_dbfs": neg, "conjugated": dev.conjugate, "ok": pos > neg + 10}


def cmd_levels(a, dev, gen, out):
    fs = 80_000_000
    f_bb = a.bb_mhz * 1e6
    f_rf = a.esp_freq * 1e6 + f_bb
    gen.cw(f_rf / 1e6, level=a.level, dbm=a.dbm)
    dev.set_gain(a.esp_gain)
    out["orientation"] = o = orientation(dev, f_bb, fs)
    print(f"orientation: +f_bb {o['plus_dbfs']:.1f} dBFS, -f_bb {o['minus_dbfs']:.1f} dBFS, "
          f"client conjugates={o['conjugated']} -> {'OK' if o['ok'] else 'WRONG: fix espsdr.CONJUGATE'}")

    print("A. generator level sweep")
    rows = []
    settings = [(None, x) for x in a.dbm_sweep] if gen.expansion else [(i, None) for i in range(len(LEVELS))]
    for level, dbm in settings:
        gen.cw(f_rf / 1e6, level=level, dbm=dbm)
        r = measure(dev, f_bb, a.count, fs) | {"gen_level": level, "gen_dbm": dbm, "gen_est_dbm": gen.estimated_dbm()}
        rows.append(r)
        print(f"  gen {r['gen_est_dbm']:6.1f} dBm: {r['power_dbfs']:6.1f} dBFS  SNR {r['snr_db']:5.1f}  IRR {r['irr_db']:5.1f}")
    out["A_gen_level_sweep"] = rows

    print("B. ESP gain-index sweep")
    gen.cw(f_rf / 1e6, level=a.level, dbm=a.dbm)
    rows = []
    gain_max = int(dev.command("GAIN?").split()[4])
    for gi in range(0, min(83, gain_max + 1), a.gain_step):
        dev.set_gain(gi)
        r = measure(dev, f_bb, max(a.count // 2, 5), fs) | {"esp_gain_index": gi}
        rows.append(r)
        print(f"  idx {gi:2d}: {r['power_dbfs']:6.1f} dBFS  noise {r['noise_dbfs_hz']:6.1f} dBFS/Hz")
    out["B_esp_gain_sweep"] = rows
    dev.set_gain(a.esp_gain)

    print("C. frequency error against the generator TCXO")
    d = measure(dev, f_bb, a.count * 2, fs)
    d["expected_bb_hz"] = f_bb
    d["error_hz"] = d["freq_hz"] - f_bb
    d["error_ppm_at_rf"] = d["error_hz"] / f_rf * 1e6
    d["reference"] = "RFE6GEN internal TCXO, +/-1 ppm"
    out["C_freq_error"] = d
    print(f"  {d['error_hz']:+.0f} Hz = {d['error_ppm_at_rf']:+.2f} ppm (ESP LO low => positive)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    A = out["A_gen_level_sweep"]
    ax[0].plot([r["gen_est_dbm"] - a.atten_db for r in A], [r["power_dbfs"] for r in A], "o-")
    ax[0].set_xlabel("generator estimate - nominal attenuation (dBm)"); ax[0].set_ylabel("tone (dBFS)")
    ax[0].set_title(f"A. level linearity, gain idx {a.esp_gain}")
    B = out["B_esp_gain_sweep"]
    ax[1].plot([r["esp_gain_index"] for r in B], [r["power_dbfs"] for r in B], "o-", ms=3)
    ax[1].set_xlabel("ESP gain index"); ax[1].set_ylabel("tone (dBFS)"); ax[1].set_title("B. gain table")
    fig.tight_layout()
    return fig


def cmd_passband(a, dev, gen, out):
    dev.set_gain(a.esp_gain)
    out["rates"] = {}
    for fs in a.rates:
        dev.set_freq(a.esp_freq)
        rows = []
        for frac in np.linspace(-0.47, 0.47, a.points):
            f_bb = float(round(frac * fs / 1e3) * 1e3)
            if abs(f_bb) < 0.3e6:
                continue
            gen.cw((a.esp_freq * 1e6 + f_bb) / 1e6, level=a.level, dbm=a.dbm)
            r = measure(dev, f_bb, max(a.count // 2, 5), int(fs)) | {"bb_hz": f_bb}
            rows.append(r)
            print(f"  {fs / 1e6:4.0f} MS/s {f_bb / 1e6:+7.2f} MHz: {r['power_dbfs']:6.1f} dBFS  IRR {r['irr_db']:5.1f}")
        out["rates"][str(int(fs))] = rows
    fig, ax = plt.subplots(figsize=(7, 4))
    for fs, rows in out["rates"].items():
        p = np.array([r["power_dbfs"] for r in rows])
        ax.plot([r["bb_hz"] / 1e6 for r in rows], p - p.max(), "o-", ms=3, label=f"{int(fs) / 1e6:.0f} MS/s")
    ax.axhline(-3, color="grey", ls=":")
    ax.set_xlabel("baseband offset (MHz)"); ax.set_ylabel("relative tone (dB)"); ax.legend()
    ax.set_title(f"Q7 passband, {a.esp_freq} MHz, conducted (generator flatness not removed)")
    fig.tight_layout()
    return fig


def cmd_stats(a, dev, gen, out):
    f_bb = a.bb_mhz * 1e6
    gen.cw((a.esp_freq * 1e6 + f_bb) / 1e6, level=a.level, dbm=a.dbm)
    dev.set_gain(a.esp_gain)
    rows, r_glob = snapshot_stats.run(dev, f_bb, a.count, 16380, "n16380")
    out["n16380"] = snapshot_stats.summarize(rows, r_glob, f_bb)
    print(json.dumps(out["n16380"], indent=1))
    tag = f"_{a.tag}" if a.tag else ""
    with open(a.rd / f"stats_{a.esp_freq}{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return None


def cmd_drift(a, dev, gen, out):
    fs = 80_000_000
    f_bb = a.bb_mhz * 1e6
    f_rf = a.esp_freq * 1e6 + f_bb
    gen.cw(f_rf / 1e6, level=a.level, dbm=a.dbm)
    dev.set_gain(a.esp_gain)
    rows = []
    t0 = time.monotonic()
    path = a.rd / f"drift_{a.esp_freq}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = None
        while time.monotonic() - t0 < a.minutes * 60:
            t = time.monotonic() - t0
            r = measure(dev, f_bb, a.count, fs) | {"t_s": t}
            r["ppm"] = (r["freq_hz"] - f_bb) / f_rf * 1e6
            rows.append(r)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=[k for k in r if k != "power_p10_p90"], extrasaction="ignore")
                w.writeheader()
            w.writerow(r)
            fh.flush()
            print(f"  {t / 60:6.1f} min: {r['ppm']:+.3f} ppm  {r['power_dbfs']:6.1f} dBFS")
            time.sleep(max(0.0, a.interval - (time.monotonic() - t0 - t)))
    ppm = np.array([r["ppm"] for r in rows])
    out["drift"] = {"samples": len(rows), "ppm_first": float(ppm[0]), "ppm_last": float(ppm[-1]),
                    "ppm_span": float(ppm.max() - ppm.min()), "csv": path.name}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["t_s"] / 60 for r in rows], ppm, ".-", ms=3)
    ax.set_xlabel("time since start (min)"); ax.set_ylabel("ESP LO error vs RFE6GEN (ppm)")
    ax.set_title(f"Q10 drift, {a.board}, {a.esp_freq} MHz")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="ESP32 COM port")
    ap.add_argument("--baud", type=int, default=2_000_000,
                    help="UART baud after connect; 1000000 for a CH340 bridge")
    ap.add_argument("--board", required=True, help="label, e.g. s3-wroom1u-01 / c5-01")
    ap.add_argument("--gen", help="RFE6GEN COM port (default: auto-detect among CP210x ports)")
    ap.add_argument("--atten-db", type=float, required=True, help="nominal attenuation generator -> ESP32")
    ap.add_argument("--chain", nargs="*", default=[], help="part names in results/level_calib.json")
    ap.add_argument("--esp-freq", type=int, default=2442, help="MHz")
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--bb-mhz", type=float, default=10.0, help="tone offset from the ESP centre")
    ap.add_argument("--level", type=int, default=3, help="generator level 0..7 (3 = L3, about -30 dBm)")
    ap.add_argument("--dbm", type=float, default=-30.0, help="generator dBm if its expansion board is active")
    ap.add_argument("--dbm-sweep", type=float, nargs="+", default=[-40, -35, -30, -25, -20, -15, -10, -5, 0])
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--gate-pin", type=int, default=None,
                    help="drive this GPIO high for the run (SPDT control, 1 = pass)")
    ap.add_argument("--tag", default="", help="suffix on the result file, so a second level does not overwrite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("levels")
    p.add_argument("--gain-step", type=int, default=2)
    p = sub.add_parser("passband")
    p.add_argument("--rates", type=float, nargs="+", default=[80e6, 40e6, 16e6])
    p.add_argument("--points", type=int, default=25)
    sub.add_parser("stats")
    p = sub.add_parser("drift")
    p.add_argument("--minutes", type=float, default=70)
    p.add_argument("--interval", type=float, default=30)
    a = ap.parse_args()
    if a.cmd == "stats" and a.count < 1000:
        a.count = 1000
    if a.atten_db < MIN_ATTEN_DB:
        raise SystemExit(f"--atten-db {a.atten_db} < {MIN_ATTEN_DB}: put more attenuation in front of the ESP32")

    a.rd = ROOT / "results" / "conducted" / a.board
    a.rd.mkdir(parents=True, exist_ok=True)
    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found: plug it in or pass --gen")
    out = {"utc": utc(), "cmd": a.cmd, "board": a.board,
           "args": {k: v for k, v in vars(a).items() if k != "rd"},
           "setup": f"conducted: RFE6GEN -> {a.atten_db:g} dB nominal -> IPEX pigtail -> ESP32; bench, no enclosure"}
    with RfeGenerator(gp) as gen, open_esp(a.port, a.baud) as dev:
        out["generator"] = gen.info()
        out["firmware"] = dev.info()["version"]
        out["identity"] = dev.identity
        if a.gate_pin is not None:
            dev.gpio(a.gate_pin, "1")
        dev.set_freq(a.esp_freq)
        dev.set_bandwidth(0)
        fig = {"levels": cmd_levels, "passband": cmd_passband, "stats": cmd_stats, "drift": cmd_drift}[a.cmd](
            a, dev, gen, out)
        gen.off()
    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()
    stem = f"{a.cmd}_{a.esp_freq}" + (f"_{a.tag}" if a.tag else "")
    (a.rd / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if fig is not None:
        fig.savefig(a.rd / f"{stem}.png", dpi=130)
    print("wrote", a.rd / f"{stem}.json")


if __name__ == "__main__":
    main()
