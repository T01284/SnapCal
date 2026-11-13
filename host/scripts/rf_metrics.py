"""Q8 conducted MDS + single-tone SFDR (and later two-tone IIP3) -> results/conducted/<board>/.

MDS (conducted definition, no enclosure P_fa):
  RF-off noise density at fixed gain, plus a generator level sweep. Absolute power at the
  ESP antenna port is gen_est_dbm - --atten-db (use Q13: att30_a ~30.7 dB at 2442 plus
  ipex_pigtail_direct_retest 0.5 dB). Report:
    - SNR-vs-Pin curve
    - Pin where median SNR crosses --snr-thresh (default 10 dB), interpolated
    - analytic MDS = N0_dbfs/Hz + 10 log10(ENBW) + SNR_thresh, referred to Pin via the
      mid-range tone calibration (slope ~1)

SFDR:
  At the strongest non-clipping tone, average PSD; exclude DC / main / image; strongest
  remaining bin is the spur. SFDR = carrier - spur in the same binning. RFE6GEN harmonics
  are typically ~-10 dBc and may dominate — recorded as spur_note.

Example:
  python host/scripts/rf_metrics.py --port COM15 --board s3-com15 --atten-db 31.2 mds_sfdr
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _path  # noqa: F401
from _path import ROOT
from snapcal import envinfo
from snapcal.dsp import clip_fraction
from snapcal.espsdr import EspSdr, EspSdrError
from snapcal.ric import estimate_tone
from snapcal.rfe import LEVELS, RfeGenerator, discover

MIN_ATTEN_DB = 30.0
FS = 80_000_000
N = 16380


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_esp(port, baud):
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"), flush=True)
        dev.ser.baudrate = baud
        return dev


def capture_retry(dev, n=N, fs=FS, fmt="iq8", retries=4):
    """UART bridges drop bytes at 2 Mbaud; iq8 + retry is the stable path on CH340/CH9102."""
    last = None
    for _ in range(retries):
        try:
            return dev.capture(n, fs, fmt)
        except EspSdrError as e:
            last = e
            try:
                dev.ser.reset_input_buffer()
            except Exception:
                pass
    raise last


def measure_tone(dev, f_bb, count, n=N, fs=FS, fmt="iq8"):
    es, clips = [], []
    for _ in range(count):
        x = capture_retry(dev, n, fs, fmt).iq
        clips.append(clip_fraction(x))
        es.append(estimate_tone(x, fs, f_bb))
    p = np.array([e.power_dbfs for e in es])
    return {
        "power_dbfs": float(np.median(p)),
        "power_p10_p90": [float(np.percentile(p, 10)), float(np.percentile(p, 90))],
        "snr_db": float(np.median([e.snr_db for e in es])),
        "irr_db": float(np.median([e.irr_db for e in es])),
        "noise_dbfs_hz": float(np.median([e.noise_dbfs_hz for e in es])),
        "freq_hz": float(np.median([e.freq_hz for e in es])),
        "clip_fraction_max": float(np.max(clips)),
        "enbw_hz": float(1.5 * fs / n),
        "fmt": fmt,
    }


def averaged_psd(dev, count, n=N, fs=FS, fmt="iq8"):
    w = np.hanning(n)
    acc = np.zeros(n)
    clips = []
    for _ in range(count):
        x = capture_retry(dev, n, fs, fmt).iq.astype(np.complex128)
        clips.append(clip_fraction(x))
        x = x - np.mean(x)
        acc += np.abs(np.fft.fft(x * w)) ** 2
    P = np.fft.fftshift(acc / count) / np.sum(w) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    return f, P, float(np.max(clips))


def find_sfdr(f, P, f_tone, guard_hz=150e3, dc_hz=200e3):
    P_db = 10 * np.log10(P + 1e-30)
    k_tone = int(np.argmin(np.abs(f - f_tone)))
    # parabolic refine in dB around peak
    lo, hi = max(1, k_tone - 2), min(len(P_db) - 2, k_tone + 2)
    k_tone = lo + int(np.argmax(P_db[lo: hi + 1]))
    carrier = float(P_db[k_tone])
    f_c = float(f[k_tone])
    mask = (np.abs(f) > dc_hz) & (np.abs(f - f_c) > guard_hz) & (np.abs(f + f_c) > guard_hz)
    if not np.any(mask):
        return {"carrier_dbfs_bin": carrier, "spur_dbfs_bin": None, "sfdr_db": None}
    k_spur = int(np.flatnonzero(mask)[np.argmax(P_db[mask])])
    spur = float(P_db[k_spur])
    return {
        "carrier_dbfs_bin": carrier,
        "carrier_f_bb_hz": f_c,
        "spur_dbfs_bin": spur,
        "spur_f_bb_hz": float(f[k_spur]),
        "spur_f_rf_mhz": None,  # filled by caller
        "sfdr_db": float(carrier - spur),
        "guard_hz": guard_hz,
        "dc_hz": dc_hz,
        "bin_hz": float(f[1] - f[0]),
    }


def interpolate_mds(rows, snr_thresh):
    """Pin (dBm) where SNR crosses snr_thresh, walking from strong to weak."""
    pts = sorted(
        [(r["pin_dbm"], r["snr_db"]) for r in rows if r.get("pin_dbm") is not None],
        key=lambda t: -t[0],
    )
    if not pts:
        return None
    if pts[-1][1] >= snr_thresh:
        return {"pin_dbm": pts[-1][0], "snr_db": pts[-1][1],
                "note": f"SNR still >= {snr_thresh} dB at weakest setting; MDS is an upper bound"}
    if pts[0][1] < snr_thresh:
        return {"pin_dbm": pts[0][0], "snr_db": pts[0][1],
                "note": f"SNR already < {snr_thresh} dB at strongest setting"}
    for (p0, s0), (p1, s1) in zip(pts, pts[1:]):
        if s0 >= snr_thresh > s1:
            if s0 == s1:
                pin = p1
            else:
                frac = (snr_thresh - s0) / (s1 - s0)
                pin = p0 + frac * (p1 - p0)
            return {"pin_dbm": float(pin), "snr_db": snr_thresh,
                    "bracket_dbm": [p0, p1], "bracket_snr_db": [s0, s1]}
    return None


def cmd_mds_sfdr(a, dev, gen, out):
    f_bb = a.bb_mhz * 1e6
    f_rf = a.esp_freq * 1e6 + f_bb
    dev.set_gain(a.esp_gain)

    print("A. RF-off floor", flush=True)
    gen.off()
    off = measure_tone(dev, f_bb, a.count)
    out["rf_off"] = off
    print(f"  noise {off['noise_dbfs_hz']:.1f} dBFS/Hz  "
          f"tone-bin {off['power_dbfs']:.1f} dBFS  SNR {off['snr_db']:.1f} dB", flush=True)

    print("B. generator level sweep (MDS)", flush=True)
    rows = []
    settings = [(None, x) for x in a.dbm_sweep] if gen.expansion else [(i, None) for i in range(len(LEVELS))]
    for level, dbm in settings:
        gen.cw(f_rf / 1e6, level=level, dbm=dbm)
        r = measure_tone(dev, f_bb, a.count)
        r["gen_level"] = level
        r["gen_dbm"] = dbm
        r["gen_est_dbm"] = gen.estimated_dbm()
        r["pin_dbm"] = r["gen_est_dbm"] - a.atten_db
        rows.append(r)
        print(f"  gen {r['gen_est_dbm']:6.1f} dBm  Pin {r['pin_dbm']:6.1f} dBm: "
              f"{r['power_dbfs']:6.1f} dBFS  SNR {r['snr_db']:5.1f}  "
              f"clip {r['clip_fraction_max']:.1e}", flush=True)
    out["mds_sweep"] = rows

    mds = interpolate_mds(rows, a.snr_thresh)
    # analytic: scale from non-clipping, still-linear points (prefer SNR 25..65)
    cal = [r for r in rows if r["clip_fraction_max"] < 1e-3 and 15 <= r["snr_db"] <= 65]
    if len(cal) < 2:
        cal = [r for r in rows if r["clip_fraction_max"] < 5e-2 and r["power_dbfs"] < -10]
    analytic = None
    if cal:
        deltas = [r["power_dbfs"] - r["pin_dbm"] for r in cal]
        scale = float(np.median(deltas))  # dBFS - dBm
        n0 = off["noise_dbfs_hz"]
        enbw = off["enbw_hz"]
        mds_dbfs = n0 + 10 * np.log10(enbw) + a.snr_thresh
        analytic = {
            "scale_dbfs_per_dbm": scale,
            "cal_points": len(cal),
            "noise_in_enbw_dbfs": float(n0 + 10 * np.log10(enbw)),
            "mds_dbfs": float(mds_dbfs),
            "mds_pin_dbm": float(mds_dbfs - scale),
            "snr_thresh_db": a.snr_thresh,
            "enbw_hz": enbw,
        }
    out["mds"] = {
        "definition": (
            f"conducted Pin (gen_est - {a.atten_db:g} dB) where median tone SNR "
            f"crosses {a.snr_thresh:g} dB at gain {a.esp_gain}; RF-off N0 also reported"
        ),
        "snr_crossing": mds,
        "analytic_from_floor": analytic,
        "atten_db": a.atten_db,
        "atten_note": a.atten_note,
    }
    if mds:
        print(f"  SNR-crossing MDS Pin ≈ {mds['pin_dbm']:.1f} dBm  ({mds.get('note', 'interpolated')})",
              flush=True)
    if analytic:
        print(f"  analytic MDS Pin ≈ {analytic['mds_pin_dbm']:.1f} dBm "
              f"(N0+ENBW+{a.snr_thresh:g} dB, scale {analytic['scale_dbfs_per_dbm']:.1f})", flush=True)

    print("C. single-tone SFDR", flush=True)
    # strongest non-clipping row, else mid level
    candidates = [r for r in rows if r["clip_fraction_max"] < 1e-3]
    if not candidates:
        candidates = rows
    sfdr_row = max(candidates, key=lambda r: r["power_dbfs"])
    gen.cw(f_rf / 1e6, level=sfdr_row["gen_level"], dbm=sfdr_row["gen_dbm"])
    f, P, clip = averaged_psd(dev, a.sfdr_count)
    tone_est = measure_tone(dev, f_bb, max(a.count // 2, 5))
    sfdr = find_sfdr(f, P, tone_est["freq_hz"], dc_hz=200e3)
    sfdr_ex_lo = find_sfdr(f, P, tone_est["freq_hz"], dc_hz=1e6)  # drop near-LO skirt
    for block in (sfdr, sfdr_ex_lo):
        if block.get("spur_f_bb_hz") is not None:
            block["spur_f_rf_mhz"] = a.esp_freq + block["spur_f_bb_hz"] / 1e6
    meta = {
        "gen_est_dbm": sfdr_row["gen_est_dbm"],
        "pin_dbm": sfdr_row["pin_dbm"],
        "tone_power_dbfs": tone_est["power_dbfs"],
        "irr_db": tone_est["irr_db"],
        "clip_fraction_max": clip,
        "spur_note": (
            "RFE6GEN has no harmonic filter (typ. -10 dBc); in-band spur may be source-limited. "
            "sfdr_exclude_lo_1mhz drops |f_bb|<1 MHz (LO/DC skirt)."
        ),
    }
    out["sfdr"] = meta | sfdr | {"exclude_lo_1mhz": sfdr_ex_lo}
    print(f"  carrier {sfdr['carrier_dbfs_bin']:.1f} dBFS @ {sfdr.get('carrier_f_bb_hz', 0)/1e6:+.3f} MHz; "
          f"spur {sfdr.get('spur_dbfs_bin')} @ {sfdr.get('spur_f_bb_hz', 0)/1e6:+.3f} MHz; "
          f"SFDR {sfdr.get('sfdr_db')} dB; "
          f"excl-LO SFDR {sfdr_ex_lo.get('sfdr_db')} dB @ {sfdr_ex_lo.get('spur_f_bb_hz', 0)/1e6:+.3f} MHz",
          flush=True)

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pins = [r["pin_dbm"] for r in rows]
    ax[0].plot(pins, [r["snr_db"] for r in rows], "o-", label="SNR")
    ax[0].axhline(a.snr_thresh, color="grey", ls=":", label=f"thresh {a.snr_thresh:g} dB")
    if mds:
        ax[0].axvline(mds["pin_dbm"], color="C3", ls="--", label=f"MDS {mds['pin_dbm']:.1f} dBm")
    if analytic:
        ax[0].axvline(analytic["mds_pin_dbm"], color="C2", ls="--",
                      label=f"analytic {analytic['mds_pin_dbm']:.1f} dBm")
    ax[0].set_xlabel("Pin at ESP (dBm, gen_est − atten)")
    ax[0].set_ylabel("tone SNR (dB)")
    ax[0].set_title(f"MDS sweep, {a.board}, {a.esp_freq} MHz, gain {a.esp_gain}")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(f / 1e6, 10 * np.log10(P + 1e-30), lw=0.6)
    if sfdr.get("spur_f_bb_hz") is not None:
        ax[1].plot(sfdr["carrier_f_bb_hz"] / 1e6, sfdr["carrier_dbfs_bin"], "g^", ms=7, label="carrier")
        ax[1].plot(sfdr["spur_f_bb_hz"] / 1e6, sfdr["spur_dbfs_bin"], "rv", ms=7, label="spur")
    ax[1].set_xlabel("baseband (MHz)")
    ax[1].set_ylabel("PSD (dBFS / bin)")
    ax[1].set_title(f"SFDR {sfdr.get('sfdr_db')} dB" if sfdr.get("sfdr_db") else "SFDR")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=1_000_000,
                    help="UART baud; 1e6 is reliable on CH340/CH9102 (2e6 drops bytes)")
    ap.add_argument("--board", required=True)
    ap.add_argument("--gen", help="RFE6GEN COM (default: auto)")
    ap.add_argument("--atten-db", type=float, required=True,
                    help="nominal gen→ESP loss (att30_a≈30.7 + pigtail≈0.5 at 2442)")
    ap.add_argument("--atten-note", default="att30_a + ipex pigtail (Q13)",
                    help="what --atten-db covers")
    ap.add_argument("--esp-freq", type=int, default=2442)
    ap.add_argument("--esp-gain", type=int, default=44)
    ap.add_argument("--bb-mhz", type=float, default=10.0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--sfdr-count", type=int, default=40)
    ap.add_argument("--snr-thresh", type=float, default=10.0)
    ap.add_argument("--dbm-sweep", type=float, nargs="+",
                    default=[-40, -35, -30, -25, -20, -15, -10, -5, 0])
    ap.add_argument("--tag", default="")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mds_sfdr")
    a = ap.parse_args()
    if a.atten_db < MIN_ATTEN_DB:
        raise SystemExit(f"--atten-db {a.atten_db} < {MIN_ATTEN_DB}: add more attenuation")

    a.rd = ROOT / "results" / "conducted" / a.board
    a.rd.mkdir(parents=True, exist_ok=True)
    gp = a.gen or discover().get("generator", {}).get("port")
    if not gp:
        raise SystemExit("RFE6GEN not found")

    out = {
        "utc": utc(), "cmd": a.cmd, "board": a.board,
        "args": {k: v for k, v in vars(a).items() if k != "rd"},
        "setup": (
            f"conducted A-chain: RFE6GEN -> {a.atten_db:g} dB ({a.atten_note}) "
            f"-> IPEX -> ESP; MDS/SFDR, no enclosure"
        ),
    }
    with RfeGenerator(gp) as gen, open_esp(a.port, a.baud) as dev:
        out["generator"] = gen.info()
        out["firmware"] = dev.info()["version"]
        out["identity"] = dev.identity
        dev.set_freq(a.esp_freq)
        dev.set_bandwidth(0)
        fig = {"mds_sfdr": cmd_mds_sfdr}[a.cmd](a, dev, gen, out)
        gen.off()
        dev.set_gain(None)
    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()
    stem = f"mds_sfdr_{a.esp_freq}" + (f"_{a.tag}" if a.tag else "")
    (a.rd / f"{stem}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if fig is not None:
        fig.savefig(a.rd / f"{stem}.png", dpi=130)
    print("wrote", a.rd / f"{stem}.json")

    # rolling Q8 summary
    summary_path = ROOT / "results" / "rf_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"runs": []}
    summary.setdefault("runs", []).append({
        "utc": out["utc"], "board": a.board, "file": str(a.rd / f"{stem}.json"),
        "mds": out.get("mds"), "sfdr": {k: out["sfdr"].get(k) for k in
                                        ("sfdr_db", "spur_f_bb_hz", "carrier_dbfs_bin", "spur_dbfs_bin",
                                         "pin_dbm", "spur_note") if "sfdr" in out},
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("updated", summary_path)


if __name__ == "__main__":
    main()
