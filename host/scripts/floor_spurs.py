"""Q6 step 0: terminated noise floor and self-spurs -> results/<out>/floor_spurs_<tag>.json.

The antenna connector is terminated (50 ohm load, or an idle LimeSDR TX port as a stand-in);
the ESP32 takes back-to-back snapshots at fixed gain and rate. Per (frequency, gain index):

  floor        median over bins of the snapshot-averaged PSD (DC +-dc_bins excluded), dBFS/Hz
  persistent   bins whose across-snapshot MEDIAN PSD stands >= --spur-db above the local floor
               -> on-chip / bench spurs present in most snapshots
  intermittent bins whose MAX-HOLD stands >= --burst-db above the local floor but are not
               persistent -> coupled-in traffic; reported with the fraction of snapshots hit
  rms          per-snapshot total power with DC removed; outliers (> median + 3 dB) are counted

Persistent spurs are re-located on the full-length averaged spectrum (n-point Hann, ~5 kHz bins at
80 MS/s) and matched to harmonics of 40 MHz (XTAL) and 48 MHz (USB). A harmonic of the ESP32's own
crystal sits at its nominal baseband offset; one from another instrument is shifted by the relative
crystal error (e.g. +4.5 ppm * f_RF against the LimeSDR on s3-wroom1u-01).

Run once per environment with --tag (box / bench / hotspot) and compare the JSON files.

Example:
  python host/scripts/floor_spurs.py --port COM14 --board s3-wroom1u-01 --tag box \
      --termination "idle LimeSDR Mini TX port (no process running)"
"""
import argparse
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
from snapcal.dsp import clip_fraction, write_cf32
from snapcal.espsdr import EspSdr, EspSdrError

NFFT = 1024


def snapshot_psd(x, nfft=NFFT):
    """Welch periodogram (Hann, 50 % overlap) in power per bin, fftshifted, linear."""
    w = np.hanning(nfft)
    step = nfft // 2
    segs = 1 + (len(x) - nfft) // step
    idx = np.arange(segs)[:, None] * step + np.arange(nfft)[None, :]
    X = np.fft.fft(x[idx] * w, axis=1)
    return np.fft.fftshift(np.mean(np.abs(X) ** 2, axis=0)) / np.sum(w) ** 2


def enbw_hz(fs, nfft=NFFT):
    w = np.hanning(nfft)
    return fs * np.sum(w ** 2) / np.sum(w) ** 2


def local_floor(p_db, half=32):
    """Running median over +-half bins (spur-robust floor estimate)."""
    n = len(p_db)
    pad = np.concatenate([p_db[-half:], p_db, p_db[:half]])
    return np.array([np.median(pad[k: k + 2 * half + 1]) for k in range(n)])


def find_lines(excess_db, thr_db, f_hz, mask):
    """Group adjacent bins above thr into lines; return [(f_peak, excess_peak, bins)]."""
    hit = (excess_db >= thr_db) & mask
    lines, k = [], 0
    while k < len(hit):
        if hit[k]:
            j = k
            while j + 1 < len(hit) and hit[j + 1]:
                j += 1
            m = k + int(np.argmax(excess_db[k: j + 1]))
            lines.append((float(f_hz[m]), float(excess_db[m]), j - k + 1, m))
            k = j + 1
        else:
            k += 1
    return lines


HARMONIC_BASES_HZ = {"XTAL 40 MHz": 40e6, "USB 48 MHz": 48e6}


def refine_peak(P_full, fs, f_coarse_hz, search_hz=250e3):
    """Peak of the full-length averaged spectrum near f_coarse, parabolic in dB. Returns Hz."""
    n = len(P_full)
    df = fs / n
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    k0 = int(np.argmin(np.abs(f - f_coarse_hz)))
    h = int(search_hz / df)
    lo, hi = max(1, k0 - h), min(n - 2, k0 + h)
    m = lo + int(np.argmax(P_full[lo: hi + 1]))
    a_, b_, c_ = np.log(P_full[m - 1: m + 2] + 1e-30)
    den = a_ - 2 * b_ + c_
    d = 0.5 * (a_ - c_) / den if den != 0 else 0.0
    return float(f[m] + d * df)


def harmonic_match(f_rf_hz, tol_hz=50e3):
    """All clock harmonics within tol (2400 MHz is both 60 x 40 and 50 x 48 MHz)."""
    out = []
    for name, base in HARMONIC_BASES_HZ.items():
        k = round(f_rf_hz / base)
        if abs(f_rf_hz - k * base) <= tol_hz:
            out.append({"source_guess": name, "order": int(k), "offset_hz": float(f_rf_hz - k * base)})
    return out


def measure(dev, freq, gain, fs, n, count, a, raw_sink=None):
    dev.set_freq(freq)
    dev.set_gain(gain)
    for _ in range(3):
        dev.capture(n, fs, "iq10")
    P = np.empty((count, NFFT), dtype=np.float32)
    P_full = np.zeros(n)
    w_full = np.hanning(n)
    rms, clip, dc, t = [], [], [], []
    t0 = time.perf_counter()
    for i in range(count):
        x = dev.capture(n, fs, "iq10").iq.astype(np.complex128)
        if raw_sink is not None:
            raw_sink.append(x.astype(np.complex64))
        m = np.mean(x)
        dc.append(m)
        rms.append(10 * np.log10(np.mean(np.abs(x - m) ** 2) + 1e-20))
        clip.append(clip_fraction(x))
        P[i] = snapshot_psd(x)
        P_full += np.abs(np.fft.fft((x - m) * w_full)) ** 2
        t.append(time.perf_counter() - t0)
    P_full = np.fft.fftshift(P_full / count)

    f = np.fft.fftshift(np.fft.fftfreq(NFFT, 1.0 / fs))
    mask = np.abs(f) > a.dc_bins * fs / NFFT
    mean_db = 10 * np.log10(P.mean(axis=0) + 1e-20)
    med_db = 10 * np.log10(np.median(P, axis=0) + 1e-20)
    max_db = 10 * np.log10(P.max(axis=0) + 1e-20)
    floor_db = local_floor(med_db)
    eb = enbw_hz(fs)

    persistent = find_lines(med_db - floor_db, a.spur_db, f, mask)
    pers_bins = np.zeros(NFFT, bool)
    for _, _, w, m in persistent:
        pers_bins[max(0, m - w): m + w + 1] = True
    intermittent = [ln for ln in find_lines(max_db - floor_db, a.burst_db, f, mask & ~pers_bins)]
    P_db = 10 * np.log10(P + 1e-20)

    def line_rows(lines, kind):
        rows = []
        for fp, ex, w, m in lines:
            hits = float(np.mean(P_db[:, m] - floor_db[m] >= (a.spur_db if kind == "p" else a.burst_db)))
            row = {"f_bb_hz": fp, "f_rf_mhz": freq + fp / 1e6, "bins": w,
                   "excess_db": ex, "level_dbfs_bin": float(med_db[m] if kind == "p" else max_db[m]),
                   "snapshot_fraction": hits}
            if kind == "p":
                fb = refine_peak(P_full, fs, fp)
                row["f_bb_fine_hz"] = fb
                row["f_rf_fine_mhz"] = freq + fb / 1e6
                row["harmonic"] = harmonic_match(freq * 1e6 + fb)
            rows.append(row)
        return sorted(rows, key=lambda r: -r["excess_db"])

    rms = np.array(rms)
    med_rms = float(np.median(rms))
    floor_bin = float(np.median(mean_db[mask]))
    res = {
        "freq_mhz": freq, "gain_index": gain, "rate_hz": fs, "n": n, "count": count,
        "duration_s": t[-1],
        "floor_dbfs_per_bin": floor_bin,
        "floor_dbfs_per_hz": floor_bin - 10 * np.log10(eb),
        "floor_median_psd_dbfs_per_hz": float(np.median(med_db[mask])) - 10 * np.log10(eb),
        "enbw_hz": eb, "nfft": NFFT, "dc_bins_excluded": a.dc_bins,
        "rms_dbfs": {"median": med_rms, "p10": float(np.percentile(rms, 10)),
                     "p90": float(np.percentile(rms, 90)), "max": float(rms.max()),
                     "outlier_fraction_3db": float(np.mean(rms > med_rms + 3.0))},
        "dc": {"median_re": float(np.median(np.real(dc))), "median_im": float(np.median(np.imag(dc))),
               "dbfs": float(20 * np.log10(abs(np.median(np.real(dc)) + 1j * np.median(np.imag(dc))) + 1e-20))},
        "clip_fraction_max": float(np.max(clip)),
        "persistent_spurs": line_rows(persistent, "p"),
        "intermittent_lines": line_rows(intermittent, "i"),
    }
    spectra = {"f": f, "mean": mean_db, "median": med_db, "max": max_db, "floor": floor_db,
               "rms": rms, "t": np.array(t)}
    return res, spectra


def open_esp(port, baud):
    """C5's CH340 stays at the last BAUD until reboot."""
    try:
        return EspSdr(port, baud=baud)
    except EspSdrError:
        other = 1_000_000 if baud != 1_000_000 else 2_000_000
        dev = EspSdr(port, baud=other)
        print(dev.command(f"BAUD {baud}"), flush=True)
        dev.ser.baudrate = baud
        return dev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=2_000_000)
    ap.add_argument("--board", required=True)
    ap.add_argument("--tag", required=True, help="environment label: box / bench / hotspot ...")
    ap.add_argument("--termination", required=True, help="what terminates the antenna connector")
    ap.add_argument("--freqs", type=int, nargs="+", default=[2412, 2442, 2472])
    ap.add_argument("--gains", type=int, nargs="+", default=[20, 44, 60])
    ap.add_argument("--rate", type=int, default=80_000_000)
    ap.add_argument("--n", type=int, default=16380)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--spur-db", type=float, default=6.0, help="persistent-spur threshold over local floor")
    ap.add_argument("--burst-db", type=float, default=10.0, help="max-hold threshold for intermittent lines")
    ap.add_argument("--dc-bins", type=int, default=3)
    ap.add_argument("--save-raw", action="store_true", help="keep IQ of the first (freq, gain) in data/raw/")
    ap.add_argument("--out", default=None, help="results subdirectory (default conducted/<board>)")
    a = ap.parse_args()
    rd = ROOT / "results" / (a.out or f"conducted/{a.board}")
    rd.mkdir(parents=True, exist_ok=True)

    out = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "board": a.board,
           "tag": a.tag, "termination": a.termination, "args": vars(a), "runs": []}
    spectra = []
    raw = [] if a.save_raw else None
    with open_esp(a.port, a.baud) as dev:
        out["firmware"] = dev.info()["version"]
        dev.set_bandwidth(0)
        for freq in a.freqs:
            for g in a.gains:
                sink = raw if (raw is not None and not out["runs"]) else None
                r, s = measure(dev, freq, g, a.rate, a.n, a.count, a, sink)
                out["runs"].append(r)
                spectra.append(s)
                print(f"{freq} MHz gain {g:2d}: floor {r['floor_dbfs_per_hz']:7.1f} dBFS/Hz "
                      f"(median PSD {r['floor_median_psd_dbfs_per_hz']:7.1f}), "
                      f"rms {r['rms_dbfs']['median']:6.1f} dBFS (p90 {r['rms_dbfs']['p90']:6.1f}, "
                      f"outliers {100 * r['rms_dbfs']['outlier_fraction_3db']:.1f} %), "
                      f"DC {r['dc']['dbfs']:6.1f} dBFS, {len(r['persistent_spurs'])} persistent / "
                      f"{len(r['intermittent_lines'])} intermittent lines", flush=True)
        dev.set_gain(None)
    out["toolchain"] = envinfo.toolchain()
    out["host"] = envinfo.host()

    p = rd / f"floor_spurs_{a.tag}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if raw:
        rawdir = ROOT / "data" / "raw" / f"floor_{a.board}_{a.tag}"
        rawdir.mkdir(parents=True, exist_ok=True)
        write_cf32(rawdir / "snapshots.cf32", raw)
        first = out["runs"][0]
        (rawdir / "meta.json").write_text(json.dumps({k: first[k] for k in ("freq_mhz", "gain_index", "rate_hz", "n", "count")}
                                                     | {"tag": a.tag, "termination": a.termination}, indent=2), encoding="utf-8")

    nf, ng = len(a.freqs), len(a.gains)
    fig, ax = plt.subplots(ng, nf + 1, figsize=(5 * (nf + 1), 3.2 * ng), squeeze=False)
    for k, (r, s) in enumerate(zip(out["runs"], spectra)):
        i, j = k % ng, k // ng
        x = s["f"] / 1e6
        ax[i, j].plot(x, s["max"], lw=0.5, color="0.7", label="max hold")
        ax[i, j].plot(x, s["mean"], lw=0.7, label="mean")
        ax[i, j].plot(x, s["median"], lw=0.7, label="median")
        for sp in r["persistent_spurs"]:
            ax[i, j].plot(sp["f_bb_hz"] / 1e6, sp["level_dbfs_bin"], "rv", ms=4)
        ax[i, j].set_title(f"{r['freq_mhz']} MHz, gain {r['gain_index']}: floor {r['floor_dbfs_per_hz']:.1f} dBFS/Hz", fontsize=9)
        ax[i, j].set_xlabel("baseband (MHz)")
        if k == 0:
            ax[i, j].legend(fontsize=7)
        ax[i, nf].plot(s["t"], s["rms"], ".", ms=1, label=f"{r['freq_mhz']} MHz")
        ax[i, nf].set_title(f"gain {r['gain_index']}: snapshot power, DC removed (dBFS)", fontsize=9)
        ax[i, nf].set_xlabel("time (s)")
    for i in range(ng):
        ax[i, nf].legend(fontsize=7)
    fig.suptitle(f"{a.board} [{a.tag}] termination: {a.termination}", fontsize=10)
    fig.tight_layout()
    fig.savefig(rd / f"floor_spurs_{a.tag}.png", dpi=120)
    print("wrote", p)


if __name__ == "__main__":
    main()
