"""Host / toolchain / chip metadata written next to every result (plan: results/env.json)."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIO_HOME = Path(os.environ.get("PLATFORMIO_CORE_DIR", Path.home() / ".platformio"))


def _run(cmd, **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_revision(path: Path = ROOT) -> str:
    rev = _run(["git", "-C", str(path), "rev-parse", "HEAD"]).strip() or "unknown"
    if _run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]).strip():
        rev += "-dirty"
    return rev


def esptool_chip(port: str) -> dict:
    """Reads chip identity over the ROM bootloader. Resets the board."""
    esptool = PIO_HOME / "packages" / "tool-esptoolpy" / "esptool.py"
    py = PIO_HOME / "penv" / "Scripts" / "python.exe"
    if not py.exists():
        py = PIO_HOME / "penv" / "bin" / "python"
    out = _run([str(py), str(esptool), "--port", port, "flash_id"])
    info = {}
    for key, pat in {
        "chip": r"Chip is (.+?) \(",
        "silicon_revision": r"\(revision (v[\d.]+)\)",
        "features": r"Features: (.+)",
        "crystal": r"Crystal is (\S+)",
        "mac": r"MAC: (\S+)",
        "flash_size": r"Detected flash size: (\S+)",
        "flash_manufacturer": r"Manufacturer: (\S+)",
        "flash_device": r"Device: (\S+)",
    }.items():
        m = re.search(pat, out)
        if m:
            info[key] = m.group(1).strip()
    return info


def toolchain() -> dict:
    ini = (ROOT / "firmware" / "snapcal_fw" / "platformio.ini").read_text(encoding="utf-8")
    m = re.search(r"platform\s*=\s*espressif32@(\S+)", ini)
    plat = m.group(1) if m else "unknown"
    idf = "unknown"
    ver = PIO_HOME / "packages" / "framework-espidf" / "version.txt"
    if ver.exists():
        idf = ver.read_text().strip()
    upstream = (ROOT / "firmware" / "snapcal_fw" / "UPSTREAM_COMMIT").read_text().split()
    return {
        "platformio_platform": f"espressif32@{plat}",
        "esp_idf": idf,
        "esp_sdr_upstream_commit": upstream[0] if upstream else "unknown",
        "pio_core": _run([str(PIO_HOME / "penv" / "Scripts" / "pio.exe"), "--version"]).strip(),
    }


def host() -> dict:
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "snapcal_git": git_revision(),
    }
