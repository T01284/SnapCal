"""RF Explorer Signal Generator RFE6GEN / 6G Combo PLUS analyzer over the vendor Python library.

Only CP210x ports (VID 10C4, PID EA60) are opened: RFECommunicator.GetConnectedPorts()
opens every COM port on the machine, which would touch the ESP32 boards and the Pluto.

Generator levels are indexed 0..7 = (high-power switch, power level) in the order the
vendor calibration table uses: 0..3 = low-power L0..L3 (about -40..-30 dBm),
4..7 = high-power H0..H3 (about -10..0 dBm). The Combo generator with the expansion
board takes a dBm value instead.

6G Combo PLUS: left connector 0.05-960 MHz, right connector 240-6100 MHz (2.4 and 5.8 GHz
are both on the right one). Readings come in 0.5 dB steps; the vendor quotes linearity of
about +/-3 dB only at constant RBW, and RBW follows the span, so thru and inserted-part
readings must share span and input stage.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from serial.tools import list_ports

from RFExplorer import RFE_Common, RFECommunicator

VID, PID = 0x10C4, 0xEA60
BAUD = 500000
LEVELS = [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0), (True, 1), (True, 2), (True, 3)]
INPUT_STAGES = {"direct": RFE_Common.eInputStage.Direct, "att30": RFE_Common.eInputStage.Attenuator_30dB,
                "lna25": RFE_Common.eInputStage.LNA_25dB, "att60": RFE_Common.eInputStage.Attenuator_60dB}


class RfeError(RuntimeError):
    pass


def find_ports() -> list:
    return [p for p in list_ports.comports() if p.vid == VID and p.pid == PID]


def cw_command(f_mhz: float, level: int) -> str:
    hp, pl = LEVELS[level]
    return f"C3-F:{int(round(f_mhz * 1000)):07d},{int(hp)},{pl}"


def cw_command_expansion(f_mhz: float, dbm: float) -> str:
    return f"C5-F:{int(round(f_mhz * 1000)):07d},{dbm:+05.1f}"


class _Rfe:
    def __init__(self, port: str, intended_analyzer: bool, timeout: float = 10.0):
        infos = [p for p in find_ports() if p.device == port]
        if not infos:
            raise RfeError(f"{port} is not a CP210x port; RF Explorer ports: {[p.device for p in find_ports()]}")
        self.port = port
        self.c = RFECommunicator()
        self.c.VerboseLevel = 0
        self.c.m_bIntendedAnalyzer = intended_analyzer
        self.c.m_arrValidCP2102Ports = infos
        if not self.c.ConnectPort(port, BAUD):
            self.c.Close()
            raise RfeError(f"cannot open {port}")
        if not self._wait_model(timeout):
            self.c.SendCommand("r")
            time.sleep(8)
            self.c.SendCommand_RequestConfigData()
            if not self._wait_model(timeout):
                self.close()
                raise RfeError(f"{port}: no RF Explorer configuration received")

    def _pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.c.ProcessReceivedString(True)
            time.sleep(0.02)

    def _wait_model(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.c.ProcessReceivedString(True)
            if self.c.ActiveModel != RFE_Common.eModel.MODEL_NONE:
                self._pump(1.0)
                return True
            time.sleep(0.05)
        return False

    def info(self) -> dict:
        c = self.c
        return {
            "port": self.port,
            "kind": "generator" if c.IsGenerator(True) else "analyzer",
            "main_model": c.MainBoardModel.name,
            "expansion_model": c.ExpansionBoardModel.name,
            "active_model": c.ActiveModel.name,
            "expansion_active": bool(c.ExpansionBoardActive),
            "firmware": c.RFExplorerFirmwareDetected,
            "serial": c.SerialNumber,
            "min_freq_mhz": c.MinFreqMHZ,
            "max_freq_mhz": c.MaxFreqMHZ,
            "input_stage": c.InputStage.name,
            "embedded_cal": bool(c.GetRFE6GENCal().GetCalSize() > 0 if c.IsGenerator(True)
                                 else c.IsAnalyzerEmbeddedCal()),
        }

    def close(self) -> None:
        thread = getattr(self.c, "m_objThread", None)
        try:
            self.c.Close()
        except Exception:
            pass
        # Close() drops the thread without joining, so a failed run keeps the COM port open.
        if thread is not None and thread.is_alive():
            self.c.RunReceiveThread = False
            thread.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RfeGenerator(_Rfe):
    def __init__(self, port: str, timeout: float = 10.0):
        super().__init__(port, intended_analyzer=False, timeout=timeout)
        if not self.c.IsGenerator(True):
            self.close()
            raise RfeError(f"{port} is an analyzer, not a generator")
        self.c.SendCommand_GeneratorRFPowerOFF()
        self.f_mhz: float | None = None
        self.level: int | None = None
        self.dbm: float | None = None

    @property
    def expansion(self) -> bool:
        return bool(self.c.ExpansionBoardActive)

    def cw(self, f_mhz: float, level: int | None = None, dbm: float | None = None, settle: float = 0.5) -> None:
        if self.expansion:
            if dbm is None:
                raise RfeError("expansion board active: give dbm")
            self.c.SendCommand(cw_command_expansion(f_mhz, dbm))
        else:
            if level is None or not 0 <= level < len(LEVELS):
                raise RfeError("give level 0..7")
            self.c.SendCommand(cw_command(f_mhz, level))
        self.f_mhz, self.level, self.dbm = f_mhz, level, dbm
        self.c.SendCommand_GeneratorRFPowerON()
        self._pump(settle)

    def off(self, settle: float = 0.3) -> None:
        self.c.SendCommand_GeneratorRFPowerOFF()
        self._pump(settle)

    def estimated_dbm(self) -> float:
        """Nominal output of the current CW: the set dBm with the expansion board, otherwise the
        vendor estimate from the unit's embedded -30 dBm table plus the per-level deltas."""
        if self.expansion:
            return float(self.dbm)
        hp, pl = LEVELS[self.level]
        return float(self.c.GetRFE6GENCal().GetEstimatedAmplitude(self.f_mhz, hp, pl))

    def has_embedded_cal(self) -> bool:
        return self.c.GetRFE6GENCal().GetCalSize() > 0

    def close(self) -> None:
        try:
            self.c.SendCommand_GeneratorRFPowerOFF()
            time.sleep(0.2)
        except Exception:
            pass
        super().close()


@dataclass
class Sweep:
    f_mhz: np.ndarray
    dbm: np.ndarray


class RfeAnalyzer(_Rfe):
    def __init__(self, port: str, timeout: float = 10.0):
        super().__init__(port, intended_analyzer=True, timeout=timeout)
        if self.c.IsGenerator(True):
            self.close()
            raise RfeError(f"{port} is a generator, not an analyzer")
        self.c.UseMaxHold = False
        self.c.SendCommand_Realtime()
        self._pump(0.5)

    def _select_module(self, f_lo: float, f_hi: float) -> None:
        c = self.c
        if c.MinFreqMHZ <= f_lo and f_hi <= c.MaxFreqMHZ:
            return
        if c.ExpansionBoardModel == RFE_Common.eModel.MODEL_NONE:
            raise RfeError(f"{f_lo}-{f_hi} MHz outside {c.MinFreqMHZ}-{c.MaxFreqMHZ} MHz")
        c.SendCommand("CM" + ("\x00" if c.ExpansionBoardActive else "\x01"))
        time.sleep(1.0)
        c.SendCommand_RequestConfigData()
        self._pump(2.0)
        if not (c.MinFreqMHZ <= f_lo and f_hi <= c.MaxFreqMHZ):
            raise RfeError(f"{f_lo}-{f_hi} MHz not covered by either module "
                           f"(now {c.ActiveModel.name} {c.MinFreqMHZ}-{c.MaxFreqMHZ} MHz)")

    def configure(self, start_mhz: float, stop_mhz: float) -> None:
        self._select_module(start_mhz, stop_mhz)
        self.c.UpdateDeviceConfig(start_mhz, stop_mhz)
        self._pump(1.0)
        self.c.ResetInternalBuffers()
        self.c.CleanSweepData()

    def set_input_stage(self, name: str, timeout: float = 3.0) -> None:
        """direct / att30 / lna25 / att60. The library adds the nominal stage offset to the
        readings only when the unit reports embedded calibration, so compare like with like."""
        stage = INPUT_STAGES[name]
        # IsInputStageAvailable is a property. The vendor setter calls it as a method and raises.
        if not self.c.IsInputStageAvailable:
            if stage != RFE_Common.eInputStage.Direct:
                raise RfeError(f"{self.c.ActiveModel.name} has no selectable input stage")
            return
        self.c.SendCommand("a" + str(stage.value))
        end = time.monotonic() + timeout
        while self.c.InputStage != stage:
            if time.monotonic() > end:
                raise RfeError(f"input stage still {self.c.InputStage.name}, wanted {stage.name}")
            self._pump(0.1)
        self.c.CleanSweepData()

    @property
    def input_stage(self) -> str:
        return next((k for k, v in INPUT_STAGES.items() if v == self.c.InputStage), self.c.InputStage.name)

    def sweeps(self, n: int, skip: int = 2, timeout: float = 30.0) -> list[Sweep]:
        """Next n complete sweeps after discarding `skip` (the first ones may straddle a level change)."""
        c = self.c
        c.CleanSweepData()
        end = time.monotonic() + timeout
        while c.SweepData.Count < skip + n:
            if time.monotonic() > end:
                raise RfeError(f"only {c.SweepData.Count} sweeps in {timeout} s")
            c.ProcessReceivedString(True)
            time.sleep(0.02)
        out = []
        for i in range(skip, skip + n):
            s = c.SweepData.GetData(i)
            k = np.arange(s.TotalDataPoints)
            out.append(Sweep(np.array([s.GetFrequencyMHZ(j) for j in k]),
                             np.array([s.GetAmplitude_DBM(j) for j in k])))
        return out

    def tone(self, f_mhz: float, n: int = 10, window_mhz: float = 1.0) -> dict:
        """Peak level of a CW near f_mhz over n sweeps (call configure() first)."""
        peaks, freqs = [], []
        for s in self.sweeps(n):
            m = np.abs(s.f_mhz - f_mhz) <= window_mhz / 2
            if not m.any():
                raise RfeError(f"{f_mhz} MHz not in sweep {s.f_mhz[0]:.3f}-{s.f_mhz[-1]:.3f} MHz")
            i = np.flatnonzero(m)[np.argmax(s.dbm[m])]
            peaks.append(s.dbm[i])
            freqs.append(s.f_mhz[i])
        p = np.array(peaks)
        # mean, not median: readings are quantised to 0.5 dB and sweep-to-sweep noise dithers them
        return {"dbm": float(p.mean()), "median_dbm": float(np.median(p)), "std_db": float(np.std(p)), "n": n,
                "peak_mhz": float(np.median(freqs)), "rbw_khz": float(self.c.RBW_KHZ),
                "input_stage": self.input_stage,
                "span_mhz": round(float(self.c.CalculateFrequencySpanMHZ()), 3)}


def discover(timeout: float = 10.0) -> dict[str, dict]:
    """Connects to every CP210x port once and returns {"generator"|"analyzer": info}."""
    found: dict[str, dict] = {}
    for p in find_ports():
        try:
            with _Rfe(p.device, intended_analyzer=True, timeout=timeout) as d:
                info = d.info()
        except RfeError as e:
            print(f"{p.device}: {e}")
            continue
        if info["kind"] in found:
            raise RfeError(f"two {info['kind']}s: {found[info['kind']]['port']} and {p.device}; pass ports explicitly")
        found[info["kind"]] = info
    return found
