# SnapCal changes to ESP-SDR

Base: `ESPARGOS/esp-sdr` at the commit in `UPSTREAM_COMMIT` (GPL-3.0-or-later).
Upstream pins an ESP-IDF v6 development commit; this fork builds with the locally
installed PlatformIO `espressif32@6.13.0` (ESP-IDF 5.5.3).

| File | Change | Reason |
| --- | --- | --- |
| `platformio.ini` | added | PIO build, `src_dir = main` |
| `CMakeLists.txt` | `COMPONENTS main __pio_env` under PIO instead of `MINIMAL_BUILD` | PIO reads flags from its `__pio_env` stub component, which `MINIMAL_BUILD` drops |
| `main/CMakeLists.txt` | also generate `build_version.h` at configure time | PIO compiles from the CMake code model and never runs custom targets |
| `tools/pio_version.py` + `extra_scripts = pre:` (2026-10-08) | regenerate `build_version.h` before every PIO build | PIO reruns the CMake configure step only when build files change, so the embedded revision stayed at the first commit (recorded as `3d932a8`; `f281ecf` after the commit messages were rewritten) |
| `tools/generate_version.py` | adds `upstream` field from `UPSTREAM_COMMIT` | `VERSION?` reports both the SnapCal and ESP-SDR commits |
| `main/targets/esp32s3/s3_fft_rnd.S`, `s3_unpack.S` | `.section .iram1.*` instead of `.text` | PIO links `main/` as objects, so the `libmain.a` mapping in `linker.lf` does not match; SPEC mode needs these in IRAM |
| `dependencies.lock`, `.github/`, `docs/testing/` | removed | S31-only / CI artefacts |

Verified on the build output (2026-10-08): calls into `chip_v7_set_chan_ana` go through
`__wrap_chip_v7_set_chan_ana`; `sram_guard.ld` is on the ELF link line; all `s3_*`
kernels are at `0x4037xxxx` (IRAM); `_bss_end = 0x3fcafc88 < 0x3fcb0000` (ring base),
only ~0.9 KiB of margin, so new static data in this firmware will trip the guard.

Planned SnapCal additions (configs B–D in the paper plan) go below this line with dates.

### 2026-10-20 · on-chip energy detect (`CAPE*`, paper mode B)

| Command | Reply | Effect |
| --- | --- | --- |
| `CAPE <n> <rate> <thresh>` / `CAPE20` / `CAPE16` | `MISS <code> <us>` or `HIT <n> <crc> <us> <code>` + payload | capture like `CAP*`, mean \|z\|² → wire code (~0.5 dB/step); upload IQ only when `code >= thresh` |

`CAPS` gains `ENERGY`. Shared helper `main/common/energy_detect.h`. Wired on S3 and C5/C6/C61 receivers. No new static BSS (S3 SRAM margin unchanged). Note: advertised `DCT` is DC correction (`DC?`/`DC`), not energy detect.

### 2026-10-08 · snapshot scheduling and timing hooks (`main/targets/esp32s3/receiver.c`)

| Command | Reply | Effect |
| --- | --- | --- |
| `JITTER <max_us>` (0–100000) / `JITTER?` | `OK` / `JITTER <max_us>` | each `CAP`/`CAP20`/`CAP16` busy-waits `esp_random() % (max_us+1)` µs before the capture starts |
| `POLL 0\|1` | `OK` | 1: busy-poll the command FIFO between FreeRTOS ticks (still yields one tick every 50 ms); 0: stock one-tick sleep |
| `TS?` | `TS <cmd_us> <start_us>` | `esp_timer` µs when the last capture command line was parsed and when that capture started |
| `MARK <snap_pin> <xfer_pin>` / `MARK OFF` | `OK` | SNAP_MARK high from just before capture start until capture end; XFER_DONE ~1 µs pulse after the payload is queued to the USB FIFO (not host receipt). Pins must be set to outputs with `GPIO n 0` first |

`CAPS` gains `JITTER POLL TS MARK`. Reason: with `CONFIG_FREERTOS_HZ=1000` the idle loop
sleeps one tick between polls, so host-paced snapshots start on a 1 ms grid, which
creates a sync trap with periodic bursts commensurate with 1 ms
(`results/phase0_bringup/burst_intercept_Tb10ms.json`). Static data added: 40 B;
`_bss_end = 0x3fcafcb0` (848 B margin).

### 2026-10-12 · off-USB snapshot log (`main/targets/esp32s3/snaplog.h`, PIO env `esp32s3_wroom1u_snaplog`)

| Command | Reply | Effect |
| --- | --- | --- |
| `LOG ARM <count> <n> <rate> <fmt> <gain> <delay_s> <gap_ms>` | `OK <slot_bytes> <capacity>` | stores the run config (current frequency and filter) in the `snaplog` partition and erases the slots |
| `LOG RUN` | none | runs the armed log now with USB attached (tethered reference) |
| `LOG?` | `LOG <EMPTY\|ARMED\|RUNNING\|DONE> ...` / `LOG NONE` | config and progress |
| `LOG READ <i>` | `LOGDATA <i> <n> <fmt> <start_us> <elapsed> <crc> <gap_us>` + payload | export one snapshot |
| `LOG CLEAR` | `OK` | erase the config sector |

An armed log also starts on its own once the USB SOF frame counter has not moved for 2 s
(battery attached and cable pulled, or boot on battery; 3 s grace after boot), waits
`delay_s`, then takes `count` snapshots with a random 0–`gap_ms` gap and writes each to
flash. `capture` was split into `capture_raw` (capture + pack + CRC) and the USB send so
the RF path is the same as `CAP`. `CAPS` gains `SNAPLOG`. The env uses 16 MB flash and
`partitions-snaplog.csv`; `sdkconfig.defaults.snaplog` is passed through
`cmake_extra_args` because `board_upload.flash_size` / `board_build.partitions` alone did
not reach sdkconfig. Main dependencies gain `esp_partition`. `_bss_end = 0x3fcafcc8`
(824 B margin).
