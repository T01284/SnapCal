# third_party

Reference clones only, not tracked. Recreate with:

```powershell
git clone https://github.com/ESPARGOS/esp-sdr.git third_party/esp-sdr
git -C third_party/esp-sdr checkout 06a5ca4027d28626c8e8a8b38176814dfe774a1c
git -C third_party/esp-sdr submodule update --init --recursive
```

The firmware actually built and flashed is the vendored fork in `firmware/snapcal_fw/`
(GPL-3.0-or-later, same license as upstream). Its upstream base commit is in
`firmware/snapcal_fw/UPSTREAM_COMMIT`; local changes are listed in
`firmware/snapcal_fw/SNAPCAL_CHANGES.md`.
