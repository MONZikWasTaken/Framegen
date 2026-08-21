# Display Hz browser check

This Windows-only headed check exercises the real unpacked extension in
`Display Hz` mode at a temporarily selected primary-monitor refresh rate. It is
separate from the authoritative target-FPS matrix.

```powershell
node tools/run_display_hz_check.mjs --hz 60 --source-fps 24
```

The runner validates the requested mode before applying it and uses a
session-only display change. On normal completion or a handled `SIGINT`/`SIGTERM`
it makes up to three attempts to restore the original device, resolution, scan
mode, and refresh rate before other cleanup. A failed or unverifiable
restoration makes the report red and prints a manual recovery command. A forced
process kill can still require that command; a Windows restart discards the
session-only mode. Reports are write-once JSON files inside `output/display-hz`.

Safe helper checks that do not change the active mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/set_primary_refresh.ps1 -Query
powershell -NoProfile -ExecutionPolicy Bypass -File tools/set_primary_refresh.ps1 -Hz 60 -TestOnly
```

The check requires a foreground headed browser, a primary monitor that supports
the requested mode at its current resolution, and a clean Git worktree. It fails
on source or staged-extension drift, browser diagnostics, source-producer skips,
presentation drops, estimator plan resets, or restoration failure. Successful
GPUCanvas submissions are observable; physical panel scan-out is not.
