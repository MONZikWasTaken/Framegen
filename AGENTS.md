# AGENTS.md - Framegen

Framegen is a browser extension (Chrome MV3, Firefox) that generates frames for any `<video>` on a page in real time,
on the GPU through WebGPU. Since 2.0 the generation, the stream of frames and the pacing are the FlyHigh engine
(`getflyhigh/flyhigh-engine`, wasm build of the Rust crate); the extension is the host: the page, the video, the canvas,
the settings and the UI. `docs/architecture.md` explains the mapping.

## Layout

```
extension/              the loadable extension (Load unpacked points here)
  content.js            classic bootstrap: one guard, then import('src/main.js')
  src/                  ES modules of the host
    main.js             wiring: settings, player, UI, popup protocol, navigation, hover
    player.js           the pipeline: video -> stream slots, Pacer ticks -> canvas; backend, autotune, device loss
    engine.js           the FlyHigh backend (FhEngine/FhModel/FhStream/FhPacer + adapters.js), stream spec -> options
    settings.js         chrome.storage.local sync, sanitized through profiles.js
    profiles.js         the settings schema and the profile store (shared with the options page and the tests)
    overlay.js          the canvas over the video (placement, object-fit mirroring, visibility)
    present.js          canvas present path: SDR/HDR, sharpening, compare split
    upscale.js          TinySR 2x (rt/sr.js) on the present path
    capture.js          the CORS reload that makes a cross-origin video readable
    bench.js            the product benchmark bridge (loopback pages with a token): Pacer report + trace v1
    sites.js            site quirks and video discovery
    ui/                 controls rail + quick panel, control bar, notices (HUD, plates), subtitles, icons
  engine/               wasm module, adapters.js, graph.json, flyhighfg.json/.bin (copied from the engine repo)
  rt/                   sr.js (TinySR, the upscaler) and conv.js (its WGSL conv kernels, taken from the previous engine)
  assets/               rt_sr.* weights of the upscaler
  options.*, popup.*, background.js, rules.json (CORS headers on media), icons/
tools/                  build_extension.ps1 (validate + zips), node tests, the WebGPU bench of the previous engine
web/                    the previous engine's sources (rt/), its bench pages
packages/rt             the previous engine on npm (framegen 1.x); its v7s weights left the tree with 2.0, so it is
                        packed from the v1.5.5 tag
crates/, benchmarks/, experiments/, demo/, models/   the pre-1.0 RIFE candle/TensorRT work; history, not shipped
docs/                   architecture.md (2.0), TECHNICAL.md (the previous engine), media
```

## Build, test, run

```pwsh
node --test tools/*.test.mjs                       # schema, layout, the upscaler's kernel contract
pwsh tools/build_extension.ps1                      # validate payloads, framegen-{extension,webstore,firefox}.zip
pwsh tools/build_extension.ps1 -EngineRepo E:\FlyHighProject\flyhigh-runtime   # also refresh extension/engine
```

Load unpacked: `chrome://extensions` → Developer mode → Load unpacked → `extension/`. Firefox: `about:debugging` →
Load Temporary Add-on → `extension/manifest.json` (the Firefox zip from the build script carries the Gecko keys).

The engine module is built in the engine repo (`sh tools/build_web.sh` → `examples/web/pkg/`); refresh it here with
`-EngineRepo`. Model files come from the engine's `models/`.

## Conventions

- Comments say why, never what. No narrating comments.
- No emoji, no em dashes in text or code.
- Settings keys are declared once, in `src/profiles.js`; the options page, the quick panel and the player read them
  from there. A new setting means: schema (and a schema version bump if a key is removed), options.html row, the
  player's reaction in `applySettings`, a test.
- Everything measured about the pacing belongs in the engine's Pacer (and its traces), not here. The host records a
  trace through `pacer.record` (debug on, or the bench bridge) and the engine's `flyhigh-dev replay` reproduces it.
- Pushes and releases happen only on the maintainer's word; local commits otherwise.
