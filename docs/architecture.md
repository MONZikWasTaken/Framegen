# Framegen 2.0 on the FlyHigh engine

Framegen 1.x carried its own WGSL network (`web/rt/rt.js`), its own scheduler (`cadence.js`) and a 5000-line content
script that joined them. In 2.0 the generation, the stream and the pacing are the FlyHigh engine's JavaScript classes;
the extension is a host of about 3000 lines in ES modules: the page, the video, the canvas, the settings, the UI.

## The loop

Per source frame (`requestVideoFrameCallback`), in `src/player.js`:

1. the pool size for this frame (the canvas backing, capped at 1920 x 1080; the native size when the upscaler needs
   small input), dead-banded so player chrome does not rebuild the stream every tick;
2. `stream.nextInput(inUse)`: a slot the stream no longer reads and the host no longer shows; the frame goes in through
   the adapters' `VideoSource` (a copy at equal size, an `importExternalTexture` pass otherwise and in Firefox);
3. `stream.classify(previous, slot)`: motion, cut or hold (the held-frame thresholds are on in Anime mode);
4. `pacer.sourceFrame(now, { id, ptsMs, kind, motion, skipped })`: the requests it returns are the previous pair's mids
   that must go out before the pair changes; they are generated at once;
5. `stream.push(slot, pts)` (`pushTimed` every thirtieth call: the pair cost for the Pacer); in video mode the pair
   statistics are awaited when the stream says so.

Per display tick (`requestAnimationFrame`): `pacer.tick(now)` names the frame to present (a source slot or an output
slot) and the mids to generate just in time; the host blits the texture onto the overlay canvas (HDR, sharpening and
the compare split live in `src/present.js`, the 2x upscaler in `src/upscale.js`) and generates through the adapters'
`CostSampler`, which times every thirtieth mid and every probe for the Pacer.

Output slots are a small ring: a slot is free once its frame was presented, or 250 ms after its due time when the
Pacer dropped it. Input slots the host still needs (the Pacer's history, the compare ring, the slot on screen) are
passed to `nextInput`; the ring grows up to 24 slots.

## Settings to engine options

| Setting | Engine |
|---|---|
| Quality (288 to 1080) | session `internal` = the rung, clamped to the frame |
| Anime mode | stream `heldFrames` on, `repair` 2, Pacer motion units from the classifier |
| Anime cadence smoothing (with Anime mode and repair) | stream `mode: 'video'`, `cadence: true`, Pacer `lookahead` |
| Periodic-structure repair | session `repair` 1 (2 in Anime mode), 0 when off |
| Sharper edges (bicubic) | session `render: 'bicubic'`, else `'bilinear'` |
| Static protection | session `staticProtection: true` |
| Output rate, FPS limit, Custom FPS | Pacer `mode` (`auto`, `display_hz`, `{ factor }`, `{ target }`), `fpsLimit` |
| Limit to display, Full refresh rate, Motion-adaptive | Pacer `limitToDisplay`, `fillDisplay`, `motionAdaptive` |
| Frame generation off | no Pacer: each source frame is presented at once (upscale, HDR, sharpening, compare) |

Realtime mode runs with `fastRender` and `linearT` when the rung is below the frame size; video mode renders the exact
path, which the cadence re-timing needs. Scene cuts use the extension's threshold (a mean cell difference above 90 of
765), not the engine's default 0.08.

The autotune (`engine.autotune`, 5 to 15 s of GPU time) runs when the video is paused or generation is off, or after
two minutes of playback, once per GPU, rung and pool size; the table is stored in `chrome.storage.local` (`fcTune`) and
applied at the next stream build.

## Pages whose CSP blocks WebAssembly

Chrome applies the page's Content Security Policy to the content script's world when WebAssembly compiles: a
`script-src` without `'wasm-unsafe-eval'` or `'unsafe-eval'` refuses the engine module. Neither the extension's own
CSP nor an iframe of the extension gets around it: the wasm runs in such an iframe, but the page's frames cannot be
handed across the origin boundary (`VideoFrame`, `ImageBitmap` and `MediaStreamTrack` all fail to transfer, measured
2026-11-02). Framegen says so in the HUD, the panel and the popup and does nothing else on such a page. Of 24 video
sites checked the same day only reddit.com sends such a policy; YouTube, Twitch, Vimeo, Dailymotion, Kick, Bilibili,
Rumble, Odysee, VK, TikTok and the anime sites allow it. Firefox applies the extension's policy to content scripts and
is not affected.

## What a bug report attaches

With the Debug setting on, the Pacer records its session (`pacer.record`); `window.__framegenDiagnostics()` in the
content script's console shows the status. On a loopback page with the bench token (`src/bench.js`) the `trace`
command returns the recording as `trace v1`, which the engine's `flyhigh-dev replay` reproduces bit for bit.

## Tests

`node --test tools/*.test.mjs`: the settings schema and store migrations, the fixed pacer, the extension layout (every
import resolves, the engine payload is complete, the runtimes mirror `web/rt`). Browser runs are manual: load unpacked
and play any 24 fps video; the quick panel's status block shows the engine class, the measured costs and the Pacer's
decisions.
