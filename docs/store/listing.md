# Chrome Web Store listing

## Name
Framegen - Frame Interpolation

## Summary (132 chars max)
Real-time neural frame generation for any video: 24/30 fps becomes 60-240 fps, fully on your GPU. No cloud, no account.

## Detailed description

Framegen generates the frames between the real frames of any HTML5 video, in real time, in the tab: 24 or 30 fps
footage plays at 60, 120 or your display's refresh rate. Version 2.0 runs on the FlyHigh engine: the FlyHighFG
network and its WGSL kernels compiled to WebAssembly, with the engine's scheduler deciding what to show on every
display tick. About a millisecond per generated frame on a mid-range GPU.

Works on any site with a <video> element: YouTube, Twitch, streaming sites, local files opened in the browser.

Features:
- Auto mode that fits your GPU and display, with an optional FPS limit; Match display and Custom FPS on an exact
  output grid; fixed 2x to 6x
- Anime mode: held drawings are detected and the real motion is interpolated; an optional cadence smoothing
  spreads each drawing step evenly over the frames it is held for
- Optional 2x neural upscale when the player is larger than the video
- Static protection for burnt-in subtitles and logos
- Compare slider: original on the left, Framegen on the right
- Profiles in a full settings page; quick controls inside the player; an optional fps counter, watermark and notices

2.0 is marked experimental: the engine is new, and the previous engine (v7 small) is gone. Known limits: a few
sites forbid WebAssembly for extensions through their security policy (reddit.com among the ones checked) and
Framegen says so there; on clean film-like footage the new network is slightly below the previous one, on anime and
games it is above.

Requirements: a WebGPU-capable browser (Chrome or Edge 121+; Firefox 141+ on Windows through the Firefox build).
GPUs without shader-f16 run the f32 kernels, slower but the same picture. No WebGPU = the extension does nothing.

Privacy: everything runs locally on your GPU. Framegen has no servers, makes no network requests beyond loading its
own bundled files, and collects nothing. See the privacy policy.

Free for personal use. Source: https://github.com/MONZikWasTaken/Framegen

## Category
Photos (or: Fun / Entertainment)

## Permission justifications (store review form)
- host_permissions <all_urls>: the content script must run on any site the user watches video on; there is no fixed
  site list for "any <video>".
- storage: persists the user's settings, local profiles and the per-GPU autotune tables.
- declarativeNetRequest: relaxes CORS on media requests so cross-origin video frames can be read into GPU textures
  (rules.json, media only).

## Assets checklist (manual)
- [ ] Screenshots 1280x800: docs/store/assets/shot{1..5}.png (re-shoot with the 2.0 panel)
- [x] Small promo tile 440x280: docs/store/assets/promo_tile.png
- [x] Privacy policy URL:
      https://github.com/MONZikWasTaken/Framegen/blob/main/docs/store/privacy-policy.md
