# Chrome Web Store listing (draft)

## Name
Framecast - Frame Interpolation

## Summary (132 chars max)
Real-time neural frame interpolation for any video: 24/30 fps becomes 60-240+ fps, fully on your GPU. No cloud, no account.

## Detailed description

Framecast doubles (or more) the frame rate of any HTML5 video in real time,
right in the tab. A small neural network - our own WebGPU runtime, ~5 ms per
frame on a mid-range GPU - synthesizes the frames between the real ones, so
24/30 fps footage plays at 60, 120 or up to your display's refresh rate.

Works on any site with a <video> element: YouTube, Twitch, streaming sites,
local files opened in the browser.

Features:
- 2x to 6x interpolation, an auto mode that follows your GPU headroom, and a
  "display Hz" mode that locks output to your monitor's refresh rate
- Anime mode: detects animation drawn "on twos" and interpolates the real
  drawing cadence instead of duplicated frames
- Optional 2x neural upscale (SR) on interpolated frames, trained to repair
  interpolation artifacts, not just sharpen
- Self-calibrating kernels: the runtime benchmarks itself on your GPU once and
  picks the fastest shader variants
- HUD with live fps / latency stats

Requirements: a WebGPU-capable browser (Chrome/Edge 113+) and a GPU with
shader-f16 support. No WebGPU = the extension politely does nothing.

Privacy: everything runs locally on your GPU. Framecast has no servers, makes
no network requests beyond loading its own bundled files, and collects
nothing. See the privacy policy.

Free for personal use. Source: https://github.com/MONZikWasTaken/Framecast

## Category
Photos (or: Fun / Entertainment)

## Permission justifications (store review form)
- host_permissions <all_urls>: the content script must run on any site the
  user watches video on; there is no fixed site list for "any <video>".
- storage: persists the user's settings (quality, factor, HUD) and the
  per-GPU kernel autotune result.
- declarativeNetRequest: relaxes CORS on media requests so cross-origin
  video frames can be read into GPU textures (rules.json, media only).

## Assets checklist (manual)
- [ ] Screenshots 1280x800: player page + HUD, settings panel, before/after
- [ ] Small promo tile 440x280
- [ ] Privacy policy URL (docs/store/privacy-policy.md - needs a public URL
      once the repo is unlocked)
