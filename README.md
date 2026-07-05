<div align="center">

<img src="docs/media/logo.png" width="96" alt="Framegen">

# Framegen

**Silky-smooth video in your browser.** A Chrome extension that turns 24-30 fps
video into 60-240 fps in real time - with a neural network running entirely on
your GPU. No servers, no accounts, nothing leaves your computer.

[![Download](https://img.shields.io/badge/download-latest_release-19c37d)](https://github.com/MONZikWasTaken/Framegen/releases/latest)
[![License](https://img.shields.io/badge/code-AGPLv3-blue)](LICENSE)
[![npm](https://img.shields.io/npm/v/framegen?label=npm&color=cb3837)](https://www.npmjs.com/package/framegen)
[![Ko-fi](https://img.shields.io/badge/support-ko--fi-ff5e5b)](https://ko-fi.com/monzikxd)

<img src="docs/media/hero.gif" width="880" alt="15 fps source vs Framegen x4 interpolation, side by side">

*Real output of the shipped model (v7 small), not a mockup. In the browser
this runs in real time: 3.75 ms per generated frame at 720p on an RTX 4060 Ti.*

**[Watch a 50-second live demo](https://github.com/MONZikWasTaken/Framegen/releases/download/v1.0.0/framegen-live-demo.mp4)** -
real YouTube, the compare slider, the debug HUD, recorded at 60 fps on an
RTX 4060 Ti. *(Footage: Sintel © Blender Foundation, CC-BY.)*

</div>

## What it does

- **2×-6× more frames** on any `<video>` - movies, series, sports, anime,
  screen recordings; YouTube and most video sites
- **Auto mode** picks the highest factor your GPU actually sustains, and backs
  off before you'd see a stutter
- **Anime mode** detects animation drawn "on twos" and interpolates the real
  motion instead of the duplicated frames
- **Display-Hz mode** locks output to your monitor's refresh grid (great on
  120-240 Hz screens)
- **Compare slider** - drag a divider across the video: original on the left,
  Framegen on the right
- **Private by construction** - the whole pipeline runs on your GPU; we collect
  literally nothing

An interpolated frame costs ~3 ms on a mid-range GPU (RTX 4060 Ti) - the
model and inference runtime are custom-built for this (a 2.9 MB network on
hand-written WebGPU kernels; details in [docs/TECHNICAL.md](docs/TECHNICAL.md)).

## Install (2 minutes)

1. Download **`framegen-extension.zip`** from the
   [latest release](https://github.com/MONZikWasTaken/Framegen/releases/latest)
   and extract it anywhere.
2. Open `chrome://extensions` in Chrome.
3. Turn on **Developer mode** (top-right toggle).
4. Click **Load unpacked** and select the extracted `framegen-extension`
   folder.

That's it. Requirements: **Chrome 121+** on a machine with a GPU (Windows,
macOS with Apple Silicon, Linux). Firefox and Safari don't ship the WebGPU
features we need yet.

## How to use

1. Open any video and hover over it - a round **FC** button appears at the
   left edge of the player.
2. Click it. The button turns green, an fps readout appears, and the video is
   now interpolated. Click again to turn it off.
3. The **gear** button next to it opens settings:

| Setting | What it does |
|---|---|
| **Factor** | `auto` is right for most people. Fixed 2×-6× if you want control, `display Hz` to sync exactly to your monitor |
| **Quality** | Resolution of the inserted frames. `480` is the sweet spot; raise it on a strong GPU |
| **Model** | `v7` (default, fastest) or `v6` (previous generation) |
| **Anime mode** | Keep on for anime; harmless elsewhere |
| **SR 2×** | Neural upscale of inserted frames - costs GPU, sharper result |
| **Compare** | The split slider, for seeing the difference yourself |

**Good first test:** anything shot at 24 fps - a movie trailer, a film scene
with a slow camera pan, an anime opening. That's where the difference hits
hardest. On a 60 Hz screen you'll see 24→60; on a 144-240 Hz screen,
considerably more.

## FAQ

**It says "no video found" / the button doesn't appear.**
Make sure the video is actually playing. On some players the button appears
only when the mouse is over the video itself.

**Does it work on Netflix / Crunchyroll?**
No, and it can't: DRM-protected video is invisible to extensions by design -
the browser hands us black frames. YouTube and most other sites work.

**My fps counter shows less than the promised factor.**
Auto mode adapts to your GPU's real headroom - it will never stutter to hit a
number. Lower the quality setting or the factor ceiling if you want more.

**Does it phone home?**
No. There is no server, no telemetry, no analytics. The extension is a local
GPU pipeline; the code is right here to check.

**Is my GPU good enough?**
If it can run the video at all, 2× at 480p almost certainly fits. The HUD
shows the per-frame cost in ms - budget is roughly `(factor-1) × cost <
frame interval`.

## The story

Framegen is older than this repo. The idea - and the first prototype - date
back six months before the first commit here. That prototype never got
published: it worked far too poorly to show anyone. But the idea refused to
go away, and for half a year I kept watching the space - and nobody shipped
it properly: real-time neural frame interpolation, in the browser, on any
video, for anyone. So I decided to build it myself. That's how Framegen
happened.

## Support the project

Framegen is built by **one person** with one mid-range GPU. The extension is
free and will stay free - but the models behind it are not free to make:
every training experiment runs on rented cloud GPUs paid out of pocket
($5-30 per run, and a new model generation takes dozens of runs before one
is good enough to ship). The next, bigger model is designed and waiting -
mostly for GPU-hours.

If Framegen made your video smoother and you want the next model to exist
sooner:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/monzikxd)

Starring the repo helps too - visibility is the other currency.

## Under the hood (the short version)

A distilled RIFE-family student (2.9 MB) runs on a hand-written WGSL runtime -
raw WebGPU compute shaders, no ML framework, matching the PyTorch reference to
1 LSB. The pipeline is fully GPU-resident: frames never cross to the CPU. From
the first naive browser attempt to today is a **×200-330 speedup**
(1957 ms → 2.6-3.75 ms per frame).

Full story, numbers, model ladder and training instructions:
**[docs/TECHNICAL.md](docs/TECHNICAL.md)**

## License

Code: **AGPLv3** ([LICENSE](LICENSE)). The inference runtime is also
available as a library under **LGPL-3.0** - [`framegen`](packages/rt) -
so it can be embedded in other projects. Model weights: non-commercial
research/personal use ([WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md)).
