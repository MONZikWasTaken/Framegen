<div align="center">

<img src="docs/media/logo.png" width="96" alt="Framegen">

# Framegen

**Smooth video in your browser.** An extension that turns 24 to 30 fps video into 60 to 240 fps in real time, with a
neural network running on your GPU. No servers, no accounts, nothing leaves your computer.

[![Chrome Web Store](https://img.shields.io/chrome-web-store/v/hpdpcjakhclhljfdkpjolonjlopbdhfk?label=chrome%20web%20store&color=19c37d)](https://chromewebstore.google.com/detail/framegen-frame-interpolat/hpdpcjakhclhljfdkpjolonjlopbdhfk)
[![License](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Built on FlyHigh](https://img.shields.io/badge/built%20on-FlyHigh%20engine-7c5cff)](https://github.com/getflyhigh/flyhigh-engine)
[![Ko-fi](https://img.shields.io/badge/support-ko--fi-ff5e5b)](https://ko-fi.com/monzikxd)

<img src="docs/media/hero.gif" width="880" alt="15 fps source vs Framegen x4 interpolation, side by side">

</div>

## What it does

- **2x to 6x more frames** on any `<video>`: movies, series, sports, anime, screen recordings; YouTube and most video
  sites
- **Auto mode** picks the highest factor your GPU and display sustain and backs off before you see a stutter
- **Anime mode** detects drawings held on twos and threes and interpolates the real motion; the optional cadence
  smoothing spreads each drawing step evenly over the frames it is held for
- **Match display and Custom FPS** place the output on an exact grid at your monitor's rate or any rate you ask for
- **Compare slider**: original on the left, Framegen on the right
- **Private by construction**: the whole pipeline runs on your GPU; nothing is collected

Since 2.0 the generation runs on the [FlyHigh engine](https://github.com/getflyhigh/flyhigh-engine): the FlyHighFG
network (0.78 M parameters) and its WGSL kernels compiled to WebAssembly, with the engine's Pacer deciding what to show
on every display tick. A generated frame costs about a millisecond on a mid-range GPU.

## Install

**[Add to Chrome from the Web Store](https://chromewebstore.google.com/detail/framegen-frame-interpolat/hpdpcjakhclhljfdkpjolonjlopbdhfk)**.

Manual install: download `framegen-extension.zip` from the
[latest release](https://github.com/MONZikWasTaken/Framegen/releases/latest), extract it, open `chrome://extensions`,
enable **Developer mode**, click **Load unpacked** and select the extracted folder. Firefox: `framegen-firefox.zip`
from the same release, loaded through `about:debugging` until it is listed on AMO.

Requirements: **Chrome or Edge 121+** (Windows, macOS with Apple Silicon, Linux) or **Firefox 141+ on Windows**, and
a GPU with WebGPU support. GPUs without `shader-f16` run the engine's f32 kernels, slower but the same picture.

## How to use

1. Open any video and hover over it: the **FG** button appears at the left edge of the player.
2. Click it. The button turns green, the fps readout appears, and the video is now interpolated. Click again to turn
   it off.
3. The two buttons under it open the panels inside the player: **Picture** (looks, sharpness, upscale, HDR) and
   the quick settings. **Advanced settings** there or in the extension popup opens the full page with profiles.

| Setting | What it does |
|---|---|
| **Output rate** | `Auto` fits your GPU and display, with an optional FPS limit. `Match display` and `Custom FPS` pace on an exact grid; fixed 2x to 6x multiply the source. **Limit to display refresh rate** (on) makes the screen a ceiling; off, the GPU generates as many frames as its budget allows. **Motion-adaptive Auto** (on) generates fewer frames on fast scenes |
| **Quality** | The size the network runs at. `Balanced` is the model's native scale; higher costs more GPU time and rarely looks better |
| **Anime mode** | Held-frame detection and the aggressive periodic-structure repair; keep on for anime |
| **Anime cadence smoothing** | Experimental, off by default. Drawings on twos slide instead of stepping; every frame becomes a generated in-between, so big steps look softer; one source frame of lookahead |
| **Picture mode** | A color and detail look on top of the video: Cinema, Cinema Home, Noir, Vivid, Sports, Detail Boost. `Original` changes nothing |
| **Upscale** | 2x neural upscaling of every presented frame when the player is larger than the video |
| **Static protection** | Pixels that did not change over the last frames (burnt-in subtitles, a logo) are taken from the source |
| **Compare** | The split slider |

Profiles on the full settings page keep any number of setups; existing settings migrate without losing values.

**Good first test:** anything shot at 24 fps: a film scene with a slow pan, an anime opening. On a 60 Hz screen you
see 24 to 60; on a 144 to 240 Hz screen, considerably more.

## FAQ

**It says "no video found" / the button does not appear.** Make sure the video is playing. On some players the button
appears only when the mouse is over the video itself.

**Does it work on Netflix or Crunchyroll?** No: DRM-protected video is invisible to extensions by design; the browser
hands over black frames. YouTube and most other sites work.

**The fps counter shows less than the factor I chose.** Auto adapts to the GPU's real headroom and never stutters to
hit a number; a fixed factor is capped the same way. Lower the quality or the factor if you want more.

**The panel says the site forbids WebAssembly.** A few sites (reddit.com among the ones checked) send a security
policy that Chrome also applies to extensions' WebAssembly. Framegen cannot run there; Firefox is not affected.

**Does it phone home?** No. There is no server, no telemetry, no analytics; the code is here to check.

## Development

```pwsh
node --test tools/*.test.mjs
pwsh tools/build_extension.ps1                      # validates every payload, writes the three release zips
pwsh tools/build_extension.ps1 -EngineRepo <path>   # also refreshes extension/engine from an engine checkout
```

`extension/` is directly loadable (Load unpacked). [docs/architecture.md](docs/architecture.md) explains how the host
maps onto the engine. The previous engine stays in
`web/rt` and on npm as [`framegen`](https://www.npmjs.com/package/framegen) ([packages/rt](packages/rt)); its notes are
in [docs/TECHNICAL.md](docs/TECHNICAL.md). New integrations should use the engine directly.

## Contributors

- [@NimaJafariComp](https://github.com/NimaJafariComp): picture modes
  ([#25](https://github.com/MONZikWasTaken/Framegen/pull/25), in 2.0.10).

Pull requests are welcome against the `dev` branch; [docs/architecture.md](docs/architecture.md) explains how the code is laid out.

## Support the project

Framegen is built by one person with one mid-range GPU. The extension is free and will stay free; the models behind it
are trained on rented cloud GPUs paid out of pocket. If Framegen made your video smoother:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/monzikxd)

## License

Code: **MIT** ([LICENSE](LICENSE)). Model weights: the FlyHighFG weights under
[CC BY-NC 4.0](extension/engine/WEIGHTS_LICENSE.md); the previous engine's weights for personal and research use
([WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md)).
