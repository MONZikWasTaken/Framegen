<div align="center">

<img src="docs/media/logo.png" width="96" alt="Framegen">

# Framegen

**Smooth video in your browser.** An extension that turns 24 to 30 fps video into 60 to 240 fps in real time, with a
neural network running on your GPU. No servers, no accounts, nothing leaves your computer.

[![Chrome Web Store](https://img.shields.io/chrome-web-store/v/hpdpcjakhclhljfdkpjolonjlopbdhfk?label=chrome%20web%20store&color=19c37d)](https://chromewebstore.google.com/detail/framegen-frame-interpolat/hpdpcjakhclhljfdkpjolonjlopbdhfk)
[![License](https://img.shields.io/badge/code-MIT-blue)](LICENSE)
[![Built on FlyHigh](https://img.shields.io/badge/built%20on-FlyHigh%20engine-7c5cff)](https://github.com/getflyhigh/flyhigh-engine)
[![Ko-fi](https://img.shields.io/badge/support-ko--fi-ff5e5b)](https://ko-fi.com/monzikxd)

<a href="https://github.com/MONZikWasTaken/Framegen/releases/download/v1.0.0/framegen-live-demo.mp4">
  <img src="docs/media/hero.gif" width="880" alt="15 fps source vs Framegen x4 interpolation, side by side">
</a>

**[Watch the original 50-second comparison video](https://github.com/MONZikWasTaken/Framegen/releases/download/v1.0.0/framegen-live-demo.mp4)**

*Legacy Framegen demo showing the compare slider on YouTube. Footage: Sintel © Blender Foundation (CC BY).*

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

Download the archive for your browser from the [latest release](https://github.com/MONZikWasTaken/Framegen/releases/latest).

- **Chrome and Edge** — extract `framegen-chrome.zip` to a folder. Open `chrome://extensions` or `edge://extensions`,
  enable **Developer mode**, click **Load unpacked**, and select the folder containing `manifest.json`.
- **Firefox** — open `about:debugging#/runtime/this-firefox`, click **Load Temporary Add-on**, and select
  `framegen-firefox.zip`.

Requirements: **Chrome or Edge 121+** (Windows, macOS with Apple Silicon, Linux) or **Firefox 142+ on Windows**, and
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
| **[Picture mode](#picture-modes)** | A color and detail look on top of the video: Cinema, Cinema Home, Noir, Vivid, Sports, Detail Boost. `Original` changes nothing |
| **Upscale** | 2x neural upscaling of every presented frame when the player is larger than the video |
| **Static protection** | Pixels that did not change over the last frames (burnt-in subtitles, a logo) are taken from the source |
| **Compare** | The split slider |

Profiles on the full settings page keep any number of setups; existing settings migrate without losing values.

**Good first test:** anything shot at 24 fps: a film scene with a slow pan, an anime opening. On a 60 Hz screen you
see 24 to 60; on a 144 to 240 Hz screen, considerably more.

## Picture modes

Compare the pink petals, green background, and dark wing lines across seven modes. The sample grades and sharpening
are intentionally stronger where needed so the differences show at thumbnail size. Framegen's in-player presets
are unchanged. Click any tile to see it full size.

| Original | Cinema | Cinema Home |
| :---: | :---: | :---: |
| [![Original monarch butterfly on a pink zinnia](docs/media/picture-modes/original.webp)](docs/media/picture-modes/original.webp) | [![Cinema warms the wing and softens the flower colors](docs/media/picture-modes/cinema.webp)](docs/media/picture-modes/cinema.webp) | [![Cinema Home brightens the background and lifts wing shadows](docs/media/picture-modes/cinema-home.webp)](docs/media/picture-modes/cinema-home.webp) |
| Unchanged reference | Warmer, more subdued color | Warmer and brighter midtones |
| **Noir** | **Vivid** | **Sports** |
| [![Noir butterfly and zinnia in black and white](docs/media/picture-modes/noir.webp)](docs/media/picture-modes/noir.webp) | [![Vivid brings out the orange wing and pink petals](docs/media/picture-modes/vivid.webp)](docs/media/picture-modes/vivid.webp) | [![Sports brightens colors and sharpens the wing and petals](docs/media/picture-modes/sports.webp)](docs/media/picture-modes/sports.webp) |
| Black and white with lifted shadows | Stronger saturation and contrast | Brighter color and crisper edges |
| **Detail Boost** | **Original close-up** | **Detail Boost close-up** |
| [![Detail Boost sharpens the wing without changing the color grade](docs/media/picture-modes/detail.webp)](docs/media/picture-modes/detail.webp) | [![Original: magnified wing veins and scales](docs/media/picture-modes/original-closeup.webp)](docs/media/picture-modes/original-closeup.webp) | [![Detail Boost: the same wing crop with more defined veins and scales](docs/media/picture-modes/detail-closeup.webp)](docs/media/picture-modes/detail-closeup.webp) |

The last two tiles show the same wing crop at 2× magnification to make the sharpening visible. Neural upscaling
is not shown here.

Photo: [Monarch butterfly on a zinnia](https://commons.wikimedia.org/wiki/File:Monarch_Butterfly_(20210717-FPAC-PJK-2022).jpg)
by Preston Keres, USDA/FPAC (public domain). Cropped to a square and graded with Framegen.

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
pwsh tools/build_extension.ps1                      # validates every payload, writes Chrome and Firefox ZIPs
pwsh tools/build_extension.ps1 -EngineRepo <path>   # also refreshes FlyHigh's browser module and model
```

`extension/` is directly loadable (Load unpacked). [docs/architecture.md](docs/architecture.md) explains how the host
maps onto the engine. The previous engine stays in
`web/rt` and on npm as [`framegen`](https://www.npmjs.com/package/framegen) ([packages/rt](packages/rt)); its notes are
in [docs/TECHNICAL.md](docs/TECHNICAL.md). New integrations should use the engine directly.

### Translating Framegen

Framegen follows the browser interface language by default and falls back to English. The **Interface language**
setting overrides that choice for the settings page, toolbar popup, and player controls. Technical diagnostics in the
player use English in every locale.

To add a translation, copy [the English catalog](extension/_locales/en/messages.json) to
`extension/_locales/<locale>/messages.json` using a WebExtensions locale code, then translate the `message` values.
Keep the keys and placeholders such as `{fps}` and `{name}`. Add the locale and its native language name to
`LANGUAGES` in [i18n.js](extension/src/i18n.js). Run the tests and open the unpacked extension to review the translation.

## Support the project

Framegen is built by one person with one mid-range GPU. The extension is free and will stay free; the models behind it
are trained on rented cloud GPUs paid out of pocket. If Framegen made your video smoother:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/monzikxd)

## License

Code: **MIT** ([LICENSE](LICENSE)). Model weights: the FlyHighFG weights under
[CC BY-NC 4.0](extension/engine/WEIGHTS_LICENSE.md); the previous engine's weights for personal and research use
([WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md)).
