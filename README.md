# InterModule — real-time frame interpolation для браузера

Rust-библиотека + браузерное расширение: берёт видео на сайте (аниме 24 fps) и
в реальном времени апсэмплит до 60 fps через RIFE на WebGPU. Ниша свободна —
готовых real-time RIFE-реализаций под браузер (WASM/WebGPU) с anime-режимом нет:
SVP и `rife-ncnn-vulkan` заточены под десктоп-плеер, не под веб.

## Идея

Заходишь на сайт аниме → видео 24 fps → расширение перехватывает кадры через
**WebCodecs**, гоняет RIFE-Lite на **WebGPU**, рисует интерполированные кадры в
**Canvas** поверх `<video>`. На выходе 60 fps плавности, локально, без сервера.

## Почему Rust + RIFE + WebGPU

| | |
|---|---|
| **Rust** | оверхед на обвязке ниже, чем у JS/Python; memory safety; одна кодовая база натив + wasm |
| **RIFE-Lite (RIFEm)** | direct intermediate-flow estimation, поддерживает произвольный timestep; ~30 fps @ 720p/2x на 2080Ti — реальный кандидат в real-time |
| **WebGPU** | единственный способ гонять нейросеть на GPU прямо в браузере без плагинов |
| **WebCodecs** | декод видео в raw-кадры без ffmpeg-in-browser |

Rust даст плюс в оверхеде и безопасности, но **bottleneck не язык, а backend
инференса** — скорость определится эффективностью шейдер-пайплайна и квантизацией
модели, а не обвязкой. «Вдвое быстрее всего существующего» автоматически не
гарантируется — модель ещё оптимизировать под WebGPU придётся.

## Anime-режим (ключевой дифференциатор)

Классический блендинг кадров плохо работает на аниме: контуры слишком резкие,
блендинг создаёт «смазанные» артефакты на статичных линиях. SVP решает это
отдельным «Sharp»/«Animation» режимом — **детектирует только крупное движение**
(панорамирование, зум) и интерполирует именно его, оставляя резкие статичные
объекты нетронутыми. Без этого аниме выглядит «мыльным».

В InterModule это будет отдельный pass поверх RIFE: глобальный motion detection →
interpolate только там, где смещение заметно → статика остаётся пиксель-в-пиксель.

## Архитектура

```
┌─ browser extension ──────────────────────────────┐
│  <video> → WebCodecs decode → raw frames         │
│            → InterModule (Rust/wasm, WebGPU)     │
│              RIFE-Lite forward + anime-mode pass │
│            → Canvas paint (замена video-рендера)  │
└──────────────────────────────────────────────────┘
         ↑ одна кодовая база с нативом
┌─ native CLI (offline) ──────────────────────────┐
│  mp4 → ffmpeg decode → RIFE forward → encode     │
│  валидация корректности и замер fps ДО браузера   │
└──────────────────────────────────────────────────┘
```

## Roadmap

1. **CPU offline** — корректный 2x @ 720p, замер fps. *(текущий этап)*
   - Полный forward-pass RIFE-Lite на candle ✅
   - mp4-харнес (ffmpeg decode → 2x → encode) *(в работе)*
   - Валидация против `inference_img.py` paper-RIFE_m
2. **wasm32 + wasm-pack** — тот же код, что и натив, собирается под wasm.
3. **candle wgpu backend** — цель: стабильный 60 fps на 2x/720p.
4. **Браузерное расширение** — WebCodecs decode → lib interpolate → Canvas paint.
5. **Anime-mode** — global-motion-only интерполяция, статика нетронута.

Принцип roadmap: **сначала «трудная» ML-часть отдельно от «простой» браузерной
обёртки**. CPU-оффлайн отлаживать в разы проще, чем сразу лезть в WebCodecs/wgpu.

## Стек

| Слой | Технология |
|---|---|
| ML inference | Rust + [candle](https://github.com/huggingface/candle) 0.11 (далеко — wgpu backend) |
| Модель | RIFE-Lite / RIFEm (`hzwer/ECCV2022-RIFE`, `model/IFNet_m.py`), fp32 → fp16 |
| Offline decode/encode | ffmpeg subprocess |
| Browser decode | WebCodecs API |
| Browser GPU | WebGPU (через candle `wgpu` feature) |
| Browser render | Canvas / WebGL compositor |
| wasm-экспорт | wasm-bindgen + wasm-pack |

## Что уже работает

- Полный порт `IFNet_m` на candle: IFBlock, Contextnet, Unet, backward warp.
  Воспроизведение `grid_sample(align_corners=True, border)` вручную через `gather`
  (в candle 0.11 нет `grid_sample`).
- `cargo check` / `cargo clippy -D warnings` / `cargo test` / `cargo build --release` — зелёные.
- CLI `rife-interpolate` (пара PNG → интерполированный PNG) + `rife-smoke` (загрузка весов, один forward).
- Конвертер весов `scripts/convert_weights.py` (`flownet.pkl` → `rife_lite.safetensors`).

См. `AGENTS.md` для инструкций сборки и `docs/rife_lite_reference.md` для
PyTorch-референса (ground truth порта).

## Риски (честно)

- **WebGPU ещё не везде стабилен** — в Firefox за флагом, в Safari недавно.
- **60 fps @ 720p/2x на wgpu** — амбициозно; нативный RIFE-HD это не везде тянет,
  потому и выбран Lite-вариант. Если не вытяну — целиться в 30→60 для 480p аниме
  тоже ок для старта.
- **Anime-режим** — эвристика global-motion detection, потребует тюнинга под
  разные стили аниме.
- **Браузерное расширение** — перехват `<video>` + Canvas-composite нетривиален
  на разных сайтах (DRM-видео вообще не отдаст кадры через WebCodecs).
