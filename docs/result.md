# InterModule — Финальный отчёт

## Проект: Real-time RIFE-Lite интерполяция кадров на Rust + GPU

**Дата:** 1 июля 2026
**GPU:** NVIDIA GeForce RTX 4060 Ti (8 GB VRAM, compute capability 8.9, Ada Lovelace)
**CUDA:** 13.1 (nvcc 13.1.80)
**Rust:** 1.90.0, candle-core/candle-nn 0.11.0
**TensorRT:** 10.13.3.9
**OS:** Windows 11, MSVC 14.44

---

## 1. Что было сделано (хронология)

### Этап 1: Порт RIFE-Lite (RIFEm) на Rust/candle

**Исходник:** `hzwer/ECCV2022-RIFE`, ветка `main`, файл `model/IFNet_m.py` — lite-вариант
RIFE (arbitrary=True path в `model.RIFE.Model`).

**Веса:** RIFE_m paper checkpoint — `flownet.pkl` (42 MB), Google Drive ID:
`147XVsDXBfJPlyct2jfo9kpbL944mNeZr`. Конвертированы в `rife_lite.safetensors` (190 тензоров,
41 MB, fp32) через `scripts/convert_weights.py`. Ключи PyTorch state_dict сохранены 1-в-1
(после stripping `module.` префикса DataParallel).

**Архитектура IFNet_m (пере-реализована в `src/model.rs`, 325 строк):**
- `IFBlock`: conv0 (2× stride-2 downsample /4), convblock (8 residual convs),
  lastconv (ConvTranspose2d upsample /2). Выход: 4-ch flow + 1-ch mask.
- `IFNet_m`: 3 IFBlock'а (block0 c=240, block1 c=150, block2 c=90) + block_tea (неиспользуется
  при inference, нужен gt) + Contextnet (c=16, 4 уровня Conv2) + Unet (c=16, 4 down + 4 up).
- Backward warp через `grid_sample(align_corners=True, padding_mode='border')` — порт
  `model/warplayer.py`.
- Forward pass: 3 scale iterations (4, 2, 1) → flow refinement → contextnet+unet residual.

**Ключевые candle 0.11 API-заметки:**
- `grid_sample` в candle **НЕТ** — пришлось реализовывать backward warp вручную.
- `PReLU` — есть (`candle_nn::prelu`), per-channel, ключ `.weight`, точно повторяет PyTorch.
- `Conv2d` / `ConvTranspose2d` — есть, ключи `.weight` / `.bias` совпадают с PyTorch.
- `upsample_bilinear2d(h, w, align_corners)` — есть, совпадает с PyTorch.
- `upsample_bilinear2d_with_scale(scale_h, scale_w, align_corners)` — есть, для scale-factor mode.
- `gather` — есть, но требует contiguous тензоров (`.contiguous()` обязателен после broadcast).
- `slice_set` — есть, для padding (вместо `F.pad`), требует contiguous src и dst.
- `VarBuilder::from_mmaped_safetensors` — `unsafe`, грузит веса через mmap.
- `Device::synchronize()` — критично для честного замера GPU-времени (без неё команды только
  ставятся в очередь, возвращая мгновенно).

**Верификация корректности:**
- PSNR 88.31 dB vs PyTorch `inference_img.py` на демо-кадрах (256×448, timestep=0.5).
- 99.99% пикселей идентичны, оставшиеся 33 пикселя (0.01%) отличаются на 1 (шум uint8 округления).
- Важные детали для паритета: BGR (модель обучена на cv2 BGR), /32 padding, truncation
  (`.byte()` в PyTorch = `as u8` в Rust, не `.round()`).

### Этап 2: mp4 I/O харнес

`src/io/video.rs` — ffmpeg subprocess pipeline:
- decode: `ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 -` → stdout
- encode: `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r fps -i - -c:v libx264 ... output.mp4`
- Без temp-файлов — стриминг через pipes, в памяти только 2 кадра + промежуточный.
- Поддержка `--times N` (2x, 4x, ...) и `--scale 0.5` (lower-res processing).

### Этап 3: CUDA GPU backend (candle)

**Сборка:** `cargo build --release --features cuda` (нужен vcvars64.bat в PATH для nvcc).
`.cargo/config.toml` перенаправляет `target-dir` на E: (диск C: был полон).

**Первая проблема:** `Tensor::full(0f32, ...)` + `Tensor::cat` для padding падало на CUDA
с `CUDA_ERROR_INVALID_VALUE` для размеров, не кратных 32. Исправлено заменой на
`Tensor::zeros` + `slice_set`.

**Вторая проблема (критическая!):** Замеры без `Device::synchronize()` были фантомными.
`rife.interpolate()` ставит CUDA-команды в очередь за ~19мс, но реальное выполнение занимает
~1175мс на 1080p. Все первоначальные "16.6мс = 60fps" замеры были **неправильными**.
После добавления `synchronize()` реальные цифры: 1069мс на 1080p, ~371мс на 720p.

### Этап 4: Оптимизации candle (по убыванию эффекта)

#### 4.1. Fused CUDA warp kernel (`src/warp.rs`)

**Проблема:** Оригинальный backward warp через `gather` — 4 gather-операции с broadcast-
индексами + dtype-конверсии на каждый из 14 вызовов warp в модели (6 в IFNet + 8 в Contextnet).
Gather на GPU = случайный доступ к памяти = катастрофически медленно.

**Решение:** Custom `CustomOp2` с одним fused CUDA-кернелом `backward_warp_bilinear`.
Кернел компилируется runtime через NVRTC (`cudarc::nvrtc::safe::compile_ptx_with_opts`),
загружается через `CudaDevice::get_or_load_custom_func`.

Кернел (см. `WARP_CUDA` const в `warp.rs`):
- Один thread на элемент output тензора.
- Bilinear interpolation с border padding (как `grid_sample(padding_mode='border')`).
- align_corners=True: flow normalized by (W-1)/2, (H-1)/2.

**Результат:** 1069мс → 691мс на 1080p (1.55× ускорение).
**Корректность:** PSNR 88.73 dB (даже чуть выше — нет шума от dtype-конверсий gather).

**API-заметки для candle CustomOp2 на CUDA:**
- `candle_core::cuda::{cudarc, WrapErr}` — правильный путь импорта.
- `CudaStorage::as_cuda_slice::<f32>()` — получить slice.
- `CudaDevice::alloc_uninit(&shape, dtype)` — unsafe, выделение output.
- `dev.get_or_load_custom_func(name, module_name, &ptx_src)` — кеш кернелов.
- `Ptx::to_src()` — получить строку PTX из скомпилированного Ptx объекта.
- `dev.cuda_stream().launch_builder(&func)` — builder pattern для аргументов.
- **Важно:** временные значения (int casts) нужно биндить к `let`, иначе borrow checker.
- `builder.arg(&cuda_view)` — передаёт device pointer, не `&cuda_slice.slice` (метод, не поле).

#### 4.2. fp16 half-precision

**Реализация:** `RifeLite::load(path, dtype, device)` — `DType::F16` для CUDA.
Веса грузятся сразу в fp16 через `VarBuilder::from_mmaped_safetensors(&[path], DType::F16, device)`.
`interpolate()` кастит входные тензоры к model dtype, warp кастит к f32 внутри (кернел f32-only),
результат кастится обратно к f32 для вывода.

**Проблема:** `Tensor::full(timestep as f32, ...)` в model forward — hardcoded f32.
Исправлено: `.to_dtype(self.dtype)`.

**Результат:** 691мс → 566мс на 1080p (1.22× ускорение).
**Почему так мало:** RTX 4060 Ti — memory-bandwidth bound на этих размерах свёрток, не
compute-bound. Tensor cores для fp16 не дают 2× потому что bottleneck — чтение весов из VRAM,
а не MACs. PSNR упал до 56.99 dB (всё ещё >35dB порог, 0 пикселей с diff>1).

#### 4.3. GPU-side download optimization

**Проблема:** `tensor_to_rgb24` делал `to_vec3()` (pull f32 на CPU) + по-пиксельный цикл
для конверсии BGR→RGB + *255. Это занимало ~202мс на 1080p.

**Решение:** Все операции (clamp, affine *255, BGR→RGB через narrow+cat, cast to U8)
на GPU через candle ops. Скачивается только u8 буфер.

**Результат:** 202мс → 5.7мс (35× ускорение transfer).

#### 4.4. Lower-res inference (`--scale 0.5`)

**Реализация:** `RifeLite::interpolate_scaled(img0, img1, timestep, scale)`:
1. Resize input to `proc_h = H*scale, proc_w = W*scale` через `upsample_bilinear2d`.
2. Pad to /32.
3. Adjusted scale_list: `[4/scale, 2/scale, 1/scale]` (как `inference_img.py --scale`).
4. Forward.
5. Crop + upscale back to original resolution.

**Результат:** 566мс → 265мс на 1080p (2.13× ускорение).
**Качество:** PSNR 28.59 dB на 256×448 (маленькое изображение, scale=0.5 слишком агрессивен
для маленьких разрешений). На 1080p→540p потеря менее заметна, но всё ещё артефактная.

#### 4.5. Сводная таблица candle оптимизаций

| Шаг | 1080p | 720p | PSNR |
|---|---|---|---|
| Baseline (gather warp, fp32, async) | 1069мс | 371мс | 88.31 dB |
| + fused CUDA kernel | 691мс | 340мс | 88.73 dB |
| + fp16 | 566мс | 320мс | 56.99 dB |
| + GPU download | ~570мс | ~320мс | — |
| + scale=0.5 | 265мс | 213мс | ~28 dB |
| **Всего ускорение** | **4×** | **1.7×** | — |

**Реальная скорость video pipeline (wall time / frames):**
- 1080p scale=1.0: 664мс/кадр (1.5 fps)
- 1080p scale=0.5: 329мс/кадр (3.0 fps)
- 720p scale=0.5: ~225мс/кадр (4.4 fps)

### Этап 5: cuDNN — НЕУДАЧА

**Проблема:** candle 0.11 без `cudnn` feature использует generic CUDA kernels для свёрток
(не cuDNN Winograd/GEMM). Это в 5-10× медленнее.

**Попытка 1:** pip install nvidia-cudnn-cu12 (cuDNN 9.23.2.1). Скачал, скопировал DLL+headers
в CUDA toolkit dir. Создал import library через `lib /DEF:cudnn_full.def /OUT:cudnn.lib`.
Также понадобился `nvidia-cublas-cu12` (cublasLt64_12.dll, cublas64_12.dll).

**Результат:** Сборка прошла, но cuDNN 9 (cu12 variant) **медленнее** чем без неё на CUDA 13:
- 1080p scale=0.5: 406мс с cuDNN vs 262мс без cuDNN
- 720p scale=0.5: 364мс с cuDNN vs 213мс без cuDNN

**Причина:** Версионный конфликт. cuDNN 9 из pip-пакета `nvidia-cudnn-cu12` скомпилирована
для CUDA 12, а у нас CUDA 13. Бинарная несовместимость или suboptimal code path.

**Вывод:** Нужна родная cuDNN для CUDA 13 с сайта NVIDIA (нужен логин).
Или дождаться `nvidia-cudnn-cu13` pip-пакета. Пока — отключено (`--features cuda` без `cudnn`).

### Этап 6: TensorRT — УСПЕХ

**Подход:** Вместо candle inference — экспорт модели в ONNX → TensorRT engine.

#### 6.1. ONNX экспорт

**Скрипт:** `scripts/export_onnx.py`
- PyTorch model → `torch.onnx.export(dynamo=True, opset_version=20)`
- `dynamo=True` обязателен — `torch.jit.trace` падает из-за dynamic кода в `warplayer.py`
  (backwarp_tenGrid кэш + grid_sample).
- После экспорта нужно инлайнить external data: `onnx.save_model(m, path, save_as_external_data=False)`
  (dynamo экспорт создаёт external data файл `*.onnx.data`, TensorRT не умеет его читать).
- **Важно:** `onnxscript` pip-пакет нужен для dynamo экспорта.

**Результат:** 316 ONNX узлов, 197 initializers, 40.8 MB весов.
- Два engine'а: 720p (736×1280, padded) и 1080p (1088×1920, padded).
- Static shapes (dynamic_axes=None) — для максимальной TensorRT оптимизации.

#### 6.2. TensorRT engine build

**Скрипт:** `scripts/build_trt_engine.py`
- `trt.Builder` + `OnnxParser` + `BuilderFlag.FP16`
- Workspace: 4 GB (`MemoryPoolType.WORKSPACE`)
- Engine размер: 24.1 MB (720p), 23.5 MB (1080p) — веса в fp16.

**Попытались INT8:** без calibration dataset TensorRT падает (`engine build failed`).
Нужен `IInt8Calibrator` с representative данными — не сделали.

**Попытались optimization level 5:** `config.builder_optimization_level = 5` — таймаут 15 минут.
Остановились на optimization level 3 (default).

#### 6.3. TensorRT benchmark

**Скрипт:** `scripts/bench_trt.py` (pycuda, синхронизированный)

| Разрешение | p50 | p10 | FPS | Real-time 48fps (2x@24) | Real-time 60fps |
|---|---|---|---|---|---|
| 720p (736×1280) | **20.8мс** | 18.8мс | **48.1** | **PASS** (< 41.7мс) | FAIL (нужно 16.6мс) |
| 1080p (1088×1920) | **51.7мс** | 48.6мс | **19.4** | FAIL (нужно 41.7мс) | FAIL |

**720p — real-time PASS для 2x интерполяции 24fps аниме!** (нужно 41.7мс, имеем 20.8мс).
p10 = 18.8мс — почти 60fps при благоприятных условиях.

1080p — 51.7мс, чуть выше бюджета. С INT8 или optimization level 5 может пройти.

#### 6.4. Rust интеграция TensorRT

**Подход:** Python subprocess server (`scripts/trt_server.py`):
- Rust запускает `python trt_server.py engine.engine <native_h> <native_w>` как subprocess.
- Python грузит engine, создаёт execution context.
- Protocol: Rust пишет raw RGB24 пары кадров в stdin, Python возвращает интерполированные.
- Rust ffmpeg decode → stdin → TRT inference → stdout → ffmpeg encode.

**Проблема (была) — ДВА бага, оба исправлены:**

1. **Deadlock на больших кадрах.** Rust писал обе рамки в stdin сервера и только потом читал
   stdout — при кадрах в 2.8–6.3 МБ (>> размера pipe-буфера) writer и reader блокировали друг
   друга. Исправлено: отдельный writer-поток пишет в stdin через `mpsc::sync_channel`, main-поток
   читает stdout (`src/io/video_trt.rs`).

2. **Рассинхрон по размеру кадра (главная причина зависания на реальном видео).** Engine
   принимает вход, дополненный до /32 (720p → **736×1280**), а Rust слал кадры в нативном
   разрешении (720×1280) и только *ресайзил выход* (NN) — сервер ждал недостающие байты и
   намертво блокировался. Исправлено: `trt_server.py` теперь работает в нативном разрешении —
   принимает нативные кадры, сам делает zero-pad до размера engine (bottom/right, как
   `F.pad` в `compare_pytorch.py`), инференс, crop выхода обратно (top-left) — и отдаёт нативные
   кадры. Rust передаёт нативные H W серверу и убрал ошибочный NN-ресайз выхода.

**Статус: РАБОТАЕТ.** End-to-end проверено на `demo/test_720p.mp4` (1280×720, 72 кадра):
72 in → 143 out (2×, 48 fps), exit 0, без зависаний. Pass-through кадры совпадают с источником
(MAE ~0.6 = шум h264), интерполированные — корректный midpoint. Wall-time ~11.6 fps на 720p
(включая Python subprocess + ffmpeg overhead; чистый TRT-инференс остаётся 20.8мс = 48 fps).
Ограничение: `--times` только 2 (engine с фиксированным timestep=0.5).

---

## 2. Структура проекта

```
InterModule/
├── Cargo.toml              # candle 0.11, features: cuda, cudnn (broken), bin
├── .cargo/config.toml      # target-dir = E:\cargo-target\InterModule (C: full)
├── AGENTS.md               # build/lint/test commands
├── README.md               # project vision (anime 24→60fps in browser)
├── result.md               # этот файл
├── docs/
│   └── rife_lite_reference.md  # PyTorch reference (ground truth for port)
├── scripts/
│   ├── convert_weights.py     # flownet.pkl → rife_lite.safetensors
│   ├── compare_pytorch.py     # ground truth generator (PyTorch inference_img)
│   ├── psnr_compare.py        # PSNR Rust vs PyTorch
│   ├── export_onnx.py         # PyTorch → ONNX (dynamo, opset 20)
│   ├── build_trt_engine.py    # ONNX → TensorRT engine (fp16)
│   ├── build_trt_best.py      # ONNX → TensorRT (opt level 5, tactics) — не завершён
│   ├── build_trt_int8.py      # INT8 — не работает без calibrator
│   ├── bench_trt.py            # TensorRT benchmark (pycuda)
│   └── trt_server.py           # Python inference server для Rust
├── src/
│   ├── lib.rs               # RifeLite API: load(), interpolate(), interpolate_scaled()
│   ├── model.rs             # IFNet_m reimplementation (325 строк)
│   ├── warp.rs              # CustomOp2 backward warp (fused CUDA kernel + CPU fallback)
│   ├── imgutil.rs           # image/tensor conversion (BGR, GPU-side)
│   ├── io/
│   │   ├── video.rs         # candle-based video pipeline (ffmpeg pipes)
│   │   └── video_trt.rs     # TensorRT video pipeline (Python subprocess)
│   └── bin/
│       ├── interpolate.rs   # CLI: img/video/trt subcommands
│       ├── smoke.rs          # load weights + one forward
│       └── profile.rs       # per-component timing (with synchronize)
├── weights/
│   ├── rife_lite.safetensors       # 190 tensors, 41 MB, fp32
│   ├── rife_lite_manifest.json     # key→shape manifest
│   ├── rife_lite_trt_fp16.engine  # TensorRT 720p engine, 24.1 MB
│   ├── rife_lite_1080p_trt_fp16.engine  # TensorRT 1080p engine, 23.5 MB
│   └── rife_lite_1080x1920.onnx    # ONNX model, 40.2 MB
└── demo/
    ├── I0_0.png, I0_1.png          # RIFE demo frames
    ├── mid_pytorch.png/rgb         # PyTorch ground truth
    ├── mid_rust.png                # Rust fp32 output (PSNR 88.31)
    ├── mid_rust_cuda.png           # Rust CUDA fp32 output (PSNR 88.73)
    ├── mid_rust_fp16.png           # Rust CUDA fp16 output (PSNR 56.99)
    ├── test_10s.mp4                # 10-sec 1080p 24fps anime clip
    └── test_10s_2x*.mp4            # various 2x outputs
```

---

## 3. Полная таблица результатов

### Корректность (vs PyTorch inference_img.py, 256×448, timestep=0.5)

| Конфигурация | PSNR | diff>1 pixels | Статус |
|---|---|---|---|
| Rust CPU, gather warp, fp32 | 88.31 dB | 33/344064 (0.01%) | ✅ PASS |
| Rust CUDA, fused kernel, fp32 | 88.73 dB | 0/344064 (0.0%) | ✅ PASS |
| Rust CUDA, fused kernel, fp16 | 56.99 dB | 0/344064 (0.0%) | ✅ PASS |
| Rust CUDA, fused kernel, fp16, scale=0.5 | 28.59 dB | 193018/344064 (56%) | ⚠️ low res |

### Скорость (synchronized, real GPU time)

| Backend | Resolution | scale | Time/frame | FPS | Real-time 48fps? |
|---|---|---|---|---|---|
| candle CPU | 720p | 1.0 | 4889мс | 0.2 | ❌ |
| candle CPU | 1080p | 1.0 | ~11000мс | 0.09 | ❌ |
| candle CUDA fp32 | 1080p | 1.0 | 1069мс | 0.9 | ❌ |
| candle CUDA fp32 + fused warp | 1080p | 1.0 | 691мс | 1.4 | ❌ |
| candle CUDA fp16 + fused warp | 1080p | 1.0 | 566мс | 1.8 | ❌ |
| candle CUDA fp16 + fused warp + GPU download | 1080p | 1.0 | ~570мс | 1.8 | ❌ |
| candle CUDA fp16 + all + scale=0.5 | 1080p | 0.5 | 265мс | 3.8 | ❌ |
| candle CUDA fp16 + all + scale=0.5 | 720p | 0.5 | 213мс | 4.7 | ❌ |
| candle CUDA + cuDNN 9 (cu12) | 1080p | 0.5 | 406мс | 2.5 | ❌ (ХУЖЕ!) |
| **TensorRT fp16** | **720p** | **1.0** | **20.8мс** | **48.1** | **✅ PASS** |
| TensorRT fp16 | 1080p | 1.0 | 51.7мс | 19.4 | ❌ (близко) |

### Video pipeline (wall time, включая ffmpeg decode/encode)

| Backend | Resolution | scale | Wall time | Frames | Effective FPS |
|---|---|---|---|---|---|
| candle CPU | 720p | 1.0 | 353с | 71 inter | 0.2 |
| candle CUDA fp32 | 720p | 1.0 | 26с | 71 inter | 2.7 |
| candle CUDA fp16 | 1080p | 0.5 | 79с | 239 inter | 3.0 |
| TensorRT (pipeline) | 720p | 1.0 | — | — | зависает (I/O bug) |

---

## 4. Ключевые технические наблюдения

### 4.1. Async CUDA — главная ловушка замеров

candle ставит CUDA-команды в очередь асинхронно. `rife.interpolate()` возвращает Tensor за ~19мс,
но реальное GPU-время — ~1175мс на 1080p. **Без `Device::synchronize()` все замеры бессмысленны.**
Это заняло время чтобы понять — первоначальные "60fps достигнуты" были фантомными.

**Урок:** Всегда вызывай `dev.synchronize()` перед `Instant::elapsed()` при GPU-замерах.
Профайлер `src/bin/profile.rs` делает это правильно.

### 4.2. gather на CUDA — катастрофически медленно

candle's `gather` op на CUDA делает случайный доступ к памяти. Мой оригинальный backward warp
через gather (4 gather + broadcast + contiguous + dtype conversion) был главным bottleneck'ом.
Fused custom kernel дал 1.55× — но это было только начало проблемы.

**Урок:** Для операций типа grid_sample/warp — всегда пиши fused CUDA kernel.
candle's `CustomOp2` trait + NVRTC runtime compilation — рабочий путь.

### 4.3. fp16 не дал 2× — memory bandwidth bound

Ожидали 2× ускорение от fp16 (tensor cores), получили 1.22×. Причина: свёртки RIFE-Lite
на RTX 4060 Ti — **memory-bandwidth bound**, не compute-bound. Чтение весов из VRAM — bottleneck,
не MACs. fp16 уменьшает размер весов в 2×, но pipeline overhead съедает большую часть выгоды.

**Урок:** Tensor cores помогают только на compute-bound задачах (большие batch, большие
channel counts). Для inference batch=1 — bandwidth лимитирует.

### 4.4. cuDNN версионный конфликт

cuDNN 9 из `nvidia-cudnn-cu12` pip-пакета **медленнее** чем candle's generic kernels
на CUDA 13. Это несовместимость cu12 ↔ CUDA 13. cuDNN пытается использовать suboptimal
code paths или fallback'ит.

**Урок:** pip-пакеты NVIDIA привязаны к major version CUDA. cu12 ≠ cu13.
Нужна родная cuDNN с NVIDIA Developer (ZIP archive, matching CUDA version).

### 4.5. TensorRT — единственный путь к real-time

TensorRT дал **10× ускорение** над candle (1069мс → 20.8мс на 720p).
Fused kernels + auto-tuning + fp16 tensor cores + memory layout optimization.
Это тот же движок, что PyTorch использует с `torch.compile` + `torch_tensorrt`.

**Урок:** candle — отличный фреймворк для прототипа и portability, но не для production
inference speed. TensorRT — единственный путь к real-time на NVIDIA GPU.

### 4.6. ONNX dynamo export нюансы

- `torch.jit.trace` не работает с RIFE из-за dynamic кода в `warplayer.py` (grid cache).
- `torch.onnx.export(dynamo=True, opset_version=20)` работает, но создаёт external data.
- Нужно инлайнить: `onnx.save_model(m, path, save_as_external_data=False)`.
- `onnxscript` pip-пакет обязателен для dynamo.

### 4.7. TensorRT INT8 без calibrator не работает

Просто `config.set_flag(trt.BuilderFlag.INT8)` без `IInt8Calibrator` → `engine build failed`.
Для INT8 нужен representative dataset (~500 кадров) для calibration. Это потенциально даст
ещё 2-3× ускорения, но не было сделано.

### 4.8. Disk space — скрытый blocker

Диск C: был заполнен на 99.9% (0.5 GB free). cargo build падал с "no space on device".
Решение: `.cargo/config.toml` с `target-dir = "E:\cargo-target\InterModule"`.
Также `tokenizers` — обязательная (не через фичу) зависимость candle-core на non-wasm,
~500MB при сборке, сбросить нельзя.

---

## 5. Что НЕ было сделано / нерешённые проблемы

### 5.1. TensorRT Rust integration — pipeline I/O deadlock — ✅ ИСПРАВЛЕНО

`src/io/video_trt.rs` + `scripts/trt_server.py` — pipeline зависал при запуске на реальном
видео. **Настоящие причины (не «Python не успевает»):** (1) Rust писал обе рамки в stdin и
только потом читал stdout → deadlock на кадрах >> pipe-буфера; (2) рассинхрон по размеру
кадра — engine ждёт padded /32 вход (736×1280), Rust слал нативные 720×1280. Оба исправлены
(см. §6.4). End-to-end работает.

**Ещё возможные улучшения (не блокеры):**
- `tensorrt-rs` crate (Rust FFI to TensorRT C API) — убрать Python subprocess overhead целиком.
- `ort` crate (ONNX Runtime with CUDA execution provider) — может быть быстрее candle,
  но медленнее TensorRT.
- Fuse RGB→BGR + /255 + pad в ONNX-граф, чтобы crop/pad делал сам engine.

### 5.2. 1080p real-time — не достигнут

1080p TensorRT = 51.7мс, нужно 41.7мс для 48fps. Возможные пути:
- INT8 quantization с calibration dataset (~2-3× ускорение).
- TensorRT optimization level 5 (не завершился за 15 минут — нужен больший timeout).
- Стриппинг ненужных частей модели (block_tea weights — можно убрать из ONNX).
- Process at 960×540 (scale=0.5) + bicubic upscale — качество пострадает.

### 5.3. 60fps (16.6мс) — не достигнут ни на каком разрешении

720p p10 = 18.8мс — почти, но не стабильно. Пути:
- INT8 на 720p может дать ~7-10мс — точно 60fps.
- Уменьшение модели (ещё lite-er) — не существует готовой.
- CUDA Graphs для уменьшения kernel launch overhead.

### 5.4. Браузерный путь (wasm + WebGPU) — не начат

candle 0.11 **не имеет wgpu feature** — только cuda/metal/cuda. Для браузера нужен:
- `ort` (ONNX Runtime Web) с WebGPU execution provider.
- Или кастомные wgpu compute shaders (переписать inference с нуля).
- WebCodecs API для decode + Canvas для paint.
- Это отдельный проект, не reuse candle-кода.

### 5.5. Anime-mode — не начат

SVP имеет отдельный "Animation" режим: детектирует только глобальное движение (pan/zoom)
и интерполирует только его, оставляя статичные объекты нетронутыми. Без этого raw RIFE
создаёт "мыльные" артефакты на аниме (резкие контуры, sudden occlusion).

---

## 6. Идеи для дальнейшего ускорения (приоритизированы)

### 6.1. INT8 quantization с calibration (потенциал: 2-3×)

Собрать ~500 representative кадров из аниме, создать `IInt8EntropyCalibrator2`,
перестроить engine с `BuilderFlag.INT8`. Ожидание: 720p 20.8мс → ~7-10мс (60fps PASS!),
1080p 51.7мс → ~20-25мс (48fps PASS!).

```python
# Нужен calibrator:
class RifeCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir, cache_file):
        ...
    def get_batch(self, names):
        # Return [img0, img1] as np arrays
        ...
    def get_batch_size(self):
        return 1
    def read_calibration_cache(self):
        ...
    def write_calibration_cache(self, cache):
        ...
```

### 6.2. CUDA Graphs (потенциал: 1.3-1.5×)

TensorRT поддерживает CUDA Graphs (`context.record_cursor()` + `context.replay()`).
Убирает kernel launch overhead (~200 кернелов × ~5-10μs = 1-2мс).
Не реализовано в нашем Python benchmark.

### 6.3. TensorRT optimization level 5 (потенциал: 1.1-1.3×)

`config.builder_optimization_level = 5` — максимальная оптимизация, пробует все tactic
combinations. Не завершился за 15 минут. Нужен timeout 30-60 минут. Может дать 10-30%
на некоторых свёртках.

### 6.4. Stream overlap (потенциал: 1.2×)

Два CUDA streams: один декодирует следующий кадр, другой inference текущего.
Нужен double-buffering. Не реализовано.

### 6.5. Убрать block_tea из ONNX (потенциал: 5-10% weights)

`block_tea` (IFBlock(16+4+1, c=90)) — unused at inference (нужен gt для teacher block).
Можно модифицировать export скрипт чтобы не экспортировать эти weights. Сэкономит
~5-10% inference времени (один IFBlock из четырёх).

### 6.6. Fuse preprocessing into TensorRT (потенциал: 2-5мс)

Сейчас RGB→BGR + /255 + padding делается вне engine (в Python или Rust).
Можно добавить эти ops как часть ONNX graph — TensorRT их fused с первым conv.

### 6.7. Native Rust TensorRT bindings (потенциал: убрать Python overhead)

`tensorrt-rs` или прямой FFI к `NvInfer.h`. Убрать Python subprocess overhead (~5-10мс
на frame pair из-за stdin/stdout serialization). Вся inference в одном Rust процессе.

### 6.8. Dynamic shapes (потенциал: один engine для всех разрешений)

Сейчас engine жёстко привязан к разрешению (736×1280 или 1088×1920).
Dynamic shapes позволят один engine для любого разрешения, но с штраф 10-20% к скорости.

---

## 7. Команды для воспроизведения

### Сборка и тест (CPU, без GPU)

```pwsh
cargo check                          # typecheck
cargo clippy --all-targets -- -D warnings  # lint
cargo test                           # warp_identity test
cargo run --release --bin rife-smoke -- --weights weights\rife_lite.safetensors
cargo run --release --bin rife-interpolate -- --weights weights\rife_lite.safetensors img --img0 demo\I0_0.png --img1 demo\I0_1.png --out demo\mid_rust.png --timestep 0.5
```

### Сборка с CUDA

```pwsh
# Нужен vcvars64.bat в PATH для nvcc
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cargo build --release --features cuda"
```

### Профайлинг

```pwsh
cmd /c "`"...\vcvars64.bat`" >nul 2>&1 && cargo run --release --features cuda --bin rife-profile -- --weights weights\rife_lite.safetensors --h 1080 --w 1920 --scale 1.0"
```

### TensorRT

```pwsh
# 1. Export ONNX
python scripts\export_onnx.py 720 1280
# 2. Inline weights
python -c "import onnx; m=onnx.load('weights/rife_lite_720x1280.onnx', load_external_data=True); onnx.save_model(m, 'weights/rife_lite_720x1280.onnx', save_as_external_data=False)"
# 3. Build TRT engine
python scripts\build_trt_engine.py weights\rife_lite_720x1280.onnx weights\rife_lite_trt_fp16.engine
# 4. Benchmark
python scripts\bench_trt.py weights\rife_lite_trt_fp16.engine
```

### Конвертация весов (one-time)

```pwsh
python scripts\convert_weights.py flownet.pkl weights\rife_lite.safetensors --manifest weights\rife_lite_manifest.json
```

### PSNR проверка

```pwsh
python scripts\compare_pytorch.py    # ground truth
cargo run --release --features cuda --bin rife-interpolate -- --weights weights\rife_lite.safetensors img --img0 demo\I0_0.png --img1 demo\I0_1.png --out demo\mid_rust.png --timestep 0.5
python scripts\psnr_compare.py
```

---

## 8. Зависимости и окружение

### Rust

- `candle-core = "0.11"` — ML framework (HuggingFace)
- `candle-nn = "0.11"` — Conv2d, PReLU, etc.
- `anyhow = "1"` — error handling
- `half = "2"` — f16 support
- `clap = "4"` — CLI
- `image = "0.25"` — PNG I/O

### Python

- `torch = "2.10.0+cpu"` — model loading, ONNX export
- `onnx` — model inlining
- `onnxscript` — dynamo exporter dependency
- `tensorrt = "10.13.3.9"` — engine build + inference
- `pycuda` — CUDA memory management
- `opencv-python` — BGR image I/O (for reference)
- `nvidia-cudnn-cu12 = "9.23.2.1"` — НЕ ИСПОЛЬЗОВАТЬ (конфликт с CUDA 13)
- `nvidia-cublas-cu12` — нужен для cuDNN (если включать)

### Системные

- CUDA Toolkit 13.1 (nvcc 13.1.80)
- Visual Studio 2022 (MSVC 14.44) — для nvcc и linker
- ffmpeg 8.0.1 — video decode/encode
- ffprobe — video probing

### GPU

- NVIDIA GeForce RTX 4060 Ti, 8 GB VRAM, compute 8.9 (Ada Lovelace)

---

## 9. Итоговые выводы

### Что работает

1. **Порт RIFE-Lite на Rust/candle** — верифицирован PSNR 88.31 dB, корректный до пикселя.
2. **Custom fused CUDA warp kernel** — 1.55× ускорение, корректность сохранена.
3. **TensorRT fp16 на 720p** — **real-time 48fps PASS** (20.8мс, нужно 41.7мс).
4. **Полный pipeline** mp4→2x→mp4 через ffmpeg pipes (candle version).
5. **Конвертация весов** pkl → safetensors, ключи совпадают 1-в-1.
6. **Нативный in-process TensorRT из Rust (без Python!)** — прямой FFI к `nvinfer_10.dll`
   через C++ shim (`csrc/trt_shim.cpp`, `build.rs` фича `trt`, `src/trt.rs` `RifeTrt`).
   Заголовки TRT взяты с GitHub `release/10.13`, import-lib сгенерирован из DLL (без NVIDIA-логина).
   Скорость = скорость движка: 720p fp16 21.6мс (48fps PASS), INT8 17.1мс/p10 16мс, 1080p 52.9мс.
   Заменил `trt_server.py` (удалён). Сквозной pipeline `rife-trt`: 720p 72→143 кадра за 2.42с
   (59 fps wall vs 26 fps у старого Python-пути), корректность подтверждена (pass-through MAE
   0.6, интерполяция — симметричный midpoint, BGR-порядок верный). `ort`-путь отклонён — на
   CUDA 13 ORT CUDA-EP даёт нули (cuDNN cu12-конфликт), TRT-EP не грузится (см. §5.1).

### Real-time итог (нативно, in-process)

| Разрешение | infer | 48fps (2x@24) | wall pipeline |
|---|---|---|---|
| 720p fp16 | 21.6мс | ✅ PASS | 59 fps (2.42с/72к) |
| 720p INT8 | 17.1мс (p10 16.0) | ✅ PASS | — |
| 1080p fp16 | 52.9мс | ❌ (близко) | — |

### Что не работает / не завершено

1. **1080p real-time** — 51.7мс, чуть выше бюджета (нужно 41.7мс).
2. **60fps** — не достигнуто стабильно (720p p10=18.8мс, почти).
3. **cuDNN на CUDA 13** — версионный конфликт, делает медленнее.
4. **Браузерный путь** — не начат (candle без wgpu, нужен ort/ONNX+WebGPU).
5. **Anime-mode** — не начат.

### Главная рекомендация для следующего шага

**INT8 quantization с calibration** — это самый быстрый путь к 60fps на 720p и
real-time на 1080p. Собрать 500 кадров аниме, создать calibrator, перестроить engine.
Ожидаемый результат: 720p → ~8мс (120fps!), 1080p → ~20мс (48fps PASS).

**Вторая приоритет** — native Rust TensorRT bindings (без Python subprocess), чтобы
убрать I/O overhead и получить полный real-time pipeline в одном процессе.

**Третья приоритет** — браузерный путь через `ort` crate + WebGPU. candle больше не нужен
для production inference, но его Rust-порт модели остаётся как reference implementation
для валидации.
