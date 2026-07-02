# Framecast Phase 0 — `rife-core` + `FrameInterpolator` + единый pre/post

Дата: 2026-07-02
Статус: утверждено (пользователь делегировал детали)

## Зачем

Сейчас логика pre/post (порядок каналов BGR, `/255`, паддинг до кратного 32,
кроп, обратная конвертация в RGB, усечение `as u8`) продублирована в трёх местах:

- `src/imgutil.rs` — candle-путь (PNG и raw RGB24 → тензор, тензор → PNG/RGB24);
- `src/trt.rs` — `fill_input` / `read_output` (сырые байты ↔ engine-буфер);
- `tools/build_trt_int8.py` — калибратор INT8.

Семантика во всех трёх **идентична** (проверено): BGR, `/255`, zero-pad вправо-вниз,
кроп из левого-верхнего угла, усечение. Отличается только *где* она выполняется
(тензор vs сырые байты) и цель паддинга (candle → round32(w,h); trt → размер движка `ew×eh`).

Без общего трейта и общего pre/post каждый новый бэкенд (браузер/wgpu позже) = копипаст.
Это ROADMAP «Фаза 0 — Фундамент», делается первым.

## Что делаем

### 1. Структура воркспейса

```
crates/rife-core/          # НОВЫЙ крейт, без тяжёлых зависимостей (только anyhow)
  src/lib.rs               # Frame, FrameInterpolator, pad32
  src/prepost.rs           # to_input / from_output — единственный источник правды
Cargo.toml (корень)        # становится [workspace]; framecast — член воркспейса
framecast (в корне)        # существующий крейт, добавляет dep на rife-core
```

Решение: `framecast` **остаётся в корне** репозитория как член воркспейса (не переносим
под `crates/`), чтобы не трогать пути `build.rs`, `csrc/`, `models/`, `assets/`.

### 2. rife-core (нейтральный к бэкенду, без candle/cuda)

```rust
pub struct Frame { pub w: u32, pub h: u32, pub rgb: Vec<u8> }  // HWC, RGB8

pub trait FrameInterpolator {
    fn interpolate(&self, f0: &Frame, f1: &Frame, timestep: f32) -> anyhow::Result<Frame>;
}

pub fn pad32(x: u32) -> u32;                                 // округление вверх до /32

// pw,ph — цель паддинга: candle передаёт pad32(w),pad32(h); trt передаёт ew,eh движка.
pub fn to_input(rgb: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32>;
//   RGB8 HWC -> CHW f32, порядок BGR, /255, zero-pad вправо-вниз до pw×ph.

pub fn from_output(chw: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<u8>;
//   CHW f32 (BGR) -> RGB8 HWC, кроп левого-верхнего w×h, clamp*255, усечение as u8.
```

Юнит-тесты `rife-core` владеют контрактом корректности:
round-trip `to_input`→`from_output`, геометрия паддинга, порядок каналов.

### 3. Бэкенды реализуют трейт

- **RifeTrt** (`src/trt.rs`): `to_input(rgb, w,h, ew,eh)` → `infer` → `from_output`.
  Удаляем `fill_input` / `read_output` / `interpolate_rgb` (заменены общими функциями).
  FFI-часть (`trt_create`/`trt_dims`/`trt_infer`/`trt_destroy`, буферы) без изменений.

- **RifeCandle** (переименование `RifeLite`): trait-impl делает
  `to_input` → `Tensor::from_vec((c,ph,pw))` → `model.forward` →
  скачивание CHW f32 с GPU → `from_output`.
  Оставляем **инхерентный** метод `interpolate_scaled` (на тензорах, режим scale<1
  с даунскейлом) для PNG-CLI — это чисто candle-функциональность, не общая pre/post-логика.

- **imgutil** (`src/imgutil.rs`): теряет циклы BGR/нормализации. Остаётся только
  `DynamicImage → Frame` и `Frame → DynamicImage` (PNG-граница). Вся цвет/норм/паддинг —
  в rife-core.

**Компромисс (осознанный):** сейчас candle делает пост-обработку выхода на GPU
(`tensor_to_rgb24`: clamp/*255/BGR→RGB/u8 на устройстве). Прогон через `from_output`
переносит это на CPU и скачивает в 3× больше данных (f32 вместо u8). Это небольшая
просадка на **референсном/oracle** пути (не на real-time trt-пути). Выбор в пользу
единого источника правды: oracle приоритезирует корректность, а не скорость.

### 4. Дедуп ffmpeg-пайплайна

`src/io/video.rs` и `src/io/video_trt.rs` дублируют `read_exact_or_eof` и обвязку
probe/декод/энкод. Выносим общий `RawVideoReader`/writer; каждый пайплайн оставляет
только свой per-frame вызов в свой бэкенд.

### 5. Критерий готовности («Done when»)

Гейтед интеграционный тест (нужны gitignore-ные engine+weights, поэтому `#[ignore]`,
запускается вручную): один и тот же `Frame` прогоняется через **RifeCandle и RifeTrt**,
результаты сравниваются в пределах допуска. Это ровно ROADMAP-гейт Фазы 0.

## Вне области этой спеки (отложено)

- CUDA-автодискавери в `build.rs` (сейчас хардкод `v13.1`) + скрипт `bootstrap_trt`
  (регенерация import-lib + fetch headers) — пункт «воспроизводимая сборка» из Фазы 0.
  Независим от трейт-работы, отдельным шагом, чтобы не раздувать этот проход.

## Проверка после реализации

- `cargo check` (candle) и `cargo clippy --all-targets -- -D warnings` — чисто.
- `cargo build --release --features trt --bin rife-trt` — собирается.
- `rife-trt` на `demo/test_720p.mp4` даёт тот же результат, что и до рефактора
  (побайтовая проверка выхода или визуально — интерполяция корректна).
- Юнит-тесты rife-core проходят.
