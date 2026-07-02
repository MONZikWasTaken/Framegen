# Framecast Phase 1 — Browser demo (two images → middle frame)

Дата: 2026-07-02
Статус: утверждено (пользователь делегировал; выбран вариант A — ручной JS pre/post)

## Зачем

Первый demo-able артефакт браузерного трека (ROADMAP Фаза 1): открыл страницу, дал 2 кадра,
получил интерполированный средний кадр — через ort-web + WebGPU (wasm-fallback). Полезно
независимо от того, что покажет замер fps: доказывает сквозной браузерный путь и даёт то, что
можно показать. Медленно — ок.

Не real-time и не видео (это Фаза 4). Ровно примитив интерполяции в браузере, наглядно.

## Что делаем

### Файлы

- `web/demo.html` — UI + оркестрация (грузит ort-web как `web/probe.html`).
- `web/rife_prepost.js` — изолированный порт pre/post (`toInput`/`fromOutput`), зеркалит
  `rife-core::prepost`. ES-модуль, без зависимостей, тестируемый в отрыве.
- `web/prepost.test.html` — крошечная страница-раннер юнит-тестов для `rife_prepost.js`
  (assert'ы в консоль/на страницу), чтобы parity ловилась автоматически.

### JS pre/post (`web/rife_prepost.js`) — зеркало rife-core

Браузер даёт пиксели как **RGBA8** (canvas `ImageData`), поэтому сигнатуры чуть отличаются от
Rust (там RGB8), но семантика цвета/нормализации/паддинга идентична.

```js
// RGBA8 HWC (w*h*4) -> CHW f32, BGR, /255, zero-pad bottom/right to pw x ph. len = 3*pw*ph.
export function toInput(rgba, w, h, pw, ph) -> Float32Array
// CHW f32 (BGR, pw x ph) -> RGBA8 HWC (w*h*4), crop top-left w x h, clamp*255 trunc, alpha=255.
export function fromOutput(chw, w, h, pw, ph) -> Uint8ClampedArray
export function pad32(x) -> number   // round up to /32
```

Контракт слоёв: CHW плоскости B,G,R; stride строки = pw; кроп из левого-верхнего угла. Точное
соответствие `crates/rife-core/src/prepost.rs` (порядок B/G/R, `(v*255) clamp truncate`).

### Поток данных (`web/demo.html`)

1. Ввод: drag-drop двух изображений ИЛИ кнопка «demo pair» → грузит `demo/I0_0.png`,
   `demo/I0_1.png` (448×256).
2. Для каждого изображения: нарисовать в offscreen-canvas размера native (≤ 1280×720; если
   больше — ресайз с сохранением пропорций внутрь 1280×720). `pw=pad32(w)`, `ph=pad32(h)`
   — как в нативном пайплайне (модель фикс. 736×1280, но меньший вход паддится нулями, движок
   это принимает). Для ONNX фикс-формы вход всегда набивается в поле engine-размера eW×eH
   = 1280×736: `toInput(rgba, w,h, 1280, 736)`.
3. `toInput` обоих → два `ort.Tensor('float32', …, [1,3,736,1280])` → `session.run`.
4. Выход `mid` [1,3,736,1280] BGR → `fromOutput(chw, w,h, 1280,736)` → `ImageData(w,h)` →
   рисуется в result-canvas.
5. UI: before/after слайдер (наложение img0 и результата, вертикальный раздел по X) + строка
   `p50=… ms · ep=webgpu|wasm`.

Замечание по фикс-форме: `rife_lite_inlined.onnx` имеет вход [1,3,736,1280]. Значит `pw,ph`
для `toInput`/`fromOutput` — это **engine-размер 1280×736**, а `w,h` — native кадра. Это ровно
как `RifeTrt` (native ≤ engine, pad до engine, crop назад).

### Проверка (автономно, без реального GPU)

- Юнит-тесты `web/prepost.test.html`: round-trip `toInput`→`fromOutput` для in-range значений,
  порядок BGR, геометрия паддинга/кропа, clamp/усечение — те же кейсы, что в Rust-тестах
  `prepost.rs`.
- E2E через Playwright на **wasm-EP** (WebGPU в chromium раннере нет): подать demo-пару,
  убедиться, что выход не сломан (не все нули, mean в разумном диапазоне) и близок к эталону
  `mid` (mean|Δ| мал — ort-web fp32 vs candle fp32 могут чуть отличаться движками, порог с
  запасом). Реальный WebGPU-перф — на GPU пользователя (harness уже есть).

## Одна развилка (решена)

Где брать pre/post в JS — «4-я копия» логики rife-core:
- **(A) Ручной JS-порт — ВЫБРАНО.** ~30 строк, YAGNI; дрейф гасится parity-тестами.
- (B) Компиляция `rife-core` → wasm (wasm-bindgen) — реально единый источник, но тянет
  wasm-pack тулчейн + exports. Отложено: чистый апгрейд, когда браузерный перф докажет ценность.

## Вне области

- Видео / slow-mo / WebCodecs / GPU-resident пайплайн — Фаза 4.
- fp16-переключатель в демо — не нужен для «demo-able»; перф-сравнение делает `web/probe.html`.
- 480p — заблокировано (нужен реэкспорт, веса удалены).

## Критерий готовности

Открываешь `web/demo.html` (через `python -m http.server`), жмёшь «demo pair» (или кидаешь 2
своих) → видишь интерполированный средний кадр и before/after слайдер, строка с p50. Юнит-тесты
pre/post зелёные; E2E на wasm подтверждает корректный (не сломанный) выход.
