# Framecast Phase 4-lite — browser video 2× slow-mo demo

Дата: 2026-07-02
Статус: утверждено (пользователь: «да кнщ»)

## Зачем

Наглядное видео-демо браузерного трека: короткий клип → плавное 2× замедление. **Не real-time**
— модель дорогая, реальный WebGPU-перф ещё не замерен. Модель: precompute-then-play — взять
окно кадров, посчитать промежуточные, проиграть удвоенную последовательность на исходном fps
(движение вдвое медленнее = плавный slow-mo). Полная Фаза 4 (WebCodecs, GPU-resident, real-time)
— позже.

## Что делаем

### Файлы

- `web/rife_session.js` (НОВЫЙ, DRY) — общий инференс-слой:
  - `createSession(ep) → Promise<ort.InferenceSession>` — грузит `/assets/rife_lite_inlined.onnx`.
  - `interpolate(session, imgDataA, imgDataB) → ImageData` — прогоняет пару кадров (одинакового
    размера) через модель и возвращает средний кадр. Внутри: `toInput`(pad до engine 1280×736)
    → `ort.Tensor` → `run` → `fromOutput`(crop до native). Требует `a.width===b.width` и
    `a.height===b.height`.
  - Константы движка `EW=1280, EH=736`, вход/выход имена берутся из сессии.
- `web/demo.html` (МОДИФИКАЦИЯ) — переключается на `rife_session.js` вместо инлайн-инференса
  (убрать дубль `ensureSession`/`toInput→Tensor→run→fromOutput`). Поведение и вывод неизменны.
- `web/slowmo.html` (НОВЫЙ) — страница slow-mo.
- `web/slowmo.test.js` (НОВЫЙ) — node-тест чистой функции сборки `assembleSlowmo`.

### `assembleSlowmo` (чистая, тестируемая; живёт в `web/slowmo.html`-модуле или отдельном `.js`)

```js
// frames: [F0..F_{n-1}] (n>=2), mids: [M0..M_{n-2}] (mid_i between F_i and F_{i+1}).
// -> interleaved [F0, M0, F1, M1, ..., F_{n-1}] (length 2n-1).
export function assembleSlowmo(frames, mids) -> Array
```
Вынести в `web/slowmo.js` как ES-модуль, чтобы node-тест мог импортировать без DOM.

### Поток (`web/slowmo.html`)

1. Ввод: `<input type=file accept=video/*>` ИЛИ кнопка «demo clip» → `/demo/test_720p.mp4`.
2. Загрузить в скрытый `<video>` (`preload=auto`, `muted`). Прочитать `duration`; частоту кадров
   считаем из UI-поля (default 24) — точный fps из `<video>` браузер не отдаёт, так что задаём.
3. Сэмплировать **N кадров** (UI, default 8) от начала: для `i` в `0..N-1` установить
   `video.currentTime = i / fps`, дождаться события `seeked`, `drawImage` в offscreen-canvas
   размера native (ресайз с сохранением пропорций внутрь 1280×720), `getImageData`.
4. Precompute: для каждой пары `(f_i, f_{i+1})` → `interpolate` → `mid_i`; UI-прогресс
   «computing i/(N-1)…».
5. `assembleSlowmo(frames, mids)` → последовательность длиной `2N-1`.
6. Playback: `requestAnimationFrame`-луп рисует последовательность на видимый canvas с интервалом
   `1000/fps` мс на кадр → 2× медленнее оригинала. Кнопки: play/pause, loop (default on),
   тумблер «interpolated ↔ original» (original = только `frames`, N кадров).

### Переиспользование / проверка

- `web/rife_prepost.js` и `web/rife_session.js` — общие с `demo.html`.
- **node-тест** `web/slowmo.test.js`: `assembleSlowmo` на маленьких массивах (чередование,
  длина `2n-1`, порядок).
- **E2E (Playwright, wasm)**: `slowmo.html` с маленьким N (2-3, чтобы уложиться по времени на
  CPU): подгрузить demo clip, засемплить, precompute, убедиться что собранная последовательность
  имеет длину `2N-1`, кадры не пустые (не все нули), и playback-луп рисует на canvas. Реальный
  webgpu-перф (плавно ли на 8+ кадрах) — только на GPU пользователя.
- **Регресс demo.html**: после перевода на `rife_session.js` — повторить E2E demo-пары (выход
  по-прежнему совпадает с PyTorch mid, mean|Δ|≈0.1).

## Одна развилка (решена)

Декод кадров: **`<video>` + seek → canvas** (выбрано; универсально, без демуксера). WebCodecs
(`VideoDecoder` + mp4box.js) — отложено на полную Фазу 4.

## Вне области

Real-time, WebCodecs, GPU-resident пайплайн, звук, экспорт результата в файл, произвольный fps
из контейнера. Slow-mo фиксированно 2× (timestep=0.5 — модель так и экспортирована).

## Критерий готовности

Открываешь `web/slowmo.html`, «demo clip» → «Compute» → видишь плавное 2× замедление на canvas,
тумблер interpolated/original показывает разницу. `node web/slowmo.test.js` зелёный; E2E на wasm
подтверждает семплинг+precompute+сборку+playback; demo.html после рефактора не сломан.
