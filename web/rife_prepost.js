// JS port of crates/rife-core/src/prepost.rs. Input pixels are RGBA8 (canvas);
// model buffer is CHW f32, BGR order, /255, zero-padded bottom/right.

export function pad32(x) {
  return Math.ceil(x / 32) * 32;
}

// RGBA8 HWC (w*h*4) -> CHW f32 (BGR, /255), zero-padded bottom/right to pw x ph. len 3*pw*ph.
export function toInput(rgba, w, h, pw, ph) {
  const dst = new Float32Array(3 * pw * ph); // zero-filled
  const plane = ph * pw; // row stride is pw
  for (let y = 0; y < h; y++) {
    const row = y * pw;
    const srow = y * w * 4;
    for (let x = 0; x < w; x++) {
      const s = srow + x * 4;
      const o = row + x;
      dst[o] = rgba[s + 2] / 255;         // B
      dst[plane + o] = rgba[s + 1] / 255; // G
      dst[2 * plane + o] = rgba[s] / 255; // R
    }
  }
  return dst;
}

// CHW f32 (BGR, pw x ph) -> RGBA8 HWC (w*h*4), crop top-left, BGR->RGB, clamp*255 trunc, A=255.
export function fromOutput(chw, w, h, pw, ph) {
  const dst = new Uint8ClampedArray(4 * w * h);
  const plane = ph * pw;
  const px = (v) => Math.min(255, Math.max(0, v * 255)) | 0; // clamp then truncate
  for (let y = 0; y < h; y++) {
    const row = y * pw;
    const drow = y * w * 4;
    for (let x = 0; x < w; x++) {
      const o = row + x;
      const d = drow + x * 4;
      dst[d] = px(chw[2 * plane + o]); // R
      dst[d + 1] = px(chw[plane + o]); // G
      dst[d + 2] = px(chw[o]);         // B
      dst[d + 3] = 255;                // A
    }
  }
  return dst;
}
