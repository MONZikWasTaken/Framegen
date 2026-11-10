export const PICTURE_MODES = Object.freeze({
  original: { label: 'Original', contrast: 0, saturation: 1, lift: 0, warmth: 0, detail: 0 },
  cinema: { label: 'Cinema', contrast: 0.38, saturation: 0.84, lift: 0, warmth: 0.065, detail: 0 },
  'cinema-home': { label: 'Cinema Home', contrast: 0.16, saturation: 1.06, lift: 0.28, warmth: 0.045, detail: 0 },
  noir: { label: 'Noir', contrast: 0.12, saturation: 0, lift: 0.16, warmth: 0, detail: 0 },
  vivid: { label: 'Vivid', contrast: 0.40, saturation: 1.40, lift: 0.06, warmth: 0, detail: 0 },
  sports: { label: 'Sports', contrast: 0.22, saturation: 1.22, lift: 0.18, warmth: 0, detail: 0.7 },
  detail: { label: 'Detail Boost', contrast: 0, saturation: 1, lift: 0, warmth: 0, detail: 1.2 },
});

// CSS approximations of the looks for the swatches in the UI; the picture itself is graded by the shader below.
export const PICTURE_SWATCH_FILTERS = Object.freeze({
  original: 'none',
  cinema: 'saturate(.8) contrast(1.2) sepia(.22)',
  'cinema-home': 'brightness(1.14) saturate(1.05) sepia(.14)',
  noir: 'grayscale(1) brightness(1.1)',
  vivid: 'saturate(1.5) contrast(1.18)',
  sports: 'saturate(1.25) brightness(1.1) contrast(1.08)',
  detail: 'contrast(1.12)',
});

export function pictureMode(value) {
  return Object.hasOwn(PICTURE_MODES, value) ? PICTURE_MODES[value] : PICTURE_MODES.original;
}

export function pictureGradeWGSL(value) {
  const p = pictureMode(value);
  if (p === PICTURE_MODES.original || value === 'detail') return 'fn grade(c: vec3<f32>) -> vec3<f32> { return c; }';
  const f = n => n.toFixed(4);
  return `
fn grade(c: vec3<f32>) -> vec3<f32> {
  let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  let tone = clamp(y + ${f(p.contrast)} * y * (1.0 - y) * (2.0 * y - 1.0)
    + ${f(p.lift)} * y * (1.0 - y), 0.0, 1.0);
  let neutral = vec3<f32>(tone);
  let tint = vec3<f32>(${f(p.warmth)}, 0.0, ${f(-p.warmth)}) * (4.0 * y * (1.0 - y));
  let chroma = (c - vec3<f32>(y)) * ${f(p.saturation)} + tint;
  var limit = 1.0;
  for (var i = 0u; i < 3u; i++) {
    if (chroma[i] > 0.00001) { limit = min(limit, (1.0 - tone) / chroma[i]); }
    if (chroma[i] < -0.00001) { limit = min(limit, -tone / chroma[i]); }
  }
  return clamp(neutral + chroma * limit, vec3<f32>(0.0), vec3<f32>(1.0));
}`;
}
