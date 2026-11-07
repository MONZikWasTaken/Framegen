export const PICTURE_MODES = Object.freeze({
  original: { label: 'Original', contrast: 0, saturation: 1, lift: 0, warmth: 0, detail: 0 },
  cinema: { label: 'Cinema', contrast: 0.10, saturation: 0.94, lift: 0, warmth: 0.025, detail: 0 },
  'cinema-home': { label: 'Cinema Home', contrast: 0.04, saturation: 0.98, lift: 0.08, warmth: 0.018, detail: 0 },
  noir: { label: 'Noir', contrast: 0.20, saturation: 0, lift: 0.015, warmth: 0, detail: 0 },
  vivid: { label: 'Vivid', contrast: 0.14, saturation: 1.18, lift: 0.025, warmth: 0, detail: 0 },
  sports: { label: 'Sports', contrast: 0.06, saturation: 1.08, lift: 0.06, warmth: 0, detail: 0.35 },
  detail: { label: 'Detail Boost', contrast: 0, saturation: 1, lift: 0, warmth: 0, detail: 0.65 },
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
