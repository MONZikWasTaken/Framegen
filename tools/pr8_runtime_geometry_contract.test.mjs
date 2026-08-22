import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const extensionRuntime = readFileSync(new URL('../extension/rt/rt.js', import.meta.url), 'utf8');
const webRuntime = readFileSync(new URL('../web/rt/rt.js', import.meta.url), 'utf8');

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, `missing ${startMarker}`);
  const end = source.indexOf(endMarker, start);
  assert.ok(end > start, `missing ${endMarker} after ${startMarker}`);
  return source.slice(start, end);
}

function refineUv(model, source, position, flow) {
  const sourcePosition = position.map((value, axis) =>
    (value + 0.5) * source[axis] / model[axis] - 0.5);
  const sourceFlow = flow.map((value, axis) => value * source[axis] / model[axis]);
  return sourcePosition.map((value, axis) =>
    (value + sourceFlow[axis] + 0.5) / source[axis]);
}

function modelUv(model, position, flow) {
  return position.map((value, axis) =>
    (value + flow[axis] + 0.5) / model[axis]);
}

test('extension and web WebGPU runtimes stay byte-identical', () => {
  assert.equal(extensionRuntime, webRuntime);
});

test('refine source-pixel coordinates normalize against the bound texture', () => {
  const refine = section(extensionRuntime, 'function wgslRefinePrep', '\nfunction wgslRefineTiles');
  const warp = section(refine, 'fn warpT', '@compute');

  assert.match(warp,
    /let uv = \(vec2<f32>\(sx, sy\) \+ 0\.5\) \/ vec2<f32>\(textureDimensions\(t\)\);/);
  assert.doesNotMatch(warp, /vec2<f32>\(\$\{W\}\.0, \$\{H\}\.0\)/);
  assert.match(refine, /let srcPos = .* \* srcDim \/ vec2<f32>\(\$\{W\}\.0, \$\{H\}\.0\) - 0\.5;/);
  assert.match(refine, /let f4 = flowModel \* vec4<f32>\(srcDim\.x \/ \$\{W\}\.0, srcDim\.y \/ \$\{H\}\.0,/);
});

test('refine full-resolution remap preserves model-space normalized sampling', () => {
  const cases = [
    { model: [848, 480], source: [1920, 1080] },
    { model: [848, 480], source: [640, 360] },
    { model: [1280, 720], source: [1920, 1080] },
    { model: [1920, 1088], source: [1920, 1080] },
  ];
  const flows = [[0, 0], [7.25, -3.5], [-11.75, 8.125]];

  for (const { model, source } of cases) {
    const positions = [[0, 0], [model[0] * 0.37, model[1] * 0.61],
      [model[0] - 1, model[1] - 1]];
    for (const position of positions) {
      for (const flow of flows) {
        const actual = refineUv(model, source, position, flow);
        const expected = modelUv(model, position, flow);
        assert.ok(Math.abs(actual[0] - expected[0]) < 1e-12,
          `x remap drifted for ${model} -> ${source}`);
        assert.ok(Math.abs(actual[1] - expected[1]) < 1e-12,
          `y remap drifted for ${model} -> ${source}`);
      }
    }
  }
});

test('edge-guided upsampling does not compute an unconsumed mask variant', () => {
  const flowOut = section(extensionRuntime, 'function wgslFlowOutTexDirect',
    '\nfunction wgslDebugWarpsDirect');
  const upsample = section(flowOut, 'fn edgeUpsample', '@compute');

  assert.match(upsample, /-> vec4<f32>/);
  assert.doesNotMatch(upsample, /FlowMask|textureLoad\(t8m/);
  assert.match(flowOut, /flow = edgeUpsample\(uv8, guide\);/);
  assert.doesNotMatch(flowOut, /edgeUpsample\(uv8, guide\)\.flow/);
});

test('production timing isolates pair prep and complete per-mid GPU work', () => {
  const ring = section(extensionRuntime, 'function createGpuTimestampRing', '\n// NOTE:');
  const runtime = section(extensionRuntime, '  function prepPair', '\n  const runTDebug');

  assert.match(ring, /slotCount = 4, maxQueryCount = 4/);
  assert.match(ring, /if \(!slot\) return null/,
    'a saturated timing ring must skip sampling instead of allocating');
  assert.match(runtime, /function prepPair\(a, b, \{ measure = false \} = \{\}\)/);
  assert.match(runtime, /function runT\(t, outTex, \{ measure = false \} = \{\}\)/);
  assert.match(runtime,
    /timestampRing\.collect\(timingSlot, timingQueryCount, 0, timingQueryCount - 1\)/,
    'sparse v7s timing must span its render clear and compute pass');
  assert.match(runtime, /timestampRing\.collect\(timingSlot, 4, 0, 3\)/,
    'classic timing must span both compute passes');
  assert.match(extensionRuntime, /hasGpuTimestamps, w, h/);
});
