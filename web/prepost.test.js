import { pad32, toInput, fromOutput } from './rife_prepost.js';
import assert from 'node:assert/strict';

// pad32
assert.equal(pad32(720), 736);
assert.equal(pad32(1280), 1280);
assert.equal(pad32(1), 32);
assert.equal(pad32(32), 32);
assert.equal(pad32(33), 64);

// toInput: one 2x1 RGBA image, pad to 4x2. Mirrors rife-core to_input_bgr_and_pad.
{
  const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
  const [w, h, pw, ph] = [2, 1, 4, 2];
  const out = toInput(rgba, w, h, pw, ph);
  const plane = pw * ph; // 8
  const near = (a, b) => Math.abs(a - b) < 1e-6;
  assert.ok(near(out[0], 30 / 255), 'B0');
  assert.ok(near(out[1], 60 / 255), 'B1');
  assert.equal(out[2], 0);
  assert.equal(out[3], 0);
  assert.ok(near(out[plane], 20 / 255), 'G0');
  assert.ok(near(out[plane + 1], 50 / 255), 'G1');
  assert.ok(near(out[2 * plane], 10 / 255), 'R0');
  assert.ok(near(out[2 * plane + 1], 40 / 255), 'R1');
  assert.equal(out[4], 0, 'padded row1');
}

// round-trip crop
{
  const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]);
  const [w, h, pw, ph] = [2, 2, 32, 32];
  const chw = toInput(rgba, w, h, pw, ph);
  const back = fromOutput(chw, w, h, pw, ph);
  assert.deepEqual(Array.from(back), [10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]);
}

// fromOutput clamp/truncate
{
  const chw = new Float32Array([2.0, -1.0, 0.5]);
  const out = fromOutput(chw, 1, 1, 1, 1);
  assert.deepEqual(Array.from(out), [127, 0, 255, 255]);
}

console.log('rife_prepost.js: ALL TESTS PASSED');
