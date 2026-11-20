import assert from 'node:assert/strict';
import test from 'node:test';
import { phaseCases, frameIndices } from '../web/flyhigh_phase_bench.mjs';

test('the phase benchmark changes only the render path at matched model and repair settings', () => {
  const cases = phaseCases(1920, 1080);
  assert.deepEqual(cases.map(([name, { fastRender, linearT }]) => [name, fastRender, linearT]), [
    ['fast-linear', true, true], ['direct-exact', true, false], ['full-buffer-exact', false, false],
  ]);
  const rest = cases.map(([, { fastRender, linearT, ...options }]) => options);
  assert.deepEqual(rest, [rest[0], rest[0], rest[0]]);
  assert.deepEqual(rest[0].internal, [848, 480]);
  assert.equal(rest[0].staticProtection, true);
  assert.equal(rest[0].repair, 1);
  assert.equal(rest[0].nearestSource, true);
  assert.throws(() => phaseCases(848, 480), /smaller/);
});

test('every fourth captured frame is an anchor and quarter phases use withheld frames', () => {
  assert.deepEqual(frameIndices(10), [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]);
  assert.throws(() => frameIndices(4), /five/);
});
