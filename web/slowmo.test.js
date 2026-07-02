import { assembleSlowmo } from './slowmo.js';
import assert from 'node:assert/strict';

assert.deepEqual(assembleSlowmo(['a', 'b', 'c'], ['m0', 'm1']), ['a', 'm0', 'b', 'm1', 'c']);
assert.deepEqual(assembleSlowmo(['a', 'b'], ['m0']), ['a', 'm0', 'b']);
assert.equal(assembleSlowmo(['a', 'b', 'c', 'd'], ['m0', 'm1', 'm2']).length, 2 * 4 - 1);
// frames-only (no mids) returns the frames unchanged
assert.deepEqual(assembleSlowmo(['a', 'b', 'c'], []), ['a', 'b', 'c']);

console.log('slowmo.js: ALL TESTS PASSED');
