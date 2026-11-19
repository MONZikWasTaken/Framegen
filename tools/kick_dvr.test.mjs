import assert from 'node:assert/strict';
import test from 'node:test';
import { positionToRestore } from '../extension/src/kick-dvr.js';

const url = 'https://kick.com/somebody';
// the recorded case: 6 h 31 min into a rewound live stream, a quality change, the new source plays from 0.03 s
const before = { url, duration: 23708.3, position: 23467.29 };

test('a rebuilt recording that starts at 0 goes back to where it was', () => {
  assert.equal(positionToRestore(before, { url, duration: 23738.3, position: 0.03 }), 23467.29);
});

test('the jump back to live is left alone', () => {
  assert.equal(positionToRestore(before, { url, duration: Infinity, position: 0.03 }), null);
});

test('a rewind out of live has no position to restore', () => {
  assert.equal(positionToRestore({ url, duration: Infinity, position: 812 }, { url, duration: 23616.3, position: 0 }), null);
  assert.equal(positionToRestore(null, { url, duration: 23616.3, position: 0 }), null);
});

test('a player that kept the place is left alone', () => {
  assert.equal(positionToRestore(before, { url, duration: 23738.3, position: 23467.3 }), null);
});

test('a viewer who went to the start, or watched to the end, is left there', () => {
  assert.equal(positionToRestore({ url, duration: 23708.3, position: 12 }, { url, duration: 23738.3, position: 0 }), null);
  assert.equal(positionToRestore({ url, duration: 600, position: 599.5 }, { url, duration: 600, position: 0 }), null);
});

test('another video is never seeked into', () => {
  assert.equal(positionToRestore(before, { url: 'https://kick.com/somebody/videos/1', duration: 23738.3, position: 0 }), null);
  assert.equal(positionToRestore(before, { url, duration: 900, position: 0 }), null);
  assert.equal(positionToRestore(before, { url, duration: 40000, position: 0 }), null);
});
