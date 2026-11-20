import assert from 'node:assert/strict';
import test from 'node:test';
import { positionToRestore, watchKickDvr } from '../extension/src/kick-dvr.js';

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

function watchFixture(t) {
  const handlers = new Map();
  let now = 0;
  class Video {
    constructor(currentTime) { this.currentTime = currentTime; }
    duration = 120;
    offsetWidth = 640;
    readyState = 4;
  }
  const originals = { document: globalThis.document, location: globalThis.location, HTMLVideoElement: globalThis.HTMLVideoElement };
  t.after(() => Object.assign(globalThis, originals));
  const location = { href: url };
  Object.assign(globalThis, {
    location,
    HTMLVideoElement: Video,
    document: {
      addEventListener: (name, handler) => handlers.set(name, handler),
      createElement: () => ({ style: {} }),
      documentElement: { appendChild() {} },
    },
  });
  t.mock.method(performance, 'now', () => now);
  t.mock.method(globalThis, 'setTimeout', () => {});
  watchKickDvr();
  return {
    Video,
    dispatch: (name, target) => handlers.get(name)({ target }),
    advance: ms => { now += ms; },
    location,
  };
}

test('replacing the video clears pending restoration and tracks the new element', t => {
  const { Video, dispatch } = watchFixture(t);
  const old = new Video(72);
  dispatch('timeupdate', old);
  dispatch('emptied', old);
  const video = new Video(0.034);
  dispatch('playing', video);
  assert.equal(video.currentTime, 0.034);

  video.currentTime = 67;
  dispatch('timeupdate', video);
  dispatch('emptied', video);
  video.currentTime = 0.034;
  dispatch('playing', video);
  assert.equal(video.currentTime, 67);
});

test('an expired restore does not override playback on the same element', t => {
  const { Video, dispatch, advance } = watchFixture(t);
  const video = new Video(72);
  dispatch('timeupdate', video);
  dispatch('emptied', video);
  advance(10_001);
  video.currentTime = 0.034;
  dispatch('playing', video);
  assert.equal(video.currentTime, 0.034);
});

test('navigation discards a restore even when the old video element is reused', t => {
  const { Video, dispatch, location } = watchFixture(t);
  const video = new Video(72);
  dispatch('timeupdate', video);
  dispatch('emptied', video);
  location.href = 'https://kick.com/another-channel';
  dispatch('playing', video);
  location.href = url;
  video.currentTime = 0.034;
  dispatch('playing', video);
  assert.equal(video.currentTime, 0.034);
});
