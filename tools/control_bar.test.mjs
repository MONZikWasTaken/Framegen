import assert from 'node:assert/strict';
import test from 'node:test';
import { timeline } from '../extension/src/ui/bar.js';

test('a live stream reads LIVE where the duration goes and has no position to seek to', () => {
  const live = timeline(Infinity, 776);
  assert.equal(live.current, '12:56');
  assert.equal(live.end, 'LIVE');
  assert.equal(live.total, 0);
  assert.equal(live.progress, 0);
});

test('a video with a duration keeps its total and its progress', () => {
  const vod = timeline(3725, 931.25);
  assert.equal(vod.current, '15:31');
  assert.equal(vod.end, '1:02:05');
  assert.equal(vod.total, 3725);
  assert.equal(vod.progress, 25);
});

test('a video without metadata yet reads 0:00 and cannot be sought', () => {
  const empty = timeline(NaN, 0);
  assert.equal(empty.end, '0:00');
  assert.equal(empty.total, 0);
  assert.equal(empty.progress, 0);
});
