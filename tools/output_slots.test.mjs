import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.chrome = { runtime: { getURL: path => path } };
const { Player } = await import('../extension/src/player.js');
delete globalThis.chrome;

function playerForTest(initialOutputs, write = () => true) {
  const player = Object.create(Player.prototype);
  const textures = Array(initialOutputs).fill(null);
  const pending = new Map();
  const skipped = [];
  const presented = [];
  let nextFrame;
  player.settings = { cfg: { fg: true } };
  player.stats = {};
  player.nextUiAt = Infinity;
  player.hudAt = Infinity;
  player.resetSlots();
  player.stream = {
    report: () => ({ outputs: textures.length }),
    addOutput: () => textures.push(null) - 1,
    outputTexture: slot => textures[slot],
    inputTexture: slot => `source-${slot}`,
  };
  player.costs = {
    generate(_stream, request, slot) {
      if (!write(request)) return false;
      textures[slot] = request.id;
      return true;
    },
  };
  player.pacer = {
    generated(_now, id, output) { pending.set(id, { output }); },
    skipped(id) { skipped.push(id); },
    tick() {
      return { present: { frame: nextFrame }, generate: [], factor: 2, dropped: 0 };
    },
  };
  player.stats.dropWin = [];
  player.present = frame => presented.push(frame);
  return {
    player, textures, pending, skipped, presented,
    show(id, now) {
      nextFrame = pending.get(id);
      assert.ok(nextFrame, `request ${id} should be queued`);
      pending.delete(id);
      player.tickBody(now);
    },
  };
}

const request = (id, due_ms) => ({ id, due_ms });

test('a full pool skips extra generation and presents the original frame before reusing its slot', () => {
  const { player, textures, skipped, presented, show } = playerForTest(8);
  player.generate(100, Array.from({ length: 21 }, (_, i) => request(i + 1, 120 + i)));
  assert.deepEqual(textures, Array.from({ length: 20 }, (_, i) => i + 1));
  assert.deepEqual(skipped, [21]);

  show(1, 120);
  player.generate(121, [request(22, 142)]);
  show(22, 142);
  assert.deepEqual(presented, [1, 22]);
});

test('a late presentation cannot display or release a recycled slot', () => {
  const { player, textures, skipped, presented, show } = playerForTest(20);
  player.generate(100, Array.from({ length: 20 }, (_, i) => request(i + 1, i === 0 ? 120 : 500)));
  player.generate(400, [request(21, 430)]);
  player.lastShownSource = 3;
  show(1, 410);
  player.generate(411, [request(22, 440)]);

  assert.deepEqual(presented, ['source-3']);
  assert.deepEqual(skipped, [22]);
  assert.equal(textures[0], 21);
  show(21, 430);
  assert.deepEqual(presented, ['source-3', 21]);
});

test('a generator fallback frees its slot for the following request in the same batch', () => {
  const { player, textures, skipped, presented, show } = playerForTest(20, r => r.id !== 20);
  player.generate(100, Array.from({ length: 21 }, (_, i) => request(i + 1, 120 + i)));
  assert.deepEqual(skipped, [20]);
  assert.equal(textures[19], 21);
  show(21, 140);
  assert.deepEqual(presented, [21]);
});
