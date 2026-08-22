import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const extensionSr = await readFile(new URL('../extension/rt/sr.js', import.meta.url), 'utf8');
const webSr = await readFile(new URL('../web/rt/sr.js', import.meta.url), 'utf8');

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, `missing ${startMarker}`);
  const end = source.indexOf(endMarker, start);
  assert.ok(end > start, `missing ${endMarker} after ${startMarker}`);
  return source.slice(start, end);
}

function safeCoc(co, preferred) {
  const candidate = Math.max(1, Math.floor(Number(preferred) || 1));
  if (co % candidate === 0) return candidate;
  return co % 8 === 0 ? 8 : co % 4 === 0 ? 4 : 1;
}

function selectedCocs(channels, outputChannels, baseCoc) {
  return {
    mid: channels % 16 === 0 ? 16 : safeCoc(channels, baseCoc),
    output: outputChannels <= 16 ? outputChannels : safeCoc(outputChannels, baseCoc),
  };
}

test('extension and web TinySR runtimes stay byte-identical', () => {
  assert.equal(extensionSr, webSr);
});

test('per-size state compilation never benchmarks kernels on the playback queue', () => {
  const state = section(extensionSr, 'function stateFor', '  const gx');
  assert.doesNotMatch(state,
    /pickConv|onSubmittedWorkDone|device\.queue\.submit|performance\.now|run\(3\)|run\(20\)/);
  assert.match(state, /const compile = async \(v\) => \(\{ pipe: await pipeAsync\(/);
  assert.match(state,
    /Promise\.all\(\[\s*compile\(spec\(C, midCoc\)\),\s*compile\(spec\(C4, outCoc\)\),/);
});

test('kernel selection uses deterministic wide blocks and safe divisors', () => {
  const state = section(extensionSr, 'function stateFor', '  const gx');
  assert.match(state, /const useSg = !!baseT\.sg && device\.features\.has\('subgroups'\);/);
  assert.match(state, /const midCoc = C % 16 === 0 \? 16 : safeCoc\(C, baseT\.coc\);/);
  assert.match(state, /const outCoc = C4 <= 16 \? C4 : safeCoc\(C4, baseT\.coc\);/);
  assert.deepEqual(selectedCocs(16, 12, 8), { mid: 16, output: 12 });
  assert.deepEqual(selectedCocs(24, 48, 16), { mid: 8, output: 16 });
  assert.deepEqual(selectedCocs(15, 17, 8), { mid: 1, output: 1 });
});

test('shared-memory fitting remains fail-closed at the WebGPU minimum budget', () => {
  const state = section(extensionSr, 'function stateFor', '  const gx');
  assert.match(state, /while \(slab > 1 && sharedBytes\(slab\) > 16384\) slab--;/);
  assert.match(state,
    /if \(sharedBytes\(slab\) > 16384\) throw new Error\('TinySR kernel exceeds shared-memory budget'\);/);
});

test('bundled rt_sr weights match the deterministic 16-channel 2x contract', async () => {
  const manifest = JSON.parse(await readFile(
    new URL('../extension/assets/rt_sr.json', import.meta.url), 'utf8'));
  const weights = await readFile(new URL('../extension/assets/rt_sr.bin', import.meta.url));
  const channels = manifest['c1.weight'].shape[0];
  const outputChannels = manifest['c4.weight'].shape[0];
  const manifestWords = Math.max(...Object.values(manifest).map(({ offset, shape }) =>
    offset + shape.reduce((product, value) => product * value, 1)));

  assert.equal(channels, 16);
  assert.equal(outputChannels, 12);
  assert.deepEqual(selectedCocs(channels, outputChannels, 8), { mid: 16, output: 12 });
  assert.equal(weights.byteLength / 4, manifestWords);
});

test('TinySR timing is optional, bounded, and backward compatible', () => {
  const ring = section(extensionSr, 'function createGpuTimestampRing', '\nfunction wgslIn');
  const process = section(extensionSr, '  function submitProcess', '\n  function destroy');

  assert.match(ring, /slotCount = 4/);
  assert.match(ring, /if \(!slot\) return null/,
    'a saturated timing ring must skip sampling instead of allocating');
  assert.match(process, /function process\(srcTex, dstTex, w, h\)[\s\S]*?!== null/);
  assert.match(process, /function processTimed\(srcTex, dstTex, w, h\)/);
  assert.match(process, /timestampRing\.collect\(timingSlot\)/);
  assert.match(extensionSr,
    /return \{ process, processTimed, destroy, hasGpuTimestamps, scale: SCALE \};/);
});
