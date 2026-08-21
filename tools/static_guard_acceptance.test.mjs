import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import {
  STATIC_GUARD_CONTRACT,
  analyzeStaticGuardFrames,
  finalizeStaticGuardReport,
} from './static_guard_acceptance.mjs';

const WIDTH = 8;
const HEIGHT = 4;
const PIXELS = WIDTH * HEIGHT;
const TIMESTEPS = [...STATIC_GUARD_CONTRACT.timesteps];

function frame(fill) {
  const out = new Uint8Array(PIXELS * 4);
  for (let pixel = 0; pixel < PIXELS; pixel++) {
    const value = fill(pixel);
    out.set([value[0], value[1], value[2], 255], pixel * 4);
  }
  return out;
}

function fixture({ frozenMidpoint = false, parityError = false, motionError = false } = {}) {
  const textCoreMask = new Uint8Array(PIXELS);
  const textPlateMask = new Uint8Array(PIXELS);
  const ordinaryMotionMask = new Uint8Array(PIXELS);
  for (let pixel = 0; pixel < PIXELS; pixel++) {
    if (pixel < 8) textCoreMask[pixel] = 1;
    else if (pixel < 16) textPlateMask[pixel] = 1;
    else ordinaryMotionMask[pixel] = 1;
  }
  const anchorA = frame(pixel => pixel < 8 ? [238, 240, 242] : pixel < 16 ? [20, 22, 24] : [18, 60, 110]);
  const anchorB = frame(pixel => pixel < 8 ? [244, 246, 248] : pixel < 16 ? [24, 26, 28] : [210, 120, 24]);
  const interpolate = (a, b, t) => Math.round(a + (b - a) * t);
  const defaultGuardOffFrames = TIMESTEPS.map(t => frame(pixel => {
    if (pixel < 16) return [80 + Math.round(20 * t), 90 + Math.round(10 * t), 100];
    return [Math.round(18 + 192 * t), Math.round(60 + 60 * t), Math.round(110 - 86 * t)];
  }));
  const explicitGuardOffFrames = defaultGuardOffFrames.map(value => value.slice());
  if (parityError) explicitGuardOffFrames[2][0]++;
  const guardFrames = TIMESTEPS.map((t, frameIndex) => frame(pixel => {
    if (pixel < 8) {
      const sampleT = frozenMidpoint ? 0.5 : t;
      return [interpolate(238, 244, sampleT), interpolate(240, 246, sampleT), interpolate(242, 248, sampleT)];
    }
    if (pixel < 16) {
      return [interpolate(20, 24, t), interpolate(22, 26, t), interpolate(24, 28, t)];
    }
    const offset = pixel * 4;
    const source = defaultGuardOffFrames[frameIndex];
    return [source[offset] + (motionError ? 3 : 0), source[offset + 1], source[offset + 2]];
  }));
  return {
    width: WIDTH,
    height: HEIGHT,
    timesteps: TIMESTEPS,
    anchorA,
    anchorB,
    guardFrames,
    defaultGuardOffFrames,
    explicitGuardOffFrames,
    textCoreMask,
    textPlateMask,
    ordinaryMotionMask,
  };
}

function reportFor(input) {
  const metrics = analyzeStaticGuardFrames(input);
  return finalizeStaticGuardReport({
    schemaVersion: STATIC_GUARD_CONTRACT.schemaVersion,
    benchmarkId: STATIC_GUARD_CONTRACT.benchmarkId,
    status: 'complete',
    workload: {
      patternId: STATIC_GUARD_CONTRACT.patternId,
      model: STATIC_GUARD_CONTRACT.model,
      width: STATIC_GUARD_CONTRACT.width,
      height: STATIC_GUARD_CONTRACT.height,
      timesteps: [...STATIC_GUARD_CONTRACT.timesteps],
    },
    metrics,
  });
}

test('time-consistent candidate passes the text-over-motion contract', () => {
  const report = reportFor(fixture());
  assert.equal(report.passed, true, report.validation.checks.filter(check => !check.passed).map(check => check.name).join(', '));
});

test('the former fixed midpoint fails temporal and anchor checks', () => {
  const report = reportFor(fixture({ frozenMidpoint: true }));
  assert.equal(report.passed, false);
  const failed = report.validation.checks.filter(check => !check.passed).map(check => check.name);
  assert.ok(failed.includes('text-candidate-mae'));
  assert.ok(failed.includes('anchor-step-continuity'));
});

test('guard-off parity corruption is rejected', () => {
  const report = reportFor(fixture({ parityError: true }));
  assert.equal(report.passed, false);
  assert.equal(report.validation.checks.find(check => check.name === 'guard-off-byte-parity').passed, false);
});

test('ordinary moving content changes are rejected', () => {
  const report = reportFor(fixture({ motionError: true }));
  assert.equal(report.passed, false);
  assert.equal(report.validation.checks.find(check => check.name === 'ordinary-motion-max').passed, false);
});

test('runtime shader binds and consumes the per-mid guard timestep', async () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const source = await readFile(path.join(here, '..', 'web', 'rt', 'rt.js'), 'utf8');
  assert.ok((source.match(/guardT\[0\]/g) ?? []).length >= 2);
  assert.match(source, /@group\(0\) @binding\(5\) var<storage, read> guardT/);
  assert.match(source, /mix\(warpT\(tex0,[\s\S]+warpT\(tex1,[\s\S]+, t\)/);
  assert.match(source, /flowBgFor\(outTexs\[i\], tbufs\[i\]\)/);
  assert.match(source, /flowBgFor\(outTex, guardTbuf\)/);
  assert.doesNotMatch(source, /let stat = \(warpT\(tex0,[^\n]+\+ warpT\(tex1,[^\n]+\) \* 0\.5/);
});

test('browser fixture hashes the exact runtime response bytes', async () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const source = await readFile(path.join(here, '..', 'web', 'static_guard_fixture.html'), 'utf8');
  assert.match(source, /runtimeResponse\.arrayBuffer\(\)/);
  assert.match(source, /runtimeSha256:\s*await sha256\(runtimeBytes\)/);
  assert.doesNotMatch(source, /runtimeResponse\.text\(\)/);
});
