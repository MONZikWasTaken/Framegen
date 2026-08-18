import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const Cadence = require('../extension/cadence.js');
const extensionDirectory = fileURLToPath(new URL('../extension/', import.meta.url));

function resolveTarget(targetFps, sourceHz, displayHz, options = {}) {
  return Cadence.resolveOutputRate('target', 1000 / displayHz, {
    targetFps,
    sourceHz,
    sourceReady: true,
    displayReady: true,
    ...options,
  });
}

function simulateConstantCadence(sourceFps, targetHz, seconds = 120) {
  const sourceStep = 1000 / sourceFps;
  let startAt = 1000;
  let nextAt = 0;
  let ticks = 0;
  let maximumTicksPerInterval = 0;
  const intervals = Math.ceil(sourceFps * seconds);
  for (let index = 0; index < intervals; index += 1) {
    const endAt = startAt + sourceStep;
    const plan = Cadence.planCadenceInterval({ nextAt, startAt, endAt, outputHz: targetHz });
    assert.equal(plan.overflowed, false);
    assert.ok(plan.ticks.every(tick => Number.isFinite(tick.at) && Number.isFinite(tick.t)));
    maximumTicksPerInterval = Math.max(maximumTicksPerInterval, plan.ticks.length);
    ticks += plan.ticks.length;
    nextAt = plan.nextAt;
    startAt = endAt;
  }
  const durationSeconds = intervals * sourceStep / 1000;
  return { observedHz: ticks / durationSeconds, maximumTicksPerInterval, nextAt, startAt };
}

function simulateProductPresentations(sourceFps, targetHz, {
  seconds = 120,
  interpolate = true,
  gpuFallback = false,
} = {}) {
  const sourceStep = 1000 / sourceFps;
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  let scheduled = 0;
  const intervals = Math.ceil(sourceFps * seconds);
  for (let index = 0; index < intervals; index += 1) {
    const plan = Cadence.planSourceCadencePresentations({
      nextAt, phaseMs, startAt, sourceIntervalMs: sourceStep,
      outputHz: targetHz, interpolate,
    });
    assert.equal(plan.overflowed, false);
    const presentations = gpuFallback
      ? Cadence.fallbackCadencePresentations(plan.presentations)
      : plan.presentations;
    assert.equal(presentations.length, plan.ticks.length);
    assert.ok(presentations.every(item => ['previous', 'current', 'interpolate'].includes(item.kind)));
    scheduled += presentations.length;
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += sourceStep;
  }
  return scheduled / (intervals * sourceStep / 1000);
}

function simulatePresentationSplit(sourceFps, targetHz, seconds = 600) {
  const sourceStep = 1000 / sourceFps;
  const intervals = Math.ceil(sourceFps * seconds);
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  let anchors = 0;
  let mids = 0;
  let total = 0;
  for (let index = 0; index < intervals; index += 1) {
    const plan = Cadence.planSourceCadencePresentations({
      nextAt, phaseMs, startAt, sourceIntervalMs: sourceStep,
      outputHz: targetHz, interpolate: true,
    });
    assert.equal(plan.overflowed, false);
    anchors += plan.presentations.filter(item => item.kind !== 'interpolate').length;
    mids += plan.presentations.filter(item => item.kind === 'interpolate').length;
    total += plan.presentations.length;
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += sourceStep;
  }
  const duration = intervals / sourceFps;
  return { anchorsHz: anchors / duration, midsHz: mids / duration, totalHz: total / duration };
}

function simulateProductComponents(sourceFps, targetHz, {
  seconds = 120,
  interpolate = Cadence.targetNeedsInterpolation(sourceFps, targetHz),
} = {}) {
  const sourceStep = 1000 / sourceFps;
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  const counts = { previous: 0, current: 0, interpolate: 0 };
  const intervals = Math.ceil(sourceFps * seconds);
  for (let index = 0; index < intervals; index += 1) {
    const plan = Cadence.planSourceCadencePresentations({
      nextAt, phaseMs, startAt, sourceIntervalMs: sourceStep,
      outputHz: targetHz, interpolate,
    });
    assert.equal(plan.overflowed, false);
    if (interpolate && targetHz > sourceFps) {
      assert.equal(plan.presentations.filter(item => item.kind !== 'interpolate').length, 1);
      const anchor = plan.presentations.find(item => item.kind !== 'interpolate');
      const distance = Math.min(Math.abs(anchor.t), Math.abs(1 - anchor.t));
      assert.ok(plan.presentations.every(item => distance
        <= Math.min(Math.abs(item.t), Math.abs(1 - item.t)) + 1e-12));
    }
    for (const presentation of plan.presentations) counts[presentation.kind]++;
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += sourceStep;
  }
  const durationSeconds = intervals * sourceStep / 1000;
  return {
    anchorHz: (counts.previous + counts.current) / durationSeconds,
    midHz: counts.interpolate / durationSeconds,
    totalHz: (counts.previous + counts.current + counts.interpolate) / durationSeconds,
  };
}

for (const sourceFps of [24000 / 1001, 24, 25, 30000 / 1001, 30]) {
  for (const targetHz of [60, 120]) {
    test(`${sourceFps.toFixed(3)} fps source maps deterministically to ${targetHz} FPS`, () => {
      const result = simulateConstantCadence(sourceFps, targetHz);
      assert.ok(Math.abs(result.observedHz - targetHz) < 0.03,
        `observed ${result.observedHz} Hz instead of ${targetHz} Hz`);
      assert.ok(result.maximumTicksPerInterval <= Math.ceil(targetHz / sourceFps) + 1);
      assert.ok(result.nextAt <= result.startAt + 1000 / targetHz + 1e-6);
    });
  }
}

test('VFR-like source intervals keep one continuous target grid', () => {
  const intervals = [41.708, 33.367, 40, 50, 16.683, 41.625, 34.1, 39.4];
  const targetHz = 120;
  const targetStep = 1000 / targetHz;
  let startAt = 500;
  let nextAt = 0;
  const ticks = [];
  for (let cycle = 0; cycle < 200; cycle += 1) {
    for (const interval of intervals) {
      const endAt = startAt + interval;
      const plan = Cadence.planCadenceInterval({ nextAt, startAt, endAt, outputHz: targetHz });
      assert.equal(plan.overflowed, false);
      ticks.push(...plan.ticks.map(tick => tick.at));
      nextAt = plan.nextAt;
      startAt = endAt;
    }
  }
  for (let index = 1; index < ticks.length; index += 1) {
    assert.ok(Math.abs(ticks[index] - ticks[index - 1] - targetStep) < 1e-7);
  }
  assert.ok(nextAt <= startAt + targetStep + 1e-6);
});

test('decoded cadence estimator rejects callback jitter and normalizes common video rates', () => {
  const jittered60 = [16.8, 16.4, 33.3, 16.9, 16.6, 15.9, 17.1, 16.7, 16.5];
  const sixty = Cadence.estimateSourceCadence(jittered60, 42);
  assert.ok(Math.abs(sixty.sourceHz - 60) < 0.1);
  assert.equal(Cadence.targetNeedsInterpolation(sixty.sourceHz, 60), false);

  const ntsc = Cadence.estimateSourceCadence(Array(9).fill(1000 / (60000 / 1001)), 42);
  assert.equal(ntsc.sourceHz, 60000 / 1001);
  assert.equal(Cadence.targetNeedsInterpolation(ntsc.sourceHz, 60), false);

  const fifty = Cadence.estimateSourceCadence([20.1, 19.9, 20, 20.05, 19.95], 42);
  assert.equal(fifty.sourceHz, 50);
  assert.equal(Cadence.targetNeedsInterpolation(fifty.sourceHz, 60), true);

  const arbitrary55 = Cadence.estimateSourceCadence(Array(7).fill(1000 / 55), 42);
  assert.ok(Math.abs(arbitrary55.sourceHz - 55) < 1e-9);
  assert.equal(Cadence.targetNeedsInterpolation(arbitrary55.sourceHz, 60), true);
});

test('decoded cadence estimator validates inputs and preserves its fallback', () => {
  assert.deepEqual(Cadence.estimateSourceCadence([], 20), {
    intervalMs: 20, sourceHz: 50, rawHz: 50, sampleCount: 0, normalized: false,
  });
  assert.throws(() => Cadence.estimateSourceCadence(null, 20), TypeError);
  assert.throws(() => Cadence.estimateSourceCadence([], 0), RangeError);
  assert.throws(() => Cadence.normalizeVideoRate(0), RangeError);
  assert.throws(() => Cadence.targetNeedsInterpolation(60, 0), RangeError);
});

test('decoded cadence estimator becomes ready for a stable 1 FPS source', () => {
  const estimate = Cadence.estimateSourceCadence(Array(8).fill(1000), 42);
  assert.equal(estimate.sampleCount, 8);
  assert.equal(estimate.sourceHz, 1);
  assert.equal(estimate.intervalMs, 1000);

  const target = resolveTarget(2, estimate.sourceHz, 60);
  assert.equal(target.state, 'active');
  assert.equal(target.minimumHz, 2);
  assert.equal(target.outputHz, 2);
});

test('decoded cadence estimator becomes ready for a native 240 FPS source', () => {
  const estimate = Cadence.estimateSourceCadence(Array(8).fill(1000 / 240), 42);
  assert.equal(estimate.sampleCount, 8);
  assert.ok(Math.abs(estimate.sourceHz - 240) < 1e-9);
  assert.ok(Math.abs(estimate.intervalMs - 1000 / 240) < 1e-9);
});

test('low-FPS cadence delay keeps the complete source pair ahead of presentation deadlines', () => {
  for (const sourceHz of [10, 15]) {
    const sourceIntervalMs = 1000 / sourceHz;
    const delayMs = Cadence.computePresentationDelayMs({
      cadenceMode: true,
      sourceIntervalMs,
      midCostMs: 3,
      burstPadMs: 0,
      floorMs: 60,
      maxDelayMs: 2500,
    });
    const currentFrameAt = 1000;
    const pairStartAt = currentFrameAt - sourceIntervalMs + delayMs;
    assert.ok(pairStartAt >= currentFrameAt + 25,
      `${sourceHz} FPS pair must retain classify/submit safety after the current frame arrives`);
  }

  assert.equal(Cadence.computePresentationDelayMs({
    cadenceMode: false,
    sourceIntervalMs: 100,
    midCostMs: 3,
    floorMs: 60,
    maxDelayMs: 180,
  }), 60);
  assert.throws(() => Cadence.computePresentationDelayMs({
    cadenceMode: true,
    sourceIntervalMs: 0,
  }), RangeError);
});

test('arbitrary targets enforce the 2x source floor and measured display cap', () => {
  const fifty = resolveTarget(50, 24, 60);
  assert.deepEqual({ state: fifty.state, outputHz: fifty.outputHz, clamped: fifty.clamped },
    { state: 'active', outputHz: 50, clamped: false });

  const belowFloor = resolveTarget(40, 24, 60);
  assert.equal(belowFloor.outputHz, 48);
  assert.equal(belowFloor.clampReason, 'minimum');
  assert.match(belowFloor.warning, /minimum is 2x/);

  const aboveDisplay = resolveTarget(300, 60, 240);
  assert.equal(aboveDisplay.outputHz, 240 * Cadence.DISPLAY_CLAMP_HEADROOM);
  assert.equal(aboveDisplay.clampReason, 'display');

  const fractional = resolveTarget(59.94, 24, 60);
  assert.equal(fractional.outputHz, 59.94);
  assert.equal(fractional.clamped, false);

  const ntscBoundary = resolveTarget(120, 60000 / 1001, 119.88);
  assert.equal(ntscBoundary.state, 'active');
  assert.equal(ntscBoundary.outputHz, 119.88);
  assert.equal(ntscBoundary.clampReason, null);

  const noRange = resolveTarget(120, 60, 100);
  assert.equal(noRange.state, 'no-2x-display-range');
  assert.equal(noRange.interpolationAllowed, false);
  assert.equal(noRange.outputHz, null);
  assert.match(noRange.warning, /Needs at least 120 Hz/);
});

test('an explicit target at the measured display ceiling keeps recovery headroom', () => {
  const plan = Cadence.resolveOutputRate('target', 1000 / 239.52, {
    targetFps: 240,
    sourceHz: 60,
    sourceReady: true,
    displayReady: true,
    midCostMs: 1,
  });
  assert.equal(plan.clampReason, 'display');
  assert.ok(plan.outputHz < plan.capacityHz);
  assert.ok(Math.abs(plan.outputHz - plan.capacityHz * Cadence.DISPLAY_CLAMP_HEADROOM) < 1e-9);
});

test('playback-adjusted source cadence keeps the exact 2x floor', () => {
  const playbackAdjustedSourceHz = 60 * 1.01;
  const plan = resolveTarget(120, playbackAdjustedSourceHz, 240);
  assert.equal(plan.state, 'active');
  assert.ok(Math.abs(plan.minimumHz - 121.2) < 1e-12);
  assert.ok(Math.abs(plan.outputHz - 121.2) < 1e-12);
  assert.equal(plan.clampReason, 'minimum');
});

test('target mode fails safe until source and display cadence are stable', () => {
  const result = Cadence.resolveOutputRate('target', 100, {
    targetFps: 120, sourceHz: 60, sourceReady: true, displayReady: false,
  });
  assert.equal(result.measured, false);
  assert.equal(result.outputHz, null);
  assert.equal(result.interpolationAllowed, false);
  assert.equal(result.state, 'measuring');

  const noSource = Cadence.resolveOutputRate('target', 1000 / 240, {
    targetFps: 120, sourceReady: false, displayReady: true,
  });
  assert.equal(noSource.state, 'measuring');
  assert.equal(noSource.outputHz, null);
});

test('target resolver separates GPU clamping from an impossible 2x budget', () => {
  const reduced = resolveTarget(60, 24, 120, { midCostMs: 20 });
  assert.equal(reduced.state, 'active');
  assert.equal(reduced.outputHz, 48);
  assert.equal(reduced.clampReason, 'gpu');

  const unavailable = resolveTarget(60, 24, 120, { midCostMs: 40 });
  assert.equal(unavailable.state, 'no-2x-gpu-range');
  assert.equal(unavailable.outputHz, null);
  assert.equal(unavailable.interpolationAllowed, false);
});

test('low source rates cannot exceed scheduler queue and texture bounds', () => {
  const plan = resolveTarget(240, 1, 240);
  assert.equal(plan.state, 'active');
  assert.equal(plan.outputHz, Cadence.MAX_MIDS_PER_PAIR + 1);
  assert.equal(plan.clampReason, 'runtime');
  const split = simulatePresentationSplit(1, plan.outputHz, 60);
  assert.ok(split.midsHz <= Cadence.MAX_MIDS_PER_PAIR + 1e-9);
});

test('ten-minute arbitrary target grids preserve source anchors and generate only the remainder', () => {
  const cases = [
    { source: 24000 / 1001, requested: 47.952, display: 120 },
    { source: 24, requested: 50, display: 60 },
    { source: 24, requested: 59.94, display: 60 },
    { source: 30000 / 1001, requested: 60, display: 120 },
    { source: 55, requested: 90, display: 144 },
    { source: 60, requested: 144, display: 240 },
    { source: 60, requested: 300, display: 240 },
  ];
  for (const item of cases) {
    const plan = resolveTarget(item.requested, item.source, item.display);
    assert.equal(plan.state, 'active');
    const split = simulatePresentationSplit(item.source, plan.outputHz);
    assert.ok(Math.abs(split.totalHz - plan.outputHz) < 0.01,
      `${item.source} -> ${plan.outputHz} total drifted to ${split.totalHz}`);
    assert.ok(Math.abs(split.anchorsHz - item.source) < 0.01,
      `${item.source} -> ${plan.outputHz} source anchors drifted to ${split.anchorsHz}`);
    assert.ok(Math.abs(split.midsHz - (plan.outputHz - item.source)) < 0.01,
      `${item.source} -> ${plan.outputHz} mids drifted to ${split.midsHz}`);
  }
});

test('display estimator requires ten stable samples and ignores one fast outlier', () => {
  const state = { floorMs: 100, ready: false, stableSamples: 0 };
  for (let index = 0; index < Cadence.REFRESH_TRANSITION_SAMPLES - 1; index += 1) {
    assert.equal(Cadence.updateDisplayInterval(state, 1000 / 60), false);
    assert.equal(state.ready, false);
    assert.equal(Cadence.measureDisplayHz(state.floorMs).measured, false);
  }
  assert.equal(Cadence.updateDisplayInterval(state, 1000 / 60), true);
  assert.equal(state.ready, true);
  assert.ok(Math.abs(state.floorMs - 1000 / 60) < 1e-9);

  Cadence.updateDisplayInterval(state, 1000 / 240);
  assert.ok(Math.abs(state.floorMs - 1000 / 60) < 1e-9);
  assert.equal(Cadence.measureDisplayHz(state.floorMs).capacityHz, 60);

  for (let index = 0; index < Cadence.REFRESH_TRANSITION_SAMPLES - 1; index += 1) {
    Cadence.updateDisplayInterval(state, 1000 / 240);
  }
  assert.equal(state.ready, true);
  assert.ok(Math.abs(state.floorMs - 1000 / 240) < 1e-9);
});

test('confirmed display survives isolated slow callbacks and fails safe on a sustained slowdown', () => {
  const state = { floorMs: 100, ready: false, stableSamples: 0 };
  for (let index = 0; index < Cadence.REFRESH_TRANSITION_SAMPLES; index += 1) {
    Cadence.updateDisplayInterval(state, 1000 / 240);
  }
  assert.equal(state.ready, true);

  Cadence.updateDisplayInterval(state, 1000 / 120);
  assert.equal(state.ready, true, 'one scheduling outlier must not disable a confirmed display');
  Cadence.updateDisplayInterval(state, 1000 / 120);
  assert.equal(state.ready, true, 'two scheduling outliers remain transient');
  Cadence.updateDisplayInterval(state, 1000 / 120);
  assert.equal(state.ready, false, 'three agreeing slow samples begin fail-safe transition');

  for (let index = 3; index < Cadence.REFRESH_TRANSITION_SAMPLES; index += 1) {
    Cadence.updateDisplayInterval(state, 1000 / 120);
  }
  assert.equal(state.ready, true);
  assert.ok(Math.abs(state.floorMs - 1000 / 120) < 1e-9);
});

test('quantized 4/4/5ms callbacks never authorize more service than they deliver', () => {
  const state = { floorMs: 100, ready: false, stableSamples: 0 };
  const pattern = [4, 4, 5];
  for (let index = 0; index < 30; index += 1) {
    Cadence.updateDisplayInterval(state, pattern[index % pattern.length]);
    if (index < Cadence.REFRESH_TRANSITION_SAMPLES - 1) assert.equal(state.ready, false);
  }
  const deliveredHz = 1000 / (pattern.reduce((sum, value) => sum + value, 0) / pattern.length);
  const measured = Cadence.measureDisplayHz(state.floorMs);
  assert.equal(state.ready, true);
  assert.ok(measured.capacityHz <= deliveredHz + 1e-9,
    `capacity ${measured.capacityHz} exceeded delivered ${deliveredHz}`);
  assert.ok(measured.capacityHz > 220, `capacity ${measured.capacityHz} was excessively conservative`);
});

test('display estimator accepts faster and slower transitions only after ten stable samples', () => {
  for (const [fromHz, toHz] of [[60, 120], [60, 50], [144, 120]]) {
    const state = { floorMs: 100, ready: false, stableSamples: 0 };
    for (let index = 0; index < Cadence.REFRESH_TRANSITION_SAMPLES; index += 1) {
      Cadence.updateDisplayInterval(state, 1000 / fromHz);
    }
    assert.equal(state.ready, true);
    assert.ok(Math.abs(state.floorMs - 1000 / fromHz) < 1e-9);
    for (let index = 0; index < Cadence.REFRESH_TRANSITION_SAMPLES - 1; index += 1) {
      assert.equal(Cadence.updateDisplayInterval(state, 1000 / toHz), false);
      assert.ok(Math.abs(state.floorMs - 1000 / fromHz) < 1e-9);
    }
    assert.equal(Cadence.updateDisplayInterval(state, 1000 / toHz), true);
    assert.ok(Math.abs(state.floorMs - 1000 / toHz) < 1e-9);
  }
});

for (const sourceFps of [50, 60000 / 1001, 60, 75, 120]) {
  for (const targetHz of [60, 120]) {
    test(`product scheduling decimates ${sourceFps.toFixed(3)} to ${targetHz} without duplicate fallbacks`, () => {
      const observedHz = simulateProductPresentations(sourceFps, targetHz);
      assert.ok(Math.abs(observedHz - targetHz) < 0.03,
        `observed ${observedHz} scheduled presentations/s instead of ${targetHz}`);
    });
  }
}

for (const [sourceFps, targetHz, expectedAnchors, expectedMids] of [
  [24, 60, 24, 36],
  [25, 60, 25, 35],
  [30, 120, 30, 90],
  [50, 60, 50, 10],
  [55, 60, 55, 5],
  [60, 120, 60, 60],
]) {
  test(`${sourceFps} to ${targetHz} keeps one source anchor per interval and fills only the remainder`, () => {
    const result = simulateProductComponents(sourceFps, targetHz);
    assert.ok(Math.abs(result.anchorHz - expectedAnchors) < 0.03,
      `observed ${result.anchorHz} source anchors/s instead of ${expectedAnchors}`);
    assert.ok(Math.abs(result.midHz - expectedMids) < 0.03,
      `observed ${result.midHz} mids/s instead of ${expectedMids}`);
    assert.ok(Math.abs(result.totalHz - targetHz) < 0.03);
  });
}

test('60 to 60 and no-interpolation intervals never invoke the model', () => {
  const equalRate = simulateProductComponents(60, 60);
  assert.ok(Math.abs(equalRate.totalHz - 60) < 0.03);
  assert.equal(equalRate.midHz, 0);

  const cutOrDuplicate = simulateProductComponents(24, 120, { interpolate: false });
  assert.ok(Math.abs(cutOrDuplicate.totalHz - 120) < 0.03);
  assert.equal(cutOrDuplicate.midHz, 0);
  assert.equal(cutOrDuplicate.anchorHz, cutOrDuplicate.totalHz);
});

test('one nearest anchor per interval is stable across target-grid phase and source jitter', () => {
  const intervals = [16.2, 17.05, 16.45, 16.95, 16.55, 16.8];
  const outputHz = 120;
  const stepMs = 1000 / outputHz;
  for (const phase of [0.01, 0.25, 0.5, 0.75, 0.99]) {
    let startAt = 1000;
    let nextAt = startAt + phase * stepMs;
    for (let cycle = 0; cycle < 40; cycle += 1) {
      for (const intervalMs of intervals) {
        const plan = Cadence.planCadencePresentations({
          nextAt, startAt, endAt: startAt + intervalMs, outputHz, interpolate: true,
        });
        assert.equal(plan.presentations.filter(item => item.kind !== 'interpolate').length, 1);
        const anchor = plan.presentations.find(item => item.kind !== 'interpolate');
        const anchorDistance = Math.min(Math.abs(anchor.t), Math.abs(1 - anchor.t));
        assert.ok(plan.presentations.every(item => anchorDistance
          <= Math.min(Math.abs(item.t), Math.abs(1 - item.t)) + 1e-12));
        nextAt = plan.nextAt;
        startAt += intervalMs;
      }
    }
  }
});

test('60 to 120 media phase stays factor2 when wall callback timing jitters independently', () => {
  const wallIntervals = [13.4, 19.8, 15.1, 18.2, 16.5, 17.0];
  const sourceIntervalMs = 1000 / 60;
  const outputHz = 120;
  const deadlines = [];
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  for (let index = 0; index < 960; index += 1) {
    const plan = Cadence.planSourceCadencePresentations({
      nextAt,
      phaseMs,
      startAt,
      sourceIntervalMs,
      outputHz,
      interpolate: true,
    });
    if (index >= 240) {
      assert.equal(plan.presentations.length, 2);
      assert.equal(plan.presentations.filter(item => item.kind === 'interpolate').length, 1);
      assert.equal(plan.presentations.filter(item => item.kind !== 'interpolate').length, 1);
      deadlines.push(...plan.presentations.map(item => item.at));
    }
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += wallIntervals[index % wallIntervals.length];
  }
  const stepMs = 1000 / outputHz;
  for (let index = 1; index < deadlines.length; index += 1) {
    assert.ok(Math.abs(deadlines[index] - deadlines[index - 1] - stepMs) < 1e-7);
  }
});

test('noisy nominal-60 media deltas use the robust cadence phase instead of raw per-frame jitter', () => {
  const nominalIntervalMs = 1000 / 60;
  // Chromium can quantize nominal-60 media timestamps into alternating short
  // and long deltas; their pair average, not either median bucket, is the rate.
  const mediaIntervals = [20.733333333333, 12.6];
  const samples = [];
  const deadlines = [];
  let sourceIntervalMs = 42;
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  for (let index = 0; index < 960; index += 1) {
    samples.push(mediaIntervals[index % mediaIntervals.length]);
    if (samples.length > 32) samples.shift();
    sourceIntervalMs = Cadence.estimateSourceCadence(samples, sourceIntervalMs).intervalMs;
    const plan = Cadence.planSourceCadencePresentations({
      nextAt, phaseMs, startAt, sourceIntervalMs, outputHz: 120, interpolate: true,
    });
    if (index >= 240) {
      assert.ok(Math.abs(sourceIntervalMs - nominalIntervalMs) < 1e-9);
      assert.equal(plan.presentations.length, 2);
      assert.equal(plan.presentations.filter(item => item.kind === 'interpolate').length, 1);
      assert.equal(plan.presentations.filter(item => item.kind !== 'interpolate').length, 1);
      deadlines.push(...plan.presentations.map(item => item.at));
    }
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += nominalIntervalMs;
  }
  const targetStepMs = 1000 / 120;
  for (let index = 1; index < deadlines.length; index += 1) {
    assert.ok(Math.abs(deadlines[index] - deadlines[index - 1] - targetStepMs) < 1e-7);
  }
});

test('a sustained decoded-rate change replaces the nominal cadence lock', () => {
  const samples = [];
  let intervalMs = 42;
  for (let index = 0; index < 64; index += 1) {
    samples.push(1000 / 60);
    if (samples.length > 32) samples.shift();
    intervalMs = Cadence.estimateSourceCadence(samples, intervalMs).intervalMs;
  }
  assert.ok(Math.abs(intervalMs - 1000 / 60) < 1e-9);
  for (let index = 0; index < 64; index += 1) {
    samples.push(1000 / 30);
    if (samples.length > 32) samples.shift();
    intervalMs = Cadence.estimateSourceCadence(samples, intervalMs).intervalMs;
  }
  assert.ok(Math.abs(intervalMs - 1000 / 30) < 1e-9);
});

test('a 70ms wall stall resyncs by whole target steps without queuing stale deadlines', () => {
  const sourceIntervalMs = 1000 / 60;
  const outputHz = 120;
  const targetStepMs = 1000 / outputHz;
  let startAt = 1000;
  let nextAt = 0;
  let phaseMs = 0;
  let lastDeadline = null;
  for (let index = 0; index < 30; index += 1) {
    const plan = Cadence.planSourceCadencePresentations({
      nextAt, phaseMs, startAt, sourceIntervalMs, outputHz, interpolate: true,
    });
    if (plan.presentations.length) lastDeadline = plan.presentations.at(-1).at;
    nextAt = plan.nextAt;
    phaseMs = plan.nextPhaseMs;
    startAt += sourceIntervalMs;
  }

  startAt += 70;
  const phaseBeforeStallPlan = phaseMs;
  const stalled = Cadence.planSourceCadencePresentations({
    nextAt, phaseMs, startAt, sourceIntervalMs, outputHz, interpolate: true,
  });
  assert.equal(stalled.resynced, true);
  assert.equal(stalled.presentations.length, 2);
  assert.equal(stalled.presentations.filter(item => item.kind === 'interpolate').length, 1);
  const desiredFirstDeadline = startAt + phaseBeforeStallPlan;
  assert.ok(stalled.presentations[0].at >= desiredFirstDeadline - 1e-7);
  assert.ok(stalled.presentations[0].at < desiredFirstDeadline + targetStepMs + 1e-7);
  const gridStepsAcrossStall = (stalled.presentations[0].at - lastDeadline) / targetStepMs;
  assert.ok(gridStepsAcrossStall > 1);
  assert.ok(Math.abs(gridStepsAcrossStall - Math.round(gridStepsAcrossStall)) < 1e-7);
  assert.ok(Math.abs(stalled.presentations[1].at - stalled.presentations[0].at - targetStepMs) < 1e-7);
});

test('cuts, anime duplicates and GPU fallback preserve the same target cadence', () => {
  const regular = simulateProductPresentations(24, 120);
  const duplicate = simulateProductPresentations(24, 120, { interpolate: false });
  const cut = simulateProductPresentations(24, 120, { interpolate: false });
  const overloaded = simulateProductPresentations(24, 120, { gpuFallback: true });
  for (const observedHz of [regular, duplicate, cut, overloaded]) {
    assert.ok(Math.abs(observedHz - 120) < 0.03, `cadence fell to ${observedHz}`);
  }
});

test('pathological source gaps cannot create an unbounded tick queue', () => {
  const plan = Cadence.planCadenceInterval({ nextAt: 0, startAt: 1000, endAt: 11000, outputHz: 120 });
  assert.equal(plan.overflowed, true);
  assert.deepEqual(plan.ticks, []);
  assert.equal(plan.nextAt, 11000 + 1000 / 120);
});

test('presentation queue evicts the oldest deadline at a fixed upper bound', () => {
  const queue = [];
  let evictions = 0;
  for (let index = 0; index < 10000; index += 1) {
    if (Cadence.enqueuePresentation(queue, { at: index, id: index })) evictions++;
  }
  assert.equal(queue.length, Cadence.MAX_PENDING_PRESENTATIONS);
  assert.equal(evictions, 10000 - Cadence.MAX_PENDING_PRESENTATIONS);
  assert.equal(queue[0].id, 10000 - Cadence.MAX_PENDING_PRESENTATIONS);

  const outOfOrder = Array.from({ length: Cadence.MAX_PENDING_PRESENTATIONS }, (_, index) => ({ at: index + 10 }));
  outOfOrder[11].at = -1;
  const evicted = Cadence.enqueuePresentation(outOfOrder, { at: 999 });
  assert.equal(evicted.at, -1);
  assert.equal(outOfOrder.length, Cadence.MAX_PENDING_PRESENTATIONS);

  const current = Array.from({ length: Cadence.MAX_PENDING_PRESENTATIONS }, (_, index) => ({ at: index + 100 }));
  const alreadyLate = { at: 1, id: 'already-late' };
  assert.equal(Cadence.enqueuePresentation(current, alreadyLate), alreadyLate);
  assert.equal(current.length, Cadence.MAX_PENDING_PRESENTATIONS);
  assert.equal(current.some(entry => entry.id === 'already-late'), false);
});

test('arbitrary targets recover a transient due backlog oldest-first when display service has headroom', () => {
  const queue = [
    { at: 100, id: 'anchor' },
    { at: 100 + 1000 / 120, id: 'mid' },
    { at: 100 + 2000 / 120, id: 'next-anchor' },
  ];
  const untouched = queue.map(entry => ({ ...entry }));
  const first = Cadence.selectDuePresentation(queue, 112, {
    targetHz: 120, displayCapacityHz: 239.8,
  });
  assert.deepEqual(first, {
    presentIndex: 0, dueCount: 2, dropCount: 0, recovering: true, recoveryCapacity: 3,
  });
  assert.deepEqual(queue, untouched, 'selection helper must stay pure');

  const presented = [];
  let now = 112;
  while (queue.length) {
    const selection = Cadence.selectDuePresentation(queue, now, {
      targetHz: 120, displayCapacityHz: 239.8,
    });
    if (selection.presentIndex >= 0) {
      assert.equal(selection.dropCount, 0);
      presented.push(queue[selection.presentIndex].id);
      queue.splice(0, selection.presentIndex + 1);
    }
    now += 1000 / 240;
  }
  assert.deepEqual(presented, ['anchor', 'mid', 'next-anchor']);
});

test('newest-due policy remains without recovery headroom or an exact target', () => {
  const queue = [{ at: 100, id: 'old' }, { at: 108, id: 'new' }, { at: 120, id: 'future' }];
  for (const options of [
    { targetHz: 240, displayCapacityHz: 240 },
    { targetHz: 0, displayCapacityHz: 240 },
  ]) {
    assert.deepEqual(Cadence.selectDuePresentation(queue, 110, options), {
      presentIndex: 1, dueCount: 2, dropCount: 1, recovering: false, recoveryCapacity: 1,
    });
  }
});

test('catch-up is bounded and requires deadline-ordered queues', () => {
  const threeDue = [
    { at: 90, id: 0 }, { at: 98, id: 1 }, { at: 106, id: 2 }, { at: 120, id: 3 },
  ];
  assert.deepEqual(Cadence.selectDuePresentation(threeDue, 110, {
    targetHz: 120, displayCapacityHz: 240,
  }), {
    presentIndex: 0, dueCount: 3, dropCount: 0, recovering: true, recoveryCapacity: 3,
  });

  const fourDue = [
    { at: 82, id: 0 }, { at: 90, id: 1 }, { at: 98, id: 2 },
    { at: 106, id: 3 }, { at: 120, id: 4 },
  ];
  assert.deepEqual(Cadence.selectDuePresentation(fourDue, 110, {
    targetHz: 120, displayCapacityHz: 240,
  }), {
    presentIndex: 3, dueCount: 4, dropCount: 3, recovering: false, recoveryCapacity: 3,
  });

  const twoDue = [{ at: 100, id: 0 }, { at: 108, id: 1 }];
  const cappedRecovery = Cadence.selectDuePresentation(twoDue, 110, {
    targetHz: 60, displayCapacityHz: 240,
  });
  assert.equal(cappedRecovery.presentIndex, 0);
  assert.equal(cappedRecovery.recovering, true);
  assert.equal(cappedRecovery.recoveryCapacity, Cadence.MAX_RECOVERY_PRESENTATIONS);

  assert.throws(() => Cadence.selectDuePresentation([
    { at: 108 }, { at: 100 },
  ], 110, { targetHz: 120, displayCapacityHz: 240 }), /ordered by deadline/);
});

test('seek/reset discards the old cadence epoch and resyncs at the new timeline', () => {
  const beforeSeek = Cadence.planCadenceInterval({
    nextAt: 0, startAt: 1000, endAt: 1000 + 1000 / 24, outputHz: 60,
  });
  const seekStart = 500000;
  const afterSeek = Cadence.planCadenceInterval({
    nextAt: 0, startAt: seekStart, endAt: seekStart + 1000 / 24, outputHz: 60,
  });
  const staleState = Cadence.planCadenceInterval({
    nextAt: beforeSeek.nextAt, startAt: seekStart, endAt: seekStart + 1000 / 24, outputHz: 60,
  });
  assert.equal(afterSeek.resynced, true);
  assert.equal(staleState.resynced, true);
  assert.equal(afterSeek.ticks[0].at, seekStart + 1000 / 60);
  assert.deepEqual(staleState.ticks, afterSeek.ticks);
});

test('storage values preserve new modes and reject invalid targets', () => {
  assert.equal(Cadence.sanitizeOutputRate('target'), 'target');
  assert.equal(Cadence.sanitizeOutputRate('fps60'), 'target');
  assert.equal(Cadence.sanitizeOutputRate('fps120'), 'target');
  assert.equal(Cadence.sanitizeOutputRate('4'), 4);
  assert.equal(Cadence.sanitizeOutputRate('fps144'), 'auto');
  assert.equal(Cadence.sanitizeTargetFps('143.5'), 143.5);
  assert.equal(Cadence.sanitizeTargetFps(''), null);
  assert.equal(Cadence.sanitizeTargetFps(Infinity), null);
  assert.equal(Cadence.sanitizeTargetFps(-1), null);
  assert.equal(Cadence.outputRateLabel('target', 143.5), '143.5 FPS');
  assert.equal(Cadence.outputRateLabel('fps120'), '120 FPS');
});

test('extension loads the helper first and exposes every output-rate choice', () => {
  const manifest = JSON.parse(readFileSync(`${extensionDirectory}/manifest.json`, 'utf8'));
  assert.deepEqual(manifest.content_scripts[0].js, ['cadence.js', 'content.js']);
  const content = readFileSync(`${extensionDirectory}/content.js`, 'utf8');
  for (const value of ['auto', 'hz', 'target', '2', '3', '4', '5', '6']) {
    assert.match(content, new RegExp(`<option value="${value}">`));
  }
  assert.doesNotMatch(content, /<option value="fps(?:60|120)">/);
  assert.match(content, /id="fcTargetFps"/);
  assert.match(content, /<span>Output rate/);
  assert.match(content, /Cadence\.isCadenceMode\(cfg\.factor\)/);
  assert.match(content, /Cadence\.planSourceCadencePresentations\(/);
  assert.match(content, /Cadence\.fallbackCadencePresentations\(/);
  assert.match(content, /Cadence\.estimateSourceCadence\(/);
  assert.match(content, /Cadence\.targetNeedsInterpolation\(/);
  assert.match(content, /Cadence\.selectDuePresentation\(/);
  assert.match(content, /const wallPairMs = decodedIntervalMs \/ playbackRate/);
  assert.match(content, /Cadence\.computePresentationDelayMs\(/);
  assert.match(content, /startAt: schedT - schedulingIntervalMs \+ delayMs/);
  assert.match(content, /sourceIntervalMs: wallPairMs/);
  assert.match(content, /cadenceIntervalMs: wallPairMs/);
  assert.doesNotMatch(content, /decodedPairIntervalMs/);
  assert.match(content, /sampleMs \/= frameDelta/);
  assert.match(content, /decodedFrameDelta > 1[\s\S]*?lastTex = null[\s\S]*?hzNext = 0/,
    'a missed source callback must break pair history before the next interpolation');
  assert.match(content, /sourceGapHistoryBreaks\+\+/,
    'source-gap recovery must remain observable in product evidence');
  assert.match(content, /pumpWorkMs[\s\S]*?sourceWorkMs/,
    'product evidence must separate runtime callback work from browser scheduling stalls');
  assert.match(content, /decodedIntervalSamples\.length > 32/);
  const cadencePlan = content.slice(content.indexOf('function planPairCadence'),
    content.indexOf('function scheduleCadenceAnchors'));
  assert.match(cadencePlan, /1000 \/ timing\.sourceIntervalMs/);
  assert.doesNotMatch(cadencePlan, /uniqueIntervalMs/);
  assert.doesNotMatch(content, /hzTs\.length === 0/);
  assert.match(content, /addEventListener\('seeking', onSrcChange\)/);
  assert.match(content, /ratePlan = syncOutputRatePlan\(ratePlan\)/,
    'the cadence clock must consume the committed, not transient, rate plan');
  const rateIdentityChange = content.slice(content.indexOf('function outputRateIdentityChange'),
    content.indexOf('function stableOutputRatePlan'));
  assert.doesNotMatch(rateIdentityChange, /minimumHz/,
    'source-floor refinement alone must not reset an unchanged output clock');
  assert.match(content, /minimumHz: plan\.minimumHz/,
    'the committed plan must expose the current exact 2x source floor');
  assert.match(content, /OUTPUT_RATE_TRANSITION_SAMPLES = 16/,
    'material runtime-capacity changes require sustained confirmation');
  assert.match(content, /resetOutputCadence\(false\);[\s\S]*?outputRatePlanIdentity = nextIdentity/,
    'a committed rate change must preserve already scheduled presentations');
  assert.match(content, /midCostSamples\.length > 16[\s\S]*?0\.25/,
    'GPU admission must reject queue-drain outliers with a rolling clean-cost estimate');
  assert.match(content, /UI_UPDATE_INTERVAL_MS = 1000 \/ 15/,
    'control UI service must stay fixed at 15 Hz on high-refresh displays');
  assert.match(content, /if \(now >= nextUiUpdateAt\)[\s\S]*?nextUiUpdateAt = now \+ UI_UPDATE_INTERVAL_MS[\s\S]*?updateBar\(\)/,
    'geometry and control-bar maintenance must share the fixed UI service tick');
  assert.ok(content.indexOf('present(queue[due].tex, queue[due].mid)')
    < content.indexOf('// Queue the current canvas blit before future inference'),
  'the current presentation must be queued before future inference work');
});
