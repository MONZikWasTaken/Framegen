import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import {
  ProductHfrAcceptanceError,
  validateProductHfrReport,
} from './product_hfr_acceptance.mjs';

const HASH = 'a'.repeat(64);
const HASH_NAMES = [
  'contentScript', 'cadenceScript', 'extensionManifest', 'fixture', 'runtime', 'weights',
  'weightsManifest', 'runner', 'gateManifest', 'validator',
];

function expected() {
  return {
    profile: 'v7s-720p-product-hfr',
    factors: [3, 4],
    fixtureName: 'web/product_hfr_fixture.html',
    pattern: 'moving-primitives-v1',
    width: 1280,
    height: 720,
    sourceFps: 60,
    warmupMs: 5000,
    measureMs: 1000,
    browserChannel: 'chromium',
    hashes: Object.fromEntries(HASH_NAMES.map(name => [name, HASH])),
    stagedHashes: {
      contentScript: HASH,
      cadenceScript: HASH,
      extensionManifest: HASH,
      runtime: HASH,
      weights: HASH,
      weightsManifest: HASH,
    },
    durationToleranceFraction: 0.02,
    sourceHzToleranceFraction: 0.03,
    minimumDisplayHzFraction: 0.95,
    minimumScheduledHzFraction: 0.95,
    minimumPresentedHzFraction: 0.94,
    maximumDroppedFrames: 0,
    maximumQueueHighWater: 24,
    maximumPendingFrames: 24,
    maximumLateP95OutputIntervals: 1.25,
    maximumLateMaxOutputIntervals: 3,
    maximumSourceBusySkipped: 0,
    maximumSourcePresentedFrameGaps: 0,
    maximumSourceMetadataDuplicates: 0,
    maximumPoolExhaustion: 0,
    maximumClassificationSuperseded: 0,
    maximumSkippedPairs: 0,
    maximumProducerSkippedScheduleSlots: 0,
    minimumSourceCallbacksFraction: 0.96,
    minimumPairCoverageFraction: 0.98,
    requiredDeviceFeatures: ['shader-f16', 'timestamp-query'],
  };
}

function repeated(count, value) {
  return Array.from({ length: count }, () => value);
}

function factorResult(factor) {
  const sourceCallbacks = 60;
  const pairPlans = 59;
  const scheduledSource = 60;
  const scheduledMid = pairPlans * (factor - 1);
  const scheduled = scheduledSource + scheduledMid;
  const pending = 5;
  const presented = scheduled - pending;
  const rafCallbacks = factor * 60;
  return {
    schemaVersion: 1,
    fixture: { name: 'web/product_hfr_fixture.html', pattern: 'moving-primitives-v1',
      width: 1280, height: 720, nominalSourceFps: 60 },
    conditions: { factor, warmupMs: 5000, measureMs: 1000 },
    bridge: { bridgeVersion: 2, extensionVersion: '1.4.5' },
    configured: { factor, targetFps: 120, fpsLimit: null, anime: false,
      resolution: 720, model: 'v7s' },
    prepared: { convTune: { coc: 8, slab: 12 }, gpu: 'test',
      deviceFeatures: ['shader-f16', 'timestamp-query'] },
    started: { running: true, width: 1280, height: 720 },
    producer: {
      durationMs: 1000,
      producedFrames: sourceCallbacks,
      skippedScheduleSlots: 0,
      observedHz: 60,
      intervalsMs: repeated(sourceCallbacks - 1, 1000 / 60),
    },
    telemetry: {
      schemaVersion: 1,
      epoch: 1,
      durationMs: 1000,
      counters: {
        sourceCallbacks,
        sourceProcessed: sourceCallbacks,
        sourceBusySkipped: 0,
        sourcePresentedFrameGaps: 0,
        sourceMetadataDuplicates: 0,
        rafCallbacks,
        scheduled,
        scheduledSource,
        scheduledMid,
        presented,
        presentedSource: scheduledSource - 1,
        presentedMid: presented - (scheduledSource - 1),
        dropped: 0,
        droppedSource: 0,
        droppedMid: 0,
        pending,
        queueHighWater: 10,
        sourcePoolExhausted: 0,
        midPoolExhausted: 0,
        classificationSuperseded: 0,
        pairPlans,
        skippedPairs: 0,
        plannedMids: scheduledMid,
      },
      plannedFactorHistogram: { [factor]: pairPlans },
      observed: { sourceCallbackHz: 60, sourceHz: 60, rafHz: factor * 60 },
      lateness: { count: presented, p95Ms: 2, maxMs: 2 },
      samples: {
        sourceCallbackIntervalsMs: repeated(sourceCallbacks - 1, 1000 / 60),
        sourceMediaIntervalsMs: repeated(sourceCallbacks - 1, 1000 / 60),
        rafIntervalsMs: repeated(rafCallbacks - 1, 1000 / (factor * 60)),
        lateMs: repeated(presented, 2),
      },
      sampleOverflow: false,
      errors: [],
      product: {
        running: true,
        factor,
        effectiveFactor: factor,
        resolution: 720,
        model: 'v7s',
        modelAsset: 'rt_v7s',
        videoWidth: 1280,
        videoHeight: 720,
        gpu: 'test',
        integrated: false,
        deviceFeatures: ['shader-f16', 'timestamp-query'],
        convTune: { coc: 8, slab: 12 },
        scheduler: { msAvg: 2, intervalMs: 1000 / 60, uniqueIntervalMs: 1000 / 60,
          delayMs: 60, lateAvg: 1, rafMs: 1000 / (factor * 60), rafFloor: 1000 / (factor * 60) },
        cuts: 0,
        duplicates: 0,
      },
    },
  };
}

function report() {
  const sourceFiles = Object.fromEntries(HASH_NAMES.map(name => [name, {
    path: name,
    sizeBytes: 1,
    sha256Start: HASH,
    sha256End: HASH,
  }]));
  return {
    schemaVersion: 1,
    profile: 'v7s-720p-product-hfr',
    source: { files: sourceFiles },
    browser: {
      channel: 'chromium',
      headed: true,
      persistentContext: true,
      unpackedExtension: true,
      extensionStageHashes: expected().stagedHashes,
    },
    workload: { model: 'v7s', resolution: 720, width: 1280, height: 720,
      sourceFps: 60, factors: [3, 4] },
    conditions: { warmupMsPerFactor: 5000, measureMsPerFactor: 1000,
      presentationMetric: 'gpu-canvas-submit-from-rAF-pump' },
    measurements: { x3: factorResult(3), x4: factorResult(4) },
    diagnostics: { consoleErrors: [], pageErrors: [], requestFailures: [] },
  };
}

function clone(value) {
  return structuredClone(value);
}

test('fixture scripts parse and authoritative manifest is frozen', async () => {
  const fixture = await readFile(new URL('../web/product_hfr_fixture.html', import.meta.url), 'utf8');
  const start = fixture.indexOf('<script>');
  const end = fixture.lastIndexOf('</script>');
  assert.ok(start >= 0 && end > start);
  assert.doesNotThrow(() => new vm.Script(fixture.slice(start + '<script>'.length, end)));
  const contentScript = await readFile(new URL('../extension/content.js', import.meta.url), 'utf8');
  assert.doesNotThrow(() => new vm.Script(contentScript));
  const cadenceScript = await readFile(new URL('../extension/cadence.js', import.meta.url), 'utf8');
  assert.doesNotThrow(() => new vm.Script(cadenceScript));
  assert.match(contentScript, /location\.hostname === '127\.0\.0\.1'/);
  assert.match(contentScript, /framegenBenchToken/);
  const manifest = JSON.parse(await readFile(new URL('./product_hfr_gate_manifest.json', import.meta.url), 'utf8'));
  assert.deepEqual(manifest.measurement.factors, [3, 4]);
  assert.equal(manifest.measurement.warmupMsPerFactor, 5000);
  assert.equal(manifest.measurement.measureMsPerFactor, 30000);
  assert.equal(manifest.playwrightCli.browserChannel, 'chromium');
  assert.equal(manifest.extension.cadenceScript, 'extension/cadence.js');
});

test('valid product HFR report passes', () => {
  const result = validateProductHfrReport(report(), expected());
  assert.equal(result.passed, true);
  assert.deepEqual(result.failures, []);
  assert.equal(result.factors.length, 2);
});

test('source cadence mismatch fails closed', () => {
  const value = report();
  value.measurements.x3.telemetry.samples.sourceMediaIntervalsMs.fill(20);
  value.measurements.x3.telemetry.observed.sourceHz = 50;
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('source cadence mismatch')));
});

test('display rAF mismatch fails closed', () => {
  const value = report();
  value.measurements.x4.telemetry.samples.rafIntervalsMs.fill(1000 / 120);
  value.measurements.x4.telemetry.observed.rafHz = 120;
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('display/rAF Hz mismatch')));
});

test('drops fail closed', () => {
  const value = report();
  const row = value.measurements.x3.telemetry;
  row.counters.presented--;
  row.counters.presentedMid--;
  row.counters.dropped = 1;
  row.counters.droppedMid = 1;
  row.samples.lateMs.pop();
  row.lateness.count--;
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('dropped frames')));
});

test('queue and pool exhaustion fail closed', () => {
  const value = report();
  value.measurements.x3.telemetry.counters.queueHighWater = 25;
  value.measurements.x4.telemetry.counters.midPoolExhausted = 1;
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('queue contract')));
  assert.ok(result.failures.some(message => message.includes('texture pool exhausted')));
});

test('lateness contract fails closed', () => {
  const value = report();
  const row = value.measurements.x4.telemetry;
  row.samples.lateMs.fill(8);
  row.lateness.p95Ms = 8;
  row.lateness.maxMs = 8;
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('lateness contract')));
});

test('browser errors fail closed', () => {
  const value = report();
  value.diagnostics.pageErrors.push({ message: 'device lost' });
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('browser errors')));
});

test('source hash drift and staged hash mismatch fail closed', () => {
  const value = report();
  value.source.files.runtime.sha256End = 'b'.repeat(64);
  value.browser.extensionStageHashes.weights = 'c'.repeat(64);
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('hash drift')));
  assert.ok(result.failures.some(message => message.includes('unpacked extension hash mismatch')));
});

test('factor step-down fails closed', () => {
  const value = report();
  const row = value.measurements.x4.telemetry;
  row.plannedFactorHistogram = { 3: 1, 4: 58 };
  const result = validateProductHfrReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('stepped down')));
});

test('missing applied tune is structurally invalid', () => {
  const value = report();
  value.measurements.x3.telemetry.product.convTune = null;
  assert.throws(() => validateProductHfrReport(value, expected()), ProductHfrAcceptanceError);
});

test('queue conservation mismatch is structurally invalid', () => {
  const value = report();
  value.measurements.x3.telemetry.counters.pending++;
  assert.throws(() => validateProductHfrReport(value, expected()), ProductHfrAcceptanceError);
});
