import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

import {
  ProductTargetFpsAcceptanceError,
  validateProductTargetFpsReport,
} from './product_target_fps_acceptance.mjs';
import {
  PRODUCT_TARGET_FPS_RUNNER_CONTRACT,
  parseArguments,
  resolveOutputPath,
  validateManifest,
} from './run_product_target_fps_gate.mjs';

const HASH = 'a'.repeat(64);
const HASH_NAMES = Object.keys(PRODUCT_TARGET_FPS_RUNNER_CONTRACT.sourceFiles);
const CASES = PRODUCT_TARGET_FPS_RUNNER_CONTRACT.cases;
const DURATION_MS = 12000;

function repeated(count, value) {
  return Array.from({ length: count }, () => value);
}

function expected() {
  return {
    gateId: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.gateId,
    profile: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.profile,
    cases: CASES,
    fixtureName: 'web/product_target_fps_fixture.html',
    pattern: 'moving-primitives-v1',
    width: 1280,
    height: 720,
    supportedSourceFps: [10, 15, 24, 60],
    expectedDisplayHz: 240,
    warmupMs: 4000,
    measureMs: DURATION_MS,
    browserChannel: 'chromium',
    browserDistribution: 'chrome-for-testing',
    hashes: Object.fromEntries(HASH_NAMES.map(name => [name, HASH])),
    stagedHashes: {
      contentScript: HASH,
      cadenceScript: HASH,
      extensionManifest: HASH,
      runtime: HASH,
      weights: HASH,
      weightsManifest: HASH,
    },
    durationToleranceFraction: 0.03,
    sourceHzToleranceFraction: 0.03,
    displayHzToleranceFraction: 0.08,
    targetHzToleranceFraction: 0.04,
    componentHzToleranceFraction: 0.05,
    minimumPresentedHzFraction: 0.95,
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
    maximumProducerSkippedScheduleSlots: 0,
    minimumSourceCallbacksFraction: 0.96,
    minimumPairCoverageFraction: 0.98,
    maximumMidPipelineLag: 3,
    requiredDeviceFeatures: ['shader-f16', 'timestamp-query'],
  };
}

function histogramFor(pairPlans, plannedMids) {
  const lowerMids = Math.floor(plannedMids / pairPlans);
  const upperCount = plannedMids - lowerMids * pairPlans;
  const lowerCount = pairPlans - upperCount;
  const histogram = {};
  if (lowerCount) histogram[lowerMids + 1] = lowerCount;
  if (upperCount) histogram[lowerMids + 2] = upperCount;
  return histogram;
}

function caseResult(targetCase) {
  const sourceCallbacks = targetCase.sourceFps * DURATION_MS / 1000;
  const pairPlans = sourceCallbacks - 1;
  const scheduledSource = sourceCallbacks;
  const scheduledMid = Math.round((targetCase.expectedEffectiveFps - targetCase.sourceFps)
    * DURATION_MS / 1000);
  const scheduled = scheduledSource + scheduledMid;
  const pendingSource = 2;
  const pendingMid = 2;
  const presentedSource = scheduledSource - pendingSource;
  const presentedMid = scheduledMid - pendingMid;
  const presented = presentedSource + presentedMid;
  const rafCallbacks = 240 * DURATION_MS / 1000;
  const displayCapacityHz = 240;
  return {
    schemaVersion: 2,
    fixture: {
      name: 'web/product_target_fps_fixture.html',
      pattern: 'moving-primitives-v1',
      width: 1280,
      height: 720,
      nominalSourceFps: targetCase.sourceFps,
    },
    conditions: {
      caseId: targetCase.id,
      factor: 'target',
      targetFps: targetCase.targetFps,
      sourceFps: targetCase.sourceFps,
      resolution: targetCase.resolution,
      warmupMs: 4000,
      measureMs: DURATION_MS,
    },
    bridge: { bridgeVersion: 2, extensionVersion: '1.4.1' },
    configured: {
      factor: 'target',
      targetFps: targetCase.targetFps,
      resolution: targetCase.resolution,
      model: 'v7s',
    },
    prepared: {
      convTune: { coc: 8, slab: 12 },
      gpu: 'test',
      deviceFeatures: ['shader-f16', 'timestamp-query'],
    },
    started: { running: true, width: 1280, height: 720 },
    producer: {
      durationMs: DURATION_MS,
      producedFrames: sourceCallbacks,
      skippedScheduleSlots: 0,
      observedHz: targetCase.sourceFps,
      intervalsMs: repeated(sourceCallbacks - 1, 1000 / targetCase.sourceFps),
    },
    telemetry: {
      schemaVersion: 1,
      epoch: 1,
      durationMs: DURATION_MS,
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
        presentedSource,
        presentedMid,
        dropped: 0,
        droppedSource: 0,
        droppedMid: 0,
        pending: pendingSource + pendingMid,
        queueHighWater: 10,
        sourcePoolExhausted: 0,
        midPoolExhausted: 0,
        classificationSuperseded: 0,
        pairPlans,
        skippedPairs: 0,
        plannedMids: scheduledMid,
      },
      plannedFactorHistogram: histogramFor(pairPlans, scheduledMid),
      observed: {
        sourceCallbackHz: targetCase.sourceFps,
        sourceHz: targetCase.sourceFps,
        rafHz: 240,
      },
      lateness: { count: presented, p95Ms: 1, maxMs: 1 },
      samples: {
        sourceCallbackIntervalsMs: repeated(sourceCallbacks - 1, 1000 / targetCase.sourceFps),
        sourceMediaIntervalsMs: repeated(sourceCallbacks - 1, 1000 / targetCase.sourceFps),
        rafIntervalsMs: repeated(rafCallbacks - 1, 1000 / 240),
        lateMs: repeated(presented, 1),
      },
      sampleOverflow: false,
      errors: [],
      product: {
        running: true,
        factor: 'target',
        targetFps: targetCase.targetFps,
        requestedTargetHz: targetCase.targetFps,
        minimumTargetHz: targetCase.sourceFps * 2,
        displayCapacityHz,
        effectiveTargetHz: targetCase.expectedEffectiveFps,
        targetState: 'active',
        targetClampReason: targetCase.expectedClampReason,
        effectiveFactor: Math.ceil(targetCase.expectedEffectiveFps / targetCase.sourceFps),
        resolution: targetCase.resolution,
        model: 'v7s',
        modelAsset: 'rt_v7s',
        videoWidth: 1280,
        videoHeight: 720,
        gpu: 'test',
        integrated: false,
        deviceFeatures: ['shader-f16', 'timestamp-query'],
        convTune: { coc: 8, slab: 12 },
        scheduler: {
          msAvg: 2,
          intervalMs: 1000 / targetCase.sourceFps,
          decodedIntervalMs: 1000 / targetCase.sourceFps,
          uniqueIntervalMs: 1000 / targetCase.sourceFps,
          delayMs: 60,
          lateAvg: 1,
          rafMs: 1000 / 240,
          rafFloor: 1000 / 240,
        },
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
    schemaVersion: 2,
    gateId: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.gateId,
    profile: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.profile,
    source: { files: sourceFiles },
    browser: {
      channel: 'chromium',
      distribution: 'chrome-for-testing',
      headed: true,
      persistentContext: true,
      unpackedExtension: true,
      extensionStageHashes: expected().stagedHashes,
    },
    workload: {
      model: 'v7s',
      resolution: 480,
      width: 1280,
      height: 720,
      supportedSourceFps: [10, 15, 24, 60],
      expectedDisplayHz: 240,
      cases: CASES,
    },
    conditions: {
      warmupMsPerCase: 4000,
      measureMsPerCase: DURATION_MS,
      presentationMetric: 'gpu-canvas-submit-from-rAF-pump',
    },
    measurements: Object.fromEntries(CASES.map(targetCase => [
      targetCase.id,
      caseResult(targetCase),
    ])),
    diagnostics: {
      consoleMessages: [],
      consoleErrors: [],
      consoleWarnings: [],
      framegenLogs: [],
      pageErrors: [],
      requestFailures: [],
      httpErrors: [],
    },
    executionError: null,
  };
}

test('fixture, manifest, and arbitrary target matrix are frozen at schema v2', async () => {
  const fixture = await readFile(new URL('../web/product_target_fps_fixture.html', import.meta.url), 'utf8');
  const start = fixture.indexOf('<script>');
  const end = fixture.lastIndexOf('</script>');
  assert.ok(start >= 0 && end > start);
  assert.doesNotThrow(() => new vm.Script(fixture.slice(start + '<script>'.length, end)));
  assert.match(fixture, /factor !== 'target'/);
  assert.match(fixture, /targetFps: TARGET_FPS/);

  const manifest = JSON.parse(await readFile(
    new URL('./product_target_fps_gate_manifest.json', import.meta.url),
    'utf8',
  ));
  assert.doesNotThrow(() => validateManifest(manifest));
  assert.equal(manifest.schemaVersion, 2);
  assert.deepEqual(manifest.fixture.supportedSourceFps, [10, 15, 24, 60]);
  assert.deepEqual(manifest.measurement.cases, CASES);
  assert.deepEqual(manifest.measurement.cases.slice(0, 2).map(({ id }) => id), [
    'source10-target50',
    'source15-target50',
  ]);
  assert.ok(manifest.measurement.cases.every(({ resolution }) => resolution === 480));
  assert.equal(manifest.measurement.warmupMsPerCase, 4000);
  assert.equal(manifest.measurement.measureMsPerCase, DURATION_MS);
  assert.equal(manifest.measurement.expectedDisplayHz, 240);
  assert.equal(manifest.playwrightCli.browserDistribution, 'chrome-for-testing');
});

test('valid arbitrary-target case matrix passes', () => {
  const result = validateProductTargetFpsReport(report(), expected());
  assert.equal(result.passed, true, result.failures.join(', '));
  assert.deepEqual(result.failures, []);
  assert.equal(result.cases.length, CASES.length);
});

test('10 to 50 and 15 to 50 prove non-preset, no-clamp interpolation', () => {
  const result = validateProductTargetFpsReport(report(), expected());
  const source10 = result.cases.find(row => row.caseId === 'source10-target50');
  const source15 = result.cases.find(row => row.caseId === 'source15-target50');
  assert.deepEqual(
    [source10.requestedHz, source10.effectiveTargetHz, source10.clampReason],
    [50, 50, null],
  );
  assert.deepEqual(
    [source15.requestedHz, source15.effectiveTargetHz, source15.clampReason],
    [50, 50, null],
  );
  assert.equal(source10.scheduledSourceHz, 10);
  assert.equal(source10.scheduledMidHz, 40);
  assert.equal(source15.scheduledSourceHz, 15);
  assert.equal(source15.scheduledMidHz, 35);
});

test('measured source jitter keeps the dynamic 2x floor structurally valid', () => {
  const value = report();
  const product = value.measurements['source10-target50'].telemetry.product;
  product.scheduler.decodedIntervalMs = 99.2;
  product.minimumTargetHz = 2000 / product.scheduler.decodedIntervalMs;
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, true, result.failures.join(', '));
});

test('minimum 2x and display ceiling are represented as explicit clamps', () => {
  const result = validateProductTargetFpsReport(report(), expected());
  const minimum = result.cases.find(row => row.caseId === 'source24-request40-floor48');
  const display = result.cases.find(row => row.caseId === 'source60-request300-display-cap');
  assert.deepEqual(
    [minimum.requestedHz, minimum.minimumTargetHz, minimum.effectiveTargetHz, minimum.clampReason],
    [40, 48, 48, 'minimum'],
  );
  assert.deepEqual(
    [display.requestedHz, display.effectiveTargetHz, display.clampReason],
    [300, 232.8, 'display'],
  );
});

test('source duplication cannot impersonate generated mids', () => {
  const value = report();
  const row = value.measurements['source15-target50'].telemetry;
  row.counters.scheduledSource += row.counters.scheduledMid;
  row.counters.scheduledMid = 0;
  row.counters.presentedSource += row.counters.presentedMid;
  row.counters.presentedMid = 0;
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('generated-mid provenance mismatch')));
});

test('factor histogram must prove the planned arbitrary cadence', () => {
  const value = report();
  const histogram = value.measurements['source15-target50'].telemetry.plannedFactorHistogram;
  histogram[3]--;
  histogram[4]++;
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes(
    'planned mid provenance does not match pair plans',
  )));
});

test('target cadence and 240Hz display mismatches fail closed', () => {
  const value = report();
  const row = value.measurements['source24-target50'].telemetry;
  row.samples.rafIntervalsMs.fill(1000 / 180);
  row.observed.rafHz = 180;
  row.durationMs = 14000;
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('display/rAF cadence mismatch')));
  assert.ok(result.failures.some(message => message.includes('target scheduler cadence mismatch')));
});

test('source callback cadence and delivery mismatch fail closed', () => {
  const value = report();
  const row = value.measurements['source15-target50'];
  row.telemetry.samples.sourceCallbackIntervalsMs.fill(100);
  row.telemetry.samples.sourceMediaIntervalsMs.fill(100);
  row.telemetry.observed.sourceCallbackHz = 10;
  row.telemetry.observed.sourceHz = 10;
  row.producer.intervalsMs.fill(100);
  row.producer.observedHz = 10;
  row.telemetry.counters.sourcePresentedFrameGaps = 1;
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('source cadence mismatch')));
  assert.ok(result.failures.some(message => message.includes('source delivery contract exceeded')));
});

test('drops, queue overflow, texture exhaustion, and browser errors fail closed', () => {
  const value = report();
  const row = value.measurements['source60-target144'].telemetry;
  row.counters.presented--;
  row.counters.presentedMid--;
  row.counters.dropped = 1;
  row.counters.droppedMid = 1;
  row.counters.queueHighWater = 25;
  row.counters.midPoolExhausted = 1;
  row.samples.lateMs.pop();
  row.lateness.count--;
  value.diagnostics.pageErrors.push({ message: 'device lost' });
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('dropped frames')));
  assert.ok(result.failures.some(message => message.includes('queue contract')));
  assert.ok(result.failures.some(message => message.includes('texture pool exhausted')));
  assert.ok(result.failures.some(message => message.includes('browser errors')));
});

test('source drift and staged cadence mismatch fail closed', () => {
  const value = report();
  value.source.files.cadenceScript.sha256End = 'b'.repeat(64);
  value.browser.extensionStageHashes.cadenceScript = 'c'.repeat(64);
  const result = validateProductTargetFpsReport(value, expected());
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(message => message.includes('hash drift during run: cadenceScript')));
  assert.ok(result.failures.some(message => message.includes(
    'unpacked extension hash mismatch: cadenceScript',
  )));
});

test('missing tune and queue conservation mismatch are structural failures', () => {
  const missingTune = report();
  missingTune.measurements['source10-target50'].telemetry.product.convTune = null;
  assert.throws(
    () => validateProductTargetFpsReport(missingTune, expected()),
    ProductTargetFpsAcceptanceError,
  );
  const badQueue = report();
  badQueue.measurements['source60-target120'].telemetry.counters.pending++;
  assert.throws(
    () => validateProductTargetFpsReport(badQueue, expected()),
    ProductTargetFpsAcceptanceError,
  );
});

test('runner CLI, output scope, source binding, and write-once behavior are fixed', async () => {
  assert.deepEqual(parseArguments(['--help']), { help: true });
  assert.deepEqual(
    parseArguments(['--output', 'output/product-target-fps/result.json']),
    { output: 'output/product-target-fps/result.json' },
  );
  assert.throws(() => parseArguments(['--manifest', 'other.json']));
  assert.ok(resolveOutputPath('output/product-target-fps/result.json', 'unused')
    .endsWith(path.join('output', 'product-target-fps', 'result.json')));
  assert.throws(() => resolveOutputPath('output/product-target-fps/../escape.json', 'unused'));
  const source = await readFile(new URL('./run_product_target_fps_gate.mjs', import.meta.url), 'utf8');
  assert.match(source, /cadenceScript: 'extension\/cadence\.js'/);
  assert.match(source, /source10-target50/);
  assert.match(source, /source15-target50/);
  assert.match(source, /--disable-extensions-except/);
  assert.match(source, /--load-extension/);
  assert.match(source, /flag: 'wx'/);
  assert.doesNotMatch(source, /install-browser|playwright install/);
  const ignore = await readFile(new URL('../.gitignore', import.meta.url), 'utf8');
  assert.match(ignore, /^\/output\/$|^\/output\/product-target-fps\/$/m);
});
