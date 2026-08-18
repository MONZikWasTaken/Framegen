import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

import {
  WebGpuBenchAcceptanceError,
  percentile,
  validateHfrReport,
} from './webgpu_bench_acceptance.mjs';

const hash = (character) => character.repeat(64);
const factors = [3, 4];
const repetitions = 30;

test('browser harness parses and manifest freezes the strict HFR profile', async () => {
  const html = await readFile(new URL('../web/bench.html', import.meta.url), 'utf8');
  const startMarker = '<script type="module">';
  const start = html.indexOf(startMarker);
  const end = html.lastIndexOf('</script>');
  assert.notEqual(start, -1);
  assert.ok(end > start);
  assert.doesNotThrow(() => new vm.Script(html.slice(start + startMarker.length, end)));

  const manifest = JSON.parse(
    await readFile(new URL('./webgpu_bench_manifest.json', import.meta.url), 'utf8'),
  );
  const profile = manifest.profiles['v7s-720p-hfr'];
  assert.equal(profile.workload.sourceFps, 60);
  assert.deepEqual(profile.measurement.intervalFactors, [3, 4]);
  assert.equal(profile.measurement.intervalWarmupsPerFactor, 2);
  assert.equal(profile.measurement.intervalRepetitions, 30);
  assert.equal(profile.acceptance.budgetFraction, 0.85);
  assert.equal(profile.acceptance.hardBudgetFraction, 1);
});

function summary(raw) {
  const sorted = [...raw].sort((left, right) => left - right);
  const mean = raw.reduce((total, value) => total + value, 0) / raw.length;
  const variance = raw.reduce((total, value) => total + ((value - mean) ** 2), 0) / raw.length;
  return {
    unit: 'ms',
    sampleCount: raw.length,
    iterationsPerSample: 1,
    samplesMs: [...raw],
    minMs: sorted[0],
    p50Ms: percentile(sorted, 0.5),
    p90Ms: percentile(sorted, 0.9),
    p95Ms: percentile(sorted, 0.95),
    maxMs: sorted.at(-1),
    meanMs: mean,
    stddevMs: Math.sqrt(variance),
  };
}

const expected = {
  budgetFraction: 0.85,
  factors,
  hardBudgetFraction: 1,
  headed: true,
  hashes: {
    acceptanceValidator: hash('a'),
    harness: hash('b'),
    manifest: hash('c'),
    modelManifest: hash('d'),
    modelWeights: hash('e'),
    runner: hash('f'),
    runtimeModules: { rt: hash('1') },
  },
  height: 720,
  minimumRepetitions: repetitions,
  minimumWarmupIntervals: 2,
  model: 'v7s',
  modules: ['rt'],
  profile: 'v7s-720p-hfr',
  requiredDeviceFeatures: ['shader-f16', 'timestamp-query'],
  resolution: '720',
  scene: 'motion',
  sourceFps: 60,
  sparseRefine: true,
  staticGuard: true,
  width: 1280,
};

function rotatedOrders() {
  return Array.from({ length: repetitions }, (_, repetition) => {
    const offset = repetition % factors.length;
    return [...factors.slice(offset), ...factors.slice(0, offset)];
  });
}

function factorRow(factor, wallValue) {
  const rawWallMs = Array.from({ length: repetitions }, () => wallValue);
  const rawCpuEncodeSubmitMs = Array.from({ length: repetitions }, () => 0.2);
  const sourceIntervalMs = 1000 / 60;
  const softDeadlineMs = sourceIntervalMs * 0.85;
  const hardDeadlineMs = sourceIntervalMs;
  const softDeadlineMisses = rawWallMs.filter((value) => value > softDeadlineMs).length;
  const hardDeadlineMisses = rawWallMs.filter((value) => value > hardDeadlineMs).length;
  return {
    acceptance: {
      hardDeadlineMisses,
      hardDeadlineMs,
      passed: percentile([...rawWallMs].sort((a, b) => a - b), 0.95) <= softDeadlineMs
        && hardDeadlineMisses === 0,
      softDeadlineMisses,
      softDeadlineMs,
    },
    cpuEncodeSubmit: summary(rawCpuEncodeSubmitMs),
    factor,
    generatedFramesPerInterval: factor - 1,
    outputHz: 60 * factor,
    rawCpuEncodeSubmitMs,
    rawWallMs,
    wall: summary(rawWallMs),
  };
}

function validReport() {
  const sourceIntervalMs = 1000 / 60;
  return {
    schemaVersion: 2,
    profile: 'v7s-720p-hfr',
    source: {
      acceptanceValidator: { sha256: hash('a') },
      harness: { sha256: hash('b') },
      manifest: { sha256: hash('c') },
      modelAssets: { manifest: { sha256: hash('d') }, weights: { sha256: hash('e') } },
      runner: { sha256: hash('f') },
      runtimeModules: { rt: { sha256: hash('1') } },
    },
    browser: { headed: true },
    webgpu: { enabledDeviceFeatures: ['shader-f16', 'timestamp-query'] },
    workload: {
      intervalFactors: factors,
      model: 'v7s',
      modelResolution: { width: 1280, height: 720 },
      modules: ['rt'],
      outputHzByFactor: { 3: 180, 4: 240 },
      requestedResolution: '720',
      scene: 'motion',
      sourceFps: 60,
      sparseRefine: true,
      staticGuard: true,
      tune: true,
    },
    conditions: {
      acceptance: {
        budgetFraction: 0.85,
        hardBudgetFraction: 1,
        hardDeadlineMs: sourceIntervalMs,
        softDeadlineMs: sourceIntervalMs * 0.85,
        sourceIntervalMs,
        strict: true,
      },
      autotune: { scope: 'once-per-module-before-all-factor-measurements' },
      samples: {
        factorOrderByRepetition: rotatedOrders(),
        fullIntervalRepetitions: repetitions,
      },
      warmup: { fullIntervalCallsPerFactor: 2 },
    },
    diagnostics: { consoleMessages: [], pageErrors: [] },
    rawBenchResult: { passed: true, schemaVersion: 3 },
    measurements: {
      rt: {
        autotune: { coc: 8, slab: 20, w4: true },
        fullIntervals: [factorRow(3, 5), factorRow(4, 6)],
        parity: { outputSha256: hash('3') },
        passed: true,
      },
    },
  };
}

test('accepts a source-bound green HFR report', () => {
  const result = validateHfrReport(validReport(), expected);
  assert.equal(result.passed, true);
  assert.equal(result.factors.length, 2);
  assert.equal(result.softDeadlineMs, (1000 / 60) * 0.85);
});

test('returns red when p95 exceeds the soft budget', () => {
  const report = validReport();
  report.measurements.rt.fullIntervals[1] = factorRow(4, 15);
  report.measurements.rt.passed = false;
  report.rawBenchResult.passed = false;
  const result = validateHfrReport(report, expected);
  assert.equal(result.passed, false);
  assert.equal(result.factors[1].hardDeadlineMisses, 0);
});

test('returns red on any hard deadline miss', () => {
  const report = validReport();
  const row = factorRow(3, 5);
  row.rawWallMs[0] = 17;
  row.wall = summary(row.rawWallMs);
  row.acceptance.softDeadlineMisses = 1;
  row.acceptance.hardDeadlineMisses = 1;
  row.acceptance.passed = false;
  report.measurements.rt.fullIntervals[0] = row;
  report.measurements.rt.passed = false;
  report.rawBenchResult.passed = false;
  const result = validateHfrReport(report, expected);
  assert.equal(result.passed, false);
  assert.equal(result.factors[0].hardDeadlineMisses, 1);
});

test('fails closed on non-rotated factor order', () => {
  const report = validReport();
  report.conditions.samples.factorOrderByRepetition[1] = [3, 4];
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});

test('fails closed on missing fixed autotune evidence', () => {
  const report = validReport();
  report.measurements.rt.autotune = null;
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});

test('fails closed on source hash drift', () => {
  const report = validReport();
  report.source.runtimeModules.rt.sha256 = hash('2');
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});

test('fails closed on too few samples', () => {
  const report = validReport();
  report.conditions.samples.fullIntervalRepetitions = 29;
  report.conditions.samples.factorOrderByRepetition.pop();
  for (const row of report.measurements.rt.fullIntervals) {
    row.rawWallMs.pop();
    row.rawCpuEncodeSubmitMs.pop();
    row.wall = summary(row.rawWallMs);
    row.cpuEncodeSubmit = summary(row.rawCpuEncodeSubmitMs);
  }
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});

test('fails closed on non-finite timing data', () => {
  const report = validReport();
  report.measurements.rt.fullIntervals[0].rawWallMs[0] = Number.NaN;
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});

test('fails closed on browser errors', () => {
  const report = validReport();
  report.diagnostics.pageErrors.push({ message: 'device lost' });
  assert.throws(
    () => validateHfrReport(report, expected),
    WebGpuBenchAcceptanceError,
  );
});
