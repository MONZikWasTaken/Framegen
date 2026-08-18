const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const NUMBER_TOLERANCE = 0.002;

export class WebGpuBenchAcceptanceError extends Error {}

function fail(message) {
  throw new WebGpuBenchAcceptanceError(`WebGPU benchmark acceptance: ${message}`);
}

function finiteNumber(value, label, { minimum = 0 } = {}) {
  if (!Number.isFinite(value) || value < minimum) fail(`${label} must be finite and >= ${minimum}`);
  return value;
}

function integer(value, label, { minimum = 0 } = {}) {
  if (!Number.isInteger(value) || value < minimum) fail(`${label} must be an integer >= ${minimum}`);
  return value;
}

function sameNumber(actual, expected, label, tolerance = 1e-9) {
  finiteNumber(actual, label);
  finiteNumber(expected, `${label} expected`);
  if (Math.abs(actual - expected) > tolerance) {
    fail(`${label} mismatch: expected ${expected}, got ${actual}`);
  }
}

function exactArray(actual, expected, label) {
  if (!Array.isArray(actual) || actual.length !== expected.length
      || actual.some((value, index) => value !== expected[index])) {
    fail(`${label} mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function object(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function sha256(value, label) {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    fail(`${label} must be a lowercase SHA-256 digest`);
  }
  return value;
}

export function percentile(sortedValues, fraction) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
    fail('percentile input must be a non-empty array');
  }
  finiteNumber(fraction, 'percentile fraction');
  if (fraction > 1) fail('percentile fraction must be <= 1');
  const sorted = sortedValues.map((value, index) => finiteNumber(value, `sample ${index}`))
    .sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * (position - lower));
}

export function summarizeSamples(samples) {
  if (!Array.isArray(samples) || samples.length === 0) fail('samples must be non-empty');
  const values = samples.map((value, index) => finiteNumber(value, `samples[${index}]`));
  const sorted = [...values].sort((left, right) => left - right);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce((total, value) => total + ((value - mean) ** 2), 0)
    / values.length;
  return {
    count: values.length,
    maxMs: sorted.at(-1),
    meanMs: mean,
    minMs: sorted[0],
    p50Ms: percentile(sorted, 0.5),
    p90Ms: percentile(sorted, 0.9),
    p95Ms: percentile(sorted, 0.95),
    populationStddevMs: Math.sqrt(variance),
  };
}

function validateSummary(summary, rawSamples, label) {
  object(summary, label);
  const recomputed = summarizeSamples(rawSamples);
  integer(summary.sampleCount, `${label}.sampleCount`, { minimum: 1 });
  if (summary.sampleCount !== recomputed.count) fail(`${label}.sampleCount does not match raw samples`);
  const fields = ['minMs', 'maxMs', 'meanMs', 'p50Ms', 'p90Ms', 'p95Ms', 'stddevMs'];
  const expected = {
    minMs: recomputed.minMs,
    maxMs: recomputed.maxMs,
    meanMs: recomputed.meanMs,
    p50Ms: recomputed.p50Ms,
    p90Ms: recomputed.p90Ms,
    p95Ms: recomputed.p95Ms,
    stddevMs: recomputed.populationStddevMs,
  };
  for (const field of fields) {
    finiteNumber(summary[field], `${label}.${field}`);
    if (Math.abs(summary[field] - expected[field]) > NUMBER_TOLERANCE) {
      fail(`${label}.${field} does not match raw samples`);
    }
  }
}

function validateHashes(source, expectedHashes, modules) {
  object(source, 'source');
  const pairs = [
    ['harness', source.harness?.sha256, expectedHashes.harness],
    ['runner', source.runner?.sha256, expectedHashes.runner],
    ['manifest', source.manifest?.sha256, expectedHashes.manifest],
    ['acceptanceValidator', source.acceptanceValidator?.sha256, expectedHashes.acceptanceValidator],
    ['model weights', source.modelAssets?.weights?.sha256, expectedHashes.modelWeights],
    ['model manifest', source.modelAssets?.manifest?.sha256, expectedHashes.modelManifest],
  ];
  for (const [label, actual, expected] of pairs) {
    sha256(actual, `source ${label}`);
    sha256(expected, `expected ${label}`);
    if (actual !== expected) fail(`source ${label} hash mismatch`);
  }
  for (const moduleName of modules) {
    const actual = source.runtimeModules?.[moduleName]?.sha256;
    const expected = expectedHashes.runtimeModules?.[moduleName];
    sha256(actual, `source runtime ${moduleName}`);
    sha256(expected, `expected runtime ${moduleName}`);
    if (actual !== expected) fail(`source runtime ${moduleName} hash mismatch`);
  }
}

function validateFactorOrder(orders, factors, repetitions) {
  if (!Array.isArray(orders) || orders.length !== repetitions) {
    fail('factorOrderByRepetition length mismatch');
  }
  for (let repetition = 0; repetition < repetitions; repetition += 1) {
    const offset = repetition % factors.length;
    const expected = [...factors.slice(offset), ...factors.slice(0, offset)];
    exactArray(orders[repetition], expected, `factorOrderByRepetition[${repetition}]`);
  }
}

export function validateHfrReport(report, expected) {
  object(report, 'report');
  object(expected, 'expected');
  if (!Array.isArray(expected.factors) || expected.factors.length === 0
      || expected.factors.some(value => !Number.isInteger(value) || value < 2)) {
    fail('expected.factors must contain interpolation factors >= 2');
  }
  if (!Array.isArray(expected.modules) || expected.modules.length === 0
      || !expected.modules.every(value => typeof value === 'string' && value.length > 0)) {
    fail('expected.modules must contain module names');
  }
  if (!Array.isArray(expected.requiredDeviceFeatures)) {
    fail('expected.requiredDeviceFeatures must be an array');
  }
  integer(expected.width, 'expected.width', { minimum: 1 });
  integer(expected.height, 'expected.height', { minimum: 1 });
  integer(expected.minimumRepetitions, 'expected.minimumRepetitions', { minimum: 1 });
  integer(expected.minimumWarmupIntervals, 'expected.minimumWarmupIntervals', { minimum: 0 });
  if (report.schemaVersion !== 2) fail(`report schemaVersion must be 2, got ${report.schemaVersion}`);
  if (report.profile !== expected.profile) fail(`profile must be ${expected.profile}`);

  const factors = [...expected.factors];
  exactArray(report.workload?.intervalFactors, factors, 'workload.intervalFactors');
  exactArray(report.workload?.modules, expected.modules, 'workload.modules');
  if (report.workload?.requestedResolution !== expected.resolution) fail('resolution mismatch');
  if (report.workload?.model !== expected.model) fail('model mismatch');
  if (report.workload?.scene !== expected.scene) fail('scene mismatch');
  if (report.workload?.sparseRefine !== expected.sparseRefine) fail('sparseRefine mismatch');
  if (report.workload?.staticGuard !== expected.staticGuard) fail('staticGuard mismatch');
  if (report.workload?.modelResolution?.width !== expected.width
      || report.workload?.modelResolution?.height !== expected.height) {
    fail('model resolution dimensions mismatch');
  }
  if (report.workload?.tune !== true) fail('strict HFR profile requires autotuning');
  if (report.browser?.headed !== expected.headed) fail('browser headed condition mismatch');
  sameNumber(report.workload?.sourceFps, expected.sourceFps, 'workload.sourceFps');
  for (const factor of factors) {
    sameNumber(
      report.workload?.outputHzByFactor?.[String(factor)],
      expected.sourceFps * factor,
      `workload.outputHzByFactor.${factor}`,
    );
  }

  const sourceIntervalMs = 1000 / expected.sourceFps;
  const softDeadlineMs = sourceIntervalMs * expected.budgetFraction;
  const hardDeadlineMs = sourceIntervalMs * expected.hardBudgetFraction;
  const acceptance = object(report.conditions?.acceptance, 'conditions.acceptance');
  if (acceptance.strict !== true) fail('conditions.acceptance.strict must be true');
  sameNumber(acceptance.budgetFraction, expected.budgetFraction, 'acceptance.budgetFraction');
  sameNumber(
    acceptance.hardBudgetFraction,
    expected.hardBudgetFraction,
    'acceptance.hardBudgetFraction',
  );
  sameNumber(acceptance.sourceIntervalMs, sourceIntervalMs, 'acceptance.sourceIntervalMs');
  sameNumber(acceptance.softDeadlineMs, softDeadlineMs, 'acceptance.softDeadlineMs');
  sameNumber(acceptance.hardDeadlineMs, hardDeadlineMs, 'acceptance.hardDeadlineMs');

  const samples = object(report.conditions?.samples, 'conditions.samples');
  integer(samples.fullIntervalRepetitions, 'samples.fullIntervalRepetitions', { minimum: 1 });
  if (samples.fullIntervalRepetitions < expected.minimumRepetitions) {
    fail(`full interval repetitions must be >= ${expected.minimumRepetitions}`);
  }
  integer(
    report.conditions?.warmup?.fullIntervalCallsPerFactor,
    'warmup.fullIntervalCallsPerFactor',
    { minimum: 0 },
  );
  if (report.conditions.warmup.fullIntervalCallsPerFactor < expected.minimumWarmupIntervals) {
    fail(`full interval warmups must be >= ${expected.minimumWarmupIntervals}`);
  }
  validateFactorOrder(
    samples.factorOrderByRepetition,
    factors,
    samples.fullIntervalRepetitions,
  );
  if (report.conditions?.autotune?.scope !== 'once-per-module-before-all-factor-measurements') {
    fail('autotune scope is not fixed across factors');
  }

  const enabledFeatures = report.webgpu?.enabledDeviceFeatures;
  if (!Array.isArray(enabledFeatures)) fail('webgpu.enabledDeviceFeatures must be an array');
  for (const feature of expected.requiredDeviceFeatures) {
    if (!enabledFeatures.includes(feature)) fail(`required WebGPU feature missing: ${feature}`);
  }
  const diagnostics = object(report.diagnostics, 'diagnostics');
  if (!Array.isArray(diagnostics.pageErrors) || diagnostics.pageErrors.length !== 0) {
    fail('browser page errors are present');
  }
  if (!Array.isArray(diagnostics.consoleMessages)
      || diagnostics.consoleMessages.some((message) => message?.type === 'error')) {
    fail('browser console errors are present');
  }

  validateHashes(report.source, expected.hashes, expected.modules);

  const factorResults = [];
  let passed = true;
  for (const moduleName of expected.modules) {
    const measurement = object(report.measurements?.[moduleName], `measurements.${moduleName}`);
    const autotune = object(measurement.autotune, `measurements.${moduleName}.autotune`);
    integer(autotune.coc, `measurements.${moduleName}.autotune.coc`, { minimum: 1 });
    integer(autotune.slab, `measurements.${moduleName}.autotune.slab`, { minimum: 1 });
    sha256(measurement.parity?.outputSha256, `measurements.${moduleName}.parity.outputSha256`);
    if (!Array.isArray(measurement.fullIntervals)
        || measurement.fullIntervals.length !== factors.length) {
      fail(`measurements.${moduleName}.fullIntervals length mismatch`);
    }
    let modulePassed = true;
    for (let index = 0; index < factors.length; index += 1) {
      const factor = factors[index];
      const row = object(measurement.fullIntervals[index], `${moduleName}.factor[${index}]`);
      if (row.factor !== factor) fail(`${moduleName} factor order mismatch`);
      if (row.generatedFramesPerInterval !== factor - 1) {
        fail(`${moduleName} factor x${factor} generated frame count mismatch`);
      }
      sameNumber(row.outputHz, expected.sourceFps * factor, `${moduleName}.x${factor}.outputHz`);
      if (!Array.isArray(row.rawWallMs)
          || row.rawWallMs.length !== samples.fullIntervalRepetitions) {
        fail(`${moduleName}.x${factor}.rawWallMs length mismatch`);
      }
      if (!Array.isArray(row.rawCpuEncodeSubmitMs)
          || row.rawCpuEncodeSubmitMs.length !== samples.fullIntervalRepetitions) {
        fail(`${moduleName}.x${factor}.rawCpuEncodeSubmitMs length mismatch`);
      }
      validateSummary(row.wall, row.rawWallMs, `${moduleName}.x${factor}.wall`);
      validateSummary(
        row.cpuEncodeSubmit,
        row.rawCpuEncodeSubmitMs,
        `${moduleName}.x${factor}.cpuEncodeSubmit`,
      );
      const recomputed = summarizeSamples(row.rawWallMs);
      const softMisses = row.rawWallMs.filter((value) => value > softDeadlineMs).length;
      const hardMisses = row.rawWallMs.filter((value) => value > hardDeadlineMs).length;
      const rowPassed = recomputed.p95Ms <= softDeadlineMs && hardMisses === 0;
      sameNumber(row.acceptance?.softDeadlineMs, softDeadlineMs, `${moduleName}.x${factor}.softDeadlineMs`);
      sameNumber(row.acceptance?.hardDeadlineMs, hardDeadlineMs, `${moduleName}.x${factor}.hardDeadlineMs`);
      if (row.acceptance?.softDeadlineMisses !== softMisses
          || row.acceptance?.hardDeadlineMisses !== hardMisses
          || row.acceptance?.passed !== rowPassed) {
        fail(`${moduleName}.x${factor} acceptance does not match raw samples`);
      }
      passed &&= rowPassed;
      modulePassed &&= rowPassed;
      factorResults.push({
        factor,
        hardDeadlineMisses: hardMisses,
        module: moduleName,
        p50Ms: recomputed.p50Ms,
        p95Ms: recomputed.p95Ms,
        passed: rowPassed,
        softDeadlineMisses: softMisses,
      });
    }
    if (measurement.passed !== modulePassed) {
      fail(`measurements.${moduleName}.passed does not match factor results`);
    }
  }
  if (report.rawBenchResult?.schemaVersion !== 3
      || report.rawBenchResult?.passed !== passed) {
    fail('raw browser result does not agree with validated acceptance');
  }
  return Object.freeze({
    factors: factorResults,
    hardDeadlineMs,
    passed,
    softDeadlineMs,
    sourceIntervalMs,
  });
}
