export class ProductHfrAcceptanceError extends Error {}

function fail(message) {
  throw new ProductHfrAcceptanceError(`Product HFR acceptance: ${message}`);
}

function object(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${name} must be an object`);
  return value;
}

function finiteNumber(value, name) {
  if (!Number.isFinite(value)) fail(`${name} must be finite`);
  return value;
}

function integer(value, name) {
  if (!Number.isInteger(value) || value < 0) fail(`${name} must be a non-negative integer`);
  return value;
}

function finiteArray(value, name) {
  if (!Array.isArray(value) || value.some(item => !Number.isFinite(item) || item < 0)) {
    fail(`${name} must be an array of finite non-negative numbers`);
  }
  return value;
}

function exactArray(actual, expected, name) {
  if (!Array.isArray(actual) || actual.length !== expected.length
      || actual.some((value, index) => value !== expected[index])) {
    fail(`${name} mismatch`);
  }
}

function sameNumber(actual, expected, name, tolerance = 0.002) {
  finiteNumber(actual, name);
  if (Math.abs(actual - expected) > tolerance) fail(`${name} mismatch`);
}

export function percentile(values, fraction) {
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
}

function rate(intervals) {
  if (!intervals.length) return null;
  const total = intervals.reduce((sum, value) => sum + value, 0);
  return total > 0 ? 1000 * intervals.length / total : null;
}

function exactHashFiles(source, expectedHashes, failures) {
  const files = object(source?.files, 'source.files');
  for (const [name, expectedHash] of Object.entries(expectedHashes)) {
    const row = object(files[name], `source.files.${name}`);
    if (row.sha256Start !== expectedHash) failures.push(`source hash mismatch before run: ${name}`);
    if (row.sha256End !== expectedHash) failures.push(`source hash drift during run: ${name}`);
  }
}

function validateTune(value, name) {
  const tune = object(value, name);
  if (!Number.isInteger(tune.coc) || tune.coc <= 0) fail(`${name}.coc must be a positive integer`);
  if (!Number.isInteger(tune.slab) || tune.slab <= 0) fail(`${name}.slab must be a positive integer`);
}

function validateFactor(raw, factor, expected, failures) {
  object(raw, `measurements.x${factor}`);
  if (raw.schemaVersion !== 1) fail(`measurements.x${factor}.schemaVersion must be 1`);
  if (typeof raw.error === 'string' && raw.error) {
    failures.push(`x${factor} fixture failed: ${raw.error}`);
    return { factor, passed: false, error: raw.error };
  }

  const fixture = object(raw.fixture, `measurements.x${factor}.fixture`);
  if (fixture.name !== expected.fixtureName || fixture.pattern !== expected.pattern
      || fixture.width !== expected.width || fixture.height !== expected.height
      || fixture.nominalSourceFps !== expected.sourceFps) {
    fail(`x${factor} fixture identity mismatch`);
  }
  if (raw.conditions?.factor !== factor
      || raw.conditions?.warmupMs !== expected.warmupMs
      || raw.conditions?.measureMs !== expected.measureMs) {
    fail(`x${factor} measurement conditions mismatch`);
  }
  if (raw.bridge?.bridgeVersion !== 2 || raw.configured?.factor !== factor
      || raw.configured?.anime !== false
      || raw.configured?.resolution !== 720 || raw.configured?.model !== 'v7s') {
    fail(`x${factor} extension bridge/config mismatch`);
  }
  validateTune(raw.prepared?.convTune, `measurements.x${factor}.prepared.convTune`);
  if (raw.started?.running !== true || raw.started?.width !== expected.width
      || raw.started?.height !== expected.height) {
    fail(`x${factor} product path did not start at the required dimensions`);
  }

  const producer = object(raw.producer, `measurements.x${factor}.producer`);
  const producerDurationMs = finiteNumber(producer.durationMs, `measurements.x${factor}.producer.durationMs`);
  const producerIntervals = finiteArray(producer.intervalsMs, `measurements.x${factor}.producer.intervalsMs`);
  integer(producer.producedFrames, `measurements.x${factor}.producer.producedFrames`);
  integer(producer.skippedScheduleSlots, `measurements.x${factor}.producer.skippedScheduleSlots`);
  if (producerIntervals.length !== Math.max(0, producer.producedFrames - 1)) {
    fail(`x${factor} producer interval count mismatch`);
  }
  const recomputedProducerHz = rate(producerIntervals);
  if (recomputedProducerHz === null) fail(`x${factor} producer has no cadence samples`);
  sameNumber(producer.observedHz, recomputedProducerHz, `x${factor} producer.observedHz`);

  const telemetry = object(raw.telemetry, `measurements.x${factor}.telemetry`);
  if (telemetry.schemaVersion !== 1 || telemetry.sampleOverflow !== false) {
    fail(`x${factor} telemetry schema/overflow mismatch`);
  }
  const counters = object(telemetry.counters, `measurements.x${factor}.telemetry.counters`);
  const counterNames = [
    'sourceCallbacks', 'sourceProcessed', 'sourceBusySkipped', 'sourcePresentedFrameGaps',
    'sourceMetadataDuplicates', 'rafCallbacks', 'scheduled', 'scheduledSource', 'scheduledMid',
    'presented', 'presentedSource', 'presentedMid', 'dropped', 'droppedSource', 'droppedMid',
    'pending', 'queueHighWater', 'sourcePoolExhausted', 'midPoolExhausted',
    'classificationSuperseded', 'pairPlans', 'skippedPairs', 'plannedMids',
  ];
  for (const name of counterNames) integer(counters[name], `x${factor} counters.${name}`);
  if (counters.sourceProcessed + counters.sourceBusySkipped !== counters.sourceCallbacks) {
    fail(`x${factor} source callback accounting mismatch`);
  }
  if (counters.scheduledSource + counters.scheduledMid !== counters.scheduled
      || counters.presentedSource + counters.presentedMid !== counters.presented
      || counters.droppedSource + counters.droppedMid !== counters.dropped) {
    fail(`x${factor} source/mid accounting mismatch`);
  }
  if (counters.scheduled !== counters.presented + counters.dropped + counters.pending) {
    fail(`x${factor} queue conservation mismatch`);
  }
  if (counters.queueHighWater < counters.pending || counters.scheduledMid > counters.plannedMids) {
    fail(`x${factor} queue/planned-mid accounting mismatch`);
  }

  const samples = object(telemetry.samples, `measurements.x${factor}.telemetry.samples`);
  const callbackIntervals = finiteArray(samples.sourceCallbackIntervalsMs, `x${factor} sourceCallbackIntervalsMs`);
  const mediaIntervals = finiteArray(samples.sourceMediaIntervalsMs, `x${factor} sourceMediaIntervalsMs`);
  const rafIntervals = finiteArray(samples.rafIntervalsMs, `x${factor} rafIntervalsMs`);
  const lateMs = finiteArray(samples.lateMs, `x${factor} lateMs`);
  if (callbackIntervals.length !== Math.max(0, counters.sourceCallbacks - 1)
      || mediaIntervals.length !== Math.max(0, counters.sourceCallbacks - 1)
      || rafIntervals.length !== Math.max(0, counters.rafCallbacks - 1)
      || lateMs.length !== counters.presented) {
    fail(`x${factor} telemetry sample count mismatch`);
  }
  const sourceCallbackHz = rate(callbackIntervals);
  const sourceHz = rate(mediaIntervals);
  const rafHz = rate(rafIntervals);
  if (sourceCallbackHz === null || sourceHz === null || rafHz === null) {
    fail(`x${factor} telemetry cadence samples are empty`);
  }
  sameNumber(telemetry.observed?.sourceCallbackHz, sourceCallbackHz, `x${factor} observed.sourceCallbackHz`);
  sameNumber(telemetry.observed?.sourceHz, sourceHz, `x${factor} observed.sourceHz`);
  sameNumber(telemetry.observed?.rafHz, rafHz, `x${factor} observed.rafHz`);
  const lateP95Ms = percentile(lateMs, 0.95);
  const lateMaxMs = lateMs.length ? Math.max(...lateMs) : null;
  if (lateP95Ms === null || lateMaxMs === null) fail(`x${factor} has no presentation samples`);
  if (telemetry.lateness?.count !== lateMs.length) fail(`x${factor} lateness count mismatch`);
  sameNumber(telemetry.lateness?.p95Ms, lateP95Ms, `x${factor} lateness.p95Ms`);
  sameNumber(telemetry.lateness?.maxMs, lateMaxMs, `x${factor} lateness.maxMs`);

  const histogram = object(telemetry.plannedFactorHistogram, `x${factor} plannedFactorHistogram`);
  let histogramTotal = 0;
  for (const [key, count] of Object.entries(histogram)) {
    integer(count, `x${factor} plannedFactorHistogram.${key}`);
    histogramTotal += count;
    if (key !== String(factor) && count) failures.push(`x${factor} stepped down to factor ${key}`);
  }
  if (histogramTotal !== counters.pairPlans) fail(`x${factor} factor histogram count mismatch`);

  const product = object(telemetry.product, `measurements.x${factor}.telemetry.product`);
  if (product.running !== true || product.factor !== factor || product.resolution !== 720
      || product.model !== 'v7s' || product.modelAsset !== 'rt_v7s'
      || product.videoWidth !== expected.width || product.videoHeight !== expected.height) {
    fail(`x${factor} product identity mismatch`);
  }
  validateTune(product.convTune, `x${factor} product.convTune`);
  if (JSON.stringify(product.convTune) !== JSON.stringify(raw.prepared.convTune)) {
    fail(`x${factor} prepared/applied convTune mismatch`);
  }
  exactArray(product.deviceFeatures, [...product.deviceFeatures].sort(), `x${factor} sorted deviceFeatures`);
  for (const feature of expected.requiredDeviceFeatures) {
    if (!product.deviceFeatures.includes(feature)) failures.push(`x${factor} missing WebGPU feature: ${feature}`);
  }
  if (!product.scheduler || Object.values(product.scheduler).some(value => !Number.isFinite(value))) {
    fail(`x${factor} scheduler state is incomplete`);
  }
  if (product.cuts !== 0) failures.push(`x${factor} deterministic fixture was classified as ${product.cuts} cuts`);
  if (!Array.isArray(telemetry.errors)) fail(`x${factor} telemetry.errors must be an array`);
  if (telemetry.errors.length) failures.push(`x${factor} extension errors are present`);
  if (product.integrated) failures.push(`x${factor} ran on an integrated GPU`);

  const durationMs = finiteNumber(telemetry.durationMs, `x${factor} telemetry.durationMs`);
  const durationError = Math.abs(durationMs - expected.measureMs) / expected.measureMs;
  const producerDurationError = Math.abs(producerDurationMs - expected.measureMs) / expected.measureMs;
  const sourceError = Math.abs(sourceHz - expected.sourceFps) / expected.sourceFps;
  const sourceCallbackError = Math.abs(sourceCallbackHz - expected.sourceFps) / expected.sourceFps;
  const producerError = Math.abs(recomputedProducerHz - expected.sourceFps) / expected.sourceFps;
  const outputHz = expected.sourceFps * factor;
  const scheduledHz = counters.scheduled * 1000 / durationMs;
  const presentedHz = counters.presented * 1000 / durationMs;
  const outputIntervalMs = 1000 / outputHz;
  const minimumSourceCallbacks = expected.sourceFps * durationMs / 1000
    * expected.minimumSourceCallbacksFraction;
  const eligiblePairs = Math.max(0, counters.sourceProcessed - 1);
  const pairCoverage = eligiblePairs ? counters.pairPlans / eligiblePairs : 0;

  if (durationError > expected.durationToleranceFraction
      || producerDurationError > expected.durationToleranceFraction) {
    failures.push(`x${factor} measurement duration mismatch`);
  }
  if (sourceError > expected.sourceHzToleranceFraction
      || sourceCallbackError > expected.sourceHzToleranceFraction
      || producerError > expected.sourceHzToleranceFraction) {
    failures.push(`x${factor} 60fps source cadence mismatch`);
  }
  if (counters.sourceCallbacks < minimumSourceCallbacks) failures.push(`x${factor} too few source callbacks`);
  if (producer.producedFrames < minimumSourceCallbacks
      || Math.abs(producer.producedFrames - counters.sourceCallbacks) > Math.max(3, counters.sourceCallbacks * 0.03)) {
    failures.push(`x${factor} source producer/rVFC delivery mismatch`);
  }
  if (rafHz < outputHz * expected.minimumDisplayHzFraction) failures.push(`x${factor} display/rAF Hz mismatch`);
  if (scheduledHz < outputHz * expected.minimumScheduledHzFraction) failures.push(`x${factor} scheduler throughput mismatch`);
  if (presentedHz < outputHz * expected.minimumPresentedHzFraction) failures.push(`x${factor} presentation throughput mismatch`);
  if (counters.dropped > expected.maximumDroppedFrames) failures.push(`x${factor} dropped frames exceed contract`);
  if (counters.queueHighWater > expected.maximumQueueHighWater
      || counters.pending > expected.maximumPendingFrames) failures.push(`x${factor} queue contract exceeded`);
  if (lateP95Ms > outputIntervalMs * expected.maximumLateP95OutputIntervals
      || lateMaxMs > outputIntervalMs * expected.maximumLateMaxOutputIntervals) {
    failures.push(`x${factor} presentation lateness contract exceeded`);
  }
  if (counters.sourceBusySkipped > expected.maximumSourceBusySkipped
      || counters.sourcePresentedFrameGaps > expected.maximumSourcePresentedFrameGaps
      || counters.sourceMetadataDuplicates > expected.maximumSourceMetadataDuplicates) {
    failures.push(`x${factor} source delivery contract exceeded`);
  }
  if (counters.sourcePoolExhausted + counters.midPoolExhausted > expected.maximumPoolExhaustion) {
    failures.push(`x${factor} texture pool exhausted`);
  }
  if (counters.classificationSuperseded > expected.maximumClassificationSuperseded) {
    failures.push(`x${factor} classifications were superseded`);
  }
  if (counters.skippedPairs > expected.maximumSkippedPairs || pairCoverage < expected.minimumPairCoverageFraction) {
    failures.push(`x${factor} factor/pair coverage contract exceeded`);
  }
  if (producer.skippedScheduleSlots > expected.maximumProducerSkippedScheduleSlots) {
    failures.push(`x${factor} fixture producer skipped source slots`);
  }

  return {
    factor,
    passed: failures.length === 0,
    sourceHz,
    sourceCallbackHz,
    rafHz,
    scheduledHz,
    presentedHz,
    dropped: counters.dropped,
    queueHighWater: counters.queueHighWater,
    lateP95Ms,
    lateMaxMs,
  };
}

export function validateProductHfrReport(report, expected) {
  object(report, 'report');
  object(expected, 'expected');
  if (report.schemaVersion !== 1) fail(`report schemaVersion must be 1, got ${report.schemaVersion}`);
  if (report.profile !== expected.profile) fail(`profile must be ${expected.profile}`);
  exactArray(report.workload?.factors, expected.factors, 'workload.factors');
  if (report.workload?.sourceFps !== expected.sourceFps || report.workload?.width !== expected.width
      || report.workload?.height !== expected.height || report.workload?.model !== 'v7s'
      || report.workload?.resolution !== 720) {
    fail('workload identity mismatch');
  }
  if (report.conditions?.warmupMsPerFactor !== expected.warmupMs
      || report.conditions?.measureMsPerFactor !== expected.measureMs
      || report.conditions?.presentationMetric !== 'gpu-canvas-submit-from-rAF-pump') {
    fail('measurement condition mismatch');
  }
  if (report.browser?.channel !== expected.browserChannel || report.browser?.headed !== true
      || report.browser?.persistentContext !== true || report.browser?.unpackedExtension !== true) {
    fail('browser/extension launch condition mismatch');
  }

  const failures = [];
  exactHashFiles(report.source, expected.hashes, failures);
  const staged = object(report.browser.extensionStageHashes, 'browser.extensionStageHashes');
  for (const [name, expectedHash] of Object.entries(expected.stagedHashes)) {
    if (staged[name] !== expectedHash) failures.push(`unpacked extension hash mismatch: ${name}`);
  }
  const diagnostics = object(report.diagnostics, 'diagnostics');
  if (!Array.isArray(diagnostics.consoleErrors) || !Array.isArray(diagnostics.pageErrors)
      || !Array.isArray(diagnostics.requestFailures)) fail('diagnostics arrays are missing');
  if (diagnostics.consoleErrors.length || diagnostics.pageErrors.length || diagnostics.requestFailures.length) {
    failures.push('browser errors are present');
  }

  const measurements = object(report.measurements, 'measurements');
  const factorResults = [];
  for (const factor of expected.factors) {
    const before = failures.length;
    const summary = validateFactor(measurements[`x${factor}`], factor, expected, failures);
    summary.passed = failures.length === before;
    factorResults.push(summary);
  }
  return { passed: failures.length === 0, failures, factors: factorResults };
}
