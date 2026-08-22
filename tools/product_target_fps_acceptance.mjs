export class ProductTargetFpsAcceptanceError extends Error {}

function fail(message) {
  throw new ProductTargetFpsAcceptanceError(`Product target-FPS acceptance: ${message}`);
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
      || actual.some((value, index) => value !== expected[index])) fail(`${name} mismatch`);
}

function exactJson(actual, expected, name) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) fail(`${name} mismatch`);
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

function withinFraction(actual, expected, tolerance) {
  return expected > 0 && Math.abs(actual - expected) / expected <= tolerance;
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

const REQUIRED_SCHEDULER_NUMERIC_FIELDS = [
  'prepCostMs', 'msAvg', 'srCostMs',
  'estimatedPrepCostMs', 'estimatedMidCostMs', 'estimatedSrCostMs',
  'intervalMs', 'decodedIntervalMs', 'uniqueIntervalMs',
  'delayMs', 'lateAvg', 'rafMs', 'rafFloor',
];

function validateScheduler(value, name) {
  const scheduler = object(value, name);
  if (!['gpu', 'defaults'].includes(scheduler.timingMode)) {
    fail(`${name}.timingMode is invalid`);
  }
  for (const field of REQUIRED_SCHEDULER_NUMERIC_FIELDS) {
    finiteNumber(scheduler[field], `${name}.${field}`);
  }
  if (Object.entries(scheduler).some(([field, fieldValue]) => (
    field !== 'timingMode' && !Number.isFinite(fieldValue)
  ))) {
    fail(`${name} contains a non-finite value`);
  }
  return scheduler;
}

function validateTargetCase(raw, targetCase, expected, failures) {
  const label = targetCase.id;
  const requestedHz = targetCase.targetFps;
  const nominalSourceHz = targetCase.sourceFps;
  const nominalMinimumHz = nominalSourceHz * 2;
  const nominalEffectiveHz = targetCase.expectedEffectiveFps;
  object(raw, `measurements.${label}`);
  if (raw.schemaVersion !== 2) fail(`measurements.${label}.schemaVersion must be 2`);
  if (typeof raw.error === 'string' && raw.error) {
    failures.push(`${label} fixture failed: ${raw.error}`);
    return { caseId: label, requestedHz, effectiveTargetHz: nominalEffectiveHz,
      passed: false, error: raw.error };
  }

  const fixture = object(raw.fixture, `measurements.${label}.fixture`);
  if (fixture.name !== expected.fixtureName || fixture.pattern !== expected.pattern
      || fixture.width !== expected.width || fixture.height !== expected.height
      || fixture.nominalSourceFps !== nominalSourceHz) fail(`${label} fixture identity mismatch`);
  if (raw.conditions?.caseId !== label || raw.conditions?.factor !== 'target'
      || raw.conditions?.targetFps !== requestedHz || raw.conditions?.sourceFps !== nominalSourceHz
      || raw.conditions?.resolution !== targetCase.resolution
      || raw.conditions?.warmupMs !== expected.warmupMs
      || raw.conditions?.measureMs !== expected.measureMs) {
    fail(`${label} measurement conditions mismatch`);
  }
  if (raw.bridge?.bridgeVersion !== 2 || raw.configured?.factor !== 'target'
      || raw.configured?.targetFps !== requestedHz
      || raw.configured?.resolution !== targetCase.resolution || raw.configured?.model !== 'v7s') {
    fail(`${label} extension bridge/config mismatch`);
  }
  validateTune(raw.prepared?.convTune, `measurements.${label}.prepared.convTune`);
  if (raw.started?.running !== true || raw.started?.width !== expected.width
      || raw.started?.height !== expected.height) fail(`${label} product path dimensions mismatch`);

  const producer = object(raw.producer, `measurements.${label}.producer`);
  const producerDurationMs = finiteNumber(producer.durationMs, `${label} producer.durationMs`);
  const producerIntervals = finiteArray(producer.intervalsMs, `${label} producer.intervalsMs`);
  integer(producer.producedFrames, `${label} producer.producedFrames`);
  integer(producer.skippedScheduleSlots, `${label} producer.skippedScheduleSlots`);
  if (producerIntervals.length !== Math.max(0, producer.producedFrames - 1)) {
    fail(`${label} producer interval count mismatch`);
  }
  const producerHz = rate(producerIntervals);
  if (producerHz === null) fail(`${label} producer has no cadence samples`);
  sameNumber(producer.observedHz, producerHz, `${label} producer.observedHz`);

  const telemetry = object(raw.telemetry, `measurements.${label}.telemetry`);
  if (telemetry.schemaVersion !== 1 || telemetry.sampleOverflow !== false) {
    fail(`${label} telemetry schema/overflow mismatch`);
  }
  const counters = object(telemetry.counters, `measurements.${label}.telemetry.counters`);
  const counterNames = [
    'sourceCallbacks', 'sourceProcessed', 'sourceBusySkipped', 'sourcePresentedFrameGaps',
    'sourceMetadataDuplicates', 'rafCallbacks', 'scheduled', 'scheduledSource', 'scheduledMid',
    'presented', 'presentedSource', 'presentedMid', 'dropped', 'droppedSource', 'droppedMid',
    'pending', 'queueHighWater', 'sourcePoolExhausted', 'midPoolExhausted',
    'classificationSuperseded', 'pairPlans', 'skippedPairs', 'plannedMids',
  ];
  for (const name of counterNames) integer(counters[name], `${label} counters.${name}`);
  if (counters.sourceProcessed + counters.sourceBusySkipped !== counters.sourceCallbacks) {
    fail(`${label} source callback accounting mismatch`);
  }
  if (counters.scheduledSource + counters.scheduledMid !== counters.scheduled
      || counters.presentedSource + counters.presentedMid !== counters.presented
      || counters.droppedSource + counters.droppedMid !== counters.dropped) {
    fail(`${label} source/mid accounting mismatch`);
  }
  if (counters.scheduled !== counters.presented + counters.dropped + counters.pending) {
    fail(`${label} queue conservation mismatch`);
  }
  if (counters.queueHighWater < counters.pending) fail(`${label} queue accounting mismatch`);

  const samples = object(telemetry.samples, `measurements.${label}.telemetry.samples`);
  const callbackIntervals = finiteArray(samples.sourceCallbackIntervalsMs, `${label} sourceCallbackIntervalsMs`);
  const mediaIntervals = finiteArray(samples.sourceMediaIntervalsMs, `${label} sourceMediaIntervalsMs`);
  const rafIntervals = finiteArray(samples.rafIntervalsMs, `${label} rafIntervalsMs`);
  const lateMs = finiteArray(samples.lateMs, `${label} lateMs`);
  if (callbackIntervals.length !== Math.max(0, counters.sourceCallbacks - 1)
      || mediaIntervals.length !== Math.max(0, counters.sourceCallbacks - 1)
      || rafIntervals.length !== Math.max(0, counters.rafCallbacks - 1)
      || lateMs.length !== counters.presented) fail(`${label} telemetry sample count mismatch`);
  const sourceCallbackHz = rate(callbackIntervals);
  const sourceHz = rate(mediaIntervals);
  const rafHz = rate(rafIntervals);
  if (sourceCallbackHz === null || sourceHz === null || rafHz === null) {
    fail(`${label} telemetry cadence samples are empty`);
  }
  sameNumber(telemetry.observed?.sourceCallbackHz, sourceCallbackHz, `${label} observed.sourceCallbackHz`);
  sameNumber(telemetry.observed?.sourceHz, sourceHz, `${label} observed.sourceHz`);
  sameNumber(telemetry.observed?.rafHz, rafHz, `${label} observed.rafHz`);
  const lateP95Ms = percentile(lateMs, 0.95);
  const lateMaxMs = lateMs.length ? Math.max(...lateMs) : null;
  if (lateP95Ms === null || lateMaxMs === null) fail(`${label} has no presentation samples`);
  if (telemetry.lateness?.count !== lateMs.length) fail(`${label} lateness count mismatch`);
  sameNumber(telemetry.lateness?.p95Ms, lateP95Ms, `${label} lateness.p95Ms`);
  sameNumber(telemetry.lateness?.maxMs, lateMaxMs, `${label} lateness.maxMs`);

  const histogram = object(telemetry.plannedFactorHistogram, `${label} plannedFactorHistogram`);
  let histogramTotal = 0;
  let histogramPlannedMids = 0;
  for (const [key, count] of Object.entries(histogram)) {
    integer(count, `${label} plannedFactorHistogram.${key}`);
    if (!/^\d+$/.test(key) || Number(key) < 1) fail(`${label} factor histogram key is invalid`);
    histogramTotal += count;
    histogramPlannedMids += count * (Number(key) - 1);
  }
  if (histogramTotal !== counters.pairPlans) fail(`${label} factor histogram count mismatch`);
  if (histogramPlannedMids !== counters.plannedMids) {
    failures.push(`${label} planned mid provenance does not match pair plans`);
  }

  const product = object(telemetry.product, `measurements.${label}.telemetry.product`);
  if (product.running !== true || product.factor !== 'target' || product.targetFps !== requestedHz
      || product.resolution !== targetCase.resolution || product.model !== 'v7s'
      || product.modelAsset !== 'rt_v7s' || product.videoWidth !== expected.width
      || product.videoHeight !== expected.height) {
    fail(`${label} product identity mismatch`);
  }
  validateTune(product.convTune, `${label} product.convTune`);
  if (JSON.stringify(product.convTune) !== JSON.stringify(raw.prepared.convTune)) {
    fail(`${label} prepared/applied convTune mismatch`);
  }
  exactArray(product.deviceFeatures, [...product.deviceFeatures].sort(), `${label} sorted deviceFeatures`);
  for (const feature of expected.requiredDeviceFeatures) {
    if (!product.deviceFeatures.includes(feature)) failures.push(`${label} missing WebGPU feature: ${feature}`);
  }
  const scheduler = validateScheduler(product.scheduler, `${label} product.scheduler`);
  integer(product.effectiveFactor, `${label} product.effectiveFactor`);
  const productRequestedHz = finiteNumber(product.requestedTargetHz, `${label} product.requestedTargetHz`);
  const productMinimumHz = finiteNumber(product.minimumTargetHz, `${label} product.minimumTargetHz`);
  const displayCapacityHz = finiteNumber(product.displayCapacityHz, `${label} product.displayCapacityHz`);
  const effectiveTargetHz = finiteNumber(product.effectiveTargetHz, `${label} product.effectiveTargetHz`);
  const decodedIntervalMs = finiteNumber(scheduler.decodedIntervalMs,
    `${label} scheduler.decodedIntervalMs`);
  const measuredMinimumHz = 2000 / decodedIntervalMs;
  sameNumber(productRequestedHz, requestedHz, `${label} requested target`, 0.01);
  sameNumber(productMinimumHz, measuredMinimumHz, `${label} measured minimum target`, 0.05);
  if (!withinFraction(productMinimumHz, nominalMinimumHz, expected.sourceHzToleranceFraction)) {
    failures.push(`${label} minimum target mismatch`);
  }
  if (product.targetState !== 'active') failures.push(`${label} target state is not active`);
  if (product.targetClampReason !== targetCase.expectedClampReason) {
    failures.push(`${label} target clamp reason mismatch`);
  }
  if (!withinFraction(effectiveTargetHz, nominalEffectiveHz, expected.targetHzToleranceFraction)) {
    failures.push(`${label} effective target mismatch`);
  }
  if (targetCase.expectedClampReason === 'display'
      && !withinFraction(effectiveTargetHz, displayCapacityHz, expected.targetHzToleranceFraction)) {
    failures.push(`${label} display clamp did not resolve to display capacity`);
  }
  if (product.cuts !== 0 || product.duplicates !== 0) {
    failures.push(`${label} deterministic source was classified as a cut or duplicate`);
  }
  if (!Array.isArray(telemetry.errors)) fail(`${label} telemetry.errors must be an array`);
  if (telemetry.errors.length) failures.push(`${label} extension errors are present`);
  if (product.integrated) failures.push(`${label} ran on an integrated GPU`);

  const durationMs = finiteNumber(telemetry.durationMs, `${label} telemetry.durationMs`);
  const scheduledHz = counters.scheduled * 1000 / durationMs;
  const scheduledSourceHz = counters.scheduledSource * 1000 / durationMs;
  const scheduledMidHz = counters.scheduledMid * 1000 / durationMs;
  const presentedHz = counters.presented * 1000 / durationMs;
  const presentedSourceHz = counters.presentedSource * 1000 / durationMs;
  const presentedMidHz = counters.presentedMid * 1000 / durationMs;
  const plannedMidHz = counters.plannedMids * 1000 / durationMs;
  const expectedMidHz = effectiveTargetHz - nominalSourceHz;
  const outputIntervalMs = 1000 / effectiveTargetHz;
  const minimumSourceCallbacks = nominalSourceHz * durationMs / 1000
    * expected.minimumSourceCallbacksFraction;
  const eligiblePairs = Math.max(0, counters.sourceProcessed - 1);
  const pairCoverage = eligiblePairs ? counters.pairPlans / eligiblePairs : 0;

  if (!withinFraction(durationMs, expected.measureMs, expected.durationToleranceFraction)
      || !withinFraction(producerDurationMs, expected.measureMs, expected.durationToleranceFraction)) {
    failures.push(`${label} measurement duration mismatch`);
  }
  if (!withinFraction(sourceHz, nominalSourceHz, expected.sourceHzToleranceFraction)
      || !withinFraction(sourceCallbackHz, nominalSourceHz, expected.sourceHzToleranceFraction)
      || !withinFraction(producerHz, nominalSourceHz, expected.sourceHzToleranceFraction)) {
    failures.push(`${label} source cadence mismatch`);
  }
  if (!withinFraction(rafHz, expected.expectedDisplayHz, expected.displayHzToleranceFraction)) {
    failures.push(`${label} display/rAF cadence mismatch`);
  }
  if (!withinFraction(scheduledHz, effectiveTargetHz, expected.targetHzToleranceFraction)) {
    failures.push(`${label} target scheduler cadence mismatch`);
  }
  if (presentedHz < effectiveTargetHz * expected.minimumPresentedHzFraction) {
    failures.push(`${label} target presentation throughput mismatch`);
  }
  if (!withinFraction(scheduledSourceHz, nominalSourceHz, expected.componentHzToleranceFraction)
      || presentedSourceHz < nominalSourceHz * expected.minimumPresentedHzFraction) {
    failures.push(`${label} source-anchor provenance mismatch`);
  }
  if (!withinFraction(scheduledMidHz, expectedMidHz, expected.componentHzToleranceFraction)
      || !withinFraction(plannedMidHz, expectedMidHz, expected.componentHzToleranceFraction)
      || presentedMidHz < expectedMidHz * expected.minimumPresentedHzFraction) {
    failures.push(`${label} generated-mid provenance mismatch`);
  }
  if (counters.skippedPairs !== 0
      || Math.abs(counters.plannedMids - counters.scheduledMid) > expected.maximumMidPipelineLag) {
    failures.push(`${label} fallback or unproven mid work is present`);
  }
  if (counters.sourceCallbacks < minimumSourceCallbacks) failures.push(`${label} too few source callbacks`);
  if (producer.producedFrames < minimumSourceCallbacks
      || Math.abs(producer.producedFrames - counters.sourceCallbacks)
        > Math.max(3, counters.sourceCallbacks * 0.03)) {
    failures.push(`${label} source producer/rVFC delivery mismatch`);
  }
  if (counters.dropped > expected.maximumDroppedFrames) failures.push(`${label} dropped frames exceed contract`);
  if (counters.queueHighWater > expected.maximumQueueHighWater
      || counters.pending > expected.maximumPendingFrames) failures.push(`${label} queue contract exceeded`);
  if (lateP95Ms > outputIntervalMs * expected.maximumLateP95OutputIntervals
      || lateMaxMs > outputIntervalMs * expected.maximumLateMaxOutputIntervals) {
    failures.push(`${label} presentation lateness contract exceeded`);
  }
  if (counters.sourceBusySkipped > expected.maximumSourceBusySkipped
      || counters.sourcePresentedFrameGaps > expected.maximumSourcePresentedFrameGaps
      || counters.sourceMetadataDuplicates > expected.maximumSourceMetadataDuplicates) {
    failures.push(`${label} source delivery contract exceeded`);
  }
  if (counters.sourcePoolExhausted + counters.midPoolExhausted > expected.maximumPoolExhaustion) {
    failures.push(`${label} texture pool exhausted`);
  }
  if (counters.classificationSuperseded > expected.maximumClassificationSuperseded) {
    failures.push(`${label} classifications were superseded`);
  }
  if (pairCoverage < expected.minimumPairCoverageFraction) {
    failures.push(`${label} pair coverage contract exceeded`);
  }
  if (producer.skippedScheduleSlots > expected.maximumProducerSkippedScheduleSlots) {
    failures.push(`${label} fixture producer skipped source slots`);
  }

  return {
    caseId: label,
    sourceFps: nominalSourceHz,
    requestedHz,
    minimumTargetHz: productMinimumHz,
    effectiveTargetHz,
    clampReason: product.targetClampReason,
    passed: failures.length === 0,
    sourceHz,
    sourceCallbackHz,
    rafHz,
    scheduledHz,
    scheduledSourceHz,
    scheduledMidHz,
    presentedHz,
    presentedSourceHz,
    presentedMidHz,
    plannedMidHz,
    dropped: counters.dropped,
    pending: counters.pending,
    queueHighWater: counters.queueHighWater,
    lateP95Ms,
    lateMaxMs,
  };
}

export function validateProductTargetFpsReport(report, expected) {
  object(report, 'report');
  object(expected, 'expected');
  if (report.schemaVersion !== 2) fail(`report schemaVersion must be 2, got ${report.schemaVersion}`);
  if (report.gateId !== expected.gateId) fail(`gateId must be ${expected.gateId}`);
  if (report.profile !== expected.profile) fail(`profile must be ${expected.profile}`);
  exactJson(report.workload?.cases, expected.cases, 'workload.cases');
  exactArray(report.workload?.supportedSourceFps, expected.supportedSourceFps,
    'workload.supportedSourceFps');
  const resolutions = [...new Set(expected.cases.map(targetCase => targetCase.resolution))];
  if (resolutions.length !== 1) fail('all authoritative cases must use one internal resolution');
  if (report.workload?.width !== expected.width || report.workload?.height !== expected.height
      || report.workload?.model !== 'v7s' || report.workload?.resolution !== resolutions[0]
      || report.workload?.expectedDisplayHz !== expected.expectedDisplayHz) {
    fail('workload identity mismatch');
  }
  if (report.conditions?.warmupMsPerCase !== expected.warmupMs
      || report.conditions?.measureMsPerCase !== expected.measureMs
      || report.conditions?.presentationMetric !== 'gpu-canvas-submit-from-rAF-pump') {
    fail('measurement condition mismatch');
  }
  if (report.browser?.channel !== expected.browserChannel
      || report.browser?.distribution !== expected.browserDistribution
      || report.browser?.headed !== true || report.browser?.persistentContext !== true
      || report.browser?.unpackedExtension !== true) fail('browser/extension launch condition mismatch');

  const failures = [];
  exactHashFiles(report.source, expected.hashes, failures);
  const staged = object(report.browser.extensionStageHashes, 'browser.extensionStageHashes');
  for (const [name, expectedHash] of Object.entries(expected.stagedHashes)) {
    if (staged[name] !== expectedHash) failures.push(`unpacked extension hash mismatch: ${name}`);
  }
  const diagnostics = object(report.diagnostics, 'diagnostics');
  for (const name of ['consoleErrors', 'pageErrors', 'requestFailures', 'httpErrors']) {
    if (!Array.isArray(diagnostics[name])) fail(`diagnostics.${name} must be an array`);
    if (diagnostics[name].length) failures.push(`browser errors are present: ${name}`);
  }
  if (report.executionError) failures.push('runner execution error is present');

  const measurements = object(report.measurements, 'measurements');
  exactArray(Object.keys(measurements).sort(), expected.cases.map(targetCase => targetCase.id).sort(),
    'measurement case ids');
  const caseResults = [];
  for (const targetCase of expected.cases) {
    const before = failures.length;
    const summary = validateTargetCase(measurements[targetCase.id], targetCase, expected, failures);
    summary.passed = failures.length === before;
    caseResults.push(summary);
  }
  return { passed: failures.length === 0, failures, cases: caseResults };
}
