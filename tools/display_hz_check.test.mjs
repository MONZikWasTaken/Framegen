import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
  CONTRACT,
  applyInfrastructureChecks,
  evaluate,
  parseArguments,
  resolveOutputPath,
} from './run_display_hz_check.mjs';

const diagnostics = () => ({
  consoleErrors: [], pageErrors: [], requestFailures: [], httpErrors: [],
});

function greenMeasurement() {
  const durationMs = 10000;
  const effectiveTargetHz = 60 * 0.97;
  return {
    producer: { observedHz: 24, producedFrames: 240, skippedScheduleSlots: 0 },
    telemetry: {
      schemaVersion: 1,
      sampleOverflow: false,
      durationMs,
      product: {
        factor: 'hz', targetState: 'active', displayCapacityHz: 60,
        effectiveTargetHz, minimumTargetHz: 48, targetClampReason: null,
        scheduler: { rafFloor: 1000 / 60 },
      },
      observed: { sourceHz: 24, rafHz: 60 },
      counters: {
        sourceCallbacks: 240,
        sourceProcessed: 240,
        scheduled: 582,
        scheduledSource: 240,
        scheduledMid: 342,
        presented: 582,
        presentedSource: 240,
        presentedMid: 342,
        dropped: 0, droppedSource: 0, droppedMid: 0,
        pending: 0, queueHighWater: 2,
        sourcePoolExhausted: 0, midPoolExhausted: 0,
        sourceBusySkipped: 0, ratePlanResets: 0,
      },
      lateness: { count: 582, p95Ms: 10, maxMs: 30 },
      errors: [],
    },
  };
}

test('Display Hz runner CLI and output stay inside the evidence directory', () => {
  assert.deepEqual(parseArguments(['--hz', '60', '--source-fps', '24']), {
    hz: 60, sourceFps: 24, output: null, help: false,
  });
  const output = resolveOutputPath('output/display-hz/unit.json', 'fixed');
  assert.match(output, /output[\\/]display-hz[\\/]unit\.json$/);
  assert.throws(() => resolveOutputPath('output/outside.json'), /inside output\/display-hz/);
  assert.throws(() => resolveOutputPath('output/display-hz/unit.txt'), /\.json file/);
  assert.throws(() => parseArguments(['--hz', '23']), /\[24, 1000\]/);
  assert.throws(() => parseArguments(['--output']), /requires a path/);
});

test('green Display Hz measurement passes the headroom and delivery contract', () => {
  const result = evaluate(greenMeasurement(), { sourceFps: 24 }, 60, {
    diagnostics: diagnostics(), rafControl: { observedHz: 60 },
  });
  assert.equal(result.passed, true, result.failures.join('\n'));
  assert.equal(result.derived.expectedHeadroomHz, 60 * 0.97);
});

test('browser errors and skipped producer slots fail closed', () => {
  const badDiagnostics = diagnostics();
  badDiagnostics.pageErrors.push('device lost');
  const measurement = greenMeasurement();
  measurement.producer.skippedScheduleSlots = 1;
  const result = evaluate(measurement, { sourceFps: 24 }, 60, {
    diagnostics: badDiagnostics, rafControl: { observedHz: 60 },
  });
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(value => value.includes('producer slots')));
  assert.ok(result.failures.some(value => value.includes('pageErrors')));
});

test('source-plan mismatch and target-clock overproduction fail closed', () => {
  const measurement = greenMeasurement();
  measurement.telemetry.product.minimumTargetHz = 44;
  measurement.telemetry.counters.scheduled = 650;
  const result = evaluate(measurement, { sourceFps: 24 }, 60, {
    diagnostics: diagnostics(), rafControl: { observedHz: 60 },
  });
  assert.equal(result.passed, false);
  assert.ok(result.failures.some(value => value.includes('scheduler source')));
  assert.ok(result.failures.some(value => value.includes('scheduled attempts')));
});

test('telemetry accounting identities fail closed independently', () => {
  const corruptions = [
    [measurement => { measurement.telemetry.counters.scheduledMid -= 1; }, 'scheduled total'],
    [measurement => { measurement.telemetry.counters.presentedMid -= 1; }, 'presented total'],
    [measurement => { measurement.telemetry.counters.dropped = 1; }, 'dropped total'],
    [measurement => { measurement.telemetry.counters.pending = 1; }, 'presented plus dropped plus pending'],
    [measurement => { measurement.telemetry.lateness.count -= 1; }, 'lateness count'],
    [measurement => { measurement.telemetry.counters.sourceProcessed -= 1; }, 'source callbacks'],
  ];
  for (const [corrupt, fragment] of corruptions) {
    const measurement = greenMeasurement();
    corrupt(measurement);
    const result = evaluate(measurement, { sourceFps: 24 }, 60, {
      diagnostics: diagnostics(), rafControl: { observedHz: 60 },
    });
    assert.equal(result.passed, false, fragment);
    assert.ok(result.failures.some(value => value.includes(fragment)), fragment);
  }
});

test('missing counters, lateness, overflow evidence, or measurement fail closed', () => {
  assert.equal(evaluate(null, { sourceFps: 24 }, 60).passed, false);
  const missingObserved = greenMeasurement();
  delete missingObserved.telemetry.observed;
  assert.doesNotThrow(() => evaluate(missingObserved, { sourceFps: 24 }, 60, {
    diagnostics: diagnostics(), rafControl: { observedHz: 60 },
  }));
  assert.equal(evaluate(missingObserved, { sourceFps: 24 }, 60, {
    diagnostics: diagnostics(), rafControl: { observedHz: 60 },
  }).passed, false);
  const measurement = greenMeasurement();
  delete measurement.telemetry.counters.dropped;
  measurement.telemetry.lateness.p95Ms = null;
  delete measurement.telemetry.sampleOverflow;
  delete measurement.producer.skippedScheduleSlots;
  const result = evaluate(measurement, { sourceFps: 24 }, 60, {
    diagnostics: diagnostics(), rafControl: { observedHz: 60 },
  });
  assert.equal(result.passed, false);
  for (const fragment of ['dropped', 'lateness p95', 'sample buffer', 'skipped slot']) {
    assert.ok(result.failures.some(value => value.includes(fragment)), fragment);
  }
});

test('restore, source identity, cleanup, and dirty Git failures make the verdict red', () => {
  const result = applyInfrastructureChecks({ passed: true, failures: [], derived: {} }, {
    hashesStart: { cadence: 'before' },
    hashesEnd: { cadence: 'after' },
    restore: { passed: false },
    cleanupErrors: ['temporary directory cleanup failed'],
    git: { dirty: true, errors: ['status unavailable'] },
  });
  assert.equal(result.passed, false);
  for (const fragment of ['source files changed', 'not restored', 'cleanup', 'dirty', 'git metadata']) {
    assert.ok(result.failures.some(value => value.includes(fragment)), fragment);
  }
});

test('runner, fixture, and temporary display helper retain fail-closed contracts', () => {
  const runner = readFileSync(new URL('./run_display_hz_check.mjs', import.meta.url), 'utf8');
  const helper = readFileSync(new URL('./set_primary_refresh.ps1', import.meta.url), 'utf8');
  const fixture = readFileSync(new URL('../web/display_hz_fixture.html', import.meta.url), 'utf8');
  const script = /<script>([\s\S]*)<\/script>/.exec(fixture)?.[1];
  assert.ok(script);
  assert.doesNotThrow(() => new Function(script));
  assert.doesNotMatch(helper, /CDS_UPDATEREGISTRY/);
  assert.match(helper, /ChangeDisplaySettingsExA\(name, ref target, IntPtr\.Zero, 0, IntPtr\.Zero\)/);
  for (const field of ['dmDisplayOrientation', 'dmDisplayFixedOutput', 'dmDisplayFlags']) {
    assert.match(helper, new RegExp(`mode\\.${field} == cur\\.${field}`));
  }
  assert.match(runner, /extension\/assets\/rt_v7s\.bin/);
  assert.match(runner, /Temporary unpacked extension differs from source files/);
  assert.match(runner, /Temporary fixture differs from its source file/);
  assert.match(runner, /flag: 'wx'/);
  assert.match(runner, /restorePrimaryMode/);
  assert.match(runner, /serializeDisplayMutation\(\(\) => setPrimaryRefreshHz/);
  assert.match(runner, /serializeDisplayMutation\(\(\) => restorePrimaryMode/);
  assert.match(runner, /process\.on\('SIGINT'/);
  assert.match(runner, /gitMetadata/);
  assert.match(fixture, /now - producerNextAtMs >= FRAME_MS/);
  assert.doesNotMatch(fixture, /FRAME_MS \* 2/);
  assert.equal(CONTRACT.sourceFiles.declarativeRules, 'extension/rules.json');
  assert.equal(CONTRACT.acceptance.maximumProducerSkippedScheduleSlots, 0);
});
