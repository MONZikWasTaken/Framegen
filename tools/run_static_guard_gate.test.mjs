import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';

import {
  STATIC_GUARD_CONTRACT,
  finalizeStaticGuardReport,
} from './static_guard_acceptance.mjs';
import {
  STATIC_GUARD_RUNNER_CONTRACT,
  parseArguments,
  resolveOutputPath,
  validateGateEvidence,
} from './run_static_guard_gate.mjs';

const HASHES = Object.freeze({
  source: 'a'.repeat(64),
  runtime: 'b'.repeat(64),
  weights: 'c'.repeat(64),
  manifest: 'd'.repeat(64),
  runner: 'e'.repeat(64),
  validator: 'f'.repeat(64),
});

function fixtureReport() {
  return {
    ...finalizeStaticGuardReport({
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
      metrics: {
        guardOffParity: { mismatchValues: 0, maxAbsLsb: 0, sampleValues: 100 },
        text: {
          candidateMaeLsb: 0.25,
          candidateMaxAbsLsb: 0.5,
          anchorStepMaxErrorLsb: 0.75,
          temporalSecondDifferenceMaxLsb: 1,
          monotonicViolations: 0,
          minimumContrastLsb: 200,
          sampleValues: 100,
        },
        ordinaryMotion: { guardVsOffMaeLsb: 0, guardVsOffMaxAbsLsb: 0, sampleValues: 100 },
      },
    }),
    identities: {
      runtimeSha256: HASHES.runtime,
      weightsSha256: HASHES.weights,
      manifestSha256: HASHES.manifest,
    },
  };
}

function cleanDiagnostics() {
  return {
    consoleMessages: [],
    consoleErrors: [],
    consoleWarnings: [],
    pageErrors: [],
    requests: [],
    requestFailures: [],
    httpErrors: [],
  };
}

function validEvidence() {
  return {
    fixtureReport: fixtureReport(),
    diagnostics: cleanDiagnostics(),
    hashesStart: { ...HASHES },
    hashesEnd: { ...HASHES },
    executionError: null,
  };
}

test('runner contract pins system Chrome, Playwright, sources, and output scope', () => {
  assert.equal(STATIC_GUARD_RUNNER_CONTRACT.playwrightPackage, '@playwright/cli@0.1.17');
  assert.equal(STATIC_GUARD_RUNNER_CONTRACT.browser, 'chrome');
  assert.equal(STATIC_GUARD_RUNNER_CONTRACT.headed, true);
  assert.deepEqual(Object.keys(STATIC_GUARD_RUNNER_CONTRACT.sourceFiles),
    ['source', 'runtime', 'weights', 'manifest', 'runner', 'validator']);
  const output = resolveOutputPath('output/static-guard/frozen.json', 'unused');
  assert.ok(output.endsWith(path.join('output', 'static-guard', 'frozen.json')));
  assert.throws(() => resolveOutputPath('output/static-guard/../escape.json', 'unused'));
  assert.throws(() => resolveOutputPath('output/static-guard/not-json.txt', 'unused'));
});

test('runner CLI accepts only help and a scoped output path', () => {
  assert.deepEqual(parseArguments(['--help']), { help: true });
  assert.deepEqual(parseArguments(['--output', 'output/static-guard/result.json']),
    { output: 'output/static-guard/result.json' });
  assert.throws(() => parseArguments(['--manifest', 'anything.json']));
  assert.throws(() => parseArguments(['--output']));
});

test('green fixture, stable identities, and clean diagnostics pass', () => {
  const validation = validateGateEvidence(validEvidence());
  assert.equal(validation.passed, true, validation.failures.join(', '));
  assert.equal(validation.nodeContract.passed, true);
  assert.equal(validation.diagnosticsClean, true);
});

test('runner fails closed on source drift, identity drift, browser errors, and red fixture', () => {
  const evidence = validEvidence();
  evidence.hashesEnd.runtime = '0'.repeat(64);
  evidence.fixtureReport.identities.weightsSha256 = '1'.repeat(64);
  evidence.diagnostics.consoleErrors.push({ type: 'error', text: 'validation error' });
  evidence.fixtureReport.passed = false;
  const validation = validateGateEvidence(evidence);
  assert.equal(validation.passed, false);
  assert.ok(validation.failures.includes('source-hash-drift:runtime'));
  assert.ok(validation.failures.includes('fixture-identity-mismatch:weights'));
  assert.ok(validation.failures.includes('browser-diagnostics:consoleErrors'));
  assert.ok(validation.failures.includes('fixture-report-not-passed'));
});

test('runner converts missing evidence and execution failures into a red result', () => {
  const validation = validateGateEvidence({
    fixtureReport: null,
    diagnostics: cleanDiagnostics(),
    hashesStart: { ...HASHES },
    hashesEnd: { ...HASHES },
    executionError: { message: 'browser unavailable' },
  });
  assert.equal(validation.passed, false);
  assert.ok(validation.failures.includes('fixture-report-structurally-invalid'));
  assert.ok(validation.failures.includes('runner-execution-error'));
});

test('runner source keeps loopback, promise wait, diagnostics, and write-once evidence', async () => {
  const source = await readFile(new URL('./run_static_guard_gate.mjs', import.meta.url), 'utf8');
  assert.match(source, /server\.listen\(0, '127\.0\.0\.1'/);
  assert.match(source, /globalThis\.__STATIC_GUARD_PROMISE__/);
  assert.match(source, /'--browser', STATIC_GUARD_RUNNER_CONTRACT\.browser, '--headed'/);
  assert.doesNotMatch(source, /playwright install|install chromium|chrome-for-testing/i);
  assert.match(source, /page\.on\('requestfailed'/);
  assert.match(source, /page\.on\('pageerror'/);
  assert.match(source, /page\.on\('console'/);
  assert.match(source, /flag: 'wx'/);
  assert.match(source, /output', 'static-guard'/);
  const ignore = await readFile(new URL('../.gitignore', import.meta.url), 'utf8');
  assert.match(ignore, /^\/output\/$|^\/output\/static-guard\/$/m);
});
