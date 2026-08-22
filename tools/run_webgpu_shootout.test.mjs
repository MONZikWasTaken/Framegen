import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  closeServer,
  parseArguments,
  resolveOutputPath,
  startStaticServer,
  summarizeSamples,
  validateStableDeviceEvidence,
  waitForServer,
} from './run_webgpu_shootout.mjs';

test('VFI shootout arguments accept only bounded supported workloads', () => {
  assert.deepEqual(parseArguments([]), {
    resolution: 480,
    repetitions: null,
    output: null,
    help: false,
  });
  assert.deepEqual(
    parseArguments(['--res', '720', '--repetitions', '3', '--output', '.bench/results/test']),
    {
      resolution: 720,
      repetitions: 3,
      output: '.bench/results/test',
      help: false,
    },
  );
  assert.throws(() => parseArguments(['--res', '1080']), /480 or 720/);
  assert.throws(() => parseArguments(['--repetitions', '0']), /integer in \[1, 20\]/);
  assert.throws(() => parseArguments(['--unknown']), /Unknown option/);
});

test('VFI shootout reports raw-sample distribution without choosing a best run', () => {
  const summary = summarizeSamples([5, 1, 3, 2, 4]);
  assert.equal(summary.count, 5);
  assert.equal(summary.minMs, 1);
  assert.equal(summary.medianMs, 3);
  assert.equal(summary.meanMs, 3);
  assert.equal(summary.p95Ms, 5);
  assert.equal(summary.maxMs, 5);
  assert.ok(Math.abs(summary.standardDeviationMs - Math.sqrt(2)) < 1e-12);
});

test('VFI shootout requires one stable, fully identified WebGPU device', () => {
  const evidence = {
    adapterIdentity: {
      vendor: '10de', architecture: 'ada', device: '2803', description: 'RTX',
    },
    deviceFeatures: ['shader-f16', 'subgroups'],
    deviceLimits: { maxTextureDimension2D: 32768, maxBufferSize: 2147483648 },
  };
  assert.deepEqual(validateStableDeviceEvidence([evidence, structuredClone(evidence)]), evidence);
  assert.throws(
    () => validateStableDeviceEvidence([
      evidence,
      { ...structuredClone(evidence), deviceFeatures: ['shader-f16'] },
    ]),
    /changed between scenes/,
  );
  assert.throws(
    () => validateStableDeviceEvidence([{ ...evidence, adapterIdentity: {} }]),
    /unidentified WebGPU adapter/,
  );
});

test('VFI shootout output is confined to ignored benchmark results', () => {
  const output = resolveOutputPath('.bench/results/vfi-test', 480, 'unused');
  assert.match(output.replaceAll('\\', '/'), /\.bench\/results\/vfi-test$/);
  assert.throws(
    () => resolveOutputPath('../outside', 480, 'unused'),
    /inside \.bench\/results/,
  );
});

test('VFI shootout manifest follows the repository Playwright contract', async () => {
  const manifest = JSON.parse(await readFile(
    new URL('../benchmarks/vfi-shootout/manifest.json', import.meta.url),
    'utf8',
  ));
  const contract = JSON.parse(await readFile(
    new URL('./webgpu_bench_manifest.json', import.meta.url),
    'utf8',
  ));
  const harness = await readFile(
    new URL('../web/shootout.html', import.meta.url),
    'utf8',
  );
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.playwrightContract, 'tools/webgpu_bench_manifest.json');
  assert.equal(contract.playwrightCli.package, '@playwright/cli@0.1.17');
  assert.equal(contract.playwrightCli.headed, true);
  assert.deepEqual(manifest.inferenceRungs, {
    480: [848, 480],
    720: [1280, 720],
  });
  assert.deepEqual(manifest.referenceMethods.ifrnet, {
    inputColorOrder: 'RGB',
    outputColorOrder: 'RGB',
    paddingDivisor: 20,
    paddingMode: 'replicate',
    scaleFactor: 0.8,
  });
  const pairLoopCounts = [...harness.matchAll(
    /for \(let index = 0; index < (\d+); index\+\+\)/g,
  )].map(match => Number(match[1]));
  assert.deepEqual(pairLoopCounts.slice(0, 2), [
    manifest.measurement.pageWarmupPairCalls,
    manifest.measurement.pageMeasuredPairCalls,
  ]);
  assert.match(harness, /adapterIdentity: adapterIdentity\(adapter\)/);
  assert.match(harness, /deviceFeatures: \[\.\.\.device\.features\]\.sort\(\)/);
});

test('VFI shootout server is loopback-only, allowlisted, ready, and closable', async () => {
  const manifestPath = new URL(
    '../benchmarks/vfi-shootout/manifest.json',
    import.meta.url,
  );
  const server = await startStaticServer([fileURLToPath(manifestPath)]);
  try {
    const address = server.address();
    assert.equal(address.address, '127.0.0.1');
    const baseUrl = 'http://127.0.0.1:' + address.port;
    const allowedUrl = baseUrl + '/benchmarks/vfi-shootout/manifest.json';
    await waitForServer(allowedUrl);
    assert.equal((await fetch(allowedUrl)).status, 200);
    assert.equal((await fetch(baseUrl + '/AGENTS.md')).status, 404);
  } finally {
    await closeServer(server);
  }
  assert.equal(server.listening, false);
});
