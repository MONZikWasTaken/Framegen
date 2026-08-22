#!/usr/bin/env node
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { access, mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(scriptPath), '..');
const shootoutManifestPath = path.join(repositoryRoot, 'benchmarks', 'vfi-shootout', 'manifest.json');
const defaultResultsRoot = path.join(repositoryRoot, '.bench', 'results');

const HELP = `Usage: node tools/run_webgpu_shootout.mjs [options]

Runs the direct v7s WebGPU midpoint shootout through the repository's pinned
Playwright CLI and installed Chrome contract. Results are written to a new,
provenance-stamped directory; existing results are never overwritten.

  --res <480|720>       inference resolution (default: 480)
  --repetitions <n>     independent page medians per scene (default: manifest)
  --output <directory>  new directory inside .bench/results
  --help
`;

function inside(parent, candidate) {
  return candidate.startsWith(`${parent}${path.sep}`);
}

async function exists(filePath) {
  try { await access(filePath); return true; } catch { return false; }
}

async function run(command, args, options = {}) {
  try {
    const result = await execFileAsync(command, args, {
      cwd: options.cwd || repositoryRoot,
      encoding: 'utf8',
      maxBuffer: 128 * 1024 * 1024,
      timeout: options.timeout,
      windowsHide: true,
    });
    return result.stdout.trim();
  } catch (cause) {
    const details = [cause.stderr?.trim(), cause.stdout?.trim()].filter(Boolean).join('\n');
    throw new Error(`${command} ${args.join(' ')} failed${details ? `:\n${details}` : ''}`, { cause });
  }
}

async function runOptional(command, args, options = {}) {
  try { return { available: true, stdout: await run(command, args, options) }; }
  catch (error) { return { available: false, error: error.message }; }
}

function npxInvocation() {
  if (process.platform === 'win32') {
    const prerequisite = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npx-cli.js');
    return { command: process.execPath, prefix: [prerequisite], prerequisite };
  }
  return { command: 'npx', prefix: [], prerequisite: null };
}

function timestampLabel() {
  return new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
}

export function parseArguments(argv) {
  const options = { resolution: 480, repetitions: null, output: null, help: false };
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (token === '--help') { options.help = true; continue; }
    if (!['--res', '--repetitions', '--output'].includes(token)) {
      throw new Error(`Unknown option: ${token}`);
    }
    const value = argv[++index];
    if (value === undefined) throw new Error(`Missing value for ${token}`);
    if (token === '--res') options.resolution = Number(value);
    if (token === '--repetitions') options.repetitions = Number(value);
    if (token === '--output') options.output = value;
  }
  if (![480, 720].includes(options.resolution)) throw new Error('--res must be 480 or 720');
  if (options.repetitions !== null
      && (!Number.isInteger(options.repetitions) || options.repetitions < 1 || options.repetitions > 20)) {
    throw new Error('--repetitions must be an integer in [1, 20]');
  }
  return options;
}

export function summarizeSamples(samples) {
  if (!Array.isArray(samples) || samples.length === 0
      || samples.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error('samples must be a non-empty array of finite non-negative numbers');
  }
  const sorted = [...samples].sort((left, right) => left - right);
  const mean = samples.reduce((sum, value) => sum + value, 0) / samples.length;
  const middle = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const variance = samples.reduce((sum, value) => sum + (value - mean) ** 2, 0) / samples.length;
  const standardDeviation = Math.sqrt(variance);
  return {
    count: samples.length,
    minMs: sorted[0],
    medianMs: median,
    meanMs: mean,
    p95Ms: sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)],
    maxMs: sorted.at(-1),
    standardDeviationMs: standardDeviation,
    coefficientOfVariationPercent: mean > 0 ? standardDeviation / mean * 100 : 0,
  };
}

export function validateStableDeviceEvidence(records) {
  if (!Array.isArray(records) || records.length === 0) {
    throw new Error('Shootout browser returned no device evidence');
  }
  let expected = null;
  for (const record of records) {
    const adapterIdentity = record?.adapterIdentity;
    if (!adapterIdentity || typeof adapterIdentity !== 'object'
        || !['vendor', 'architecture', 'device', 'description']
          .some(key => typeof adapterIdentity[key] === 'string' && adapterIdentity[key])) {
      throw new Error('Shootout browser returned an unidentified WebGPU adapter');
    }
    const deviceFeatures = record.deviceFeatures;
    if (!Array.isArray(deviceFeatures)
        || deviceFeatures.some(value => typeof value !== 'string')
        || JSON.stringify(deviceFeatures) !== JSON.stringify([...new Set(deviceFeatures)].sort())) {
      throw new Error('Shootout browser returned invalid WebGPU feature evidence');
    }
    const deviceLimits = record.deviceLimits;
    if (!deviceLimits || typeof deviceLimits !== 'object'
        || Object.keys(deviceLimits).length === 0
        || Object.values(deviceLimits).some(value => !Number.isFinite(value) || value <= 0)) {
      throw new Error('Shootout browser returned invalid WebGPU limit evidence');
    }
    const identity = JSON.stringify({ adapterIdentity, deviceFeatures, deviceLimits });
    if (expected !== null && expected !== identity) {
      throw new Error('WebGPU adapter, enabled features, or limits changed between scenes');
    }
    expected = identity;
  }
  return JSON.parse(expected);
}

function resolveRepositoryPath(relativePath, label) {
  if (typeof relativePath !== 'string' || !relativePath.trim()) throw new Error(`${label} must be a path`);
  const resolved = path.resolve(repositoryRoot, relativePath);
  if (resolved !== repositoryRoot && !inside(repositoryRoot, resolved)) {
    throw new Error(`${label} escapes the repository`);
  }
  return resolved;
}

export function resolveOutputPath(requestedPath, resolution, label = timestampLabel()) {
  const defaultPath = path.join(defaultResultsRoot, `framegen-v7s-${resolution}`, label);
  const resolved = path.resolve(repositoryRoot, requestedPath || defaultPath);
  if (!inside(defaultResultsRoot, resolved)) {
    throw new Error('--output must be a directory inside .bench/results');
  }
  return resolved;
}

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function fileMetadata(filePath) {
  const value = await stat(filePath);
  return {
    path: path.relative(repositoryRoot, filePath).replaceAll('\\', '/'),
    sizeBytes: value.size,
    sha256: await sha256(filePath),
  };
}

async function gitMetadata() {
  const safe = ['-c', `safe.directory=${repositoryRoot}`];
  const [commit, branch, status] = await Promise.all([
    runOptional('git', [...safe, 'rev-parse', 'HEAD']),
    runOptional('git', [...safe, 'rev-parse', '--abbrev-ref', 'HEAD']),
    runOptional('git', [...safe, 'status', '--porcelain=v1', '--untracked-files=all']),
  ]);
  return {
    commit: commit.available ? commit.stdout : null,
    branch: branch.available ? branch.stdout : null,
    dirty: status.available ? status.stdout.length > 0 : null,
    statusPorcelain: status.available ? status.stdout.split(/\r?\n/).filter(Boolean) : null,
    errors: [commit, branch, status].filter(value => !value.available).map(value => value.error),
  };
}

async function collectDatasetMetadata(sceneFiles) {
  return Promise.all(sceneFiles.map(async scene => ({
    id: scene.id,
    dataset: scene.dataset,
    kind: scene.kind,
    files: Object.fromEntries(await Promise.all(Object.entries(scene.files)
      .map(async ([name, filePath]) => [name, await fileMetadata(filePath)]))),
  })));
}

async function validateInputs(options) {
  const manifest = JSON.parse(await readFile(shootoutManifestPath, 'utf8'));
  if (manifest.schemaVersion !== 1 || manifest.id !== 'framegen-v7s-vfi-shootout-v1') {
    throw new Error('Unexpected VFI shootout manifest schema or id');
  }
  if (!manifest.measurement.supportedInferenceResolutions.includes(options.resolution)) {
    throw new Error(`Resolution ${options.resolution} is not supported by the shootout manifest`);
  }
  const inferenceRung = manifest.inferenceRungs?.[String(options.resolution)];
  if (!Array.isArray(inferenceRung) || inferenceRung.length !== 2
      || inferenceRung.some(value => !Number.isInteger(value) || value <= 0)) {
    throw new Error(`Resolution ${options.resolution} has an invalid inference rung`);
  }
  const repetitions = options.repetitions ?? manifest.measurement.defaultRepetitions;
  if (!Number.isInteger(repetitions)
      || repetitions < manifest.measurement.minimumRepetitions
      || repetitions > manifest.measurement.maximumRepetitions) {
    throw new Error('Requested repetitions are outside the shootout manifest limits');
  }
  const scenesPath = resolveRepositoryPath(manifest.scenes, 'manifest.scenes');
  const scenes = JSON.parse(await readFile(scenesPath, 'utf8'));
  if (!Array.isArray(scenes) || scenes.length === 0) throw new Error('Shootout scenes must be a non-empty array');
  const ids = new Set();
  const sceneFiles = [];
  for (const [index, scene] of scenes.entries()) {
    if (!scene || typeof scene !== 'object' || !/^[a-z0-9][a-z0-9-]{2,79}$/.test(scene.id || '')) {
      throw new Error(`scenes[${index}].id is invalid`);
    }
    if (ids.has(scene.id)) throw new Error(`Duplicate scene id: ${scene.id}`);
    ids.add(scene.id);
    const files = {};
    for (const field of ['i0', 'gt', 'i1']) {
      files[field] = resolveRepositoryPath(scene[field], `scenes[${index}].${field}`);
      if (!await exists(files[field])) throw new Error(`Missing ${scene.id} ${field}: ${files[field]}`);
    }
    sceneFiles.push({ ...scene, files });
  }
  const sourcePaths = {
    runner: scriptPath,
    shootoutManifest: shootoutManifestPath,
    scenes: scenesPath,
    playwrightContract: resolveRepositoryPath(manifest.playwrightContract, 'manifest.playwrightContract'),
    harness: resolveRepositoryPath(manifest.harness, 'manifest.harness'),
    runtime: resolveRepositoryPath(manifest.runtime, 'manifest.runtime'),
    weights: resolveRepositoryPath(manifest.modelAssets.weights, 'manifest.modelAssets.weights'),
    weightsManifest: resolveRepositoryPath(manifest.modelAssets.manifest, 'manifest.modelAssets.manifest'),
  };
  for (const [name, filePath] of Object.entries(sourcePaths)) {
    if (!await exists(filePath)) throw new Error(`Missing ${name}: ${filePath}`);
  }
  const harnessSource = await readFile(sourcePaths.harness, 'utf8');
  const pairLoopCounts = [...harnessSource.matchAll(
    /for \(let index = 0; index < (\d+); index\+\+\)/g,
  )].map(match => Number(match[1]));
  if (pairLoopCounts[0] !== manifest.measurement.pageWarmupPairCalls
      || pairLoopCounts[1] !== manifest.measurement.pageMeasuredPairCalls
      || !/runtime\.runT\(0\.5, output\)/.test(harnessSource)
      || !/adapterIdentity: adapterIdentity\(adapter\)/.test(harnessSource)
      || !/deviceFeatures: \[\.\.\.device\.features\]\.sort\(\)/.test(harnessSource)
      || !/deviceLimits: deviceLimitIdentity\(device\)/.test(harnessSource)) {
    throw new Error('Shootout harness measurement loops do not match the manifest');
  }
  const playwrightContract = JSON.parse(await readFile(sourcePaths.playwrightContract, 'utf8'));
  if (playwrightContract.schemaVersion !== 2
      || typeof playwrightContract.playwrightCli?.package !== 'string'
      || typeof playwrightContract.playwrightCli?.browser !== 'string'
      || playwrightContract.playwrightCli.headed !== true) {
    throw new Error('The repository WebGPU Playwright contract is incompatible with this shootout');
  }
  return { manifest, playwrightContract, repetitions, resolution: options.resolution,
    sceneFiles, sourcePaths };
}

export async function startStaticServer(allowedFiles) {
  const allowed = new Set(allowedFiles.map(filePath => path.resolve(filePath)));
  const mimeTypes = new Map([
    ['.html', 'text/html; charset=utf-8'], ['.js', 'text/javascript; charset=utf-8'],
    ['.mjs', 'text/javascript; charset=utf-8'], ['.json', 'application/json; charset=utf-8'],
    ['.bin', 'application/octet-stream'], ['.png', 'image/png'], ['.jpg', 'image/jpeg'],
    ['.jpeg', 'image/jpeg'],
  ]);
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url, 'http://127.0.0.1');
      if (requestUrl.pathname === '/favicon.ico') return response.writeHead(204).end();
      const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, '');
      const filePath = path.resolve(repositoryRoot, relativePath);
      if (!inside(repositoryRoot, filePath) || !allowed.has(filePath)) {
        return response.writeHead(404).end('not found');
      }
      const value = await stat(filePath);
      if (!value.isFile()) return response.writeHead(404).end('not found');
      response.writeHead(200, {
        'Content-Type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404).end('not found');
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server;
}

export async function closeServer(server) {
  if (!server) return;
  await new Promise(resolve => server.close(resolve));
}

export async function waitForServer(url) {
  let lastError = null;
  for (let attempt = 0; attempt < 30; attempt++) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (response.ok) {
        await response.body?.cancel();
        return;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Shootout server did not become ready: ${lastError?.message || 'unknown error'}`,
    { cause: lastError });
}

function urlPath(filePath) {
  return path.relative(repositoryRoot, filePath).split(path.sep).map(encodeURIComponent).join('/');
}

async function runBrowser({ manifest, playwrightContract, repetitions, resolution, sceneFiles, sourcePaths }) {
  const npx = npxInvocation();
  if (npx.prerequisite && !await exists(npx.prerequisite)) {
    throw new Error(`npx prerequisite is missing: ${npx.prerequisite}`);
  }
  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-vfi-shootout-'));
  const session = `framegen-vfi-shootout-${process.pid}-${Date.now()}`;
  const cliPrefix = [...npx.prefix, '--yes', '--package', playwrightContract.playwrightCli.package,
    'playwright-cli'];
  let server = null;
  let cliVersion = null;
  let browserResult = null;
  let primaryError = null;
  try {
    const allowedFiles = [sourcePaths.harness, sourcePaths.runtime, sourcePaths.weights,
      sourcePaths.weightsManifest, ...sceneFiles.flatMap(scene => [scene.files.i0, scene.files.i1])];
    server = await startStaticServer(allowedFiles);
    const baseUrl = `http://127.0.0.1:${server.address().port}`;
    const harnessUrl = `${baseUrl}/${urlPath(sourcePaths.harness)}`;
    await waitForServer(harnessUrl);
    const entries = sceneFiles.map(scene => {
      const query = new URLSearchParams({
        res: String(resolution),
        i0: `${baseUrl}/${urlPath(scene.files.i0)}`,
        i1: `${baseUrl}/${urlPath(scene.files.i1)}`,
      });
      return { id: scene.id, url: `${harnessUrl}?${query}` };
    });
    cliVersion = await run(npx.command, [...cliPrefix, '--version'], {
      cwd: tempDirectory, timeout: 60000,
    });
    const openArgs = [...cliPrefix, `-s=${session}`, 'open', 'about:blank',
      '--browser', playwrightContract.playwrightCli.browser];
    if (playwrightContract.playwrightCli.headed) openArgs.push('--headed');
    await run(npx.command, openArgs, { cwd: tempDirectory, timeout: 60000 });
    const evaluationPath = path.join(tempDirectory, 'run-vfi-shootout.js');
    await writeFile(evaluationPath, `async (page) => {
      const diagnostics = { consoleErrors: [], consoleWarnings: [], pageErrors: [], requestFailures: [], httpErrors: [] };
      const onConsole = message => {
        const row = { type: message.type(), text: message.text() };
        if (message.type() === 'error') diagnostics.consoleErrors.push(row);
        if (message.type() === 'warning') diagnostics.consoleWarnings.push(row);
      };
      const onPageError = error => diagnostics.pageErrors.push({ message: error.message, stack: error.stack || null });
      const onRequestFailed = request => diagnostics.requestFailures.push({ url: request.url(), errorText: request.failure()?.errorText || null });
      const onResponse = response => { if (response.status() >= 400) diagnostics.httpErrors.push({ status: response.status(), url: response.url() }); };
      page.on('console', onConsole);
      page.on('pageerror', onPageError);
      page.on('requestfailed', onRequestFailed);
      page.on('response', onResponse);
      const records = [];
      let environment = null;
      let executionError = null;
      try {
        for (const entry of ${JSON.stringify(entries)}) {
          const pageMedianMsSamples = [];
          let imageData = null;
          let identity = null;
          for (let repetition = 0; repetition < ${repetitions}; repetition++) {
            await page.goto(entry.url, { waitUntil: 'domcontentloaded' });
            await page.waitForFunction(() => globalThis.__shootoutResult !== undefined, null,
              { timeout: ${manifest.measurement.timeoutMsPerPage} });
            const result = await page.evaluate(() => globalThis.__shootoutResult);
            if (result.error) throw new Error(entry.id + ': ' + result.error + '\\n' + (result.stack || ''));
            if (!Number.isFinite(result.ms) || result.ms < 0) throw new Error(entry.id + ': invalid timing');
            const currentIdentity = JSON.stringify({
              source: result.source,
              rung: result.rung,
              adapterIdentity: result.adapterIdentity,
              deviceFeatures: result.deviceFeatures,
              deviceLimits: result.deviceLimits,
            });
            if (identity !== null && identity !== currentIdentity) throw new Error(entry.id + ': environment changed between repetitions');
            identity = currentIdentity;
            pageMedianMsSamples.push(result.ms);
            if (repetition === 0) imageData = result.imageData;
          }
          records.push({ scene: entry.id, ...JSON.parse(identity), pageMedianMsSamples, imageData });
        }
        environment = await page.evaluate(() => ({
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          language: navigator.language,
          hardwareConcurrency: navigator.hardwareConcurrency,
          deviceMemoryGiB: navigator.deviceMemory ?? null,
        }));
      } catch (error) {
        executionError = { name: error.name || 'Error', message: error.message || String(error), stack: error.stack || null };
      } finally {
        page.off('console', onConsole);
        page.off('pageerror', onPageError);
        page.off('requestfailed', onRequestFailed);
        page.off('response', onResponse);
      }
      return { records, environment, diagnostics, executionError };
    }`, 'utf8');
    await run(process.execPath, ['--check', evaluationPath], {
      cwd: tempDirectory, timeout: 30000,
    });
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code',
      '--filename', evaluationPath], {
      cwd: tempDirectory,
      timeout: manifest.measurement.timeoutMsPerPage * sceneFiles.length * repetitions + 60000,
    });
    try { browserResult = JSON.parse(raw); }
    catch (cause) {
      throw new Error(`Playwright CLI did not return JSON:\n${raw.slice(0, 4000)}`, { cause });
    }
    if (browserResult.executionError) {
      throw new Error(`Shootout page failed: ${browserResult.executionError.message}\n${browserResult.executionError.stack || ''}`);
    }
    if (browserResult.diagnostics.pageErrors.length || browserResult.diagnostics.requestFailures.length
        || browserResult.diagnostics.httpErrors.length || browserResult.diagnostics.consoleErrors.length) {
      throw new Error(`Shootout browser diagnostics contain errors:\n${JSON.stringify(browserResult.diagnostics, null, 2)}`);
    }
  } catch (error) {
    primaryError = error;
  } finally {
    await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'], {
      cwd: tempDirectory, timeout: 30000,
    });
    await closeServer(server);
    await rm(tempDirectory, { recursive: true, force: true });
  }
  if (primaryError) throw primaryError;
  return { ...browserResult, cliVersion };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) { console.log(HELP); return; }
  const validated = await validateInputs(options);
  const outputPath = resolveOutputPath(options.output, options.resolution);
  if (await exists(outputPath)) throw new Error(`Output directory already exists: ${outputPath}`);
  const sourceMetadataStart = Object.fromEntries(await Promise.all(Object.entries(validated.sourcePaths)
    .map(async ([name, filePath]) => [name, await fileMetadata(filePath)])));
  const datasetMetadataStart = await collectDatasetMetadata(validated.sceneFiles);
  const [git, browserResult] = await Promise.all([
    gitMetadata(),
    runBrowser(validated),
  ]);
  if (browserResult.records.length !== validated.sceneFiles.length) {
    throw new Error('Shootout browser returned an incomplete scene set');
  }
  const deviceIdentity = validateStableDeviceEvidence(browserResult.records);
  for (let index = 0; index < validated.sceneFiles.length; index++) {
    const record = browserResult.records[index];
    if (record.scene !== validated.sceneFiles[index].id
        || !Array.isArray(record.pageMedianMsSamples)
        || record.pageMedianMsSamples.length !== validated.repetitions) {
      throw new Error('Shootout browser returned malformed or reordered scene evidence');
    }
  }
  const expectedRung = validated.manifest.inferenceRungs[String(options.resolution)];
  const outputRecords = [];
  for (const record of browserResult.records) {
    if (JSON.stringify(record.rung) !== JSON.stringify(expectedRung)) {
      throw new Error(`${record.scene}: expected rung ${expectedRung.join('x')}, got ${record.rung.join('x')}`);
    }
    if (!record.imageData?.startsWith('data:image/png;base64,')) {
      throw new Error(`${record.scene}: missing PNG output`);
    }
    const image = Buffer.from(record.imageData.slice(record.imageData.indexOf(',') + 1), 'base64');
    const relativeOutput = `${record.scene}/framegen-v7s.png`;
    outputRecords.push({
      scene: record.scene,
      source: record.source,
      rung: record.rung,
      adapterIdentity: record.adapterIdentity,
      deviceFeatures: record.deviceFeatures,
      deviceLimits: record.deviceLimits,
      pageMedianMsSamples: record.pageMedianMsSamples,
      pageMedianMsSummary: summarizeSamples(record.pageMedianMsSamples),
      output: relativeOutput,
      outputSha256: createHash('sha256').update(image).digest('hex'),
      image,
    });
  }
  const sourceMetadataEnd = Object.fromEntries(await Promise.all(Object.entries(validated.sourcePaths)
    .map(async ([name, filePath]) => [name, await fileMetadata(filePath)])));
  for (const name of Object.keys(sourceMetadataStart)) {
    if (sourceMetadataStart[name].sha256 !== sourceMetadataEnd[name].sha256) {
      throw new Error(`${name} changed during the shootout`);
    }
  }
  const datasetMetadataEnd = await collectDatasetMetadata(validated.sceneFiles);
  for (let sceneIndex = 0; sceneIndex < datasetMetadataStart.length; sceneIndex++) {
    for (const field of ['i0', 'gt', 'i1']) {
      if (datasetMetadataStart[sceneIndex].files[field].sha256
          !== datasetMetadataEnd[sceneIndex].files[field].sha256) {
        throw new Error(`${datasetMetadataStart[sceneIndex].id} ${field} changed during the shootout`);
      }
    }
  }
  const datasetMetadata = datasetMetadataStart.map((scene, sceneIndex) => ({
    ...scene,
    files: Object.fromEntries(Object.entries(scene.files).map(([field, value]) => [field, {
      ...value, sha256End: datasetMetadataEnd[sceneIndex].files[field].sha256,
    }])),
  }));
  const report = {
    schemaVersion: 1,
    kind: 'framegen-webgpu-vfi-shootout',
    createdAt: new Date().toISOString(),
    source: {
      git,
      files: Object.fromEntries(Object.entries(sourceMetadataStart).map(([name, value]) => [name, {
        ...value, sha256End: sourceMetadataEnd[name].sha256,
      }])),
      dataset: datasetMetadata,
    },
    host: {
      platform: process.platform,
      arch: process.arch,
      osRelease: os.release(),
      node: process.version,
      cpu: os.cpus()[0]?.model || null,
    },
    browser: {
      channel: validated.playwrightContract.playwrightCli.browser,
      headed: validated.playwrightContract.playwrightCli.headed,
      playwrightCliPackage: validated.playwrightContract.playwrightCli.package,
      playwrightCliVersion: browserResult.cliVersion,
      environment: browserResult.environment,
      ...deviceIdentity,
    },
    workload: {
      model: 'v7s',
      inferenceResolution: options.resolution,
      inferenceRung: expectedRung,
      timestep: 0.5,
      scenes: validated.sceneFiles.map(scene => scene.id),
    },
    measurement: {
      repetitionsPerScene: validated.repetitions,
      sampleUnit: `one page-reported median of ${validated.manifest.measurement.pageMeasuredPairCalls} synchronized prepPair+runT calls`,
      warmupPairCallsPerPage: validated.manifest.measurement.pageWarmupPairCalls,
      measuredPairCallsPerPage: validated.manifest.measurement.pageMeasuredPairCalls,
      synchronization: 'GPUQueue.onSubmittedWorkDone after every measured pair',
      comparativePerformanceClaimReady: false,
      comparativePerformanceClaimBlocker: 'This run contains one revision and no equivalent-condition baseline.',
    },
    records: outputRecords.map(({ image, ...record }) => record),
    diagnostics: browserResult.diagnostics,
    limitations: [
      'This exercises direct runtime midpoint output, not extension presentation, sharpness, cadence, dropped frames, or VRAM usage.',
      'Raw values are independent page medians because web/shootout.html exposes its median, not each inner timing sample.',
      'Performance comparisons require separate baseline and candidate runs on the same host, browser, adapter, inputs, and configuration.',
      'PyTorch reference timings are descriptive per method and are not cross-backend throughput comparators.',
      'Adapter identity, enabled features, and relevant limits are browser-exposed; the GPU driver version may be unavailable.',
    ],
  };
  await mkdir(path.dirname(outputPath), { recursive: true });
  const stagedOutput = await mkdtemp(path.join(path.dirname(outputPath), '.framegen-vfi-stage-'));
  try {
    for (const record of outputRecords) {
      const destination = path.join(stagedOutput, record.output);
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, record.image, { flag: 'wx' });
    }
    await writeFile(path.join(stagedOutput, 'run.json'), `${JSON.stringify(report, null, 2)}\n`, {
      encoding: 'utf8', flag: 'wx',
    });
    await rename(stagedOutput, outputPath);
  } catch (error) {
    await rm(stagedOutput, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({ output: outputPath, scenes: outputRecords.length,
    resolution: options.resolution, adapter: report.browser.adapterIdentity }, null, 2));
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === scriptPath;
if (invokedDirectly) {
  main().catch(error => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}
