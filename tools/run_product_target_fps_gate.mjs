import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { access, copyFile, cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { validateProductTargetFpsReport } from './product_target_fps_acceptance.mjs';

const execFileAsync = promisify(execFile);
const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(scriptPath), '..');
const manifestPath = path.join(repositoryRoot, 'tools', 'product_target_fps_gate_manifest.json');
const outputRoot = path.join(repositoryRoot, 'output', 'product-target-fps');

const TARGET_CASES = Object.freeze([
  Object.freeze({ id: 'source10-target50', sourceFps: 10, targetFps: 50, resolution: 480,
    expectedEffectiveFps: 50, expectedClampReason: null }),
  Object.freeze({ id: 'source15-target50', sourceFps: 15, targetFps: 50, resolution: 480,
    expectedEffectiveFps: 50, expectedClampReason: null }),
  Object.freeze({ id: 'source24-target50', sourceFps: 24, targetFps: 50, resolution: 480,
    expectedEffectiveFps: 50, expectedClampReason: null }),
  Object.freeze({ id: 'source24-request40-floor48', sourceFps: 24, targetFps: 40, resolution: 480,
    expectedEffectiveFps: 48, expectedClampReason: 'minimum' }),
  Object.freeze({ id: 'source60-target120', sourceFps: 60, targetFps: 120, resolution: 480,
    expectedEffectiveFps: 120, expectedClampReason: null }),
  Object.freeze({ id: 'source60-target144', sourceFps: 60, targetFps: 144, resolution: 480,
    expectedEffectiveFps: 144, expectedClampReason: null }),
  Object.freeze({ id: 'source60-request300-display-cap', sourceFps: 60, targetFps: 300, resolution: 480,
    expectedEffectiveFps: 232.8, expectedClampReason: 'display' }),
]);

export const PRODUCT_TARGET_FPS_RUNNER_CONTRACT = Object.freeze({
  schemaVersion: 2,
  gateId: 'framecast-v7s-product-target-fps-v2',
  profile: 'v7s-480p-product-target-fps-v2',
  cases: TARGET_CASES,
  playwrightPackage: '@playwright/cli@0.1.17',
  browser: 'chromium',
  browserDistribution: 'chrome-for-testing',
  headed: true,
  persistentContext: true,
  warmupMsPerCase: 4000,
  measureMsPerCase: 12000,
  expectedDisplayHz: 240,
  sourceFiles: Object.freeze({
    contentScript: 'extension/content.js',
    cadenceScript: 'extension/cadence.js',
    extensionManifest: 'extension/manifest.json',
    fixture: 'web/product_target_fps_fixture.html',
    runtime: 'web/rt/rt.js',
    weights: 'extension/assets/rt_v7s.bin',
    weightsManifest: 'extension/assets/rt_v7s.json',
    runner: 'tools/run_product_target_fps_gate.mjs',
    gateManifest: 'tools/product_target_fps_gate_manifest.json',
    validator: 'tools/product_target_fps_acceptance.mjs',
  }),
});

const HELP = `Usage: node tools/run_product_target_fps_gate.mjs [options]

Runs the generic factor=target case matrix through the real unpacked
extension/WebGPU product path in one persistent headed Chrome-for-Testing
context. Evidence is written once under output/product-target-fps.

  --output <path>  unique JSON path inside output/product-target-fps
  --help
`;

export function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (token === '--help') { options.help = true; continue; }
    if (token !== '--output') throw new Error(`Unknown option: ${token}`);
    const value = argv[++index];
    if (value === undefined) throw new Error(`Missing value for ${token}`);
    options.output = value;
  }
  return options;
}

function timestampLabel() {
  return new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
}

function inside(parent, candidate) {
  return candidate.startsWith(`${parent}${path.sep}`);
}

export function resolveOutputPath(requestedPath, label = timestampLabel()) {
  const candidate = path.resolve(repositoryRoot,
    requestedPath || path.join(outputRoot, `product-target-fps-${label}.json`));
  if (!inside(outputRoot, candidate) || path.extname(candidate).toLowerCase() !== '.json') {
    throw new Error('Output must be a .json file inside output/product-target-fps');
  }
  return candidate;
}

function requireNumber(value, name, { integer = false, min = 0 } = {}) {
  if (!Number.isFinite(value) || value < min || (integer && !Number.isInteger(value))) {
    throw new Error(`${name} is invalid`);
  }
}

export function validateManifest(manifest) {
  if (manifest.schemaVersion !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.schemaVersion
      || manifest.profile !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.profile) {
    throw new Error('Unexpected product target-FPS manifest schema/profile');
  }
  const playwright = manifest.playwrightCli;
  if (playwright?.package !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.playwrightPackage
      || playwright?.browser !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.browser
      || playwright?.browserDistribution !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.browserDistribution
      || playwright?.headed !== true || playwright?.persistentContext !== true) {
    throw new Error('Target-FPS gate requires persistent headed Chrome-for-Testing');
  }
  if (manifest.fixture?.width !== 1280 || manifest.fixture?.height !== 720
      || manifest.fixture?.pattern !== 'moving-primitives-v1'
      || JSON.stringify(manifest.fixture?.supportedSourceFps) !== JSON.stringify([10, 15, 24, 60])) {
    throw new Error('Target-FPS fixture must be deterministic 1280x720 at 10/15/24/60 FPS');
  }
  const measurement = manifest.measurement;
  if (JSON.stringify(measurement?.cases) !== JSON.stringify(PRODUCT_TARGET_FPS_RUNNER_CONTRACT.cases)
      || measurement?.warmupMsPerCase !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.warmupMsPerCase
      || measurement?.measureMsPerCase !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.measureMsPerCase
      || measurement?.expectedDisplayHz !== PRODUCT_TARGET_FPS_RUNNER_CONTRACT.expectedDisplayHz) {
    throw new Error('Target-FPS measurement contract mismatch');
  }
  requireNumber(measurement.timeoutMsPerCase, 'measurement.timeoutMsPerCase', { integer: true, min: 30000 });
  for (const [name, value] of Object.entries(manifest.acceptance || {})) {
    if (name === 'requiredDeviceFeatures') continue;
    requireNumber(value, `acceptance.${name}`);
  }
  if (!Array.isArray(manifest.acceptance?.requiredDeviceFeatures)
      || !manifest.acceptance.requiredDeviceFeatures.includes('shader-f16')
      || !manifest.acceptance.requiredDeviceFeatures.includes('timestamp-query')) {
    throw new Error('Required WebGPU features are not frozen');
  }
}

async function exists(filePath) {
  try { await access(filePath); return true; } catch { return false; }
}

function resolveRepositoryPath(relativePath, name) {
  if (typeof relativePath !== 'string' || !relativePath) throw new Error(`${name} path is missing`);
  const absolute = path.resolve(repositoryRoot, relativePath);
  if (absolute !== repositoryRoot && !inside(repositoryRoot, absolute)) {
    throw new Error(`${name} escapes the repository`);
  }
  return absolute;
}

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function metadata(filePath) {
  const value = await stat(filePath);
  return { path: path.relative(repositoryRoot, filePath).replaceAll('\\', '/'), sizeBytes: value.size };
}

async function run(command, args, options = {}) {
  try {
    const result = await execFileAsync(command, args, {
      cwd: options.cwd || repositoryRoot,
      encoding: 'utf8',
      maxBuffer: 32 * 1024 * 1024,
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

async function startStaticServer(root) {
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, '');
      const filePath = path.resolve(root, relativePath);
      if (filePath !== root && !inside(root, filePath)) return response.writeHead(403).end('forbidden');
      const body = await readFile(filePath);
      const extension = path.extname(filePath);
      const mime = extension === '.html' ? 'text/html; charset=utf-8'
        : extension === '.js' || extension === '.mjs' ? 'text/javascript; charset=utf-8'
          : extension === '.json' ? 'application/json; charset=utf-8' : 'application/octet-stream';
      response.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store' });
      response.end(body);
    } catch { response.writeHead(404).end('not found'); }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server;
}

async function closeServer(server) {
  if (!server) return;
  await new Promise(resolve => server.close(resolve));
}

function serializeError(error) {
  if (!error) return null;
  return { name: error.name || 'Error', message: error.message || String(error), stack: error.stack || null };
}

async function stageExtension(manifest, tempDirectory) {
  const sourceDirectory = resolveRepositoryPath(manifest.extension.sourceDirectory, 'extension.sourceDirectory');
  const stageDirectory = path.join(tempDirectory, 'unpacked-extension');
  await cp(sourceDirectory, stageDirectory, { recursive: true,
    filter: source => !path.basename(source).startsWith('_') });
  const runtimeDirectory = path.join(stageDirectory, 'rt');
  const assetDirectory = path.join(stageDirectory, 'assets');
  await mkdir(runtimeDirectory, { recursive: true });
  await mkdir(assetDirectory, { recursive: true });
  const runtime = resolveRepositoryPath(manifest.extension.runtime, 'extension.runtime');
  const weights = resolveRepositoryPath(manifest.extension.weights, 'extension.weights');
  const weightsManifest = resolveRepositoryPath(manifest.extension.weightsManifest, 'extension.weightsManifest');
  await copyFile(runtime, path.join(runtimeDirectory, 'rt.js'));
  await copyFile(weights, path.join(assetDirectory, `${manifest.extension.modelAssetStem}.bin`));
  await copyFile(weightsManifest, path.join(assetDirectory, `${manifest.extension.modelAssetStem}.json`));
  return {
    directory: stageDirectory,
    hashes: {
      contentScript: await sha256(path.join(stageDirectory, 'content.js')),
      cadenceScript: await sha256(path.join(stageDirectory, 'cadence.js')),
      extensionManifest: await sha256(path.join(stageDirectory, 'manifest.json')),
      runtime: await sha256(path.join(runtimeDirectory, 'rt.js')),
      weights: await sha256(path.join(assetDirectory, `${manifest.extension.modelAssetStem}.bin`)),
      weightsManifest: await sha256(path.join(assetDirectory, `${manifest.extension.modelAssetStem}.json`)),
    },
  };
}

async function gitMetadata() {
  const safe = ['-c', `safe.directory=${repositoryRoot}`];
  const [commit, status] = await Promise.all([
    runOptional('git', [...safe, 'rev-parse', 'HEAD']),
    runOptional('git', [...safe, 'status', '--porcelain=v1', '--untracked-files=all']),
  ]);
  return {
    commit: commit.available ? commit.stdout : null,
    dirty: status.available ? status.stdout.length > 0 : null,
    statusPorcelain: status.available ? status.stdout.split(/\r?\n/).filter(Boolean) : null,
    errors: [commit, status].filter(value => !value.available).map(value => value.error),
  };
}

async function main() {
  const cli = parseArguments(process.argv.slice(2));
  if (cli.help) { console.log(HELP); return; }
  const outputPath = resolveOutputPath(cli.output);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  validateManifest(manifest);
  const sourcePaths = Object.fromEntries(Object.entries(PRODUCT_TARGET_FPS_RUNNER_CONTRACT.sourceFiles)
    .map(([name, relativePath]) => [name, resolveRepositoryPath(relativePath, name)]));
  for (const [name, filePath] of Object.entries(sourcePaths)) {
    if (!await exists(filePath)) throw new Error(`Required ${name} is missing: ${filePath}`);
  }
  const npx = npxInvocation();
  if (npx.prerequisite && !await exists(npx.prerequisite)) {
    throw new Error(`npx prerequisite is missing: ${npx.prerequisite}`);
  }

  const [hashPairs, metadataPairs, git] = await Promise.all([
    Promise.all(Object.entries(sourcePaths).map(async ([name, filePath]) => [name, await sha256(filePath)])),
    Promise.all(Object.entries(sourcePaths).map(async ([name, filePath]) => [name, await metadata(filePath)])),
    gitMetadata(),
  ]);
  const hashesStart = Object.fromEntries(hashPairs);
  const sourceMetadata = Object.fromEntries(metadataPairs);
  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-product-target-fps-'));
  const profileDirectory = path.join(tempDirectory, 'profile');
  await mkdir(profileDirectory, { recursive: true });
  const staged = await stageExtension(manifest, tempDirectory);
  const expectedStageHashes = Object.fromEntries(Object.keys(staged.hashes)
    .map(name => [name, hashesStart[name]]));
  if (JSON.stringify(staged.hashes) !== JSON.stringify(expectedStageHashes)) {
    throw new Error('Temporary unpacked extension differs from source files');
  }

  const session = `framegen-product-target-fps-${process.pid}-${Date.now()}`;
  const cliPrefix = [...npx.prefix, '--yes', '--package', manifest.playwrightCli.package, 'playwright-cli'];
  let server = null;
  let cliVersion = null;
  let browserResult = null;
  let primaryError = null;
  try {
    server = await startStaticServer(repositoryRoot);
    const address = server.address();
    const token = randomBytes(24).toString('hex');
    const fixtureRelative = path.relative(repositoryRoot, sourcePaths.fixture).split(path.sep).join('/');
    const entries = manifest.measurement.cases.map(targetCase => {
      const query = new URLSearchParams({
        framegenProductBench: '1',
        framegenBenchToken: token,
        caseId: targetCase.id,
        factor: 'target',
        sourceFps: String(targetCase.sourceFps),
        targetFps: String(targetCase.targetFps),
        resolution: String(targetCase.resolution),
        warmupMs: String(manifest.measurement.warmupMsPerCase),
        measureMs: String(manifest.measurement.measureMsPerCase),
      });
      return [targetCase.id, `http://127.0.0.1:${address.port}/${fixtureRelative}?${query}`];
    });
    const configPath = path.join(tempDirectory, 'playwright-cli.config.json');
    await writeFile(configPath, `${JSON.stringify({
      browser: {
        browserName: manifest.playwrightCli.browser,
        isolated: false,
        userDataDir: profileDirectory,
        launchOptions: {
          channel: manifest.playwrightCli.browser,
          headless: false,
          args: [
            `--disable-extensions-except=${staged.directory}`,
            `--load-extension=${staged.directory}`,
            '--no-first-run',
            '--disable-default-apps',
          ],
        },
        contextOptions: { viewport: { width: 1400, height: 820 }, deviceScaleFactor: 1 },
      },
      outputMode: 'stdout',
      timeouts: { action: 15000, navigation: 30000 },
    }, null, 2)}\n`, 'utf8');
    const evaluationPath = path.join(tempDirectory, 'run-product-target-fps.js');
    await writeFile(evaluationPath, `async (page) => {
      const diagnostics = { consoleMessages: [], consoleErrors: [], consoleWarnings: [], framegenLogs: [],
        pageErrors: [], requestFailures: [], httpErrors: [] };
      const onConsole = message => {
        const row = { type: message.type(), text: message.text() };
        diagnostics.consoleMessages.push(row);
        if (message.type() === 'error') diagnostics.consoleErrors.push(row);
        if (message.type() === 'warning') diagnostics.consoleWarnings.push(row);
        if (message.text().startsWith('[framegen]')) diagnostics.framegenLogs.push(row);
      };
      const onPageError = error => diagnostics.pageErrors.push({ message: error.message, stack: error.stack || null });
      const onRequestFailed = request => diagnostics.requestFailures.push({ url: request.url(),
        errorText: request.failure()?.errorText || null });
      const onResponse = response => { if (response.status() >= 400) diagnostics.httpErrors.push({
        status: response.status(), url: response.url() }); };
      page.on('console', onConsole); page.on('pageerror', onPageError);
      page.on('requestfailed', onRequestFailed); page.on('response', onResponse);
      const measurements = {};
      const controls = {};
      const measureRafControl = durationMs => page.evaluate(duration => new Promise(resolve => {
        const intervalsMs = [];
        let startedAt = null;
        let previousAt = null;
        const tick = now => {
          if (startedAt === null) startedAt = now;
          if (previousAt !== null) intervalsMs.push(now - previousAt);
          previousAt = now;
          if (now - startedAt >= duration) {
            const elapsedMs = now - startedAt;
            const ordered = [...intervalsMs].sort((left, right) => left - right);
            const totalMs = intervalsMs.reduce((sum, value) => sum + value, 0);
            resolve({
              durationMs: elapsedMs,
              callbacks: intervalsMs.length + 1,
              observedHz: totalMs > 0 ? 1000 * intervalsMs.length / totalMs : null,
              p95Ms: ordered[Math.max(0, Math.ceil(ordered.length * 0.95) - 1)] || null,
              maxMs: ordered.length ? ordered[ordered.length - 1] : null,
              intervalsMs,
            });
            return;
          }
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }), durationMs);
      let environment = null;
      try {
        for (const [caseId, url] of ${JSON.stringify(entries)}) {
          if (caseId === 'source60-request300-display-cap') {
            await page.goto('about:blank', { waitUntil: 'load' });
            controls[caseId] = await measureRafControl(3000);
          }
          await page.goto(url, { waitUntil: 'load' });
          if (!environment) environment = await page.evaluate(() => ({
            userAgent: navigator.userAgent, platform: navigator.platform,
            hardwareConcurrency: navigator.hardwareConcurrency, deviceMemoryGiB: navigator.deviceMemory || null,
            devicePixelRatio, visibilityState: document.visibilityState,
            screen: { width: screen.width, height: screen.height, availWidth: screen.availWidth,
              availHeight: screen.availHeight, colorDepth: screen.colorDepth },
          }));
          await page.waitForFunction(() => globalThis.__productTargetFpsResult !== undefined, null,
            { timeout: ${manifest.measurement.timeoutMsPerCase} });
          measurements[caseId] = await page.evaluate(() => globalThis.__productTargetFpsResult);
        }
        return { measurements, controls, environment, diagnostics, executionError: null };
      } catch (error) {
        return { measurements, controls, environment, diagnostics,
          executionError: { name: error.name, message: error.message, stack: error.stack || null } };
      } finally {
        page.off('console', onConsole); page.off('pageerror', onPageError);
        page.off('requestfailed', onRequestFailed); page.off('response', onResponse);
      }
    }`, 'utf8');

    cliVersion = await run(npx.command, [...cliPrefix, '--version'], { cwd: tempDirectory, timeout: 60000 });
    await run(npx.command, [...cliPrefix, `-s=${session}`, `--config=${configPath}`,
      'open', 'about:blank', '--headed', '--persistent'], { cwd: tempDirectory, timeout: 60000 });
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code',
      '--filename', evaluationPath], {
      cwd: tempDirectory,
      timeout: manifest.measurement.timeoutMsPerCase * manifest.measurement.cases.length + 30000,
    });
    try { browserResult = JSON.parse(raw); }
    catch (cause) { throw new Error(`Playwright CLI did not return JSON:\n${raw.slice(0, 4000)}`, { cause }); }
  } catch (error) {
    primaryError = serializeError(error);
  } finally {
    await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'], { cwd: tempDirectory, timeout: 30000 });
    await closeServer(server);
    await rm(tempDirectory, { recursive: true, force: true });
  }

  const hashesEnd = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await sha256(filePath)])));
  const sourceFiles = Object.fromEntries(Object.entries(sourceMetadata).map(([name, value]) => [name, {
    ...value, sha256Start: hashesStart[name], sha256End: hashesEnd[name],
  }]));
  const diagnostics = browserResult?.diagnostics || {
    consoleMessages: [], consoleErrors: [], consoleWarnings: [], framegenLogs: [],
    pageErrors: [], requestFailures: [], httpErrors: [],
  };
  const executionError = primaryError || browserResult?.executionError || null;
  const report = {
    schemaVersion: 2,
    gateId: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.gateId,
    profile: manifest.profile,
    createdAt: new Date().toISOString(),
    source: { git, files: sourceFiles },
    host: { platform: process.platform, arch: process.arch, osRelease: os.release(), node: process.version,
      cpu: os.cpus()[0]?.model || null },
    browser: {
      channel: manifest.playwrightCli.browser,
      distribution: manifest.playwrightCli.browserDistribution,
      headed: true,
      persistentContext: true,
      unpackedExtension: true,
      playwrightCliPackage: manifest.playwrightCli.package,
      playwrightCliVersion: cliVersion,
      environment: browserResult?.environment || null,
      controls: browserResult?.controls || {},
      extensionStageHashes: staged.hashes,
    },
    workload: { model: 'v7s', resolution: 480, width: manifest.fixture.width,
      height: manifest.fixture.height, supportedSourceFps: manifest.fixture.supportedSourceFps,
      expectedDisplayHz: manifest.measurement.expectedDisplayHz, cases: manifest.measurement.cases },
    conditions: { warmupMsPerCase: manifest.measurement.warmupMsPerCase,
      measureMsPerCase: manifest.measurement.measureMsPerCase,
      presentationMetric: 'gpu-canvas-submit-from-rAF-pump', acceptance: manifest.acceptance },
    measurements: browserResult?.measurements || {},
    diagnostics,
    executionError,
    limitations: {
      compositorAcknowledgement: false,
      note: 'presented counts successful product GPUCanvas submissions; Chromium exposes no per-canvas physical scan-out acknowledgement.',
    },
    validation: null,
    passed: false,
  };
  const expected = {
    gateId: PRODUCT_TARGET_FPS_RUNNER_CONTRACT.gateId,
    profile: manifest.profile,
    cases: manifest.measurement.cases,
    fixtureName: manifest.fixture.path,
    pattern: manifest.fixture.pattern,
    width: manifest.fixture.width,
    height: manifest.fixture.height,
    supportedSourceFps: manifest.fixture.supportedSourceFps,
    expectedDisplayHz: manifest.measurement.expectedDisplayHz,
    warmupMs: manifest.measurement.warmupMsPerCase,
    measureMs: manifest.measurement.measureMsPerCase,
    browserChannel: manifest.playwrightCli.browser,
    browserDistribution: manifest.playwrightCli.browserDistribution,
    hashes: hashesStart,
    stagedHashes: expectedStageHashes,
    ...manifest.acceptance,
  };
  let structuralError = null;
  try {
    report.validation = validateProductTargetFpsReport(report, expected);
    report.passed = report.validation.passed;
  } catch (error) {
    structuralError = error;
    report.validation = { passed: false, failures: ['report structurally invalid'],
      structuralError: error.stack || error.message };
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
  console.log(JSON.stringify({ output: outputPath, passed: report.passed,
    cases: report.validation?.cases || null, failures: report.validation?.failures || null }, null, 2));
  if (structuralError) {
    throw new Error(`Product target-FPS report is structurally invalid; red report: ${outputPath}`, {
      cause: structuralError,
    });
  }
  if (!report.passed) throw new Error(`Product target-FPS gate failed; red report preserved at ${outputPath}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch(error => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}
