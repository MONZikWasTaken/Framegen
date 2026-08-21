import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { access, copyFile, cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { validateProductHfrReport } from './product_hfr_acceptance.mjs';

const execFileAsync = promisify(execFile);
const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(scriptPath), '..');
const defaultManifestPath = path.join(repositoryRoot, 'tools', 'product_hfr_gate_manifest.json');

const HELP = `Usage: node tools/run_product_hfr_gate.mjs [options]

Runs the opt-in product extension path in one persistent headed Playwright Chromium
context. Each factor gets 5 seconds of warmup and 30 seconds of measurement.

  --manifest <path>  gate manifest (default: tools/product_hfr_gate_manifest.json)
  --output <path>    unique result JSON path
  --help
`;

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (token === '--help') { options.help = true; continue; }
    if (token !== '--manifest' && token !== '--output') throw new Error(`Unknown option: ${token}`);
    const value = argv[++index];
    if (value === undefined) throw new Error(`Missing value for ${token}`);
    options[token.slice(2)] = value;
  }
  return options;
}

async function exists(filePath) {
  try { await access(filePath); return true; } catch { return false; }
}

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function metadata(filePath) {
  const value = await stat(filePath);
  return {
    path: path.relative(repositoryRoot, filePath).replaceAll('\\', '/'),
    sizeBytes: value.size,
  };
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
    const npxScript = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npx-cli.js');
    return { command: process.execPath, prefix: [npxScript] };
  }
  return { command: 'npx', prefix: [] };
}

async function startStaticServer(root) {
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, '');
      const filePath = path.resolve(root, relativePath);
      if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) {
        response.writeHead(403).end('forbidden');
        return;
      }
      const body = await readFile(filePath);
      const mime = path.extname(filePath) === '.html' ? 'text/html; charset=utf-8'
        : path.extname(filePath) === '.js' || path.extname(filePath) === '.mjs'
          ? 'text/javascript; charset=utf-8' : 'application/octet-stream';
      response.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store' });
      response.end(body);
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

async function closeServer(server) {
  await new Promise(resolve => server.close(resolve));
}

function requireNumber(value, name, { integer = false, min = 0 } = {}) {
  if (!Number.isFinite(value) || value < min || (integer && !Number.isInteger(value))) {
    throw new Error(`${name} is invalid`);
  }
}

function validateManifest(manifest) {
  if (manifest.schemaVersion !== 1 || manifest.profile !== 'v7s-720p-product-hfr') {
    throw new Error('Unexpected product HFR manifest schema/profile');
  }
  if (manifest.playwrightCli?.browserChannel !== 'chromium' || manifest.playwrightCli?.headed !== true
      || manifest.playwrightCli?.persistentContext !== true) {
    throw new Error('Product HFR gate requires persistent headed Playwright Chromium');
  }
  if (manifest.fixture?.width !== 1280 || manifest.fixture?.height !== 720
      || manifest.fixture?.sourceFps !== 60 || manifest.fixture?.pattern !== 'moving-primitives-v1') {
    throw new Error('Product HFR fixture must be deterministic 1280x720@60');
  }
  const factors = manifest.measurement?.factors;
  if (JSON.stringify(factors) !== JSON.stringify([3, 4])
      || manifest.measurement?.warmupMsPerFactor !== 5000
      || manifest.measurement?.measureMsPerFactor !== 30000) {
    throw new Error('Product HFR measurement must be x3/x4 with 5s warmup and 30s measurement');
  }
  requireNumber(manifest.measurement.timeoutMs, 'measurement.timeoutMs', { integer: true, min: 45000 });
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

function resolveRepositoryPath(relativePath, name) {
  if (typeof relativePath !== 'string' || !relativePath) throw new Error(`${name} path is missing`);
  const absolute = path.resolve(repositoryRoot, relativePath);
  if (absolute !== repositoryRoot && !absolute.startsWith(`${repositoryRoot}${path.sep}`)) {
    throw new Error(`${name} escapes the repository`);
  }
  return absolute;
}

async function stageExtension(manifest, tempDirectory) {
  const sourceDirectory = resolveRepositoryPath(manifest.extension.sourceDirectory, 'extension.sourceDirectory');
  const stageDirectory = path.join(tempDirectory, 'unpacked-extension');
  await cp(sourceDirectory, stageDirectory, {
    recursive: true,
    filter: source => !path.basename(source).startsWith('_'),
  });
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

function timestampLabel() {
  return new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
}

async function main() {
  const cli = parseArguments(process.argv.slice(2));
  if (cli.help) { console.log(HELP); return; }
  const manifestPath = path.resolve(repositoryRoot, cli.manifest || defaultManifestPath);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  validateManifest(manifest);

  const sourcePaths = {
    contentScript: resolveRepositoryPath(manifest.extension.contentScript, 'extension.contentScript'),
    cadenceScript: resolveRepositoryPath(manifest.extension.cadenceScript, 'extension.cadenceScript'),
    extensionManifest: resolveRepositoryPath(manifest.extension.manifest, 'extension.manifest'),
    fixture: resolveRepositoryPath(manifest.fixture.path, 'fixture.path'),
    runtime: resolveRepositoryPath(manifest.extension.runtime, 'extension.runtime'),
    weights: resolveRepositoryPath(manifest.extension.weights, 'extension.weights'),
    weightsManifest: resolveRepositoryPath(manifest.extension.weightsManifest, 'extension.weightsManifest'),
    runner: scriptPath,
    gateManifest: manifestPath,
    validator: fileURLToPath(new URL('./product_hfr_acceptance.mjs', import.meta.url)),
  };
  for (const [name, filePath] of Object.entries(sourcePaths)) {
    if (!await exists(filePath)) throw new Error(`Required ${name} is missing: ${filePath}`);
  }
  const hashesStart = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await sha256(filePath)])));
  const sourceInfo = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await metadata(filePath)])));
  const gitCommit = await run('git', ['rev-parse', 'HEAD']);
  const gitStatus = await run('git', ['status', '--porcelain=v1', '--untracked-files=all']);

  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-product-hfr-'));
  const profileDirectory = path.join(tempDirectory, 'profile');
  await mkdir(profileDirectory, { recursive: true });
  const stagedExtension = await stageExtension(manifest, tempDirectory);
  const expectedStageHashes = {
    contentScript: hashesStart.contentScript,
    cadenceScript: hashesStart.cadenceScript,
    extensionManifest: hashesStart.extensionManifest,
    runtime: hashesStart.runtime,
    weights: hashesStart.weights,
    weightsManifest: hashesStart.weightsManifest,
  };
  if (JSON.stringify(stagedExtension.hashes) !== JSON.stringify(expectedStageHashes)) {
    throw new Error('Temporary unpacked extension differs from source files');
  }

  const server = await startStaticServer(repositoryRoot);
  const address = server.address();
  const token = randomBytes(24).toString('hex');
  const relativeFixture = path.relative(repositoryRoot, sourcePaths.fixture).split(path.sep).join('/');
  const urls = Object.fromEntries(manifest.measurement.factors.map(factor => {
    const query = new URLSearchParams({
      framegenProductBench: '1',
      framegenBenchToken: token,
      factor: String(factor),
      warmupMs: String(manifest.measurement.warmupMsPerFactor),
      measureMs: String(manifest.measurement.measureMsPerFactor),
    });
    return [factor, `http://127.0.0.1:${address.port}/${relativeFixture}?${query}`];
  }));
  const playwrightConfigPath = path.join(tempDirectory, 'playwright-cli.config.json');
  await writeFile(playwrightConfigPath, `${JSON.stringify({
    browser: {
      browserName: 'chromium',
      isolated: false,
      userDataDir: profileDirectory,
      launchOptions: {
        channel: manifest.playwrightCli.browserChannel,
        headless: false,
        args: [
          `--disable-extensions-except=${stagedExtension.directory}`,
          `--load-extension=${stagedExtension.directory}`,
          '--no-first-run',
          '--disable-default-apps',
        ],
      },
      contextOptions: { viewport: { width: 1400, height: 820 }, deviceScaleFactor: 1 },
    },
    outputMode: 'stdout',
    timeouts: { action: 15000, navigation: 30000 },
  }, null, 2)}\n`, 'utf8');

  const npx = npxInvocation();
  const cliPrefix = [...npx.prefix, '--yes', '--package', manifest.playwrightCli.package, 'playwright-cli'];
  const session = `framegen-product-hfr-${process.pid}-${Date.now()}`;
  const evaluationPath = path.join(tempDirectory, 'run-product-hfr.js');
  await writeFile(evaluationPath, `async (page) => {
    const consoleErrors = [];
    const consoleWarnings = [];
    const framegenLogs = [];
    const pageErrors = [];
    const requestFailures = [];
    const onConsole = message => {
      const row = { type: message.type(), text: message.text() };
      if (message.type() === 'error') consoleErrors.push(row);
      else if (message.type() === 'warning') consoleWarnings.push(row);
      if (message.text().startsWith('[framegen]')) framegenLogs.push(row);
    };
    const onPageError = error => pageErrors.push({ message: error.message, stack: error.stack || null });
    const onRequestFailed = request => requestFailures.push({ url: request.url(), errorText: request.failure()?.errorText || null });
    page.on('console', onConsole);
    page.on('pageerror', onPageError);
    page.on('requestfailed', onRequestFailed);
    const measurements = {};
    let environment = null;
    try {
      const entries = ${JSON.stringify(manifest.measurement.factors.map(factor => [factor, urls[factor]]))};
      for (const [factor, url] of entries) {
        await page.goto(url, { waitUntil: 'load' });
        if (!environment) environment = await page.evaluate(() => ({
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          hardwareConcurrency: navigator.hardwareConcurrency,
          deviceMemoryGiB: navigator.deviceMemory || null,
          devicePixelRatio,
          screen: { width: screen.width, height: screen.height, availWidth: screen.availWidth,
            availHeight: screen.availHeight, colorDepth: screen.colorDepth },
        }));
        await page.waitForFunction(() => globalThis.__productBenchResult !== undefined, null,
          { timeout: ${manifest.measurement.timeoutMs} });
        measurements['x' + factor] = await page.evaluate(() => globalThis.__productBenchResult);
      }
      return { measurements, environment,
        diagnostics: { consoleErrors, consoleWarnings, framegenLogs, pageErrors, requestFailures } };
    } finally {
      page.off('console', onConsole);
      page.off('pageerror', onPageError);
      page.off('requestfailed', onRequestFailed);
    }
  }`, 'utf8');

  let cliVersion = null;
  let browserResult = null;
  let primaryError = null;
  try {
    cliVersion = await run(npx.command, [...cliPrefix, '--version'], { cwd: tempDirectory, timeout: 60000 });
    await run(npx.command, [...cliPrefix, `-s=${session}`, `--config=${playwrightConfigPath}`,
      'open', 'about:blank', '--headed', '--persistent'], { cwd: tempDirectory, timeout: 60000 });
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code',
      '--filename', evaluationPath], {
      cwd: tempDirectory,
      timeout: manifest.measurement.timeoutMs * manifest.measurement.factors.length + 30000,
    });
    try { browserResult = JSON.parse(raw); }
    catch (cause) { throw new Error(`Playwright CLI did not return JSON:\n${raw.slice(0, 4000)}`, { cause }); }
  } catch (error) {
    primaryError = error;
  } finally {
    await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'], { cwd: tempDirectory, timeout: 30000 });
    await closeServer(server);
  }
  if (primaryError) {
    await rm(tempDirectory, { recursive: true, force: true });
    throw primaryError;
  }

  const hashesEnd = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await sha256(filePath)])));
  const sourceFiles = Object.fromEntries(Object.entries(sourceInfo).map(([name, value]) => [name, {
    ...value,
    sha256Start: hashesStart[name],
    sha256End: hashesEnd[name],
  }]));
  const report = {
    schemaVersion: 1,
    profile: manifest.profile,
    createdAt: new Date().toISOString(),
    source: {
      git: { commit: gitCommit, dirty: gitStatus.length > 0, statusPorcelain: gitStatus.split(/\r?\n/).filter(Boolean) },
      files: sourceFiles,
    },
    browser: {
      channel: manifest.playwrightCli.browserChannel,
      headed: true,
      persistentContext: true,
      unpackedExtension: true,
      playwrightCliPackage: manifest.playwrightCli.package,
      playwrightCliVersion: cliVersion,
      environment: browserResult.environment,
      extensionStageHashes: stagedExtension.hashes,
    },
    workload: { model: 'v7s', resolution: 720, width: manifest.fixture.width,
      height: manifest.fixture.height, sourceFps: manifest.fixture.sourceFps,
      factors: manifest.measurement.factors },
    conditions: {
      warmupMsPerFactor: manifest.measurement.warmupMsPerFactor,
      measureMsPerFactor: manifest.measurement.measureMsPerFactor,
      presentationMetric: 'gpu-canvas-submit-from-rAF-pump',
      acceptance: manifest.acceptance,
    },
    measurements: browserResult.measurements,
    diagnostics: browserResult.diagnostics,
    limitations: {
      compositorAcknowledgement: false,
      stableChromeAutomatedSideload: false,
      note: 'presented counts successful product pump GPUCanvas submits; Chromium exposes no per-canvas compositor scan-out acknowledgement. Branded Chrome removed automated sideload flags.',
    },
    validation: null,
    passed: false,
  };
  const expected = {
    profile: manifest.profile,
    factors: manifest.measurement.factors,
    fixtureName: manifest.fixture.path,
    pattern: manifest.fixture.pattern,
    width: manifest.fixture.width,
    height: manifest.fixture.height,
    sourceFps: manifest.fixture.sourceFps,
    warmupMs: manifest.measurement.warmupMsPerFactor,
    measureMs: manifest.measurement.measureMsPerFactor,
    browserChannel: manifest.playwrightCli.browserChannel,
    hashes: hashesStart,
    stagedHashes: expectedStageHashes,
    ...manifest.acceptance,
  };
  let validationError = null;
  try {
    report.validation = validateProductHfrReport(report, expected);
    report.passed = report.validation.passed;
  } catch (error) {
    validationError = error;
    report.validation = { passed: false, structuralError: error.stack || error.message };
  }

  const outputDirectory = resolveRepositoryPath(manifest.outputDirectory, 'outputDirectory');
  const outputPath = path.resolve(repositoryRoot, cli.output || path.join(outputDirectory,
    `product-hfr-${timestampLabel()}.json`));
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
  await rm(tempDirectory, { recursive: true, force: true });
  console.log(JSON.stringify({ output: outputPath, passed: report.passed,
    factors: report.validation?.factors || null, failures: report.validation?.failures || null }, null, 2));
  if (validationError) throw new Error(`Product HFR report is structurally invalid; red report: ${outputPath}`, { cause: validationError });
  if (!report.passed) throw new Error(`Product HFR gate failed; red report preserved at ${outputPath}`);
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
