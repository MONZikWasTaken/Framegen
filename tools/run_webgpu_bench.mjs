import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { access, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { validateHfrReport } from './webgpu_bench_acceptance.mjs';

const execFileAsync = promisify(execFile);
const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(scriptPath), '..');
const defaultManifestPath = path.join(repositoryRoot, 'tools', 'webgpu_bench_manifest.json');
const acceptanceValidatorPath = path.join(repositoryRoot, 'tools', 'webgpu_bench_acceptance.mjs');

const HELP = `Usage: node tools/run_webgpu_bench.mjs [options]

Runs web/bench.html in the installed Chrome channel through pinned Playwright CLI.

  --manifest <path>          manifest JSON (default: tools/webgpu_bench_manifest.json)
  --profile <name>           named manifest profile (for example v7s-720p-hfr)
  --output <path>            result JSON path
  --res <288|360|480|720|1080>
  --model <v7s|tfact2>
  --scene <calm|motion>
  --mods <rt,opt>            comma-separated A/B modules; first is parity reference
  --warmup <n>               warm-up runT calls
  --reps <n>                 wall/CPU sample repetitions
  --iters <n>                calls per prepPair/runT sample
  --cycle-iters <n>          prepPair+runT cycles per 2x sample
  --mids <n>                 distinct t values in the per-runT sample
  --interval-factor <n>      N in prepPair + (N-1)*runT
  --interval-factors <a,b>   factors measured in one runtime with rotated order
  --interval-warmups <n>     complete warm-up intervals per factor
  --interval-reps <n>        complete-interval samples
  --source-fps <n>
  --output-hz <n>
  --budget-fraction <n>      soft fraction of the source interval
  --hard-budget-fraction <n> hard fraction of the source interval
  --strict-acceptance | --no-strict-acceptance
  --tune | --no-tune
  --sparse | --no-sparse
  --guard | --no-guard
  --headed | --headless
  --timeout-ms <n>
  --help
`;

function parseArguments(argv) {
  const options = {};
  const valueOptions = new Set([
    'manifest', 'profile', 'output', 'res', 'model', 'scene', 'mods', 'warmup', 'reps', 'iters',
    'cycle-iters', 'mids', 'interval-factor', 'interval-factors', 'interval-warmups',
    'interval-reps', 'source-fps', 'output-hz', 'budget-fraction', 'hard-budget-fraction',
    'timeout-ms',
  ]);
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const name = token.slice(2);
    if (name === 'help') {
      options.help = true;
      continue;
    }
    if (['tune', 'sparse', 'guard', 'headed', 'strict-acceptance'].includes(name)) {
      options[name] = true;
      continue;
    }
    if (['no-tune', 'no-sparse', 'no-guard', 'headless', 'no-strict-acceptance'].includes(name)) {
      options[name.replace(/^no-/, '')] = false;
      if (name === 'headless') options.headed = false;
      continue;
    }
    if (!valueOptions.has(name)) throw new Error(`Unknown option: --${name}`);
    const value = argv[++index];
    if (value === undefined) throw new Error(`Missing value for --${name}`);
    options[name] = value;
  }
  return options;
}

function parseIntegerList(value, fallback, name, { min = 2, max = 240 } = {}) {
  const selected = value === undefined ? [...fallback] : value.split(',').map(item => Number(item));
  if (!selected.length || selected.length > 16
      || selected.some(item => !Number.isInteger(item) || item < min || item > max)
      || new Set(selected).size !== selected.length) {
    throw new Error(`${name} must contain 1..16 unique integers in [${min}, ${max}]`);
  }
  return selected;
}

function parseNumber(value, fallback, name, { integer = true, min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const parsed = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(parsed) || (integer && !Number.isInteger(parsed)) || parsed < min || parsed > max) {
    throw new Error(`${name} must be ${integer ? 'an integer' : 'a number'} in [${min}, ${max}]`);
  }
  return parsed;
}

function assertChoice(value, choices, name) {
  if (!choices.includes(value)) throw new Error(`${name} must be one of: ${choices.join(', ')}`);
  return value;
}

async function fileExists(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function assetMetadata(filePath) {
  const fileStat = await stat(filePath);
  return {
    path: path.relative(repositoryRoot, filePath).replaceAll('\\', '/'),
    sizeBytes: fileStat.size,
    sha256: await sha256(filePath),
  };
}

async function run(command, args, options = {}) {
  try {
    const result = await execFileAsync(command, args, {
      cwd: options.cwd || repositoryRoot,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
      timeout: options.timeout,
      windowsHide: true,
    });
    return result.stdout.trim();
  } catch (cause) {
    const stderr = cause.stderr?.trim();
    const stdout = cause.stdout?.trim();
    const details = [stderr, stdout].filter(Boolean).join('\n');
    throw new Error(`${command} ${args.join(' ')} failed${details ? `:\n${details}` : ''}`, { cause });
  }
}

async function runOptional(command, args, options = {}) {
  try {
    return { available: true, stdout: await run(command, args, options) };
  } catch (error) {
    return { available: false, error: error.message };
  }
}

function npxInvocation() {
  if (process.platform === 'win32') {
    const npxScript = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npx-cli.js');
    return { command: process.execPath, prefix: [npxScript] };
  }
  return { command: 'npx', prefix: [] };
}

async function startStaticServer(root) {
  const mimeTypes = new Map([
    ['.html', 'text/html; charset=utf-8'], ['.js', 'text/javascript; charset=utf-8'],
    ['.mjs', 'text/javascript; charset=utf-8'], ['.json', 'application/json; charset=utf-8'],
    ['.bin', 'application/octet-stream'], ['.png', 'image/png'], ['.jpg', 'image/jpeg'],
  ]);
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
      if (pathname === '/favicon.ico') {
        response.writeHead(204).end();
        return;
      }
      const filePath = path.resolve(root, `.${pathname}`);
      if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) {
        response.writeHead(403).end('Forbidden');
        return;
      }
      const fileStat = await stat(filePath);
      if (!fileStat.isFile()) throw new Error('Not a file');
      response.writeHead(200, {
        'Content-Type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404).end('Not found');
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return server;
}

async function closeServer(server) {
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
}

async function gitMetadata() {
  const [commit, shortCommit, branch, statusOutput] = await Promise.all([
    run('git', ['rev-parse', 'HEAD']),
    run('git', ['rev-parse', '--short=12', 'HEAD']),
    run('git', ['rev-parse', '--abbrev-ref', 'HEAD']),
    run('git', ['status', '--porcelain=v1', '--untracked-files=all']),
  ]);
  return { commit, shortCommit, branch, dirty: statusOutput.length > 0 };
}

async function chromeMetadata() {
  if (process.platform !== 'win32') return { detectedStableExecutable: null, productVersion: null };
  const candidates = [
    process.env.ProgramFiles && path.join(process.env.ProgramFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env['ProgramFiles(x86)'] && path.join(process.env['ProgramFiles(x86)'], 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'),
  ].filter(Boolean);
  const executable = (await Promise.all(candidates.map(async candidate => await fileExists(candidate) ? candidate : null))).find(Boolean);
  if (!executable) return { detectedStableExecutable: null, productVersion: null };
  const escaped = executable.replaceAll("'", "''");
  const version = await runOptional('powershell.exe', [
    '-NoProfile', '-NonInteractive', '-Command',
    `(Get-Item -LiteralPath '${escaped}').VersionInfo.ProductVersion`,
  ]);
  return {
    detectedStableExecutable: executable,
    productVersion: version.available ? version.stdout : null,
    versionProbeError: version.available ? null : version.error,
  };
}

async function gpuMetadata() {
  const nvidia = await runOptional('nvidia-smi', [
    '--query-gpu=name,driver_version,pci.bus_id,memory.total', '--format=csv,noheader,nounits',
  ]);
  const result = { nvidiaSmi: nvidia };
  if (nvidia.available) {
    result.nvidiaAdapters = nvidia.stdout.split(/\r?\n/).filter(Boolean).map(line => {
      const [name, driverVersion, pciBusId, memoryMiB] = line.split(',').map(value => value.trim());
      return { name, driverVersion, pciBusId, memoryMiB: Number(memoryMiB) };
    });
  }
  if (process.platform === 'win32') {
    const windows = await runOptional('powershell.exe', [
      '-NoProfile', '-NonInteractive', '-Command',
      'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID,AdapterRAM | ConvertTo-Json -Compress',
    ]);
    result.windowsVideoControllers = windows.available ? JSON.parse(windows.stdout || '[]') : windows;
  }
  return result;
}

function timestampLabel() {
  return new Date().toISOString().replaceAll(':', '-').replace(/\.\d{3}Z$/, 'Z');
}

async function main() {
  const cli = parseArguments(process.argv.slice(2));
  if (cli.help) {
    process.stdout.write(HELP);
    return;
  }

  const manifestPath = path.resolve(repositoryRoot, cli.manifest || defaultManifestPath);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (![1, 2].includes(manifest.schemaVersion)) {
    throw new Error(`Unsupported manifest schemaVersion: ${manifest.schemaVersion}`);
  }
  const profileId = cli.profile || null;
  const profile = profileId === null ? null : manifest.profiles?.[profileId];
  if (profileId !== null && !profile) throw new Error(`Unknown benchmark profile: ${profileId}`);
  const workload = { ...manifest.workload, ...(profile?.workload || {}) };
  const measurement = { ...manifest.measurement, ...(profile?.measurement || {}) };
  const acceptance = { ...(manifest.acceptance || {}), ...(profile?.acceptance || {}) };
  if (profile?.acceptance?.enabled && cli['strict-acceptance'] === false) {
    throw new Error(`strict acceptance cannot be disabled for profile ${profileId}`);
  }
  const resolution = assertChoice(cli.res || workload.resolution, ['288', '360', '480', '720', '1080'], 'res');
  const model = assertChoice(cli.model || workload.model, ['v7s', 'tfact2'], 'model');
  const scene = assertChoice(cli.scene || workload.scene, ['calm', 'motion'], 'scene');
  const modules = (cli.mods ? cli.mods.split(',') : workload.modules).map(value => value.trim()).filter(Boolean);
  if (!modules.length || modules.some(value => !/^[a-z0-9_-]+$/i.test(value))) throw new Error('mods contains an invalid module name');

  const fallbackFactors = measurement.intervalFactors || [measurement.intervalFactor];
  const intervalFactors = parseIntegerList(
    cli['interval-factors'] ?? (cli['interval-factor'] === undefined ? undefined : cli['interval-factor']),
    fallbackFactors,
    'interval-factors',
  );
  const config = {
    profileId,
    resolution, model, scene, modules,
    sparseRefine: cli.sparse ?? workload.sparseRefine,
    staticGuard: cli.guard ?? workload.staticGuard,
    tune: cli.tune ?? workload.tune,
    strictAcceptance: cli['strict-acceptance'] ?? acceptance.enabled ?? false,
    sourceFps: parseNumber(cli['source-fps'], workload.sourceFps, 'source-fps', { integer: false, min: 0.1, max: 1000 }),
    intervalFactors,
    intervalWarmups: parseNumber(
      cli['interval-warmups'], measurement.intervalWarmupsPerFactor ?? 0,
      'interval-warmups', { min: 0, max: 20 },
    ),
    budgetFraction: parseNumber(
      cli['budget-fraction'], acceptance.budgetFraction ?? 0.85,
      'budget-fraction', { integer: false, min: 0.01, max: 1 },
    ),
    hardBudgetFraction: parseNumber(
      cli['hard-budget-fraction'], acceptance.hardBudgetFraction ?? 1,
      'hard-budget-fraction', { integer: false, min: 0.01, max: 1 },
    ),
    warmup: parseNumber(cli.warmup, measurement.warmupRunTCalls, 'warmup', { min: 0, max: 100 }),
    repetitions: parseNumber(cli.reps, measurement.repetitions, 'reps', { min: 1, max: 100 }),
    iterations: parseNumber(cli.iters, measurement.callsPerSample, 'iters', { min: 1, max: 1000 }),
    cycleIterations: parseNumber(cli['cycle-iters'], measurement.cycleCallsPerSample, 'cycle-iters', { min: 1, max: 1000 }),
    mids: parseNumber(cli.mids, measurement.perRunTDistinctTimesteps, 'mids', { min: 1, max: 240 }),
    intervalRepetitions: parseNumber(cli['interval-reps'], measurement.intervalRepetitions, 'interval-reps', { min: 1, max: 100 }),
    timeoutMs: parseNumber(cli['timeout-ms'], measurement.timeoutMs, 'timeout-ms', { min: 1000, max: 3600000 }),
    headed: cli.headed ?? manifest.playwrightCli.headed,
  };
  if (config.hardBudgetFraction < config.budgetFraction) {
    throw new Error('hard-budget-fraction must be >= budget-fraction');
  }
  if (config.intervalFactors.length === 1) {
    config.outputHz = parseNumber(
      cli['output-hz'], workload.outputHz ?? config.sourceFps * config.intervalFactors[0],
      'output-hz', { integer: false, min: 0.1, max: 1000 },
    );
    const cadenceRatio = config.outputHz / config.sourceFps;
    if (Math.abs(cadenceRatio - config.intervalFactors[0]) > 1e-9) {
      throw new Error(
        `output-hz/source-fps (${cadenceRatio}) must equal interval-factor (${config.intervalFactors[0]}) for this integer-cadence harness`,
      );
    }
  } else {
    if (cli['output-hz'] !== undefined) throw new Error('--output-hz is ambiguous with multiple interval factors');
    config.outputHz = null;
  }
  if (config.strictAcceptance) {
    if (config.intervalWarmups < 2) throw new Error('strict acceptance needs at least 2 interval warmups per factor');
    if (config.intervalRepetitions < 30) throw new Error('strict acceptance needs at least 30 interval repetitions per factor');
    if (!config.tune) throw new Error('strict acceptance requires autotuning');
  }
  if (profileId !== null && config.strictAcceptance) {
    const exactProfile = {
      budgetFraction: acceptance.budgetFraction,
      factors: measurement.intervalFactors,
      hardBudgetFraction: acceptance.hardBudgetFraction,
      model: workload.model,
      modules: workload.modules,
      resolution: workload.resolution,
      scene: workload.scene,
      sourceFps: workload.sourceFps,
      sparseRefine: workload.sparseRefine,
      staticGuard: workload.staticGuard,
      headed: manifest.playwrightCli.headed,
    };
    if (config.budgetFraction !== exactProfile.budgetFraction
        || config.hardBudgetFraction !== exactProfile.hardBudgetFraction
        || config.model !== exactProfile.model
        || config.resolution !== exactProfile.resolution
        || config.scene !== exactProfile.scene
        || config.sourceFps !== exactProfile.sourceFps
        || config.sparseRefine !== exactProfile.sparseRefine
        || config.staticGuard !== exactProfile.staticGuard
        || config.headed !== exactProfile.headed
        || JSON.stringify(config.modules) !== JSON.stringify(exactProfile.modules)
        || JSON.stringify(config.intervalFactors) !== JSON.stringify(exactProfile.factors)) {
      throw new Error(`strict profile ${profileId} conditions cannot be overridden`);
    }
  }

  const harnessPath = path.resolve(repositoryRoot, manifest.harness.path);
  if (!harnessPath.startsWith(`${repositoryRoot}${path.sep}`) || !(await fileExists(harnessPath))) {
    throw new Error(`Harness is outside the repository or missing: ${harnessPath}`);
  }
  const selectedAssets = manifest.modelAssets?.[config.model];
  if (!selectedAssets) throw new Error(`No modelAssets entry for ${config.model}`);
  const assetPaths = Object.fromEntries(Object.entries(selectedAssets).map(([name, relativePath]) => [
    name, path.resolve(repositoryRoot, relativePath),
  ]));
  for (const [name, assetPath] of Object.entries(assetPaths)) {
    if (!assetPath.startsWith(`${repositoryRoot}${path.sep}`) || !(await fileExists(assetPath))) {
      throw new Error(`Model ${name} asset is outside the repository or missing: ${assetPath}`);
    }
  }
  if (!assetPaths.weights || !assetPaths.manifest) {
    throw new Error(`modelAssets.${config.model} must define weights and manifest`);
  }
  const assetUrls = Object.fromEntries(Object.entries(assetPaths).map(([name, assetPath]) => [
    name,
    `/${path.relative(repositoryRoot, assetPath).split(path.sep).map(encodeURIComponent).join('/')}`,
  ]));

  const query = new URLSearchParams({
    res: config.resolution, model: config.model, scene: config.scene, mods: config.modules.join(','),
    sparse: config.sparseRefine ? '1' : '0', guard: config.staticGuard ? '1' : '0',
    tune: config.tune ? '1' : '0', mids: String(config.mids), warmup: String(config.warmup),
    reps: String(config.repetitions), iters: String(config.iterations),
    cycleIters: String(config.cycleIterations), intervalFactor: String(config.intervalFactors[0]),
    intervalFactors: config.intervalFactors.join(','), intervalWarmups: String(config.intervalWarmups),
    intervalReps: String(config.intervalRepetitions), sourceFps: String(config.sourceFps),
    budgetFraction: String(config.budgetFraction), hardBudgetFraction: String(config.hardBudgetFraction),
    strictAcceptance: config.strictAcceptance ? '1' : '0',
    weights: assetUrls.weights, weightsManifest: assetUrls.manifest,
  });
  if (config.outputHz !== null) query.set('outputHz', String(config.outputHz));

  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-playwright-'));
  const server = await startStaticServer(repositoryRoot);
  const serverAddress = server.address();
  const relativeHarness = path.relative(repositoryRoot, harnessPath).split(path.sep).join('/');
  const url = `http://127.0.0.1:${serverAddress.port}/${relativeHarness}?${query}`;
  const session = `framegen-bench-${process.pid}-${Date.now()}`;
  const npx = npxInvocation();
  const cliPrefix = [...npx.prefix, '--yes', '--package', manifest.playwrightCli.package, 'playwright-cli'];
  let benchResult;
  let cliVersion;
  let primaryError;

  try {
    cliVersion = await run(npx.command, [...cliPrefix, '--version'], { cwd: tempDirectory, timeout: 60000 });
    const openArgs = [...cliPrefix, `-s=${session}`, 'open', 'about:blank', '--browser', manifest.playwrightCli.browser];
    if (config.headed) openArgs.push('--headed');
    await run(npx.command, openArgs, { cwd: tempDirectory, timeout: 60000 });
    const evaluationPath = path.join(tempDirectory, 'run-benchmark.js');
    await writeFile(evaluationPath, `async (page) => {
      const consoleMessages = [];
      const pageErrors = [];
      const onConsole = message => {
        if (message.type() === 'warning' || message.type() === 'error') {
          consoleMessages.push({ type: message.type(), text: message.text() });
        }
      };
      const onPageError = error => pageErrors.push({ message: error.message, stack: error.stack || null });
      page.on('console', onConsole);
      page.on('pageerror', onPageError);
      try {
        await page.goto(${JSON.stringify(url)});
        await page.waitForFunction(() => globalThis.__benchResult !== undefined, null, { timeout: ${config.timeoutMs} });
        return {
          benchResult: await page.evaluate(() => globalThis.__benchResult),
          diagnostics: { consoleMessages, pageErrors },
        };
      } finally {
        page.off('console', onConsole);
        page.off('pageerror', onPageError);
      }
    }`, 'utf8');
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code', '--filename', evaluationPath], {
      cwd: tempDirectory,
      timeout: config.timeoutMs + 30000,
    });
    try {
      const evaluationResult = JSON.parse(raw);
      benchResult = evaluationResult.benchResult;
      benchResult.playwrightDiagnostics = evaluationResult.diagnostics;
    } catch (cause) {
      throw new Error(`Playwright CLI eval did not return JSON:\n${raw.slice(0, 4000)}`, { cause });
    }
    if (benchResult.error) throw new Error(`Benchmark page failed: ${benchResult.error}\n${benchResult.stack || ''}`);
  } catch (error) {
    primaryError = error;
  } finally {
    await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'], { cwd: tempDirectory, timeout: 30000 });
    await closeServer(server);
    await rm(tempDirectory, { recursive: true, force: true });
  }
  if (primaryError) throw primaryError;

  if (benchResult.schemaVersion !== 3) throw new Error(`Unexpected bench schemaVersion: ${benchResult.schemaVersion}`);
  if (JSON.stringify(benchResult.workload.intervalFactors) !== JSON.stringify(config.intervalFactors)) {
    throw new Error('Harness intervalFactors do not match runner config');
  }
  if (benchResult.workload.sourceFps !== config.sourceFps
      || benchResult.conditions?.warmup?.fullIntervalCallsPerFactor !== config.intervalWarmups
      || benchResult.conditions?.samples?.fullIntervalRepetitions !== config.intervalRepetitions) {
    throw new Error('Harness source cadence or interval sampling conditions do not match runner config');
  }
  if (benchResult.workload.modelAssets?.weightsUrl !== assetUrls.weights
      || benchResult.workload.modelAssets?.manifestUrl !== assetUrls.manifest) {
    throw new Error('Harness model asset URLs do not match runner config');
  }
  for (const moduleName of config.modules) {
    const moduleResult = benchResult.mods[moduleName];
    if (!Array.isArray(moduleResult?.fullIntervals)
        || moduleResult.fullIntervals.length !== config.intervalFactors.length
        || moduleResult.fullIntervals.some(result => !Number.isFinite(result?.wall?.p95Ms))) {
      throw new Error(`Module ${moduleName} did not produce every full-interval p95`);
    }
  }
  const parityLimits = {
    maxAbsByteDiff: parseNumber(undefined, manifest.parity?.maxAbsByteDiff ?? 0, 'parity.maxAbsByteDiff', { min: 0, max: 255 }),
    maxMeanAbsByteDiff: parseNumber(undefined, manifest.parity?.maxMeanAbsByteDiff ?? 0, 'parity.maxMeanAbsByteDiff', { integer: false, min: 0, max: 255 }),
    maxDifferentBytePercent: parseNumber(undefined, manifest.parity?.maxDifferentBytePercent ?? 0, 'parity.maxDifferentBytePercent', { integer: false, min: 0, max: 100 }),
  };
  for (const moduleName of config.modules.slice(1)) {
    const parity = benchResult.mods[moduleName]?.parityDetails;
    if (!parity?.compared) throw new Error(`Module ${moduleName} has no candidate parity result`);
    if (parity.maxAbsByteDiff > parityLimits.maxAbsByteDiff
        || parity.meanAbsByteDiff > parityLimits.maxMeanAbsByteDiff
        || parity.differentBytePercent > parityLimits.maxDifferentBytePercent) {
      throw new Error(`Module ${moduleName} failed parity: ${JSON.stringify({ observed: parity, limits: parityLimits })}`);
    }
  }

  const runtimeModules = Object.fromEntries(await Promise.all(config.modules.map(async moduleName => {
    const moduleFile = moduleName === 'rt' ? 'rt.js' : `rt_${moduleName}.js`;
    return [moduleName, await assetMetadata(path.join(repositoryRoot, 'web', 'rt', moduleFile))];
  })));
  const [git, chrome, gpu, harnessHash, runnerHash, manifestHash, acceptanceValidatorAsset,
    weightsAsset, modelManifestAsset] = await Promise.all([
    gitMetadata(), chromeMetadata(), gpuMetadata(), sha256(harnessPath), sha256(scriptPath), sha256(manifestPath),
    assetMetadata(acceptanceValidatorPath), assetMetadata(assetPaths.weights), assetMetadata(assetPaths.manifest),
  ]);
  const measurements = {};
  for (const [moduleName, moduleResult] of Object.entries(benchResult.mods)) {
    measurements[moduleName] = {
      autotune: moduleResult.convTune,
      wall: moduleResult.wall,
      cpuEncodeSubmit: moduleResult.cpuEncodeSubmit,
      gpu: moduleResult.gpu,
      gpuByFactor: moduleResult.gpuByFactor,
      fullIntervals: moduleResult.fullIntervals,
      passed: moduleResult.passed,
      parity: moduleResult.parityDetails,
    };
  }
  const stability = Object.fromEntries(Object.entries(measurements).map(([moduleName, measurement]) => {
    const factors = measurement.fullIntervals.map(interval => {
      const wall = interval.wall;
      const gpuEstimate = measurement.gpuByFactor?.[String(interval.factor)];
      const fullIntervalCvPercent = wall.meanMs > 0 ? (100 * wall.stddevMs) / wall.meanMs : null;
      const gpuVsWallPercent = gpuEstimate?.available && wall.p50Ms > 0
        ? (100 * (gpuEstimate.fullIntervalEstimateMs - wall.p50Ms)) / wall.p50Ms
        : null;
      return {
        factor: interval.factor,
        fullIntervalCvPercent,
        gpuEstimateVsWallP50Percent: gpuVsWallPercent,
        stableForComparativeClaims: fullIntervalCvPercent !== null && fullIntervalCvPercent <= 10,
      };
    });
    return [moduleName, {
      factors,
      stableForComparativeClaims: factors.every(value => value.stableForComparativeClaims),
    }];
  }));
  const report = {
    schemaVersion: 2,
    profile: profileId || (config.strictAcceptance ? 'ad-hoc-strict' : 'legacy-single-factor'),
    capturedAt: new Date().toISOString(),
    source: {
      git,
      harness: { path: path.relative(repositoryRoot, harnessPath).replaceAll('\\', '/'), sha256: harnessHash },
      runner: { path: path.relative(repositoryRoot, scriptPath).replaceAll('\\', '/'), sha256: runnerHash },
      manifest: { path: path.relative(repositoryRoot, manifestPath).replaceAll('\\', '/'), sha256: manifestHash },
      acceptanceValidator: acceptanceValidatorAsset,
      runtimeModules,
      modelAssets: { weights: weightsAsset, manifest: modelManifestAsset },
    },
    host: {
      platform: process.platform, arch: process.arch, osRelease: os.release(),
      node: process.version, cpu: os.cpus()[0]?.model || null, gpu,
    },
    browser: {
      channel: manifest.playwrightCli.browser,
      headed: config.headed,
      playwrightCli: { package: manifest.playwrightCli.package, version: cliVersion },
      chrome,
      page: benchResult.environment.browser,
      dawnRevision: null,
      dawnRevisionNote: 'Chrome does not expose a stable Dawn revision to page JavaScript; adapterInfo is recorded below.',
    },
    webgpu: benchResult.environment.webgpu,
    workload: benchResult.workload,
    conditions: {
      ...benchResult.conditions,
      parityLimits,
      moduleOrderNote: 'Modules run in listed order; reverse-order confirmation is required before comparative performance claims.',
    },
    validation: {
      stability,
      comparativePerformanceClaimReady: config.modules.length > 1
        && Object.values(stability).every(value => value.stableForComparativeClaims),
      measurementCoverage: {
        prepPair: true,
        runT: true,
        fullPairInterval: true,
        cpuEncodeSubmit: true,
        gpuStageEstimate: true,
        startup: false,
        runtimeVramUsage: false,
        droppedFrames: false,
      },
      coverageNote: 'Startup, runtime VRAM usage, and dropped frames require separate product-path instrumentation.',
    },
    diagnostics: benchResult.playwrightDiagnostics,
    measurements,
    rawBenchResult: benchResult,
  };

  if (config.strictAcceptance) {
    const acceptanceResult = validateHfrReport(report, {
      budgetFraction: config.budgetFraction,
      factors: config.intervalFactors,
      hardBudgetFraction: config.hardBudgetFraction,
      headed: config.headed,
      hashes: {
        acceptanceValidator: acceptanceValidatorAsset.sha256,
        harness: harnessHash,
        manifest: manifestHash,
        modelManifest: modelManifestAsset.sha256,
        modelWeights: weightsAsset.sha256,
        runner: runnerHash,
        runtimeModules: Object.fromEntries(
          Object.entries(runtimeModules).map(([name, metadata]) => [name, metadata.sha256]),
        ),
      },
      height: acceptance.height,
      minimumRepetitions: acceptance.minimumRepetitions,
      minimumWarmupIntervals: acceptance.minimumWarmupIntervals,
      model: config.model,
      modules: config.modules,
      profile: report.profile,
      requiredDeviceFeatures: acceptance.requiredDeviceFeatures || [],
      resolution: config.resolution,
      scene: config.scene,
      sourceFps: config.sourceFps,
      sparseRefine: config.sparseRefine,
      staticGuard: config.staticGuard,
      width: acceptance.width,
    });
    report.validation.hfrAcceptance = acceptanceResult;
    report.passed = acceptanceResult.passed;
  } else {
    report.passed = null;
  }

  const outputDirectory = path.resolve(repositoryRoot, manifest.outputDirectory);
  const outputPath = path.resolve(repositoryRoot, cli.output || path.join(outputDirectory, `webgpu-bench-${timestampLabel()}.json`));
  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });

  const summary = Object.fromEntries(Object.entries(measurements).map(([name, value]) => [name, {
    factors: value.fullIntervals.map(interval => ({
      factor: interval.factor,
      hardDeadlineMisses: interval.acceptance.hardDeadlineMisses,
      p50Ms: interval.wall.p50Ms,
      p95Ms: interval.wall.p95Ms,
      passed: interval.acceptance.passed,
      softDeadlineMisses: interval.acceptance.softDeadlineMisses,
    })),
    parity: value.parity,
  }]));
  console.log(JSON.stringify({
    output: outputPath, commit: git.commit, dirty: git.dirty, passed: report.passed, summary,
  }, null, 2));
  if (config.strictAcceptance && !report.passed) {
    throw new Error(`Strict benchmark acceptance failed; red report preserved at ${outputPath}`);
  }
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
