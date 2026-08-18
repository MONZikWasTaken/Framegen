import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { access, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import { evaluateStaticGuardReport } from './static_guard_acceptance.mjs';

const execFileAsync = promisify(execFile);
const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(scriptPath), '..');
const outputRoot = path.join(repositoryRoot, 'output', 'static-guard');

export const STATIC_GUARD_RUNNER_CONTRACT = Object.freeze({
  gateId: 'framecast-static-guard-webgpu-gate-v1',
  schemaVersion: 1,
  playwrightPackage: '@playwright/cli@0.1.17',
  browser: 'chrome',
  headed: true,
  timeoutMs: 180000,
  sourceFiles: Object.freeze({
    source: 'web/static_guard_fixture.html',
    runtime: 'web/rt/rt.js',
    weights: 'extension/assets/rt_v7s.bin',
    manifest: 'extension/assets/rt_v7s.json',
    runner: 'tools/run_static_guard_gate.mjs',
    validator: 'tools/static_guard_acceptance.mjs',
  }),
});

const HELP = `Usage: node tools/run_static_guard_gate.mjs [options]

Runs the deterministic static-text-over-motion correctness gate in one headed
Playwright Chromium session. The result is written once under output/static-guard.

  --output <path>  unique JSON path inside output/static-guard
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
    requestedPath || path.join(outputRoot, `static-guard-${label}.json`));
  if (!inside(outputRoot, candidate) || path.extname(candidate).toLowerCase() !== '.json') {
    throw new Error('Output must be a .json file inside output/static-guard');
  }
  return candidate;
}

async function exists(filePath) {
  try { await access(filePath); return true; } catch { return false; }
}

async function sha256(filePath) {
  return createHash('sha256').update(await readFile(filePath)).digest('hex');
}

async function fileMetadata(filePath) {
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
    return { command: process.execPath, prefix: [npxScript], prerequisite: npxScript };
  }
  return { command: 'npx', prefix: [], prerequisite: null };
}

async function startStaticServer(root) {
  const server = createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url, 'http://127.0.0.1');
      const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, '');
      const filePath = path.resolve(root, relativePath);
      if (filePath !== root && !inside(root, filePath)) {
        response.writeHead(403).end('forbidden');
        return;
      }
      const body = await readFile(filePath);
      const extension = path.extname(filePath);
      const mime = extension === '.html' ? 'text/html; charset=utf-8'
        : extension === '.js' || extension === '.mjs' ? 'text/javascript; charset=utf-8'
          : extension === '.json' ? 'application/json; charset=utf-8' : 'application/octet-stream';
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
  if (!server) return;
  await new Promise(resolve => server.close(resolve));
}

function serializeError(error) {
  if (!error) return null;
  return { name: error.name || 'Error', message: error.message || String(error), stack: error.stack || null };
}

function validHash(value) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}

export function validateGateEvidence({ fixtureReport, diagnostics, hashesStart, hashesEnd, executionError = null }) {
  const failures = [];
  const sourceStability = {};
  for (const name of Object.keys(STATIC_GUARD_RUNNER_CONTRACT.sourceFiles)) {
    const start = hashesStart?.[name];
    const end = hashesEnd?.[name];
    const stable = validHash(start) && start === end;
    sourceStability[name] = { start, end, stable };
    if (!stable) failures.push(`source-hash-drift:${name}`);
  }

  let nodeContract = null;
  try {
    nodeContract = evaluateStaticGuardReport(fixtureReport);
    if (!nodeContract.passed) failures.push('node-validator-rejected-fixture-report');
  } catch (error) {
    nodeContract = { passed: false, structuralError: error.stack || error.message };
    failures.push('fixture-report-structurally-invalid');
  }
  if (fixtureReport?.status !== 'complete') failures.push('fixture-status-not-complete');
  if (fixtureReport?.passed !== true || fixtureReport?.validation?.passed !== true) {
    failures.push('fixture-report-not-passed');
  }

  const identityMap = {
    runtime: fixtureReport?.identities?.runtimeSha256,
    weights: fixtureReport?.identities?.weightsSha256,
    manifest: fixtureReport?.identities?.manifestSha256,
  };
  const fixtureIdentities = {};
  for (const [name, actual] of Object.entries(identityMap)) {
    const expected = hashesStart?.[name];
    const matches = validHash(actual) && actual === expected;
    fixtureIdentities[name] = { actual, expected, matches };
    if (!matches) failures.push(`fixture-identity-mismatch:${name}`);
  }

  const errorDiagnostics = {
    consoleErrors: diagnostics?.consoleErrors ?? [],
    pageErrors: diagnostics?.pageErrors ?? [],
    requestFailures: diagnostics?.requestFailures ?? [],
    httpErrors: diagnostics?.httpErrors ?? [],
  };
  for (const [name, rows] of Object.entries(errorDiagnostics)) {
    if (!Array.isArray(rows)) failures.push(`diagnostics-malformed:${name}`);
    else if (rows.length) failures.push(`browser-diagnostics:${name}`);
  }
  if (executionError) failures.push('runner-execution-error');

  return {
    passed: failures.length === 0,
    failures,
    sourceStability,
    fixtureIdentities,
    nodeContract,
    diagnosticsClean: Object.values(errorDiagnostics).every(rows => Array.isArray(rows) && rows.length === 0),
    executionError,
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
  const sourcePaths = Object.fromEntries(Object.entries(STATIC_GUARD_RUNNER_CONTRACT.sourceFiles)
    .map(([name, relativePath]) => [name, path.resolve(repositoryRoot, relativePath)]));
  for (const [name, filePath] of Object.entries(sourcePaths)) {
    if (!await exists(filePath)) throw new Error(`Required ${name} is missing: ${filePath}`);
  }
  const npx = npxInvocation();
  if (npx.prerequisite && !await exists(npx.prerequisite)) {
    throw new Error(`npx prerequisite is missing: ${npx.prerequisite}`);
  }

  const [hashPairs, metadataPairs, git] = await Promise.all([
    Promise.all(Object.entries(sourcePaths).map(async ([name, filePath]) => [name, await sha256(filePath)])),
    Promise.all(Object.entries(sourcePaths).map(async ([name, filePath]) => [name, await fileMetadata(filePath)])),
    gitMetadata(),
  ]);
  const hashesStart = Object.fromEntries(hashPairs);
  const sourceMetadata = Object.fromEntries(metadataPairs);
  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-static-guard-'));
  const session = `framegen-static-guard-${process.pid}-${Date.now()}`;
  const cliPrefix = [...npx.prefix, '--yes', '--package', STATIC_GUARD_RUNNER_CONTRACT.playwrightPackage, 'playwright-cli'];
  let server = null;
  let cliVersion = null;
  let browserResult = null;
  let primaryError = null;

  try {
    server = await startStaticServer(repositoryRoot);
    const address = server.address();
    const fixtureRelative = path.relative(repositoryRoot, sourcePaths.source).split(path.sep).map(encodeURIComponent).join('/');
    const query = new URLSearchParams({ gateRun: randomBytes(12).toString('hex') });
    const fixtureUrl = `http://127.0.0.1:${address.port}/${fixtureRelative}?${query}`;
    const evaluationPath = path.join(tempDirectory, 'run-static-guard.js');
    await writeFile(evaluationPath, `async (page) => {
      const diagnostics = {
        consoleMessages: [], consoleErrors: [], consoleWarnings: [], pageErrors: [],
        requests: [], requestFailures: [], httpErrors: [],
      };
      const onConsole = message => {
        const row = { type: message.type(), text: message.text() };
        diagnostics.consoleMessages.push(row);
        if (message.type() === 'error') diagnostics.consoleErrors.push(row);
        if (message.type() === 'warning') diagnostics.consoleWarnings.push(row);
      };
      const onPageError = error => diagnostics.pageErrors.push({ message: error.message, stack: error.stack || null });
      const onRequest = request => diagnostics.requests.push({ method: request.method(),
        resourceType: request.resourceType(), url: request.url() });
      const onRequestFailed = request => diagnostics.requestFailures.push({ method: request.method(),
        url: request.url(), errorText: request.failure()?.errorText || null });
      const onResponse = response => {
        if (response.status() >= 400) diagnostics.httpErrors.push({ status: response.status(), url: response.url() });
      };
      page.on('console', onConsole);
      page.on('pageerror', onPageError);
      page.on('request', onRequest);
      page.on('requestfailed', onRequestFailed);
      page.on('response', onResponse);
      try {
        await page.goto(${JSON.stringify(fixtureUrl)}, { waitUntil: 'load' });
        await page.waitForFunction(() => globalThis.__STATIC_GUARD_PROMISE__
          && typeof globalThis.__STATIC_GUARD_PROMISE__.then === 'function', null,
          { timeout: ${STATIC_GUARD_RUNNER_CONTRACT.timeoutMs} });
        const fixtureReport = await page.evaluate(() => globalThis.__STATIC_GUARD_PROMISE__);
        const environment = await page.evaluate(() => ({
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          hardwareConcurrency: navigator.hardwareConcurrency,
          deviceMemoryGiB: navigator.deviceMemory || null,
          devicePixelRatio,
          crossOriginIsolated,
          visibilityState: document.visibilityState,
        }));
        return { fixtureReport, environment, diagnostics, executionError: null };
      } catch (error) {
        return { fixtureReport: null, environment: null, diagnostics,
          executionError: { name: error.name, message: error.message, stack: error.stack || null } };
      } finally {
        page.off('console', onConsole);
        page.off('pageerror', onPageError);
        page.off('request', onRequest);
        page.off('requestfailed', onRequestFailed);
        page.off('response', onResponse);
      }
    }`, 'utf8');

    cliVersion = await run(npx.command, [...cliPrefix, '--version'], { cwd: tempDirectory, timeout: 60000 });
    await run(npx.command, [...cliPrefix, `-s=${session}`, 'open', 'about:blank',
      '--browser', STATIC_GUARD_RUNNER_CONTRACT.browser, '--headed'], {
      cwd: tempDirectory,
      timeout: 60000,
    });
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code',
      '--filename', evaluationPath], {
      cwd: tempDirectory,
      timeout: STATIC_GUARD_RUNNER_CONTRACT.timeoutMs + 30000,
    });
    try { browserResult = JSON.parse(raw); }
    catch (cause) { throw new Error(`Playwright CLI did not return JSON:\n${raw.slice(0, 4000)}`, { cause }); }
  } catch (error) {
    primaryError = serializeError(error);
  } finally {
    await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'], {
      cwd: tempDirectory,
      timeout: 30000,
    });
    await closeServer(server);
    await rm(tempDirectory, { recursive: true, force: true });
  }

  const hashesEnd = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await sha256(filePath)])));
  const diagnostics = browserResult?.diagnostics || {
    consoleMessages: [], consoleErrors: [], consoleWarnings: [], pageErrors: [],
    requests: [], requestFailures: [], httpErrors: [],
  };
  const executionError = primaryError || browserResult?.executionError || null;
  const validation = validateGateEvidence({
    fixtureReport: browserResult?.fixtureReport || null,
    diagnostics,
    hashesStart,
    hashesEnd,
    executionError,
  });
  const sourceFiles = Object.fromEntries(Object.entries(sourceMetadata).map(([name, value]) => [name, {
    ...value,
    sha256Start: hashesStart[name],
    sha256End: hashesEnd[name],
  }]));
  const report = {
    schemaVersion: STATIC_GUARD_RUNNER_CONTRACT.schemaVersion,
    gateId: STATIC_GUARD_RUNNER_CONTRACT.gateId,
    createdAt: new Date().toISOString(),
    source: { git, files: sourceFiles },
    host: {
      platform: process.platform,
      arch: process.arch,
      osRelease: os.release(),
      node: process.version,
      cpu: os.cpus()[0]?.model || null,
    },
    browser: {
      channel: STATIC_GUARD_RUNNER_CONTRACT.browser,
      headed: STATIC_GUARD_RUNNER_CONTRACT.headed,
      playwrightCliPackage: STATIC_GUARD_RUNNER_CONTRACT.playwrightPackage,
      playwrightCliVersion: cliVersion,
      environment: browserResult?.environment || null,
    },
    fixtureReport: browserResult?.fixtureReport || null,
    diagnostics,
    validation,
    passed: validation.passed,
  };

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });
  console.log(JSON.stringify({
    output: outputPath,
    passed: report.passed,
    failures: validation.failures,
    fixtureChecks: validation.nodeContract?.checks || null,
  }, null, 2));
  if (!report.passed) throw new Error(`Static guard gate failed; red report preserved at ${outputPath}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch(error => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}
