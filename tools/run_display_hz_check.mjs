// Display Hz browser check (GitHub issue #9).
//
// The authoritative product target-FPS gate only exercises explicit `target`
// requests at 240 Hz, so the "Display Hz" output mode had no browser coverage
// and nothing ran at the refresh rate where a mislabelled panel actually breaks
// the 2x source floor. This check is deliberately separate: it does not touch
// tools/product_target_fps_gate_manifest.json, its fixture, or its case matrix.
//
// It temporarily switches the primary display to the requested refresh rate,
// runs the real unpacked extension in Display Hz mode against the same
// deterministic moving-primitives source, then restores the original mode.
//
//   node tools/run_display_hz_check.mjs [--hz 60] [--source-fps 24] [--output <path>]

import { createHash, randomBytes } from 'node:crypto';
import { execFile } from 'node:child_process';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  access, cp, copyFile, mkdir, mkdtemp, readFile, rm, stat, writeFile,
} from 'node:fs/promises';

const execFileAsync = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('..', import.meta.url));
const outputRoot = path.join(repositoryRoot, 'output', 'display-hz');

export const CONTRACT = Object.freeze({
  checkId: 'framegen-display-hz-check',
  schemaVersion: 2,
  warmupMs: 4000,
  measureMs: 12000,
  timeoutMs: 120000,
  resolution: 480,
  playwrightPackage: '@playwright/cli@0.1.17',
  browser: 'chromium',
  sourceFiles: {
    contentScript: 'extension/content.js',
    cadenceScript: 'extension/cadence.js',
    profileStore: 'extension/profile-store.js',
    backgroundScript: 'extension/background.js',
    declarativeRules: 'extension/rules.json',
    extensionManifest: 'extension/manifest.json',
    runtime: 'web/rt/rt.js',
    weights: 'extension/assets/rt_v7s.bin',
    weightsManifest: 'extension/assets/rt_v7s.json',
    fixture: 'web/display_hz_fixture.html',
    runner: 'tools/run_display_hz_check.mjs',
    displayHelper: 'tools/set_primary_refresh.ps1',
  },
  acceptance: {
    displayHzToleranceFraction: 0.08,
    headroomToleranceFraction: 0.01,
    scheduledToleranceFraction: 0.03,
    presentedToleranceFraction: 0.05,
    sourceHzToleranceFraction: 0.03,
    maximumDropped: 0,
    maximumPoolExhaustion: 0,
    maximumSourceBusySkipped: 0,
    maximumRatePlanResets: 0,
    maximumProducerSkippedScheduleSlots: 0,
    maximumLateP95OutputIntervals: 1.25,
    maximumLateMaxOutputIntervals: 3,
  },
});

export function parseArguments(argv) {
  const options = { hz: 60, sourceFps: 24, output: null, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') options.help = true;
    else if (argument === '--hz') options.hz = Number(argv[++index]);
    else if (argument === '--source-fps') options.sourceFps = Number(argv[++index]);
    else if (argument === '--output') {
      const value = argv[++index];
      if (!value || value.startsWith('--')) throw new Error('--output requires a path');
      options.output = value;
    }
    else throw new Error(`Unknown argument: ${argument}`);
  }
  if (!Number.isInteger(options.hz) || options.hz < 24 || options.hz > 1000) {
    throw new Error('--hz must be an integer in [24, 1000]');
  }
  if (!Number.isFinite(options.sourceFps) || options.sourceFps <= 0 || options.sourceFps > 120) {
    throw new Error('--source-fps must be in (0, 120]');
  }
  return options;
}

async function exists(target) {
  try { await access(target); return true; } catch { return false; }
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

function npxInvocation() {
  if (process.platform === 'win32') {
    const prerequisite = path.join(path.dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npx-cli.js');
    return { command: process.execPath, prefix: [prerequisite], prerequisite };
  }
  return { command: 'npx', prefix: [], prerequisite: null };
}

function inside(root, target) {
  const relative = path.relative(root, target);
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function timestampLabel() {
  return new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
}

export function resolveOutputPath(requestedPath, label = timestampLabel()) {
  const candidate = path.resolve(repositoryRoot,
    requestedPath || path.join(outputRoot, `display-hz-${label}.json`));
  if (!inside(outputRoot, candidate) || path.extname(candidate).toLowerCase() !== '.json') {
    throw new Error('Output must be a .json file inside output/display-hz');
  }
  return candidate;
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

async function stageExtension(tempDirectory) {
  const stageDirectory = path.join(tempDirectory, 'unpacked-extension');
  await cp(path.join(repositoryRoot, 'extension'), stageDirectory, {
    recursive: true,
    filter: source => !path.basename(source).startsWith('_'),
  });
  const runtimeDirectory = path.join(stageDirectory, 'rt');
  await mkdir(runtimeDirectory, { recursive: true });
  await copyFile(path.join(repositoryRoot, 'web/rt/rt.js'), path.join(runtimeDirectory, 'rt.js'));
  return {
    directory: stageDirectory,
    hashes: {
      contentScript: await sha256(path.join(stageDirectory, 'content.js')),
      cadenceScript: await sha256(path.join(stageDirectory, 'cadence.js')),
      profileStore: await sha256(path.join(stageDirectory, 'profile-store.js')),
      backgroundScript: await sha256(path.join(stageDirectory, 'background.js')),
      declarativeRules: await sha256(path.join(stageDirectory, 'rules.json')),
      extensionManifest: await sha256(path.join(stageDirectory, 'manifest.json')),
      runtime: await sha256(path.join(runtimeDirectory, 'rt.js')),
      weights: await sha256(path.join(stageDirectory, 'assets', 'rt_v7s.bin')),
      weightsManifest: await sha256(path.join(stageDirectory, 'assets', 'rt_v7s.json')),
    },
  };
}

const displayHelper = path.join(repositoryRoot, 'tools', 'set_primary_refresh.ps1');

async function primaryRefreshHz() {
  const output = await run('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', displayHelper, '-Query'], { timeout: 30000 });
  const match = /^OK\s+(\S+)\s+(\d+)x(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$/.exec(output);
  if (!match) throw new Error(`Cannot read primary display mode: ${output}`);
  return {
    device: match[1], width: Number(match[2]), height: Number(match[3]), hz: Number(match[4]),
    bitsPerPixel: Number(match[5]), orientation: Number(match[6]),
    fixedOutput: Number(match[7]), displayFlags: Number(match[8]),
  };
}

async function setPrimaryRefreshHz(hz) {
  return run('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', displayHelper, '-Hz', String(hz)], { timeout: 60000 });
}

function sameDisplayMode(left, right) {
  return sameDisplayGeometry(left, right) && left?.hz === right?.hz;
}

function sameDisplayGeometry(left, right) {
  return left?.device === right?.device && left?.width === right?.width
    && left?.height === right?.height && left?.bitsPerPixel === right?.bitsPerPixel
    && left?.orientation === right?.orientation && left?.fixedOutput === right?.fixedOutput
    && left?.displayFlags === right?.displayFlags;
}

async function restorePrimaryMode(originalMode) {
  const attempts = [];
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const apply = await runOptional('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass',
      '-File', displayHelper, '-Hz', String(originalMode.hz)], { timeout: 60000 });
    await new Promise(resolve => setTimeout(resolve, 1000));
    let observed = null;
    let queryError = null;
    try { observed = await primaryRefreshHz(); }
    catch (error) { queryError = error.message; }
    attempts.push({ attempt, apply, observed, queryError });
    if (apply.available && sameDisplayMode(originalMode, observed)) {
      return { passed: true, attempts, observed };
    }
  }
  return { passed: false, attempts, observed: attempts.at(-1)?.observed || null };
}

async function closeStaticServer(server, timeoutMs = 5000) {
  if (!server) return;
  server.closeAllConnections?.();
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('static server close timed out')), timeoutMs);
    server.close(error => {
      clearTimeout(timer);
      if (error) reject(error);
      else resolve();
    });
  });
}

function rateFrom(count, durationMs) {
  return durationMs > 0 ? count * 1000 / durationMs : null;
}

function within(actual, expected, fraction) {
  if (!Number.isFinite(actual) || !Number.isFinite(expected) || expected === 0) return false;
  return Math.abs(actual - expected) / Math.abs(expected) <= fraction;
}

export function evaluate(measurement, options, appliedHz, {
  diagnostics = null,
  rafControl = null,
} = {}) {
  const failures = [];
  const accept = CONTRACT.acceptance;
  if (!measurement || typeof measurement !== 'object') {
    return { passed: false, failures: ['telemetry snapshot is missing'], derived: null };
  }
  if (measurement?.error) {
    return { passed: false, failures: [`fixture error: ${measurement.error}`], derived: null };
  }
  const telemetry = measurement.telemetry;
  const producer = measurement.producer;
  const product = telemetry?.product;
  const counters = telemetry?.counters;
  if (!telemetry || !product || !counters || !producer) {
    return { passed: false, failures: ['telemetry snapshot is missing'], derived: null };
  }
  const observedSourceHz = telemetry.observed?.sourceHz;

  const requireFinite = (value, label, { positive = false, integer = false } = {}) => {
    const valid = Number.isFinite(value) && (!positive || value > 0)
      && (!integer || Number.isInteger(value)) && (positive || value >= 0);
    if (!valid) failures.push(`${label} is missing or invalid`);
    return valid;
  };
  requireFinite(telemetry.durationMs, 'telemetry duration', { positive: true });
  requireFinite(product.displayCapacityHz, 'display capacity', { positive: true });
  requireFinite(product.effectiveTargetHz, 'effective target', { positive: true });
  requireFinite(product.minimumTargetHz, 'minimum target', { positive: true });
  requireFinite(observedSourceHz, 'observed source rate', { positive: true });
  requireFinite(producer.observedHz, 'fixture producer rate', { positive: true });
  requireFinite(producer.producedFrames, 'fixture produced frame count', { integer: true });
  requireFinite(producer.skippedScheduleSlots, 'fixture skipped slot count', { integer: true });
  for (const name of [
    'sourceCallbacks', 'sourceProcessed', 'scheduled', 'scheduledSource', 'scheduledMid',
    'presented', 'presentedSource', 'presentedMid', 'dropped', 'droppedSource', 'droppedMid', 'pending',
    'queueHighWater', 'sourcePoolExhausted', 'midPoolExhausted', 'sourceBusySkipped',
    'ratePlanResets',
  ]) requireFinite(counters[name], `counter ${name}`, { integer: true });
  requireFinite(telemetry.lateness?.count, 'lateness count', { integer: true });
  requireFinite(telemetry.lateness?.p95Ms, 'lateness p95');
  requireFinite(telemetry.lateness?.maxMs, 'lateness max');
  requireFinite(rafControl?.observedHz, 'control rAF rate', { positive: true });
  if (telemetry.schemaVersion !== 1) failures.push('telemetry schemaVersion must be 1');
  if (telemetry.sampleOverflow !== false) failures.push('telemetry sample buffer overflowed or is missing');
  if (!Array.isArray(telemetry.errors)) failures.push('telemetry errors array is missing');

  const durationMs = telemetry.durationMs;
  const derived = {
    durationMs,
    appliedDisplayHz: appliedHz,
    targetState: product.targetState,
    displayCapacityHz: product.displayCapacityHz,
    effectiveTargetHz: product.effectiveTargetHz,
    minimumTargetHz: product.minimumTargetHz,
    targetClampReason: product.targetClampReason,
    rafFloorMs: product.scheduler?.rafFloor ?? null,
    observedRafHz: telemetry.observed?.rafHz ?? null,
    observedSourceHz: observedSourceHz ?? null,
    scheduledHz: rateFrom(counters.scheduled, durationMs),
    presentedHz: rateFrom(counters.presented, durationMs),
    presentedSourceHz: rateFrom(counters.presentedSource, durationMs),
    presentedMidHz: rateFrom(counters.presentedMid, durationMs),
    dropped: counters.dropped,
    pending: counters.pending,
    queueHighWater: counters.queueHighWater,
    poolExhaustion: counters.sourcePoolExhausted + counters.midPoolExhausted,
    sourceBusySkipped: counters.sourceBusySkipped,
    lateP95Ms: telemetry.lateness?.p95Ms ?? null,
    lateMaxMs: telemetry.lateness?.maxMs ?? null,
    ratePlanResets: counters.ratePlanResets,
    producerHz: producer.observedHz ?? null,
    producerSkippedScheduleSlots: producer.skippedScheduleSlots ?? null,
    rafControlHz: rafControl?.observedHz ?? null,
    expectedHeadroomHz: null,
  };

  if (product.factor !== 'hz') failures.push(`product factor was ${product.factor}, expected hz`);
  if (product.targetState !== 'active') failures.push(`plan state was ${product.targetState}, expected active`);
  if (!within(product.displayCapacityHz, appliedHz, accept.displayHzToleranceFraction)) {
    failures.push(`measured display capacity ${product.displayCapacityHz} is not within `
      + `${accept.displayHzToleranceFraction * 100}% of the applied ${appliedHz} Hz`);
  }
  if (!within(observedSourceHz, options.sourceFps, accept.sourceHzToleranceFraction)) {
    failures.push(`observed source ${observedSourceHz} drifted from ${options.sourceFps} FPS`);
  }
  const plannedSourceHz = product.minimumTargetHz / 2;
  if (!within(plannedSourceHz, observedSourceHz, accept.sourceHzToleranceFraction)) {
    failures.push(`scheduler source ${plannedSourceHz} does not match observed source `
      + `${observedSourceHz}`);
  }
  if (!within(producer.observedHz, options.sourceFps, accept.sourceHzToleranceFraction)) {
    failures.push(`fixture producer ${producer.observedHz} drifted from ${options.sourceFps} FPS`);
  }
  if (producer.skippedScheduleSlots > accept.maximumProducerSkippedScheduleSlots) {
    failures.push(`${producer.skippedScheduleSlots} fixture producer slots were skipped`);
  }
  if (!within(rafControl?.observedHz, appliedHz, accept.displayHzToleranceFraction)) {
    failures.push(`control rAF ${rafControl?.observedHz} is not within `
      + `${accept.displayHzToleranceFraction * 100}% of the applied ${appliedHz} Hz`);
  }

  // The issue #9 headroom contract: Display Hz must reserve real presentation
  // service below the panel ceiling so a missed slot can be recovered instead of
  // dropped, unless reserving it would breach the strict 2x source floor.
  const headroomToleranceHz = Math.max(0.01, product.minimumTargetHz * 0.005);
  const headroomFloorOk = product.displayCapacityHz * 0.97 + headroomToleranceHz
    >= product.minimumTargetHz;
  derived.expectedHeadroomHz = headroomFloorOk
    ? product.displayCapacityHz * 0.97
    : product.displayCapacityHz;
  derived.headroomApplies = headroomFloorOk;
  if (!within(product.effectiveTargetHz, derived.expectedHeadroomHz, accept.headroomToleranceFraction)) {
    failures.push(`effective target ${product.effectiveTargetHz} does not match the expected `
      + `${headroomFloorOk ? 'headroom-reserved' : '2x-floor-bound'} ${derived.expectedHeadroomHz}`);
  }
  if (headroomFloorOk && !(product.effectiveTargetHz < product.displayCapacityHz)) {
    failures.push('Display Hz left no catch-up headroom below the panel ceiling');
  }
  if (product.targetClampReason !== null) {
    failures.push(`Display Hz reported clamp reason ${product.targetClampReason}, expected none`);
  }

  // A display relabel materially changes the committed output rate, so the
  // scheduler records a plan reset. Zero resets over the window is the evidence
  // that the estimator never latched a wrong panel rate during measurement.
  if (derived.ratePlanResets > accept.maximumRatePlanResets) {
    failures.push(`${derived.ratePlanResets} committed output-plan resets during measurement`);
  }
  const outputIntervalMs = 1000 / product.effectiveTargetHz;
  if (!(derived.lateP95Ms <= outputIntervalMs * accept.maximumLateP95OutputIntervals)) {
    failures.push(`late p95 ${derived.lateP95Ms?.toFixed(2)}ms exceeds `
      + `${accept.maximumLateP95OutputIntervals} output intervals (${outputIntervalMs.toFixed(2)}ms each)`);
  }
  if (!(derived.lateMaxMs <= outputIntervalMs * accept.maximumLateMaxOutputIntervals)) {
    failures.push(`late max ${derived.lateMaxMs?.toFixed(2)}ms exceeds `
      + `${accept.maximumLateMaxOutputIntervals} output intervals`);
  }
  if (!within(derived.presentedHz, product.effectiveTargetHz, accept.presentedToleranceFraction)) {
    failures.push(`presented ${derived.presentedHz} is not within `
      + `${accept.presentedToleranceFraction * 100}% of the effective target ${product.effectiveTargetHz}`);
  }
  const expectedScheduled = product.effectiveTargetHz * durationMs / 1000;
  const boundaryAllowance = Math.ceil(product.effectiveTargetHz / observedSourceHz) + 1;
  const scheduledAllowance = Math.max(expectedScheduled * accept.scheduledToleranceFraction,
    boundaryAllowance);
  if (!(Math.abs(counters.scheduled - expectedScheduled) <= scheduledAllowance)) {
    failures.push(`scheduled attempts ${derived.scheduledHz} are not within `
      + `${accept.scheduledToleranceFraction * 100}% of the effective target ${product.effectiveTargetHz}`);
  }
  const identities = [
    [counters.scheduled, counters.scheduledSource + counters.scheduledMid,
      'scheduled total does not equal source plus mid'],
    [counters.presented, counters.presentedSource + counters.presentedMid,
      'presented total does not equal source plus mid'],
    [counters.dropped, counters.droppedSource + counters.droppedMid,
      'dropped total does not equal source plus mid'],
    [counters.scheduled, counters.presented + counters.dropped + counters.pending,
      'scheduled total does not equal presented plus dropped plus pending'],
    [telemetry.lateness?.count, counters.presented,
      'lateness count does not equal presented total'],
    [counters.sourceCallbacks, counters.sourceProcessed + counters.sourceBusySkipped,
      'source callbacks do not equal processed plus busy-skipped'],
  ];
  for (const [actual, expected, failure] of identities) {
    if (actual !== expected) failures.push(failure);
  }
  if (derived.dropped > accept.maximumDropped) failures.push(`${derived.dropped} queue drops`);
  if (derived.poolExhaustion > accept.maximumPoolExhaustion) {
    failures.push(`${derived.poolExhaustion} texture-pool exhaustions`);
  }
  if (derived.sourceBusySkipped > accept.maximumSourceBusySkipped) {
    failures.push(`${derived.sourceBusySkipped} source busy skips`);
  }
  if ((telemetry.errors || []).length) {
    failures.push(`${telemetry.errors.length} extension errors`);
  }
  for (const name of ['consoleErrors', 'pageErrors', 'requestFailures', 'httpErrors']) {
    if (!Array.isArray(diagnostics?.[name])) failures.push(`browser diagnostics ${name} are missing`);
    else if (diagnostics[name].length) failures.push(`browser errors are present: ${name}`);
  }
  return { passed: failures.length === 0, failures, derived };
}

export function applyInfrastructureChecks(verdict, {
  hashesStart,
  hashesEnd,
  restore,
  cleanupErrors = [],
  git,
}) {
  if (JSON.stringify(hashesStart) !== JSON.stringify(hashesEnd)) {
    verdict.failures.push('source files changed during the run');
  }
  if (!restore?.passed) verdict.failures.push('primary display was not restored');
  verdict.failures.push(...cleanupErrors);
  if (git?.dirty !== false) {
    verdict.failures.push(git?.dirty === true ? 'git worktree is dirty' : 'git worktree state is unavailable');
  }
  if (Array.isArray(git?.errors) && git.errors.length) {
    verdict.failures.push(...git.errors.map(error => `git metadata: ${error}`));
  }
  verdict.passed = verdict.failures.length === 0;
  return verdict;
}

export async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    console.log('node tools/run_display_hz_check.mjs [--hz 60] [--source-fps 24] [--output <path>]');
    return;
  }
  const outputPath = resolveOutputPath(options.output);

  const sourcePaths = Object.fromEntries(Object.entries(CONTRACT.sourceFiles)
    .map(([name, relative]) => [name, path.join(repositoryRoot, relative)]));
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

  const originalMode = await primaryRefreshHz();
  const session = `framegen-display-hz-${process.pid}-${Date.now()}`;
  const cliPrefix = [...npx.prefix, '--yes', '--package', CONTRACT.playwrightPackage, 'playwright-cli'];

  let tempDirectory = null;
  let profileDirectory = null;
  let staged = null;
  let stagedFixtureHash = null;
  let appliedNote = null;
  let restore = null;
  let appliedMode = originalMode;
  let server = null;
  let browserResult = null;
  let primaryError = null;
  const cleanupErrors = [];
  const manualRecovery = `powershell -NoProfile -ExecutionPolicy Bypass -File tools/set_primary_refresh.ps1 -Hz ${originalMode.hz}`;
  let displayMutationTail = Promise.resolve();
  const serializeDisplayMutation = operation => {
    const result = displayMutationTail.catch(() => undefined).then(operation);
    displayMutationTail = result;
    return result;
  };
  let restorePromise = null;
  const beginRestore = () => {
    if (!restorePromise) {
      restorePromise = serializeDisplayMutation(() => restorePrimaryMode(originalMode));
    }
    return restorePromise;
  };
  let signalExitStarted = false;
  const signalHandler = signal => {
    if (signalExitStarted) return;
    signalExitStarted = true;
    console.error(`${signal} received; restoring the primary display before exit`);
    beginRestore().then(result => {
      if (!result.passed) console.error(`DISPLAY RESTORE FAILED. Run manually: ${manualRecovery}`);
      process.exit(signal === 'SIGINT' ? 130 : 143);
    }).catch(error => {
      console.error(`DISPLAY RESTORE FAILED: ${error.message}. Run manually: ${manualRecovery}`);
      process.exit(1);
    });
  };
  const signalHandlers = {
    SIGINT: () => signalHandler('SIGINT'),
    SIGTERM: () => signalHandler('SIGTERM'),
  };
  process.on('SIGINT', signalHandlers.SIGINT);
  process.on('SIGTERM', signalHandlers.SIGTERM);
  try {
    tempDirectory = await mkdtemp(path.join(os.tmpdir(), 'framegen-display-hz-'));
    profileDirectory = path.join(tempDirectory, 'profile');
    await mkdir(profileDirectory, { recursive: true });
    staged = await stageExtension(tempDirectory);
    const expectedStageHashes = Object.fromEntries(Object.keys(staged.hashes)
      .map(name => [name, hashesStart[name]]));
    if (JSON.stringify(staged.hashes) !== JSON.stringify(expectedStageHashes)) {
      throw new Error('Temporary unpacked extension differs from source files');
    }
    const siteDirectory = path.join(tempDirectory, 'site');
    await mkdir(path.join(siteDirectory, 'web'), { recursive: true });
    const stagedFixturePath = path.join(siteDirectory, 'web', 'display_hz_fixture.html');
    await copyFile(sourcePaths.fixture, stagedFixturePath);
    stagedFixtureHash = await sha256(stagedFixturePath);
    if (stagedFixtureHash !== hashesStart.fixture) {
      throw new Error('Temporary fixture differs from its source file');
    }

    appliedNote = await serializeDisplayMutation(() => setPrimaryRefreshHz(options.hz));
    if (appliedNote.startsWith('ERROR')) throw new Error(appliedNote);
    // Let the compositor settle on the new mode before measuring anything.
    await new Promise(resolve => setTimeout(resolve, 2500));
    appliedMode = await primaryRefreshHz();
    if (!sameDisplayGeometry(originalMode, appliedMode) || appliedMode.hz !== options.hz) {
      throw new Error(`primary display mode changed unexpectedly after requesting ${options.hz} Hz`);
    }

    server = await startStaticServer(siteDirectory);
    const { port } = server.address();
    const token = randomBytes(24).toString('hex');
    const query = new URLSearchParams({
      framegenProductBench: '1',
      framegenBenchToken: token,
      caseId: `display-hz-${options.hz}-source${Math.round(options.sourceFps)}`,
      factor: 'hz',
      sourceFps: String(options.sourceFps),
      resolution: String(CONTRACT.resolution),
      warmupMs: String(CONTRACT.warmupMs),
      measureMs: String(CONTRACT.measureMs),
    });
    const url = `http://127.0.0.1:${port}/web/display_hz_fixture.html?${query}`;

    const configPath = path.join(tempDirectory, 'playwright-cli.config.json');
    await writeFile(configPath, `${JSON.stringify({
      browser: {
        browserName: CONTRACT.browser,
        isolated: false,
        userDataDir: profileDirectory,
        launchOptions: {
          channel: CONTRACT.browser,
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

    const evaluationPath = path.join(tempDirectory, 'run-display-hz.js');
    await writeFile(evaluationPath, `async (page) => {
      const diagnostics = { consoleErrors: [], pageErrors: [], requestFailures: [], httpErrors: [] };
      page.on('console', message => {
        if (message.type() === 'error') diagnostics.consoleErrors.push(message.text());
      });
      page.on('pageerror', error => diagnostics.pageErrors.push(error.message));
      page.on('requestfailed', request => diagnostics.requestFailures.push(request.url()));
      page.on('response', response => {
        if (response.status() >= 400) diagnostics.httpErrors.push(response.status() + ' ' + response.url());
      });
      try {
        await page.goto('about:blank', { waitUntil: 'load' });
        const rafControl = await page.evaluate(() => new Promise(resolve => {
          const intervalsMs = [];
          let startedAt = null;
          let previousAt = null;
          const tick = now => {
            if (startedAt === null) startedAt = now;
            if (previousAt !== null) intervalsMs.push(now - previousAt);
            previousAt = now;
            if (now - startedAt >= 3000) {
              const total = intervalsMs.reduce((sum, value) => sum + value, 0);
              const ordered = [...intervalsMs].sort((a, b) => a - b);
              resolve({
                callbacks: intervalsMs.length + 1,
                observedHz: total > 0 ? 1000 * intervalsMs.length / total : null,
                p95Ms: ordered[Math.max(0, Math.ceil(ordered.length * 0.95) - 1)] || null,
                maxMs: ordered.length ? ordered[ordered.length - 1] : null,
              });
              return;
            }
            requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }));
        await page.goto(${JSON.stringify(url)}, { waitUntil: 'load' });
        const environment = await page.evaluate(() => ({
          userAgent: navigator.userAgent, devicePixelRatio,
          visibilityState: document.visibilityState,
          screen: { width: screen.width, height: screen.height },
        }));
        await page.waitForFunction(() => globalThis.__displayHzResult !== undefined, null,
          { timeout: ${CONTRACT.timeoutMs} });
        const measurement = await page.evaluate(() => globalThis.__displayHzResult);
        return { measurement, rafControl, environment, diagnostics, executionError: null };
      } catch (error) {
        return { measurement: null, rafControl: null, environment: null, diagnostics,
          executionError: { name: error.name, message: error.message } };
      }
    }`, 'utf8');

    await run(npx.command, [...cliPrefix, '--version'], { cwd: tempDirectory, timeout: 60000 });
    await run(npx.command, [...cliPrefix, `-s=${session}`, `--config=${configPath}`,
      'open', 'about:blank', '--headed', '--persistent'], { cwd: tempDirectory, timeout: 60000 });
    const raw = await run(npx.command, [...cliPrefix, `-s=${session}`, '--raw', 'run-code',
      '--filename', evaluationPath], { cwd: tempDirectory, timeout: CONTRACT.timeoutMs + 60000 });
    try { browserResult = JSON.parse(raw); }
    catch (cause) { throw new Error(`Playwright CLI did not return JSON:\n${raw.slice(0, 4000)}`, { cause }); }
  } catch (error) {
    primaryError = { name: error.name, message: error.message };
  } finally {
    try { restore = await beginRestore(); }
    catch (error) { restore = { passed: false, attempts: [], error: error.message, observed: null }; }
    process.removeListener('SIGINT', signalHandlers.SIGINT);
    process.removeListener('SIGTERM', signalHandlers.SIGTERM);
    if (!restore?.passed) console.error(`DISPLAY RESTORE FAILED. Run manually: ${manualRecovery}`);
    if (tempDirectory) {
      const close = await runOptional(npx.command, [...cliPrefix, `-s=${session}`, 'close'],
        { cwd: tempDirectory, timeout: 30000 });
      if (!close.available) cleanupErrors.push(`browser cleanup: ${close.error}`);
    }
    if (server) {
      try { await closeStaticServer(server); }
      catch (error) { cleanupErrors.push(`server cleanup: ${error.message}`); }
    }
    if (tempDirectory) {
      try { await rm(tempDirectory, { recursive: true, force: true }); }
      catch (error) { cleanupErrors.push(`temporary directory cleanup: ${error.message}`); }
    }
  }

  const hashesEnd = Object.fromEntries(await Promise.all(Object.entries(sourcePaths)
    .map(async ([name, filePath]) => [name, await sha256(filePath)])));
  const sourceFiles = Object.fromEntries(Object.entries(sourceMetadata).map(([name, value]) => [name, {
    ...value, sha256Start: hashesStart[name], sha256End: hashesEnd[name],
  }]));
  const executionError = primaryError || browserResult?.executionError || null;
  const verdict = executionError
    ? { passed: false, failures: [`execution error: ${executionError.message}`], derived: null }
    : evaluate(browserResult.measurement, options, appliedMode.hz, {
      diagnostics: browserResult.diagnostics,
      rafControl: browserResult.rafControl,
    });
  applyInfrastructureChecks(verdict, { hashesStart, hashesEnd, restore, cleanupErrors, git });
  if (!restore?.passed) {
    const index = verdict.failures.indexOf('primary display was not restored');
    if (index >= 0) verdict.failures[index] = `primary display was not restored to ${originalMode.hz} Hz`;
  }

  const report = {
    schemaVersion: CONTRACT.schemaVersion,
    checkId: CONTRACT.checkId,
    createdAt: new Date().toISOString(),
    purpose: 'Browser coverage for the Display Hz output mode at a low panel refresh rate (GitHub issue #9)',
    source: { git, files: sourceFiles },
    host: { platform: process.platform, osRelease: os.release(), node: process.version,
      cpu: os.cpus()[0]?.model || null },
    display: { original: originalMode, applied: appliedMode, requestedHz: options.hz,
      applyNote: appliedNote, restore,
      manualRecovery },
    conditions: { sourceFps: options.sourceFps, resolution: CONTRACT.resolution,
      warmupMs: CONTRACT.warmupMs, measureMs: CONTRACT.measureMs },
    acceptance: CONTRACT.acceptance,
    rafControl: browserResult?.rafControl || null,
    environment: browserResult?.environment || null,
    browser: { extensionStageHashes: staged?.hashes || null, stagedFixtureHash },
    diagnostics: browserResult?.diagnostics || null,
    measurement: browserResult?.measurement || null,
    executionError,
    derived: verdict.derived,
    failures: verdict.failures,
    passed: verdict.passed,
    limitations: [
      'presented counts successful submissions through the product rAF/GPUCanvas path; Chromium does not acknowledge physical scan-out',
      'this check is separate from the authoritative product target-FPS gate and does not replace it',
      'the estimator latch is exercised only as far as ordinary load during the run induces it',
    ],
  };

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, { encoding: 'utf8', flag: 'wx' });

  console.log(JSON.stringify({
    output: outputPath,
    passed: report.passed,
    display: { original: originalMode.hz, applied: appliedMode.hz,
      restored: restore?.observed?.hz ?? null, restorePassed: restore?.passed === true },
    rafControlHz: report.rafControl?.observedHz ?? null,
    derived: report.derived,
    failures: report.failures,
  }, null, 2));
  if (!report.passed) process.exitCode = 1;
}

const entryPoint = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : null;
if (import.meta.url === entryPoint) await main();
