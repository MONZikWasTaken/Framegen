import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const content = await readFile(new URL('../extension/content.js', import.meta.url), 'utf8');

function functionSource(name) {
  const start = content.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  const signatureEnd = content.indexOf(') {', start);
  assert.notEqual(signatureEnd, -1, `missing function body ${name}`);
  const body = signatureEnd + 2;
  let depth = 0;
  for (let index = body; index < content.length; index++) {
    if (content[index] === '{') depth++;
    if (content[index] === '}' && --depth === 0) return content.slice(start, index + 1);
  }
  throw new Error(`unterminated function ${name}`);
}

function evaluatePoolDimensions({ source, backing, sr, f16 }) {
  const context = {
    cfg: { sr },
    sys: { f16 },
    overlay: { width: backing[0], height: backing[1] },
    videoEl: { videoWidth: source[0], videoHeight: source[1] },
    result: null,
  };
  vm.runInNewContext(`${functionSource('needsNeuralUpscale')}\n${functionSource('poolDims')}\n`
    + 'result = Array.from(poolDims());', context);
  return JSON.parse(JSON.stringify(context.result));
}

function evaluateActiveSrCost({ source, backing, sr, f16, measuredCost = 2.5 }) {
  const context = {
    cfg: { sr },
    sys: { f16 },
    overlay: { width: backing[0], height: backing[1] },
    texW: source[0], texH: source[1], videoEl: null,
    srCostMs: measuredCost,
    result: null,
  };
  vm.runInNewContext(`${functionSource('needsNeuralUpscale')}\n${functionSource('activeSrCostMs')}\n`
    + 'result = activeSrCostMs();', context);
  return context.result;
}

test('native passthrough remains authoritative in frame and pump loops', () => {
  const position = functionSource('positionOverlay');
  const frame = functionSource('onFrame');
  const pump = functionSource('pumpBody');
  const present = functionSource('present');
  const reconcile = functionSource('reconcilePresentationMode');
  const native = functionSource('useNativePassthrough');

  assert.match(position, /const canvasActive = needsCanvasPresentation\(\)/);
  assert.match(position, /visibility = supported && canvasActive \? 'visible' : 'hidden'/);
  assert.match(frame, /recordBenchSource\([\s\S]*?if \(!needsCanvasPresentation\(\)\)[\s\S]*?return;[\s\S]*?processingFrame = true/);
  assert.match(pump, /if \(!needsCanvasPresentation\(\)\)[\s\S]*?return;[\s\S]*?Cadence\.selectDuePresentation/);
  assert.match(present,
    /^function present\([^]*?if \(!tex \|\| !needsCanvasPresentation\(\) \|\| !overlayFitSupported\) return;/);
  assert.match(reconcile, /resetOutputCadence\(true\)[\s\S]*?useNativePassthrough\(\)/);
  assert.match(native, /if \(nativePassthroughActive\) return true/,
    'high-refresh callbacks must not repeat allocations or DOM writes in native mode');
});

test('unsupported HDR and SR requests stay on the native compositor', () => {
  const needsCanvas = functionSource('needsCanvasPresentation');
  const check = (cfg, sys) => {
    const context = { cfg, sys, result: null };
    vm.runInNewContext(`${needsCanvas}\nresult = needsCanvasPresentation();`, context);
    return context.result;
  };
  assert.equal(check({ fg: false, sr: true, hdr: false, sharpness: 0, compare: false },
    { f16: false, hdrOk: false }), false);
  assert.equal(check({ fg: false, sr: false, hdr: true, sharpness: 0, compare: false },
    { f16: true, hdrOk: false }), false);
  assert.equal(check({ fg: false, sr: true, hdr: false, sharpness: 0, compare: false },
    { f16: true, hdrOk: false }), true);
  assert.equal(check({ fg: true, sr: false, hdr: false, sharpness: 0, compare: false },
    { f16: false, hdrOk: false }), true);
});

test('TinySR receives native pixels only when the canvas genuinely adds resolution', () => {
  assert.deepEqual(evaluatePoolDimensions({
    source: [640, 360], backing: [1280, 720], sr: true, f16: true,
  }), [640, 360]);
  assert.deepEqual(evaluatePoolDimensions({
    source: [640, 360], backing: [1280, 720], sr: false, f16: true,
  }), [1280, 720]);
  assert.deepEqual(evaluatePoolDimensions({
    source: [640, 360], backing: [1280, 720], sr: true, f16: false,
  }), [1280, 720]);
  assert.deepEqual(evaluatePoolDimensions({
    source: [1920, 1080], backing: [1280, 720], sr: true, f16: true,
  }), [1280, 720]);
  assert.deepEqual(evaluatePoolDimensions({
    source: [3840, 2160], backing: [1920, 1080], sr: true, f16: true,
  }), [1920, 1080]);
  assert.equal(evaluateActiveSrCost({
    source: [640, 360], backing: [1280, 720], sr: true, f16: true,
  }), 2.5);
  assert.equal(evaluateActiveSrCost({
    source: [1920, 1080], backing: [1920, 1080], sr: true, f16: true,
  }), 0);
});

test('mid-frame pool starts small, grows under measured pressure, and resets by size', () => {
  const constants = content.match(/const MIN_MID_TEXTURES = 8;\s*const MAX_MID_TEXTURES = Cadence\.MAX_PENDING_PRESENTATIONS;\s*const MID_TEXTURE_STEPS = \[[^\]]+\];/);
  assert.ok(constants, 'missing adaptive mid-pool constants');
  const script = `
    const Cadence = { MAX_PENDING_PRESENTATIONS: 24 };
    ${constants[0]}
    let midTexs = [], midIdx = 0, poolGen = 1, texW = 1920, texH = 1080;
    let presentationTextureGeneration = 0;
    let lastPresentedSourceTex = null, lastPresentedTex = null;
    const queued = new Set();
    const destroyed = [];
    const blitBg = new Map();
    const GPUTextureUsage = { TEXTURE_BINDING: 1, STORAGE_BINDING: 2 };
    const device = { createTexture(spec) {
      return { width: spec.size[0], height: spec.size[1], label: spec.label,
        destroy() { destroyed.push(this.label); } };
    } };
    function texQueued(texture) { return queued.has(texture); }
    ${functionSource('ensureMidTextures')}
    ${functionSource('acquireMidTexture')}
    ensureMidTextures(1920, 1080);
    const initial = midTexs.length;
    for (const texture of midTexs) queued.add(texture);
    const grownTexture = acquireMidTexture();
    const grown = midTexs.length;
    queued.clear();
    const reusedTexture = acquireMidTexture();
    const reused = midTexs.length;
    ensureMidTextures(1280, 720);
    result = { initial, grown, reused, grownLabel: grownTexture.label,
      reusedLabel: reusedTexture.label, resized: midTexs.length,
      dimensions: [midTexs[0].width, midTexs[0].height], destroyed: destroyed.length };
  `;
  const context = { result: null };
  vm.runInNewContext(script, context);
  assert.deepEqual(JSON.parse(JSON.stringify(context.result)), {
    initial: 8,
    grown: 12,
    reused: 12,
    grownLabel: 'fcmid1_8',
    reusedLabel: 'fcmid1_9',
    resized: 8,
    dimensions: [1280, 720],
    destroyed: 12,
  });
});

test('runtime rebuild leaves mid textures for the current presentation pool to restore', () => {
  const build = functionSource('buildRuntime');
  const ensureFrames = functionSource('ensureFrameTextures');
  assert.match(build, /midTexs\.forEach\(t => t\.destroy\(\)\);\s*midTexs = \[\];\s*midIdx = 0;/);
  assert.doesNotMatch(build, /for \(let i = 0; i < 24; i\+\+\)/);
  assert.match(ensureFrames,
    /frameTex\.length >= requiredCount\) \{\s*ensureMidTextures\(w, h\);\s*return;/);
});

test('private extension status exposes presentation evidence without changing the UI', () => {
  const statusHandler = content.slice(content.indexOf("if (msg && msg.type === 'fcStatus')"),
    content.indexOf("if (msg && msg.type === 'fcToggle')"));
  assert.match(statusHandler, /presentationMode: needsCanvasPresentation\(\) \? 'canvas' : 'native'/);
  assert.match(statusHandler, /presentationPool:\s*\{[\s\S]*?width: texW,[\s\S]*?generatedTextures: midTexs\.length/);
  assert.match(statusHandler,
    /superResolution:\s*\{ requested: cfg\.sr, ready: !!sr, scale: sr\?\.scale[\s\S]*?processedFrames: srProcessedFrames/);
  assert.match(statusHandler,
    /frameGeneration:\s*\{ requested: cfg\.fg, prepCalls: diag\.prepCalls,[\s\S]*?inferenceCalls: diag\.inferenceCalls/);
});

test('deferred overlay repaint uses the pre-SR source and rejects overwritten textures', () => {
  const configure = functionSource('configureOverlay');
  const present = functionSource('present');
  const capture = functionSource('captureFrame');
  const submitMid = functionSource('submitMid');
  assert.match(configure, /const configurationGeneration = \+\+overlayConfigurationGeneration/);
  assert.match(configure, /const repaintTex = lastPresentedSourceTex/);
  assert.match(configure, /const textureGeneration = presentationTextureGeneration/);
  assert.match(configure, /const textureVersion = presentationTextureVersions\.get\(repaintTex\) \|\| 0/);
  assert.match(configure,
    /configurationGeneration !== overlayConfigurationGeneration[\s\S]*?textureGeneration !== presentationTextureGeneration[\s\S]*?textureVersion !== \(presentationTextureVersions\.get\(repaintTex\) \|\| 0\)[\s\S]*?repaintTex !== lastPresentedSourceTex/);
  assert.match(present, /const sourceTex = tex/);
  assert.match(present, /lastPresentedSourceTex = sourceTex;\s*lastPresentedTex = tex/);
  assert.match(capture, /markPresentationTextureWrite\(dst\)/);
  assert.match(submitMid, /markPresentationTextureWrite\(out\)[\s\S]*?rt\.runT/);
  assert.match(present, /markPresentationTextureWrite\(out\)[\s\S]*?sr\.process/);
  assert.match(functionSource('markPresentationTextureWrite'),
    /presentationTextureVersions\.set\(tex,[\s\S]*?presentationTextureVersions\.get\(tex\)/);
  assert.doesNotMatch(functionSource('markPresentationTextureWrite'), /presentationTextureGeneration\+\+/,
    'unrelated texture writes must not cancel a valid repaint');
  assert.match(content, /presentationTextureGeneration\+\+;\s*frameTex\.forEach\(t => t\.destroy\(\)\)/);
  assert.match(content,
    /presentationTextureGeneration\+\+;[\s\S]{0,160}?midTexs\.forEach\(t => t\.destroy\(\)\)/);
});

test('FG and SR admission includes interpolation and per-presentation GPU costs', () => {
  const plan = functionSource('currentOutputRatePlan');
  const decide = functionSource('decidePair');
  const present = functionSource('present');
  const tracker = functionSource('trackSrProcessCost');
  const load = functionSource('currentEstimatedGpuLoad');
  assert.match(plan, /presentationCostMs: learnedPresentationCostMs/);
  assert.match(plan, /pairCostMs: learnedPairCostMs/);
  assert.match(plan, /const canMeasureGpuProbe = rt\?\.hasGpuTimestamps === true/);
  assert.match(plan, /gpuProbeActive[\s\S]*?!canMeasureGpuProbe/,
    'a runtime rebuild without timestamps must cancel an active exact-cadence probe');
  assert.match(plan, /!gpuProbeActive && canMeasureGpuProbe && startGpuProbe/,
    'default cost mode must not periodically probe with unmeasurable optimistic costs');
  assert.match(decide,
    /estimatedPairGpuCost\(mids\.length, cadence\.presentations\.length,[\s\S]*?modelMs, presentationMs, pairMs\)/);
  assert.match(decide,
    /estimatedLegacyFactorGpuCost\(n, modelMs, presentationMs,[\s\S]*?pairMs\)/);
  assert.match(decide,
    /estimatedActiveIntervalGpuCost\(n, modelMs, presentationMs, pairMs\)[\s\S]*?sourceIntervalMs/,
    'duplicate-heavy averages must not hide an overloaded active source interval');
  assert.match(present, /sr\.processTimed\(tex, out, tex\.width, tex\.height\)/);
  assert.match(present, /trackSrProcessCost\(key, timedResult\.timing, device\)/);
  assert.match(tracker, /Promise\.resolve\(timingPromise\)/);
  assert.doesNotMatch(content, /queue\.onSubmittedWorkDone\(\)/,
    'additive component estimates must never overlap queue-drain wall timers');
  assert.match(tracker, /updateRollingGpuCost\(srCostSamples, srCostMs, sampleMs\)/);
  assert.match(load, /Math\.max\(averageLoad, burstLoad\)/,
    'fixed-factor warnings must expose duplicate-heavy burst overloads');
  assert.match(functionSource('updateWarn'), /const load = currentEstimatedGpuLoad\(\)/);

  const context = { uniqueIntervalMs: 100, result: null };
  vm.runInNewContext(`${functionSource('estimatedPairGpuCost')}\n`
    + `${functionSource('estimatedLegacyFactorGpuCost')}\n`
    + `${functionSource('estimatedActiveIntervalGpuCost')}\n`
    + 'result = { average: estimatedLegacyFactorGpuCost(2, 2, 3, 20, 5), '
    + 'burst: estimatedActiveIntervalGpuCost(6, 6, 1, 5) };', context);
  assert.equal(context.result.average, 16,
    'pair prep, two decoded anchors, one generated mid and every SR pass must be charged');
  assert.equal(context.result.burst, 41,
    'active-interval admission must charge one anchor plus every generated mid');
});
