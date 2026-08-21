// Framegen content script: real-time frame interpolation for any <video> on the page.
// GPU-resident pipeline (own WebGPU runtime, weights bundled): video -> texture ->
// interpolation -> overlay canvas (sibling of the video; site controls stay on top).
// DRM (EME) video produces black frames - nothing any extension can do about that.
(() => {
  'use strict';
  if (window.__framegen) return;
  window.__framegen = true;

  const Cadence = globalThis.FramegenCadence;
  if (!Cadence) throw new Error('Framegen cadence helper was not loaded');
  const Profiles = globalThis.FramegenProfiles;
  if (!Profiles) throw new Error('Framegen profile store was not loaded');

  const DELAY_MS = 60;
  // runtime tiles are 16x16 - model dims must be /16 (1088, not 1080; the ~0.7%
  // vertical stretch at present time is invisible)
  const SIZES = { 288: [512, 288], 360: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1088] };

  // Deliberately unreachable from ordinary pages. The product-path gate uses a
  // fresh loopback URL and a per-run token; no benchmark state exists otherwise.
  const PRODUCT_BENCH = (() => {
    try {
      const params = new URL(location.href).searchParams;
      const token = params.get('framegenBenchToken') || '';
      const loopback = location.hostname === '127.0.0.1' || location.hostname === 'localhost';
      return window === window.top && location.protocol === 'http:' && loopback
        && params.get('framegenProductBench') === '1' && /^[a-f0-9]{32,128}$/.test(token)
        ? { token } : null;
    } catch { return null; }
  })();

  // ---------- settings (chrome.storage.local, live-applied) ----------
  // factor stores output-rate mode: auto/display/custom target, or a fixed 2..6 source multiplier.
  // model: weight set key from MODELS. v7s is the default: faster (2.57ms vs
  // 3.05ms @480p, 3.75 vs 5.51 @720p) at equal-or-better quality. v6 stays
  // selectable; users with a saved choice keep it.
  const MODELS = { v6: 'rt_tfact2', v7s: 'rt_v7s' };
  const FPS_LIMIT_STEPS = Profiles.FPS_LIMIT_PRESETS;
  const cfg = { factor: 'auto', targetFps: 120, fpsLimit: null, anime: true, debug: false, res: 480, hoverReveal: true, compare: false,
    fg: true, sr: false, hdr: false, showFps: true, showWatermark: true, showWarnings: true, guard: true, model: 'v7s' };
  function sanitizeCfg() {
    const legacyTarget = cfg.factor === 'fps60' ? 60 : cfg.factor === 'fps120' ? 120 : null;
    cfg.factor = Cadence.sanitizeOutputRate(cfg.factor);
    cfg.targetFps = Cadence.sanitizeTargetFps(legacyTarget ?? cfg.targetFps, 120);
    cfg.fpsLimit = canonicalFpsLimit(cfg.fpsLimit);
    if (!MODELS[cfg.model]) cfg.model = 'v7s';
    if (!SIZES[cfg.res]) cfg.res = 480;
    cfg.anime = !!cfg.anime; cfg.debug = !!cfg.debug;
    cfg.hoverReveal = !!cfg.hoverReveal; cfg.compare = !!cfg.compare;
    cfg.fg = !!cfg.fg; cfg.sr = !!cfg.sr; cfg.hdr = !!cfg.hdr;
    cfg.showFps = !!cfg.showFps; cfg.showWatermark = cfg.showWatermark !== false;
    cfg.showWarnings = cfg.showWarnings !== false; cfg.guard = !!cfg.guard;
  }
  let settleSettingsReady = () => {};
  let settingsSettled = !!PRODUCT_BENCH;
  const settingsReady = PRODUCT_BENCH ? Promise.resolve() : new Promise((resolve) => {
    settleSettingsReady = () => {
      if (settingsSettled) return;
      settingsSettled = true;
      resolve();
    };
  });
  if (!PRODUCT_BENCH) try {
    // Runtime creation waits for this first snapshot. Otherwise hover preload can
    // compile the defaults and then label them with the saved model/resolution.
    chrome.storage.local.get(cfg, v => {
      try {
        const storedFpsLimit = v.fpsLimit;
        Object.assign(cfg, v); sanitizeCfg(); syncPanel();
        if (storedFpsLimit !== cfg.fpsLimit) {
          chrome.storage.local.set({ fpsLimit: cfg.fpsLimit });
        }
      }
      finally { settleSettingsReady(); }
    });
    // settings changed in another tab/frame apply here live
    chrome.storage.onChanged.addListener((ch, area) => {
      if (area !== 'local') return;
      const profileStoreChanged = Object.hasOwn(ch, Profiles.STORE_KEY);
      let runtimeChanged = false;
      const previousFactor = cfg.factor;
      const previousTargetFps = cfg.targetFps;
      const previousFpsLimit = cfg.fpsLimit;
      const incomingFpsLimit = ch.fpsLimit?.newValue;
      const previousFg = cfg.fg;
      const previousCompare = cfg.compare;
      for (const k in ch) {
        if (!(k in cfg)) continue;
        if ((k === 'res' || k === 'model' || k === 'guard') && cfg[k] !== ch[k].newValue) {
          runtimeChanged = true;
        }
        cfg[k] = ch[k].newValue;
      }
      sanitizeCfg(); syncPanel();
      if (incomingFpsLimit !== undefined && incomingFpsLimit !== cfg.fpsLimit) {
        chrome.storage.local.set({ fpsLimit: cfg.fpsLimit });
      }
      if (Object.hasOwn(ch, 'showFps')) syncHudVisibility();
      if (Object.hasOwn(ch, 'showWarnings') && !cfg.showWarnings) hideWarnings();
      if (profileStoreChanged) {
        loadPanelProfiles(ch[Profiles.STORE_KEY].newValue).catch(e => log('profile sync', e));
      }
      if (cfg.factor !== previousFactor || cfg.targetFps !== previousTargetFps
          || cfg.fpsLimit !== previousFpsLimit || cfg.fg !== previousFg) {
        delayMs = DELAY_MS;
        resetOutputCadence(true);
      }
      if (previousCompare && !cfg.compare) cmpRing = [];
      if ('hdr' in ch) configureOverlay();
      if ('sr' in ch && cfg.sr && device) ensureSR().catch(e => log('sr sync', e));
      if (runtimeChanged && running && videoEl && !toggling) {
        toggling = true;
        switchRes().catch(e => log('runtime settings sync', e)).finally(() => { toggling = false; });
      }
    });
  } catch { settleSettingsReady(); /* storage unavailable in some frames */ }
  function saveCfg() {
    try { chrome.storage.local.set(cfg); } catch {}
  }
  let panelProfileStore = null;

  function fpsLimitStepIndex(value) {
    return Profiles.fpsLimitPresetIndex(value);
  }

  function canonicalFpsLimit(value) {
    return Profiles.sanitizeFpsLimit(value);
  }

  function fpsLimitFromSlider(input) {
    const index = Math.max(0, Math.min(FPS_LIMIT_STEPS.length - 1,
      Math.round(Number(input.value) || 0)));
    return FPS_LIMIT_STEPS[index];
  }

  function updateRateSliderPresentation(input = panel?.querySelector('#fcTargetFps')) {
    if (!input || !panel) return;
    const isLimit = cfg.factor === 'auto';
    const value = isLimit
      ? fpsLimitFromSlider(input)
      : Cadence.sanitizeTargetFps(input.value, cfg.targetFps);
    const output = panel.querySelector('#fcTargetFpsValue');
    const label = value === null ? 'Unlimited' : `${Number(value.toFixed(2))} FPS`;
    if (output) output.value = label;
    input.setAttribute('aria-valuetext', label);
    const parsedMin = Number(input.min);
    const parsedMax = Number(input.max);
    const min = Number.isFinite(parsedMin) ? parsedMin : 2;
    const max = Number.isFinite(parsedMax) ? parsedMax : 1000;
    const sliderValue = Number(input.value);
    const ratio = Math.max(0, Math.min(1,
      (sliderValue - min) / Math.max(1, max - min)));
    const thumbOffset = Number((7.5 * (1 - 2 * ratio)).toFixed(3));
    input.style.setProperty('--fc-fill', `calc(${ratio * 100}% + ${thumbOffset}px)`);
  }

  function renderRateScale(scale, marks) {
    scale.replaceChildren(...marks.map(({ label, position }) => {
      const mark = document.createElement('span');
      mark.textContent = label;
      mark.style.setProperty('--fc-mark-position', `${position}%`);
      const offset = Number((7.5 * (1 - 2 * position / 100)).toFixed(3));
      mark.style.setProperty('--fc-mark-offset', `${offset}px`);
      return mark;
    }));
  }

  function syncRateSlider() {
    const input = panel?.querySelector('#fcTargetFps');
    if (!input) return;
    const control = input.closest('.fc-target-control');
    const visible = cfg.factor === 'auto' || cfg.factor === 'target';
    control.hidden = !visible;
    if (!visible) return;

    const title = panel.querySelector('#fcRateSliderTitle');
    const hint = panel.querySelector('#fcRateSliderHint');
    const scale = panel.querySelector('#fcRateSliderScale');
    if (cfg.factor === 'auto') {
      title.textContent = 'FPS limit';
      hint.textContent = 'Common rates · Auto may run lower';
      input.min = '0';
      input.max = String(FPS_LIMIT_STEPS.length - 1);
      input.step = '1';
      input.value = String(fpsLimitStepIndex(cfg.fpsLimit));
      input.setAttribute('aria-label', 'FPS limit');
      renderRateScale(scale, [15, 60, 120, 240, null].map(value => ({
        label: value === null ? '∞' : String(value),
        position: fpsLimitStepIndex(value) / (FPS_LIMIT_STEPS.length - 1) * 100,
      })));
    } else {
      title.textContent = 'Custom FPS';
      hint.textContent = 'Exact target · minimum 2× source';
      input.min = '2';
      input.max = '1000';
      input.step = '0.01';
      input.value = String(cfg.targetFps);
      input.setAttribute('aria-label', 'Custom FPS');
      renderRateScale(scale, [
        { label: '2', position: 0 },
        { label: '500', position: 49.9 },
        { label: '1000', position: 100 },
      ]);
    }
    updateRateSliderPresentation(input);
  }

  function syncPanelProfileSelection() {
    const select = panel?.querySelector('#fcProfile');
    if (!select || !panelProfileStore) return;
    try {
      select.value = Profiles.findMatchingProfileId(panelProfileStore, cfg) || '';
    } catch {
      select.value = '';
    }
  }

  function renderPanelProfiles() {
    const select = panel?.querySelector('#fcProfile');
    if (!select || !panelProfileStore) return;
    const current = document.createElement('option');
    current.value = '';
    current.textContent = 'Current settings';
    const children = [current];
    const profiles = Profiles.profileList(panelProfileStore);
    if (profiles.length) {
      const group = document.createElement('optgroup');
      group.label = 'My profiles';
      for (const profile of profiles) {
        const option = document.createElement('option');
        option.value = profile.id;
        option.textContent = profile.name;
        group.append(option);
      }
      children.push(group);
    }
    select.replaceChildren(...children);
    select.disabled = false;
    syncPanelProfileSelection();
  }

  async function loadPanelProfiles(rawStore) {
    if (!panel) return;
    const select = panel.querySelector('#fcProfile');
    try {
      let candidate = rawStore;
      if (arguments.length === 0) {
        const snapshot = await chrome.storage.local.get(Profiles.STORE_KEY);
        candidate = snapshot[Profiles.STORE_KEY];
      }
      const loaded = Profiles.loadStore(candidate, cfg);
      panelProfileStore = loaded.store;
      renderPanelProfiles();
      if (loaded.needsWrite) {
        await chrome.storage.local.set({ [Profiles.STORE_KEY]: loaded.store });
      }
    } catch (error) {
      panelProfileStore = null;
      if (select) {
        const unavailable = document.createElement('option');
        unavailable.textContent = 'Profiles unavailable';
        select.replaceChildren(unavailable);
        select.disabled = true;
      }
      log('profiles', error);
    }
  }

  async function applyPanelProfile(profileId) {
    if (!profileId || !panelProfileStore) {
      syncPanelProfileSelection();
      return;
    }
    const profile = Profiles.getProfile(panelProfileStore, profileId);
    if (!profile) {
      syncPanelProfileSelection();
      return;
    }
    const nextStore = Profiles.setLastAppliedProfile(panelProfileStore, profile.id);
    panelProfileStore = nextStore;
    await chrome.storage.local.set(Profiles.toStoragePayload(nextStore, profile.settings));
  }

  function syncPanel() {
    if (!panel) return;
    panel.querySelector('#fcFactor').value = String(cfg.factor);
    syncRateSlider();
    panel.querySelector('#fcRes').value = String(cfg.res);
    panel.querySelector('#fcFG').checked = cfg.fg;
    panel.querySelector('#fcSR').checked = cfg.sr;
    panel.querySelector('#fcShowFps').checked = cfg.showFps;
    panel.querySelector('#fcWatermark').checked = cfg.showWatermark;
    panel.querySelector('#fcWarnings').checked = cfg.showWarnings;
    const hd = panel.querySelector('#fcHDR');
    hd.checked = cfg.hdr;
    if (!sys.hdrOk) { hd.disabled = true; hd.style.opacity = '.35'; }
    syncPanelProfileSelection();
  }

  let rt = null, rtRes = 0, rtModel = '', rtGuard = null, rtGeneration = 0;
  let device = null, recoveringDevice = false, deviceRecoveryEpoch = 0;
  let deviceRecoveryVideo = null, videoEl = null;
  let overlay = null, overlayCtx = null, blitPipe = null, blitSampler = null;
  const blitBg = new Map();
  let frameTex = [], frameIdx = 0, texW = 0, texH = 0, lastTex = null;
  let midTexs = [], midIdx = 0;
  let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupSampler = null;
  let dedupReads = [], dedupReadIdx = 0; // readback ring: classifies overlap now
  let queue = [], running = false, processingFrame = false;
  let videoFrameCallbackId = null, videoFrameCallbackVideo = null;
  let pumpRafId = null, playbackLoopEpoch = 0;
  let pairSeq = 0; // generation counter for in-flight classify continuations
  let pairDecisionChain = Promise.resolve();
  const classifyTextureUse = new Map();
  let hzNext = 0, hzPhaseMs = 0; // exact absolute target clock + decoded-media phase
  let intervalMs = 42, uniqueIntervalMs = 42, lastArrival = 0, lastUniqueTs = 0;
  let decodedIntervalMs = 42, lastDecodedMediaTimeMs = null, lastDecodedPresentedFrames = null;
  const decodedIntervalSamples = [];
  const decodedCadenceShift = { intervalMs: 0, samples: 0, direction: 0 };
  let msAvg = 0, lastMidCostAt = 0, dropped = 0, dups = 0, cuts = 0, fpsWin = [], effN = 2, lastStat = null;
  const midCostSamples = [];
  let btn = null, gear = null, hud = null, panel = null, statsTimer = 0;
  let bar = null, barSeeking = false, wm = null;
  let rafMs = 0, lastPumpT = 0, warnEl = null, overSince = 0;
  let splitEl = null, splitX = 0.5, toggling = false, autoSkipT = 0;
  let delayMs = DELAY_MS, dropWin = [], switching = false, preloadFailT = -1e9;
  let schedT = 0, rafFloor = 100, nextUiUpdateAt = 0, motionAvg = 0, lateAvg = 0;

  function syncHudVisibility() {
    if (!hud) return;
    const text = hud.textContent || '';
    const critical = text.startsWith('FG:') || text.startsWith('FG error:') || text.startsWith('FC:');
    hud.style.display = critical || (running && (cfg.debug || cfg.showFps)) ? 'block' : 'none';
  }
  const UI_UPDATE_INTERVAL_MS = 1000 / 15;
  const refreshEstimate = { floorMs: rafFloor, slowCandidateMs: 0, slowSamples: 0, stableSamples: 0 };
  let outputRatePlanKey = '';
  let outputRatePlanIdentity = null;
  let outputRatePlanSnapshot = null;
  let outputRatePlanCandidate = null;
  let outputRatePlanCandidateSamples = 0;
  const OUTPUT_RATE_TRANSITION_SAMPLES = 16;
  let gpuProbeActive = false, gpuProbeStartedAt = 0, gpuProbeKey = '';
  let lastVr = null, lastVrT = 0; // video rect cached by pump's UI tick - onFrame reuses it
  let barH = 0; // control-bar height, measured once (content is static)
  let overlayFit = 'fill', overlayFitRequested = 'fill', overlayFitSupported = true;
  let overlaySourceWidth = 0, overlaySourceHeight = 0;
  let autoPenalty = 0, penaltyT = 0, dropPressure = 0, lastPressureT = 0;
  const diag = {
    sourceVideo: null, sourceVideoId: 0, sourceChangedAtFullscreen: false,
    mediaTime: null, presentedFrames: null, repeatedMediaCallbacks: 0,
    rvfcCalls: 0, rafCalls: 0, inferenceCalls: 0, prepCalls: 0, presentCalls: 0,
    duplicateSkips: 0, cutSkips: 0, fullscreenEvents: 0,
    pumpStarts: 0, sourceLoopStarts: 0, loopStops: 0, sampleAt: performance.now(),
    sample: null,
  };
  const sys = { gpu: '-', f16: false, hdrOk: false, hdrOn: false };
  try { sys.hdrOk = !!(window.matchMedia && matchMedia('(dynamic-range: high)').matches); } catch {}

  const benchSessionErrors = [];
  let benchTelemetry = null, benchEpoch = 0, benchAppliedTune = null;

  function resetDecodedSourceCadence() {
    decodedIntervalMs = 42;
    lastDecodedMediaTimeMs = null;
    lastDecodedPresentedFrames = null;
    decodedIntervalSamples.length = 0;
    decodedCadenceShift.intervalMs = 0;
    decodedCadenceShift.samples = 0;
    decodedCadenceShift.direction = 0;
  }

  function updateDecodedSourceCadence(metadata, wallIntervalMs) {
    const mediaTimeMs = Number(metadata?.mediaTime) * 1000;
    const presentedFrames = Number(metadata?.presentedFrames);
    let sampleMs = wallIntervalMs;
    let frameDelta = 1;
    if (Number.isFinite(mediaTimeMs)) {
      if (lastDecodedMediaTimeMs !== null) {
        sampleMs = mediaTimeMs - lastDecodedMediaTimeMs;
        if (Number.isFinite(presentedFrames) && lastDecodedPresentedFrames !== null) {
          frameDelta = presentedFrames - lastDecodedPresentedFrames;
          if (Number.isInteger(frameDelta) && frameDelta > 1) sampleMs /= frameDelta;
        }
      }
      lastDecodedMediaTimeMs = mediaTimeMs;
    }
    if (Number.isFinite(presentedFrames)) lastDecodedPresentedFrames = presentedFrames;
    if (!Number.isFinite(sampleMs) || sampleMs <= 0.5 || sampleMs >= 2000) {
      decodedCadenceShift.intervalMs = 0;
      decodedCadenceShift.samples = 0;
      decodedCadenceShift.direction = 0;
      return frameDelta;
    }
    const update = Cadence.updateSourceInterval({
      intervalMs: decodedIntervalMs,
      samples: decodedIntervalSamples,
      transition: decodedCadenceShift,
    }, sampleMs);
    decodedIntervalMs = update.intervalMs;
    if (update.transitioned) {
      resetOutputCadence(true);
      delayMs = DELAY_MS;
      lastTex = null;
      schedT = 0;
      lastUniqueTs = 0;
      const playbackRate = Math.max(0.01,
        Math.abs(Number(videoEl?.playbackRate) || 1));
      intervalMs = decodedIntervalMs / playbackRate;
      uniqueIntervalMs = intervalMs;
    }
    return frameDelta;
  }

  function autoPolicyFactor(midCostMs) {
    const costMs = Number.isFinite(midCostMs) && midCostMs > 0 ? midCostMs : 10;
    let factor = 6;
    while (factor > 2 && (factor - 1) * costMs > uniqueIntervalMs * 0.85) factor--;
    const displayHz = rafMs > 1 ? 1000 / rafMs : 60;
    while (factor > 2 && (1000 / uniqueIntervalMs) * factor > displayHz) factor--;
    factor = Math.max(2, factor - autoPenalty);
    const motionCeiling = motionAvg > 45 ? 2 : motionAvg > 28 ? 3 : motionAvg > 16 ? 4 : 6;
    if (factor > motionCeiling) factor = motionCeiling;
    return {
      factor,
      runnable: (factor - 1) * costMs <= uniqueIntervalMs * 1.1,
    };
  }

  function cappedAutoTargetHz(limit, sourceHz, midCostMs) {
    const policy = autoPolicyFactor(midCostMs);
    if (!policy.runnable) return Math.min(limit, sourceHz);
    const uniqueHz = uniqueIntervalMs > 1 ? 1000 / uniqueIntervalMs : sourceHz;
    // Anime/on-twos still presents every decoded anchor. Only generated mids
    // follow the unique-pair cadence, so n*uniqueHz would undercount the result.
    const policyHz = sourceHz + (policy.factor - 1) * uniqueHz;
    return Math.min(limit, policyHz);
  }

  function currentOutputRatePlan({ midCostMs = null, startGpuProbe = false } = {}) {
    const sourceReady = decodedIntervalSamples.length >= 8;
    const playbackRate = Math.max(0.01, Math.abs(Number(videoEl?.playbackRate) || 1));
    const learnedCostMs = Number.isFinite(midCostMs) && midCostMs > 0
      ? midCostMs
      : (msAvg || 10);
    const context = {
      targetFps: cfg.targetFps,
      sourceHz: sourceReady ? playbackRate * 1000 / decodedIntervalMs : null,
      sourceReady,
      displayReady: refreshEstimate.ready === true || refreshEstimate.stableSamples >= 10,
    };
    const cappedAuto = cfg.factor === 'auto' && cfg.fpsLimit !== null;
    const requestedTargetFps = cappedAuto && context.sourceHz
      ? cappedAutoTargetHz(cfg.fpsLimit, context.sourceHz, learnedCostMs)
      : (cappedAuto ? cfg.fpsLimit : cfg.targetFps);
    let plan = Cadence.resolveOutputRate(cappedAuto ? 'target' : cfg.factor, rafFloor, {
      ...context,
      targetFps: requestedTargetFps,
      midCostMs: learnedCostMs,
      strictCeiling: cappedAuto,
    });
    if (cappedAuto) plan = { ...plan, mode: 'auto' };
    const planProbeKey = `${cfg.factor}:${cfg.targetFps}:${cfg.fpsLimit ?? 'unlimited'}:${Number(plan.minimumHz || 0).toFixed(2)}`;
    const now = performance.now();
    const probeTimeoutMs = sourceReady
      ? Math.max(1500, 2 * (1000 / context.sourceHz) + 250)
      : 1500;
    if (gpuProbeActive
        && (gpuProbeKey !== planProbeKey || now - gpuProbeStartedAt > probeTimeoutMs)) {
      gpuProbeActive = false;
    }
    if (!gpuProbeActive && startGpuProbe && plan.state === 'no-2x-gpu-range'
        && (!gpuProbeStartedAt || now - gpuProbeStartedAt >= 2500)) {
      gpuProbeActive = true;
      gpuProbeStartedAt = now;
      gpuProbeKey = planProbeKey;
    }
    if (!gpuProbeActive) return { ...plan, admissionCostMs: learnedCostMs, gpuProbe: false };

    const probe = Cadence.resolveOutputRate('target', rafFloor, {
      ...context,
      targetFps: plan.minimumHz,
      midCostMs: 0.1,
    });
    return {
      ...probe,
      mode: plan.mode,
      requestedHz: plan.requestedHz,
      state: 'probing-gpu',
      clampReason: 'gpu-probe',
      warning: 'Checking GPU capacity at 2x',
      admissionCostMs: 0.1,
      gpuProbe: true,
    };
  }

  function usesExactCadence() {
    return Cadence.isCadenceMode(cfg.factor)
      || (cfg.factor === 'auto' && cfg.fpsLimit !== null);
  }

  function outputRateLabel() {
    if (cfg.factor === 'auto' && cfg.fpsLimit !== null) {
      return `Auto · max ${Number(cfg.fpsLimit.toFixed(2))} FPS`;
    }
    return Cadence.outputRateLabel(cfg.factor, cfg.targetFps);
  }

  function outputRateIdentity(plan) {
    return {
      state: plan.state,
      clampReason: plan.clampReason || null,
      minimumHz: Number(plan.minimumHz) || 0,
      outputHz: Number(plan.outputHz) || 0,
    };
  }
  function outputRateIdentityChange(previous, nextIdentity) {
    const relativeChange = (left, right) => {
      if (!(left > 0) || !(right > 0)) return left === right ? 0 : Infinity;
      return Math.abs(left - right) / Math.max(left, right);
    };
    return previous.state !== nextIdentity.state
      || previous.clampReason !== nextIdentity.clampReason
      || relativeChange(previous.outputHz, nextIdentity.outputHz) > 0.02;
  }
  function stableOutputRatePlan(plan) {
    const stable = outputRatePlanSnapshot;
    if (!stable) return plan;
    return {
      ...plan,
      state: stable.state,
      clampReason: stable.clampReason,
      // Source cadence is already stabilized independently. Keep its exact 2x
      // floor current without resetting the output clock when output/state did
      // not materially change.
      minimumHz: plan.minimumHz,
      outputHz: stable.outputHz,
      interpolationAllowed: stable.interpolationAllowed,
      clamped: stable.clamped,
      warning: stable.warning,
      gpuProbe: stable.gpuProbe === true,
    };
  }
  function syncOutputRatePlan(plan) {
    if (!usesExactCadence()) return plan;
    const nextIdentity = outputRateIdentity(plan);
    const previous = outputRatePlanIdentity;
    if (!previous) {
      outputRatePlanIdentity = nextIdentity;
      outputRatePlanSnapshot = { ...plan };
      outputRatePlanCandidate = null;
      outputRatePlanCandidateSamples = 0;
      outputRatePlanKey = `${nextIdentity.state}:${nextIdentity.clampReason || ''}`;
      return plan;
    }
    if (!outputRateIdentityChange(previous, nextIdentity)) {
      outputRatePlanCandidate = null;
      outputRatePlanCandidateSamples = 0;
      return stableOutputRatePlan(plan);
    }

    if (outputRatePlanCandidate
        && !outputRateIdentityChange(outputRatePlanCandidate, nextIdentity)) {
      outputRatePlanCandidateSamples++;
    } else {
      outputRatePlanCandidate = nextIdentity;
      outputRatePlanCandidateSamples = 1;
    }
    if (outputRatePlanCandidateSamples < OUTPUT_RATE_TRANSITION_SAMPLES) {
      return stableOutputRatePlan(plan);
    }

    if (benchTelemetry) {
      benchTelemetry.ratePlanResets++;
      benchTelemetry.ratePlanBufferedFrames += queue.reduce((count, entry) =>
        count + (entry.benchEpoch === benchTelemetry.epoch ? 1 : 0), 0);
      benchTelemetry.abandonedMids += curJob
        ? Math.max(0, curJob.ts.length - curJob.next)
        : 0;
    }
    // Already scheduled textures remain valid under the old cadence epoch.
    // Preserve and drain them; only future planning switches to the new clock.
    resetOutputCadence(false);
    outputRatePlanIdentity = nextIdentity;
    outputRatePlanSnapshot = { ...plan };
    outputRatePlanCandidate = null;
    outputRatePlanCandidateSamples = 0;
    outputRatePlanKey = `${nextIdentity.state}:${nextIdentity.clampReason || ''}`;
    return plan;
  }
  const log = (...a) => {
    console.log('[framegen]', ...a);
    if (!PRODUCT_BENCH) return;
    const message = a.map(v => v instanceof Error ? (v.stack || v.message) : String(v)).join(' ');
    if (a.some(v => v instanceof Error)
        || /(^|\s)(error|failed|skipped|runT|prep|classify|decide|pump|fallback|lost)(\s|$)/i.test(message)) {
      if (benchSessionErrors.length < 100) benchSessionErrors.push({ atMs: performance.now(), message });
    }
  };

  function benchPercentile(values, fraction) {
    if (!values.length) return null;
    const ordered = [...values].sort((a, b) => a - b);
    return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
  }
  function benchRate(intervals) {
    if (!intervals.length) return null;
    const total = intervals.reduce((sum, value) => sum + value, 0);
    return total > 0 ? 1000 * intervals.length / total : null;
  }
  function resetBenchTelemetry() {
    if (!PRODUCT_BENCH) throw new Error('product benchmark telemetry is disabled');
    benchTelemetry = {
      epoch: ++benchEpoch,
      resetAtMs: performance.now(),
      sourceCallbacks: 0,
      sourceProcessed: 0,
      sourceBusySkipped: 0,
      sourcePresentedFrameGaps: 0,
      sourceMetadataDuplicates: 0,
      rafCallbacks: 0,
      scheduled: 0,
      scheduledSource: 0,
      scheduledMid: 0,
      presented: 0,
      presentedSource: 0,
      presentedMid: 0,
      dropped: 0,
      droppedSource: 0,
      droppedMid: 0,
      queued: 0,
      queueHighWater: 0,
      sourcePoolExhausted: 0,
      midPoolExhausted: 0,
      classificationSuperseded: 0,
      pairPlans: 0,
      skippedPairs: 0,
      plannedMids: 0,
      ratePlanResets: 0,
      ratePlanBufferedFrames: 0,
      abandonedMids: 0,
      sourceGapHistoryBreaks: 0,
      plannedFactorHistogram: {},
      sourceCallbackIntervalsMs: { values: new Float64Array(20000), length: 0 },
      sourceMediaIntervalsMs: { values: new Float64Array(20000), length: 0 },
      rafIntervalsMs: { values: new Float64Array(20000), length: 0 },
      pumpWorkMs: { values: new Float64Array(20000), length: 0 },
      sourceWorkMs: { values: new Float64Array(20000), length: 0 },
      lateMs: { values: new Float64Array(20000), length: 0 },
      lastSourceAtMs: null,
      lastSourceMediaTimeMs: null,
      lastSourcePresentedFrames: null,
      lastRafAtMs: null,
      sampleOverflow: false,
    };
    return snapshotBenchTelemetry();
  }
  function pushBenchSample(target, value) {
    if (!Number.isFinite(value) || value < 0) {
      if (benchTelemetry) benchTelemetry.sampleOverflow = true;
      return;
    }
    if (target.length >= target.values.length) {
      if (benchTelemetry) benchTelemetry.sampleOverflow = true;
      return;
    }
    target.values[target.length++] = value;
  }
  function benchSamples(target) {
    return Array.from(target.values.subarray(0, target.length));
  }
  function recordBenchSource(now, metadata) {
    const telemetry = benchTelemetry;
    if (!telemetry) return;
    telemetry.sourceCallbacks++;
    if (telemetry.lastSourceAtMs !== null) {
      pushBenchSample(telemetry.sourceCallbackIntervalsMs, now - telemetry.lastSourceAtMs);
    }
    telemetry.lastSourceAtMs = now;
    const mediaTimeMs = Number(metadata?.mediaTime) * 1000;
    if (Number.isFinite(mediaTimeMs)) {
      if (telemetry.lastSourceMediaTimeMs !== null) {
        pushBenchSample(telemetry.sourceMediaIntervalsMs, mediaTimeMs - telemetry.lastSourceMediaTimeMs);
      }
      telemetry.lastSourceMediaTimeMs = mediaTimeMs;
    }
    const presentedFrames = Number(metadata?.presentedFrames);
    if (Number.isFinite(presentedFrames)) {
      if (telemetry.lastSourcePresentedFrames !== null) {
        const delta = presentedFrames - telemetry.lastSourcePresentedFrames;
        if (delta <= 0) telemetry.sourceMetadataDuplicates++;
        else if (delta > 1) telemetry.sourcePresentedFrameGaps += delta - 1;
      }
      telemetry.lastSourcePresentedFrames = presentedFrames;
    }
  }
  function recordBenchRaf(now) {
    const telemetry = benchTelemetry;
    if (!telemetry) return;
    telemetry.rafCallbacks++;
    if (telemetry.lastRafAtMs !== null) {
      pushBenchSample(telemetry.rafIntervalsMs, now - telemetry.lastRafAtMs);
    }
    telemetry.lastRafAtMs = now;
  }
  function schedulePresentation(tex, at, mid) {
    const entry = { tex, at, mid };
    const telemetry = benchTelemetry;
    if (telemetry) {
      entry.benchEpoch = telemetry.epoch;
      telemetry.scheduled++;
      if (mid) telemetry.scheduledMid++;
      else telemetry.scheduledSource++;
    }
    const evicted = Cadence.enqueuePresentation(queue, entry);
    if (telemetry) {
      telemetry.queued++;
      if (evicted && evicted.benchEpoch === telemetry.epoch) {
        telemetry.queued--;
        telemetry.dropped++;
        if (evicted.mid) telemetry.droppedMid++;
        else telemetry.droppedSource++;
      }
      telemetry.queueHighWater = Math.max(telemetry.queueHighWater, telemetry.queued);
    }
    if (evicted) {
      dropped++;
      dropWin.push(performance.now());
      if (dropWin.length > 512) dropWin.splice(0, dropWin.length - 512);
    }
  }
  function recordBenchPresentationBatch(lastIndex, now) {
    const telemetry = benchTelemetry;
    if (!telemetry) return;
    for (let i = 0; i <= lastIndex; i++) {
      const entry = queue[i];
      if (entry.benchEpoch !== telemetry.epoch) continue;
      telemetry.queued--;
      if (i === lastIndex) {
        telemetry.presented++;
        if (entry.mid) telemetry.presentedMid++;
        else telemetry.presentedSource++;
        pushBenchSample(telemetry.lateMs, Math.max(0, now - entry.at));
      } else {
        telemetry.dropped++;
        if (entry.mid) telemetry.droppedMid++;
        else telemetry.droppedSource++;
      }
    }
  }
  function recordBenchPairPlan(factor, run, plannedMids) {
    const telemetry = benchTelemetry;
    if (!telemetry) return;
    telemetry.pairPlans++;
    telemetry.plannedFactorHistogram[factor] = (telemetry.plannedFactorHistogram[factor] || 0) + 1;
    if (!run) telemetry.skippedPairs++;
    telemetry.plannedMids += plannedMids;
  }
  function snapshotBenchTelemetry() {
    const telemetry = benchTelemetry;
    if (!telemetry) return null;
    const now = performance.now();
    const ratePlan = stableOutputRatePlan(currentOutputRatePlan());
    const sourceCallbackIntervalsMs = benchSamples(telemetry.sourceCallbackIntervalsMs);
    const sourceMediaIntervalsMs = benchSamples(telemetry.sourceMediaIntervalsMs);
    const rafIntervalsMs = benchSamples(telemetry.rafIntervalsMs);
    const pumpWorkMs = benchSamples(telemetry.pumpWorkMs);
    const sourceWorkMs = benchSamples(telemetry.sourceWorkMs);
    const lateMs = benchSamples(telemetry.lateMs);
    return {
      schemaVersion: 1,
      epoch: telemetry.epoch,
      durationMs: now - telemetry.resetAtMs,
      counters: {
        sourceCallbacks: telemetry.sourceCallbacks,
        sourceProcessed: telemetry.sourceProcessed,
        sourceBusySkipped: telemetry.sourceBusySkipped,
        sourcePresentedFrameGaps: telemetry.sourcePresentedFrameGaps,
        sourceMetadataDuplicates: telemetry.sourceMetadataDuplicates,
        rafCallbacks: telemetry.rafCallbacks,
        scheduled: telemetry.scheduled,
        scheduledSource: telemetry.scheduledSource,
        scheduledMid: telemetry.scheduledMid,
        presented: telemetry.presented,
        presentedSource: telemetry.presentedSource,
        presentedMid: telemetry.presentedMid,
        dropped: telemetry.dropped,
        droppedSource: telemetry.droppedSource,
        droppedMid: telemetry.droppedMid,
        pending: telemetry.queued,
        queueHighWater: telemetry.queueHighWater,
        sourcePoolExhausted: telemetry.sourcePoolExhausted,
        midPoolExhausted: telemetry.midPoolExhausted,
        classificationSuperseded: telemetry.classificationSuperseded,
        pairPlans: telemetry.pairPlans,
        skippedPairs: telemetry.skippedPairs,
        plannedMids: telemetry.plannedMids,
        ratePlanResets: telemetry.ratePlanResets,
        ratePlanBufferedFrames: telemetry.ratePlanBufferedFrames,
        abandonedMids: telemetry.abandonedMids,
        sourceGapHistoryBreaks: telemetry.sourceGapHistoryBreaks,
      },
      plannedFactorHistogram: { ...telemetry.plannedFactorHistogram },
      observed: {
        sourceCallbackHz: benchRate(sourceCallbackIntervalsMs),
        sourceHz: benchRate(sourceMediaIntervalsMs),
        rafHz: benchRate(rafIntervalsMs),
      },
      lateness: {
        count: lateMs.length,
        p95Ms: benchPercentile(lateMs, 0.95),
        maxMs: lateMs.length ? Math.max(...lateMs) : null,
      },
      samples: {
        sourceCallbackIntervalsMs, sourceMediaIntervalsMs, rafIntervalsMs,
        pumpWorkMs, sourceWorkMs, lateMs,
      },
      sampleOverflow: telemetry.sampleOverflow,
      errors: benchSessionErrors.map(error => ({ ...error })),
      product: {
        running,
        factor: cfg.factor,
        targetFps: cfg.targetFps,
        targetState: ratePlan.state,
        requestedTargetHz: ratePlan.requestedHz,
        minimumTargetHz: ratePlan.minimumHz,
        displayCapacityHz: ratePlan.capacityHz,
        effectiveTargetHz: ratePlan.outputHz,
        targetClampReason: ratePlan.clampReason,
        effectiveFactor: effN,
        resolution: cfg.res,
        model: rtModel || cfg.model,
        modelAsset: MODELS[rtModel || cfg.model] || null,
        videoWidth: videoEl ? videoEl.videoWidth : 0,
        videoHeight: videoEl ? videoEl.videoHeight : 0,
        gpu: sys.gpu,
        integrated: !!sys.integrated,
        deviceFeatures: device ? [...device.features].sort() : [],
        convTune: benchAppliedTune ? JSON.parse(JSON.stringify(benchAppliedTune)) : null,
        scheduler: { msAvg, intervalMs, decodedIntervalMs, uniqueIntervalMs, delayMs, lateAvg, rafMs, rafFloor },
        cuts,
        duplicates: dups,
      },
    };
  }

  function trackSourceVideo(v) {
    if (diag.sourceVideo === v) return;
    diag.sourceVideo = v;
    diag.sourceVideoId++;
    diag.mediaTime = null;
    diag.presentedFrames = null;
    if (cfg.debug) log('diagnostics source video', diag.sourceVideoId, v);
  }

  function diagnosticSnapshot(now = performance.now()) {
    const v = videoEl;
    const elapsed = Math.max(1, now - diag.sampleAt);
    const css = overlay?.getBoundingClientRect();
    const count = (name) => diag[name] - (diag.sample?.[name] || 0);
    const rate = (name) => (count(name) * 1000 / elapsed).toFixed(1);
    const snapshot = {
      rvfcCalls: diag.rvfcCalls,
      rafCalls: diag.rafCalls,
      inferenceCalls: diag.inferenceCalls,
      prepCalls: diag.prepCalls,
      presentCalls: diag.presentCalls,
      duplicateSkips: diag.duplicateSkips,
      cutSkips: diag.cutSkips,
    };
    const out = {
      sourceVideoId: diag.sourceVideoId,
      sourceIdentityCurrent: v === diag.sourceVideo,
      currentTime: v?.currentTime ?? null,
      mediaTime: diag.mediaTime,
      presentedFrames: diag.presentedFrames,
      paused: v?.paused ?? null,
      readyState: v?.readyState ?? null,
      playbackRate: v?.playbackRate ?? null,
      rates: { rvfc: rate('rvfcCalls'), raf: rate('rafCalls'), inference: rate('inferenceCalls'), prep: rate('prepCalls'), present: rate('presentCalls') },
      skips: { duplicate: diag.duplicateSkips, cut: diag.cutSkips, repeatedMediaCallbacks: diag.repeatedMediaCallbacks },
      loops: { pumpStarts: diag.pumpStarts, sourceStarts: diag.sourceLoopStarts, stops: diag.loopStops },
      fullscreen: { events: diag.fullscreenEvents, sourceChanged: diag.sourceChangedAtFullscreen,
        element: document.fullscreenElement?.tagName || null },
      canvas: { backing: overlay ? [overlay.width, overlay.height] : null,
        css: css ? [Math.round(css.width), Math.round(css.height)] : null, dpr: devicePixelRatio,
        objectFit: overlayFitRequested, resolvedObjectFit: overlayFit, fitSupported: overlayFitSupported },
    };
    diag.sampleAt = now;
    diag.sample = snapshot;
    return out;
  }
  window.__framegenDiagnostics = () => diagnosticSnapshot();

  // Chrome on Windows IGNORES powerPreference (crbug.com/369219127): on dual-GPU
  // machines we get whatever GPU Chrome runs on. Detect integrated ones and tell
  // the user how to move Chrome to the discrete card.
  function classifyAdapter(adapter) {
    sys.f16 = adapter.features.has('shader-f16');
    const inf = adapter.info || {};
    sys.gpu = inf.description || [inf.vendor, inf.architecture].filter(Boolean).join(' ') || 'unknown GPU';
    sys.integrated = /intel|iris|uhd|graphics 6|vega|radeon\(tm\) graphics|apu/i.test(sys.gpu)
      && !/nvidia|geforce|rtx|gtx|radeon rx|arc a|arc b/i.test(sys.gpu);
  }
  // lightweight probe for the popup: adapter info only, no device, no weights
  let probing = null;
  async function probeAdapter() {
    if (sys.gpu !== '-' || !navigator.gpu) return;
    if (!probing) probing = (async () => {
      try {
        const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (a) classifyAdapter(a);
      } catch { /* leave unknown */ }
    })();
    await probing;
  }

  // ---------- device / runtime ----------
  // memoized: the hover-preload and the FC click may race - only one build runs
  let rtBuilding = null;
  function runtimeMatches(res = cfg.res, model = cfg.model, guard = cfg.guard) {
    return !!(device && rt && rtRes === res && rtModel === model && rtGuard === guard);
  }
  async function ensureRuntime() {
    await settingsReady;
    // Converge on the latest complete identity. A profile may change while model
    // assets are fetching or WebGPU pipelines are compiling.
    while (!runtimeMatches()) {
      while (rtBuilding) await rtBuilding;
      if (runtimeMatches()) return;
      const request = Object.freeze({ res: cfg.res, model: cfg.model, guard: cfg.guard });
      rtBuilding = buildRuntime(request);
      try { await rtBuilding; } finally { rtBuilding = null; }
    }
  }

  function discardDeviceResources(lostDevice) {
    if (device !== lostDevice) return;

    clearTimeout(tuneTimer);
    tuneTimer = 0;
    rtGeneration++;
    pairSeq++;
    pairDecisionChain = Promise.resolve();
    processingFrame = false;
    curJob = null;
    queue = [];
    cmpRing = [];
    lastTex = null;
    outputRatePlanKey = '';
    outputRatePlanIdentity = null;
    outputRatePlanSnapshot = null;
    outputRatePlanCandidate = null;
    outputRatePlanCandidateSamples = 0;
    gpuProbeActive = false;
    gpuProbeKey = '';
    msAvg = 0;
    lastMidCostAt = 0;
    midCostSamples.length = 0;

    try { if (rt?.destroy) rt.destroy(); } catch {}
    try { if (sr?.destroy) sr.destroy(); } catch {}
    for (const tex of srOut.values()) {
      try { tex.destroy(); } catch {}
    }
    srOut.clear();

    rt = null;
    rtRes = 0;
    rtModel = '';
    rtGuard = null;
    sr = null;
    blitPipe = null;
    blitSampler = null;
    blitBg.clear();
    frameTex = [];
    frameIdx = 0;
    texW = 0;
    texH = 0;
    midTexs = [];
    midIdx = 0;
    dedupPipe = null;
    dedupBg = new Map();
    dedupStats = null;
    dedupSampler = null;
    dedupReads = [];
    dedupReadIdx = 0;
    capTex = null;
    downPipe = null;
    capBgs = new WeakMap();
    sys.hdrOn = false;

    try { overlayCtx?.unconfigure(); } catch {}
    device = null;
  }

  async function handleDeviceLost(lostDevice, info) {
    if (device !== lostDevice) return;
    deviceRecoveryEpoch++;
    if ((running || recoveringDevice) && videoEl?.isConnected) {
      deviceRecoveryVideo = videoEl;
    }
    invalidatePlaybackLoops();
    running = false;
    switching = true;

    if (overlay) {
      overlay.style.transition = 'none';
      overlay.style.opacity = '0';
      overlay.style.display = 'none';
    }
    if (hud) {
      hud.style.display = 'block';
      hud.textContent = 'FG: GPU reset detected - recovering...';
    }

    log('device lost', info?.reason || 'unknown', info?.message || '');
    discardDeviceResources(lostDevice);
    if (recoveringDevice) return;
    recoveringDevice = true;

    try {
      let recovered = false;
      for (let attempt = 0; attempt < 3 && deviceRecoveryVideo?.isConnected; attempt++) {
        const recoveryEpoch = deviceRecoveryEpoch;
        let recoveryError = null;
        for (let buildAttempt = 0; buildAttempt < 2; buildAttempt++) {
          try {
            await ensureRuntime();
            recoveryError = null;
            break;
          } catch (error) {
            recoveryError = error;
            await Promise.resolve();
          }
        }
        if (recoveryEpoch !== deviceRecoveryEpoch) {
          continue;
        }
        if (recoveryError) throw recoveryError;
        const resumeVideo = deviceRecoveryVideo;
        if (resumeVideo?.isConnected && videoEl === resumeVideo) {
          try {
            switching = false;
            await start(resumeVideo);
          } catch (error) {
            if (recoveryEpoch !== deviceRecoveryEpoch) {
              running = false;
              switching = true;
              continue;
            }
            throw error;
          }
          if (recoveryEpoch !== deviceRecoveryEpoch || !runtimeMatches()) {
            running = false;
            switching = true;
            continue;
          }
          recovered = true;
          deviceRecoveryVideo = null;
          advise('GPU connection restored', 3500);
          break;
        }
      }
      if (deviceRecoveryVideo && !recovered) {
        throw new Error('GPU device was lost repeatedly during recovery');
      }
    } catch (error) {
      log('device recovery failed', error);
      if (hud) {
        hud.style.display = 'block';
        hud.textContent = 'FG: GPU recovery failed - reload this page';
      }
    } finally {
      switching = false;
      recoveringDevice = false;
      deviceRecoveryVideo = null;
    }
  }
  async function loadConvTune(res = cfg.res, model = cfg.model) {
    try {
      // fcTune3: earlier generations DROPPED the tuner's w4/v2 flags on persist,
      // so every calibrated user silently ran the legacy kernels (~+30% mid,
      // ~2x SR). New key = one recalibration, then the full tune sticks.
      const key = 'fcTune3|' + sys.gpu + '|' + res + '|' + MODELS[model];
      const st = await chrome.storage.local.get('fcTune');
      return (st.fcTune && st.fcTune[key]) || null;
    } catch { return null; }
  }
  async function calibrateConvTune(rtMod, res = rtRes, model = rtModel, c1 = rtC1, c2 = rtC2) {
    // one-shot per (GPU, quality): bench kernel variants on the real conv shape,
    // persist the winner - picked up on the next runtime build
    try {
      const key = 'fcTune3|' + sys.gpu + '|' + res + '|' + MODELS[model]; // keep in sync with loadConvTune
      const st = await chrome.storage.local.get('fcTune');
      const all = st.fcTune || {};
      if (all[key]) return;
      const [mw, mh] = SIZES[res];
      const best = await rtMod.tuneConvRB(device, { ci: c2, co: c2, w16: mw / 16, h16: mh / 16, s2ci: c1 });
      // persist EVERY flag the runtime reads - dropping w4/v2 here is exactly
      // the bug that kept calibrated users on legacy kernels (fcTune2 era)
      all[key] = { coc: best.coc, slab: best.slab, sg: !!best.sg, wgx: best.wgx || 8, wgy: best.wgy || 8,
        w4: !!best.w4, v2: !!best.v2 };
      if (best.s2) all[key].s2 = { coc: best.s2.coc, w4: !!best.s2.w4 };
      await chrome.storage.local.set({ fcTune: all });
      log('conv tune', res, JSON.stringify(best));
    } catch (e) { log('tune skipped', e); }
  }
  // the calibration burst is 200-400ms of GPU work - injected 4s into playback it
  // was a guaranteed drop cascade right after the user enabled FC. Wait for an idle
  // moment (paused / FC off) instead; after 2 minutes of nonstop playback run anyway
  // (tuned kernels are worth one hiccup - up to +20% on some rungs).
  let tuneTimer = 0;
  function scheduleConvTune(rtMod, identity) {
    const t0 = performance.now();
    clearTimeout(tuneTimer);
    const tick = () => {
      if (!runtimeMatches(identity.res, identity.model, identity.guard)
          || !runtimeMatches()) return; // runtime changed - resched rides the rebuild
      if (!running || !videoEl || videoEl.paused || performance.now() - t0 > 120000) {
        calibrateConvTune(rtMod, identity.res, identity.model, identity.c1, identity.c2);
        return;
      }
      tuneTimer = setTimeout(tick, 3000);
    };
    tuneTimer = setTimeout(tick, 4000);
  }
  let rtC1 = 0, rtC2 = 0;
  async function buildRuntime({ res: buildRes, model: buildModel, guard: buildGuard }) {
    if (!device) {
      const GPU_HELP = 'WebGPU is off. Enable "Use graphics acceleration" in Chrome settings (chrome://settings/system), restart Chrome, and update your GPU driver. Very old GPUs are not supported.';
      if (!navigator.gpu) throw new Error(GPU_HELP);
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error(GPU_HELP);
      const f16 = adapter.features.has('shader-f16');
      const feats = f16 ? ['shader-f16'] : [];
      if (adapter.features.has('subgroups')) feats.push('subgroups'); // tuner may pick sg kernels
      if (adapter.features.has('timestamp-query')) feats.push('timestamp-query'); // calibration measures on GPU timestamps (2.4x faster burst)
      const createdDevice = await adapter.requestDevice({ requiredFeatures: feats });
      device = createdDevice;
      createdDevice.lost
        .then(info => handleDeviceLost(createdDevice, info))
        .catch(error => log('device loss handler', error));
      classifyAdapter(adapter);
    }
    if (runtimeMatches(buildRes, buildModel, buildGuard)) return;
    if (rt && rt.destroy) { // free the old runtime's VRAM now, not at GC time
      try { rt.destroy(); } catch {}
      rt = null;
    }
    const url = (p) => chrome.runtime.getURL(p);
    // tfact2 family: t-factored student + quarter-res refine head; the runtime
    // autodetects the trunk width from the manifest, so models are weight swaps
    const fetchSet = async (stem) => Promise.all([
      fetch(url('assets/' + stem + '.bin')).then(r => { if (!r.ok) throw 0; return r.arrayBuffer(); }),
      fetch(url('assets/' + stem + '.json')).then(r => { if (!r.ok) throw 0; return r.json(); })]);
    let bin, man;
    let runtimeModel = buildModel;
    try {
      [bin, man] = await fetchSet(MODELS[buildModel]);
    } catch (e) {
      // a dead runtime (extension was reloaded/updated) is not a missing model -
      // nothing works until the page reloads, so say exactly that and stop
      if (!chrome.runtime?.id) throw new Error('extension reloaded - refresh the page (F5)');
      if (buildModel === 'v6') throw e;
      log('model ' + buildModel + ' not bundled - falling back to v6');
      runtimeModel = 'v6';
      if (cfg.model === buildModel) cfg.model = runtimeModel;
      [bin, man] = await fetchSet(MODELS.v6);
    }
    const rtMod = await import(url('rt/rt.js'));
    const [mw, mh] = SIZES[buildRes];
    rtC1 = man['block0.conv0.0.0.weight'].shape[0];
    rtC2 = man['block0.conv0.1.0.weight'].shape[0];
    const convTune = await loadConvTune(buildRes, runtimeModel);
    rt = await rtMod.createRT(device, { w: mw, h: mh, textureInput: true, textureOutput: true,
      staticGuard: buildGuard, weightsBin: bin, weightsManifest: man, convTune });
    rtRes = buildRes; rtModel = runtimeModel; rtGuard = buildGuard;
    rtGeneration++;
    if (!convTune) scheduleConvTune(rtMod,
      { res: buildRes, model: runtimeModel, guard: buildGuard, c1: rtC1, c2: rtC2 });
    midTexs.forEach(t => t.destroy());
    midTexs = [];
    for (let i = 0; i < 24; i++) {
      midTexs.push(device.createTexture({ label: 'fcmid' + i, size: [mw, mh],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    blitBg.clear();
    log('runtime up @', buildRes, runtimeModel, buildGuard ? 'guard' : 'no-guard');
  }

  let poolGen = 0; // labels must be unique across reallocations: label-keyed
  // caches (dedup) must never hit an entry built for a destroyed generation
  const MIN_FRAME_TEXTURES = 12;
  const MAX_FRAME_TEXTURES = Cadence.MAX_PENDING_PRESENTATIONS + 2;
  const FRAME_TEXTURE_STEPS = [MIN_FRAME_TEXTURES, 16, 20, MAX_FRAME_TEXTURES];
  let resetFramePoolOnNextCapture = false;
  function requiredFrameTextureCount(nextDelayMs, wallPairMs) {
    const required = Math.ceil(nextDelayMs / Math.max(1, wallPairMs)) + 3;
    return FRAME_TEXTURE_STEPS.find(count => count >= required) || MAX_FRAME_TEXTURES;
  }
  function ensureFrameTextures(w, h, requiredCount = MIN_FRAME_TEXTURES) {
    const sameDimensions = texW === w && texH === h;
    if (sameDimensions && !resetFramePoolOnNextCapture
        && frameTex.length >= requiredCount) return;
    const textureCount = sameDimensions && !resetFramePoolOnNextCapture
      ? Math.max(frameTex.length, requiredCount)
      : requiredCount;
    resetFramePoolOnNextCapture = false;
    frameTex.forEach(t => t.destroy());
    frameTex = [];
    queue = []; curJob = null; cmpRing = []; // queued entries reference the destroyed pool
    pairSeq++; // in-flight classify continuations must not prep destroyed textures
    pairDecisionChain = Promise.resolve();
    poolGen++;
    for (let i = 0; i < textureCount; i++) {
      frameTex.push(device.createTexture({ label: 'fcfr' + poolGen + '_' + i, size: [w, h], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }));
    }
    texW = w; texH = h; dedupBg.clear(); blitBg.clear(); lastTex = null;
  }

  // pool dimensions for a source: fit inside FHD, keep the aspect ratio
  function poolDims() {
    const fw = videoEl.videoWidth, fh = videoEl.videoHeight;
    const s = Math.min(1, 1920 / fw, 1080 / fh);
    return [Math.round(fw * s), Math.round(fh * s)];
  }
  // copyExternalImageToTexture copies 1:1 and NEVER scales - for >FHD sources a
  // plain copy grabs the top-left FHD crop of the frame. Capture the full frame
  // into a scratch texture and downscale-blit it into the pool instead.
  let capTex = null, downPipe = null, capBgs = new WeakMap();
  function captureFrame(dst, vw, vh) {
    const fw = videoEl.videoWidth, fh = videoEl.videoHeight;
    if (fw <= 1920 && fh <= 1080) {
      device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: dst }, [vw, vh]);
      return;
    }
    if (!capTex || capTex.width !== fw || capTex.height !== fh) {
      if (capTex) capTex.destroy();
      capTex = device.createTexture({ label: 'fccap', size: [fw, fh], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
          | GPUTextureUsage.RENDER_ATTACHMENT });
      capBgs = new WeakMap(); // old bind groups reference the destroyed capTex
    }
    device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: capTex }, [fw, fh]);
    if (!downPipe) {
      const mod = device.createShaderModule({ code: BLIT_VS + `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, v.uv, 0.0);
}` });
      downPipe = device.createRenderPipeline({ layout: 'auto',
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] } });
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: dst.createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(downPipe);
    let cbg = capBgs.get(dst); // per-frame createBindGroup was pure garbage-churn at >FHD
    if (!cbg) {
      cbg = device.createBindGroup({ layout: downPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: capTex.createView() },
          { binding: 1, resource: blitSampler }] });
      capBgs.set(dst, cbg);
    }
    pass.setBindGroup(0, cbg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ---------- dedup / cut (GPU, 8-byte readback) ----------
  function ensureDedup() {
    if (dedupPipe) return;
    dedupSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    dedupStats = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    dedupReads = Array.from({ length: 3 }, () => ({ busy: false,
      buf: device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) }));
    dedupPipe = device.createComputePipeline({ layout: 'auto', compute: {
      module: device.createShaderModule({ code: `
@group(0) @binding(0) var t0: texture_2d<f32>;
@group(0) @binding(1) var t1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 2>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= 48 || y >= 27) { return; }
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(48.0, 27.0);
  let a = textureSampleLevel(t0, samp, uv, 0.0).rgb;
  let b = textureSampleLevel(t1, samp, uv, 0.0).rgb;
  let d = u32(dot(abs(a - b), vec3<f32>(255.0, 255.0, 255.0)));
  atomicAdd(&stats[0], d);
  atomicMax(&stats[1], d);
}`}), entryPoint: 'main' } });
  }
  const DEDUP_ZERO = new Uint32Array(2);
  async function classifyPair(ta, tb) {
    ensureDedup();
    // free readback slot: classifies overlap (the frame loop does not await
    // them), two can be in flight at 120fps sources. All busy = ordinary motion.
    let rb = null;
    for (let i = 0; i < dedupReads.length; i++) {
      const c = dedupReads[(dedupReadIdx + i) % dedupReads.length];
      if (!c.busy) { rb = c; dedupReadIdx = (dedupReadIdx + i + 1) % dedupReads.length; break; }
    }
    if (!rb) return { dup: false, cut: false, black: false };
    rb.busy = true;
    try {
      const key = ta.label + '|' + tb.label;
      if (!dedupBg.has(key)) {
        dedupBg.set(key, device.createBindGroup({ layout: dedupPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() },
          { binding: 2, resource: dedupSampler }, { binding: 3, resource: { buffer: dedupStats } }] }));
      }
      // the single stats buffer is safe across overlapping classifies: zero,
      // dispatch and the copy-out are queue-ordered per submit
      device.queue.writeBuffer(dedupStats, 0, DEDUP_ZERO);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(dedupPipe); pass.setBindGroup(0, dedupBg.get(key));
      pass.dispatchWorkgroups(6, 4);
      pass.end();
      enc.copyBufferToBuffer(dedupStats, 0, rb.buf, 0, 8);
      device.queue.submit([enc.finish()]);
      await rb.buf.mapAsync(GPUMapMode.READ);
      const s = new Uint32Array(rb.buf.getMappedRange().slice(0));
      rb.buf.unmap();
      const mean = s[0] / (48 * 27);
      lastStat = { mean, max: s[1] };
      // motion EMA feeds the artifact-aware factor cap; dups/cuts don't count as motion
      if (mean < 90) motionAvg = motionAvg * 0.7 + mean * 0.3;
      return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90, black: s[1] === 0 };
    } finally {
      rb.busy = false;
    }
  }

  // ---------- overlay presentation ----------
  function ensureOverlay() {
    if (overlay) {
      if (device && !blitSampler) {
        blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
        configureOverlay();
      }
      return;
    }
    overlay = document.createElement('canvas');
    // a SIBLING of the video with a modest z-index: above the video, below the controls
    overlay.style.cssText = 'position:absolute; pointer-events:none; z-index:2;'
      + 'opacity:0; transition:opacity .25s;';
    overlayCtx = overlay.getContext('webgpu');
    blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    // on native-controls players the overlay OWNS the pointer (pointer-events:auto
    // set in start): dblclick must never reach the native video, whose shadow-DOM
    // handler fullscreens the bare <video> where our canvas cannot exist
    overlay.addEventListener('click', () => {
      if (!videoEl || !videoEl.controls) return;
      if (videoEl.paused) videoEl.play().catch(() => {}); else videoEl.pause();
      flashCenter(svgIcon(videoEl.paused ? 'pause' : 'play', 30));
      updateBar();
    });
    overlay.addEventListener('dblclick', (e) => {
      e.preventDefault();
      if (videoEl && videoEl.controls) toggleFullscreen();
    });
    configureOverlay();
  }

  const BLIT_VS = `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4(p[i], 0.0, 1.0);
  o.uv = vec2(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}`;
  // (re)build the present path: SDR passthrough, or HDR via inverse tone mapping -
  // highlights expand past SDR white on an fp16 canvas in extended tone-mapping mode
  // (same idea as RTX Video HDR; the browser only ever hands us tonemapped SDR)
  function configureOverlay() {
    if (!overlayCtx || !device) return;
    let hdr = !!(cfg.hdr && sys.hdrOk);
    const fmt = hdr ? 'rgba16float' : 'rgba8unorm';
    try {
      overlayCtx.configure({ device, format: fmt, alphaMode: 'opaque',
        ...(hdr ? { colorSpace: 'srgb', toneMapping: { mode: 'extended' } } : {}) });
    } catch (e) {
      log('hdr configure failed, falling back to SDR', e);
      hdr = false;
      overlayCtx.configure({ device, format: 'rgba8unorm', alphaMode: 'opaque' });
    }
    const fs = hdr ? `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(tex, samp, v.uv, 0.0).rgb;
  let lin = pow(max(c, vec3(0.0)), vec3(2.2));
  let y = max(lin.r, max(lin.g, lin.b));
  let t = smoothstep(0.35, 1.0, y);
  let gain = 1.0 + 2.2 * t * t;          // shadows/midtones untouched, peaks ~3.2x SDR white
  return vec4(pow(lin * gain, vec3(1.0 / 2.2)), 1.0);
}` : `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, v.uv, 0.0);
}`;
    const mod = device.createShaderModule({ code: BLIT_VS + fs });
    blitPipe = device.createRenderPipeline({ layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: hdr ? 'rgba16float' : 'rgba8unorm' }] } });
    blitBg.clear(); // bind groups belong to the old pipeline layout
    sys.hdrOn = hdr;
  }
  // fullscreen renders in the browser's TOP LAYER: anything not inside the
  // fullscreen element is invisible there. Move the whole UI in (and back out) -
  // fired from the fullscreenchange event too, so it works with the SITE's own
  // fullscreen button and while FC is off.
  function reparentUI() {
    const uiHost = document.fullscreenElement || document.body;
    if (uiHost.tagName === 'VIDEO') return; // bare-video fullscreen: nothing can overlay it
    if (btn && btn.parentElement !== uiHost) {
      uiHost.appendChild(btn); uiHost.appendChild(gear); uiHost.appendChild(hud); uiHost.appendChild(panel);
      if (bar) uiHost.appendChild(bar);
      if (splitEl) uiHost.appendChild(splitEl);
      if (warnEl) uiHost.appendChild(warnEl);
      if (adviseEl) uiHost.appendChild(adviseEl);
      if (flashEl) uiHost.appendChild(flashEl);
      if (wm) uiHost.appendChild(wm);
    }
  }
  document.addEventListener('fullscreenchange', () => {
    diag.fullscreenEvents++;
    diag.sourceChangedAtFullscreen ||= diag.sourceVideo !== videoEl;
    if (cfg.debug) log('diagnostics fullscreenchange', diagnosticSnapshot());
    reparentUI();
    sbLeft = -1; // force button re-place at the new geometry
    // coords from the OLD geometry are garbage for a moment: hide, let the page
    // reflow (two frames), then re-place against the fresh video rect
    if (btn) { btn.style.display = 'none'; gear.style.display = 'none'; }
    uiScan = 0; // the biggest-video answer may change across fullscreen too
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (cfg.debug) log('diagnostics fullscreen settled', diagnosticSnapshot());
      const v = running ? videoEl : uiVideo;
      if (v && btn && performance.now() < revealUntil) {
        placeSideButtons(v.getBoundingClientRect());
      }
    }));
  });

  function positionOverlay(vrIn) { // caller may pass a fresh video rect to save a forced layout
    if (overlay.parentElement !== videoEl.parentElement) {
      videoEl.parentElement.insertBefore(overlay, videoEl.nextSibling);
    }
    reparentUI();
    const r = vrIn || videoEl.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    // A player may fit a cinematic stream into a 16:9 box (or crop it with
    // object-fit:cover). Mirror that presentation instead of treating the aspect
    // difference as an unsupported video. Canvas is a replaced element, so the
    // browser applies the same fit/crop rules to our rendered frame as the source.
    const videoStyle = getComputedStyle(videoEl);
    const sourceWidth = videoEl.videoWidth, sourceHeight = videoEl.videoHeight;
    const sourceDimensionsChanged = sourceWidth !== overlaySourceWidth || sourceHeight !== overlaySourceHeight;
    overlaySourceWidth = sourceWidth;
    overlaySourceHeight = sourceHeight;
    overlayFitRequested = videoStyle.objectFit || 'fill';
    let fit = overlayFitRequested;
    if (!['fill', 'contain', 'cover', 'none', 'scale-down'].includes(fit)) fit = 'fill';
    // scale-down is exactly the smaller concrete object size from none/contain.
    // Resolve it before sizing the canvas because its own intrinsic dimensions
    // otherwise influence which branch the browser chooses.
    if (fit === 'scale-down' && sourceWidth && sourceHeight) {
      fit = sourceWidth <= r.width && sourceHeight <= r.height ? 'none' : 'contain';
    }
    overlay.style.objectFit = fit;
    overlay.style.objectPosition = videoStyle.objectPosition;
    // self-calibrating placement: measure where the overlay actually landed and nudge
    // by the delta - immune to whatever containing block/margins the site uses
    const cur = overlay.getBoundingClientRect();
    const dx = r.left - cur.left, dy = r.top - cur.top;
    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      overlay.style.left = ((parseFloat(overlay.style.left) || 0) + dx) + 'px';
      overlay.style.top = ((parseFloat(overlay.style.top) || 0) + dy) + 'px';
    }
    overlay.style.width = r.width + 'px';
    overlay.style.height = r.height + 'px';
    overlay.style.outline = cfg.debug ? '3px solid #19c37d' : 'none';
    let bw = r.width * devicePixelRatio, bh = r.height * devicePixelRatio;
    let supported = fit === 'fill' || !!(sourceWidth && sourceHeight);
    if (fit === 'none' && supported) {
      // object-fit:none depends on absolute intrinsic CSS pixels, not only aspect
      // ratio. Preserve those dimensions exactly. Oversized unscaled sources
      // cannot also obey the FHD canvas safety cap, so fail open to the raw video.
      bw = sourceWidth;
      bh = sourceHeight;
      supported = bw <= 1920 && bh <= 1080;
    } else if (fit !== 'fill' && supported) {
      const sx = bw / sourceWidth, sy = bh / sourceHeight;
      const scale = fit === 'cover' ? Math.max(sx, sy) : Math.min(sx, sy);
      bw = sourceWidth * scale;
      bh = sourceHeight * scale;
    }
    if (supported !== overlayFitSupported || fit !== overlayFit || sourceDimensionsChanged) {
      overlayFitSupported = supported;
      overlayFit = fit;
      overlay.style.opacity = '0';
      queue = []; curJob = null; lastTex = null; cmpRing = [];
      pairSeq++;
      if (cfg.debug && !supported) log('object-fit fallback to raw video', overlayFitRequested, bw, bh);
    }
    overlay.style.visibility = supported ? 'visible' : 'hidden';
    overlay.style.pointerEvents = supported && videoEl.controls ? 'auto' : 'none';
    if (!supported) return;
    const cap = Math.min(1, 1920 / bw, 1080 / bh);
    bw = Math.round(bw * cap);
    bh = Math.round(bh * cap);
    if (overlay.width !== bw || overlay.height !== bh) { overlay.width = bw; overlay.height = bh; }
  }
  // ---------- TinySR 2x upscale on the present path ----------
  let sr = null, srBuilding = null;
  const srOut = new Map();
  async function ensureSR() {
    if (sr || !sys.f16 || !device) return; // SR shaders need shader-f16
    if (srBuilding) {
      await srBuilding;
      if (!sr && sys.f16 && device) return ensureSR();
      return;
    }
    srBuilding = (async () => {
      const buildDevice = device;
      const buildGeneration = rtGeneration;
      const url = (p) => chrome.runtime.getURL(p);
      const [bin, man] = await Promise.all([
        fetch(url('assets/rt_sr.bin')).then(r => r.arrayBuffer()),
        fetch(url('assets/rt_sr.json')).then(r => r.json())]);
      const { createSR } = await import(url('rt/sr.js'));
      // the SR convs run at FULL video resolution - the w4/v2 kernel variants
      // measured ~2x there. The trunk's calibrated tune transfers well enough
      // (w4v2 won every grid we benched); output stays bit-identical.
      const convTune = await loadConvTune().catch(() => null);
      // sr.js feature-gates sg itself, so the default is safe everywhere
      const built = await createSR(buildDevice, { weightsBin: bin, weightsManifest: man,
        convTune: convTune || { coc: 8, slab: 12, sg: true, w4: true, v2: true } });
      if (device !== buildDevice || rtGeneration !== buildGeneration) {
        try { if (built?.destroy) built.destroy(); } catch {}
        return;
      }
      sr = built;
      log('SR up', JSON.stringify(convTune || 'default-w4v2'));
    })();
    try { await srBuilding; } finally { srBuilding = null; }
  }

  let cmpRing = []; // source frames + their due times: compare's left half runs on
  // the ORIGINAL cadence, independent of what the output side presents (hz mode
  // rarely presents raw sources - the left half would freeze otherwise)
  function present(tex, isMid) {
    if (!overlayFitSupported) return;
    // every presented frame goes through SR when it adds pixels toward the
    // canvas. It used to be generated-frames-only as a GPU saving, but that
    // alternates sharp/soft at display rate - visible shimmer, worst on
    // low-res anime (field report 2026-07-13).
    if (cfg.sr) {
      if (!sr) { ensureSR().catch(e => log('sr', e)); }
      else if (tex.width * 1.15 < overlay.width) { // marginal upscale = invisible after the canvas downsample
        const key = tex.width + 'x' + tex.height;
        let out = srOut.get(key);
        if (!out) {
          const sc = sr.scale || 2; // weights decide: 2x today, 4x/1x when they ship
          out = device.createTexture({ label: 'fcsr' + key,
            size: [tex.width * sc, tex.height * sc], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
          srOut.set(key, out);
          if (srOut.size > 4) {
            for (const [k, t] of srOut) if (k !== key) { t.destroy(); srOut.delete(k); }
            blitBg.clear();
          }
        }
        // false while the per-size pipelines compile (async) - show the raw frame
        if (sr.process(tex, out, tex.width, tex.height)) tex = out;
      }
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: overlayCtx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(blitPipe);
    if (!blitBg.has(tex)) {
      // evict BEFORE inserting - clearing after wipes the fresh entry and the
      // setBindGroup below gets undefined (same bug class as the rt.js caches)
      if (blitBg.size > 48) blitBg.clear();
      blitBg.set(tex, device.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: blitSampler }] }));
    }
    pass.setBindGroup(0, blitBg.get(tex));
    pass.draw(3);
    // compare: the left half shows the DELAYED source frame (same pipeline clock as
    // the FC half) - revealing the live <video> instead would be off by the delay
    let cmpSrcTex = null;
    if (cfg.compare) {
      const pnow = performance.now();
      while (cmpRing.length > 1 && cmpRing[1].at <= pnow) cmpRing.shift();
      if (cmpRing.length && cmpRing[0].at <= pnow) cmpSrcTex = cmpRing[0].tex;
    }
    if (cfg.compare && cmpSrcTex && cmpSrcTex !== tex) {
      if (!blitBg.has(cmpSrcTex)) {
        blitBg.set(cmpSrcTex, device.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: cmpSrcTex.createView() }, { binding: 1, resource: blitSampler }] }));
      }
      pass.setScissorRect(0, 0, Math.max(1, Math.round(splitX * overlay.width)), overlay.height);
      pass.setBindGroup(0, blitBg.get(cmpSrcTex));
      pass.draw(3);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    diag.presentCalls++;
    if (overlay.style.opacity !== '1') {
      overlay.style.transition = ''; // back to the stylesheet fade (onSrcChange kills it)
      overlay.style.opacity = '1'; // reveal only once pixels exist
    }
    const now = performance.now();
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
  }
  // our own control bar ON TOP of the overlay: native controls render INSIDE the
  // video element and can never show above the canvas, so instead of ever revealing
  // the raw video we drive the <video> ourselves - play/seek/volume/fullscreen as
  // regular DOM above everything. Interpolation is never interrupted.
  // sites whose own DOM controls are KNOWN to render above our overlay - there we
  // don't double up with our bar. Everywhere else (jut.su-style players put their
  // bar BELOW the canvas) our controls are the only usable ones.
  const SITE_CONTROLS_OK = /(^|\.)(youtube\.com|youtu\.be|vimeo\.com|twitch\.tv)$/
    .test(location.hostname);
  let revealUntil = 0, uiVideo = null, uiScan = 0, mmLast = 0;
  document.addEventListener('mousemove', (e) => {
    const now = performance.now();
    // gaming mice fire mousemove at up to 1000Hz; the rect read below forces
    // layout - unthrottled that alone janks the main thread while the mouse moves
    if (now - mmLast < 33) return;
    mmLast = now;
    if (!running && now - uiScan > 300) { uiScan = now; uiVideo = biggestVideo(); }
    const v = running ? videoEl : uiVideo;
    if (!v || !btn) return;
    // reuse pump's cached rect while running - a per-mousemove rect read forces
    // layout up to 30x/s on heavy pages for a hit test that tolerates 250ms staleness
    const r = (running && lastVr && now - lastVrT < 250) ? lastVr : v.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
      revealUntil = now + 2000;
      placeSideButtons(r);
      // hovering a video signals intent: build the runtime NOW (weights fetch +
      // shader compilation, the expensive part) so the FC click lands instantly
      if (!rt && !rtBuilding && now - preloadFailT > 5000) {
        ensureRuntime().catch((err) => { preloadFailT = performance.now(); log('preload', err); });
      }
    } else if (revealUntil > now + 250) {
      revealUntil = now + 250; // pointer left the player: fade soon, not in 2s
    }
  }, { passive: true });

  // FC + settings live INSIDE the player: centered vertically at the left edge
  let sbLeft = -1, sbTop = -1;
  function placeSideButtons(r) {
    btn.style.display = gear.style.display = 'block';
    const left = Math.round(r.left + 12), cy = Math.round(r.top + r.height / 2);
    if (left === sbLeft && cy === sbTop) return; // no writes when nothing moved
    sbLeft = left; sbTop = cy;
    btn.style.left = gear.style.left = left + 'px';
    btn.style.top = (cy - 42) + 'px';
    gear.style.top = (cy + 4) + 'px';
  }
  // vertical feeds: every scroll is an SPA navigation to the next clip. The old
  // stream still dies with a hard stop(), but the user's FC-on intent carries
  // over - re-engage on the new player once it can decode a frame.
  const inFeed = () => /youtube\.com\/shorts|tiktok\.com/.test(location.href);
  let reattachSeq = 0;
  function reattach() {
    const seq = ++reattachSeq, t0 = performance.now();
    const tick = async () => {
      if (seq !== reattachSeq || running || !inFeed()) return;
      if (performance.now() - t0 > 12000) return; // closed player / no video: give up
      const v = biggestVideo();
      if (!v || !v.videoWidth || toggling) { setTimeout(tick, 150); return; }
      toggling = true;
      try { await start(v); }
      catch (e) { log('feed reattach', e); }
      finally { toggling = false; }
    };
    setTimeout(tick, 120);
  }
  let pageHref = location.href;
  setInterval(() => {
    // SPA navigation (YouTube next video, etc): the old stream is dead - showing
    // its frames on the new page is nonsense. Hard-off; the user re-enables -
    // except inside feeds, where the enable carries to the next clip.
    if (location.href !== pageHref) {
      pageHref = location.href;
      if (running) {
        stop();
        if (inFeed()) reattach();
      }
    }
    if (!btn) return;
    if (panel && panel.style.display === 'block') { revealUntil = performance.now() + 2000; return; }
    if (performance.now() > revealUntil) {
      btn.style.display = gear.style.display = 'none';
    }
  }, 300);
  // scrolling moves the video but not our fixed-position buttons - re-pin them
  // (capture: catches scrolling containers, not just the window)
  document.addEventListener('scroll', () => {
    if (!btn || btn.style.display === 'none') return;
    const v = running ? videoEl : uiVideo;
    if (v) placeSideButtons(v.getBoundingClientRect());
  }, { passive: true, capture: true });

  // crisp monochrome SVG icons (Feather-style) - no emoji
  const ICONS = {
    play: '<path d="M8 5.5v13a.5.5 0 0 0 .77.42l10.2-6.5a.5.5 0 0 0 0-.84L8.77 5.08A.5.5 0 0 0 8 5.5z" fill="currentColor"/>',
    pause: '<rect x="7" y="5" width="3.4" height="14" rx="1" fill="currentColor"/><rect x="13.6" y="5" width="3.4" height="14" rx="1" fill="currentColor"/>',
    vol: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
      + '<path d="M15 8.6a5 5 0 0 1 0 6.8M17.7 6a9 9 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    volX: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
      + '<path d="m15.5 9.5 5 5m0-5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    full: '<path d="M8.5 3.5H5a1.5 1.5 0 0 0-1.5 1.5v3.5m17 0V5A1.5 1.5 0 0 0 19 3.5h-3.5m0 17H19a1.5 1.5 0 0 0 1.5-1.5v-3.5m-17 0V19A1.5 1.5 0 0 0 5 20.5h3.5"'
      + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    gear: '<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1.5 14H7m2-6h6m2.5 8H21"'
      + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  };
  const svgIcon = (name, size = 16) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

  const fmt = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    const m = Math.floor(s / 60), h = Math.floor(m / 60);
    return (h ? h + ':' + String(m % 60).padStart(2, '0') : m) + ':' + String(s % 60).padStart(2, '0');
  };

  function ensureBar() {
    if (bar) return;
    barH = 0; // remeasure on (re)build
    // floating glass pill, same family as the side buttons
    bar = document.createElement('div');
    bar.style.cssText = 'position:fixed; z-index:2147483646; display:none; align-items:center; gap:10px;'
      + 'background:rgba(16,17,20,.88);'
      + 'border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:8px 14px;'
      + 'color:#fff; font:11px system-ui; box-sizing:border-box; user-select:none;'
      + 'box-shadow:0 4px 20px rgba(0,0,0,.4); opacity:0; transform:translateY(8px);'
      + 'transition:opacity .18s, transform .18s; pointer-events:none;';
    bar.innerHTML = `
      <button id="fcPlay" class="fc-btn">${svgIcon('play', 19)}</button>
      <span id="fcCur" style="min-width:34px; text-align:right">0:00</span>
      <input id="fcSeek" class="fc-range" type="range" min="0" max="1000" value="0" style="flex:1">
      <span id="fcDur" style="min-width:34px; color:rgba(255,255,255,.55)">0:00</span>
      <button id="fcMute" class="fc-btn">${svgIcon('vol')}</button>
      <input id="fcVol" class="fc-range" type="range" min="0" max="100" value="100" style="width:60px">
      <button id="fcFull" class="fc-btn">${svgIcon('full')}</button>`;
    document.body.appendChild(bar);
    const q = (id) => bar.querySelector(id);
    // per-button cooldowns: hammering the buttons must never wedge the player
    const guard = (ms) => { let t = 0; return () => {
      const n = performance.now(); if (n - t < ms) return false; t = n; return true; }; };
    const gPlay = guard(180), gMute = guard(120), gFull = guard(400);
    q('#fcPlay').onclick = () => {
      if (!videoEl || !gPlay()) return;
      if (videoEl.paused) videoEl.play().catch(() => {}); else videoEl.pause();
      flashCenter(svgIcon(videoEl.paused ? 'pause' : 'play', 30));
      updateBar();
    };
    q('#fcMute').onclick = () => { if (videoEl && gMute()) { videoEl.muted = !videoEl.muted; updateBar(); } };
    q('#fcVol').oninput = (e) => { if (videoEl) { videoEl.volume = e.target.value / 100; videoEl.muted = false; } };
    q('#fcSeek').addEventListener('pointerdown', () => { barSeeking = true; });
    q('#fcSeek').addEventListener('pointerup', () => { barSeeking = false; });
    q('#fcSeek').oninput = (e) => {
      if (videoEl && videoEl.duration) videoEl.currentTime = e.target.value / 1000 * videoEl.duration;
    };
    q('#fcFull').onclick = () => { if (gFull()) toggleFullscreen(); }; // async transition - no double-fire
    // keep the bar alive while the mouse is on it (cheap write, no throttle needed)
    bar.addEventListener('mousemove', () => { revealUntil = performance.now() + 2000; }, { passive: true });
  }

  // big centered ▶/❚❚ splash on play/pause, fades out while scaling up
  let flashEl = null;
  function flashCenter(sym) {
    if (!videoEl) return;
    if (!flashEl) {
      flashEl = document.createElement('div');
      flashEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
        + 'color:#fff; font:600 26px system-ui; background:rgba(16,17,20,.85);'
        + 'border-radius:50%; width:72px; height:72px;'
        + 'display:flex; align-items:center; justify-content:center; opacity:0;';
      document.body.appendChild(flashEl);
    }
    const r = videoEl.getBoundingClientRect();
    flashEl.innerHTML = sym;
    flashEl.style.left = (r.left + r.width / 2 - 36) + 'px';
    flashEl.style.top = (r.top + r.height / 2 - 36) + 'px';
    flashEl.style.transition = 'none';
    flashEl.style.opacity = '0.95';
    flashEl.style.transform = 'scale(0.8)';
    requestAnimationFrame(() => {
      flashEl.style.transition = 'opacity .5s ease-out, transform .5s ease-out';
      flashEl.style.opacity = '0';
      flashEl.style.transform = 'scale(1.4)';
    });
  }

  // fullscreen the PARENT (so the overlay comes along) and stretch the video to
  // fill the screen - fullscreening just the container leaves the video at its
  // layout size, which looks like fullscreen "not working"
  let fsByUs = false, fsSaved = '';
  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (videoEl) {
      fsByUs = true;
      (videoEl.parentElement || videoEl).requestFullscreen().catch(e => { fsByUs = false; log('fullscreen', e); });
    }
  }
  document.addEventListener('fullscreenchange', () => {
    if (!videoEl || !fsByUs) return;
    if (document.fullscreenElement) {
      fsSaved = videoEl.style.cssText;
      videoEl.style.cssText += ';position:fixed;left:0;top:0;width:100vw;height:100vh;'
        + 'max-width:none;max-height:none;object-fit:contain;background:#000;z-index:1;';
    } else {
      fsByUs = false;
      videoEl.style.cssText = fsSaved;
    }
  });
  const rangeFill = (el, p, color) => {
    el.style.background = `linear-gradient(to right, ${color} ${p}%, rgba(255,255,255,.22) ${p}%)`;
  };
  let barPlayIcon = '', barMuteIcon = '';
  // element refs + last-written values cached: this runs every UI tick, and
  // querySelector lookups + unconditional style writes are wasted work when
  // nothing changed (the gradient string rebuild forces a style recalc)
  let barEls = null, barVolP = -1, barCurT = '', barDurT = '', barSeekV = '', barSeekF = -1;
  function updateBar() {
    if (!bar || bar.style.display === 'none' || !videoEl) return;
    if (!barEls) {
      barEls = { play: bar.querySelector('#fcPlay'), mute: bar.querySelector('#fcMute'),
                 vol: bar.querySelector('#fcVol'), cur: bar.querySelector('#fcCur'),
                 dur: bar.querySelector('#fcDur'), seek: bar.querySelector('#fcSeek') };
    }
    const pi = videoEl.paused ? 'play' : 'pause';
    if (pi !== barPlayIcon) { barPlayIcon = pi; barEls.play.innerHTML = svgIcon(pi, 19); }
    const mi = (videoEl.muted || videoEl.volume === 0) ? 'volX' : 'vol';
    if (mi !== barMuteIcon) { barMuteIcon = mi; barEls.mute.innerHTML = svgIcon(mi); }
    const volP = Math.round((videoEl.muted ? 0 : videoEl.volume) * 100);
    if (volP !== barVolP) { barVolP = volP; barEls.vol.value = String(volP); rangeFill(barEls.vol, volP, '#fff'); }
    const d = videoEl.duration || 0, c = videoEl.currentTime || 0;
    const ct = fmt(c), dt = fmt(d);
    if (ct !== barCurT) { barCurT = ct; barEls.cur.textContent = ct; }
    if (dt !== barDurT) { barDurT = dt; barEls.dur.textContent = dt; }
    const p = d ? c / d * 100 : 0;
    if (!barSeeking && d) {
      const sv = String(Math.round(p * 10));
      if (sv !== barSeekV) { barSeekV = sv; barEls.seek.value = sv; }
    }
    const fp = Math.round(p * 10);
    if (fp !== barSeekF) { barSeekF = fp; rangeFill(barEls.seek, p, '#19c37d'); }
  }

  // compare mode: a draggable divider - raw video shows LEFT of it (the overlay is
  // clipped away), interpolated frames play on the right
  function ensureSplit() {
    if (splitEl) return;
    splitEl = document.createElement('div');
    splitEl.style.cssText = 'position:fixed; z-index:2147483645; width:18px; margin-left:-9px;'
      + 'cursor:ew-resize; touch-action:none; display:none;';
    splitEl.innerHTML = `
      <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; margin-left:-1px;
        background:rgba(255,255,255,.85); box-shadow:0 0 10px rgba(0,0,0,.7)"></div>
      <div style="position:absolute; right:14px; bottom:10px; color:#fff; font:10px system-ui;
        background:rgba(15,15,15,.6); border-radius:6px; padding:2px 6px; white-space:nowrap">orig.</div>
      <div style="position:absolute; left:14px; bottom:10px; color:#fff; font:10px system-ui;
        background:rgba(25,150,100,.65); border-radius:6px; padding:2px 6px">FG</div>`;
    splitEl.addEventListener('pointerdown', (e) => {
      splitEl.setPointerCapture(e.pointerId);
      const move = (ev) => {
        if (!videoEl) return;
        const r = videoEl.getBoundingClientRect();
        splitX = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width));
      };
      move(e);
      splitEl.onpointermove = move;
      splitEl.onpointerup = () => { splitEl.onpointermove = null; splitEl.onpointerup = null; };
      e.preventDefault();
    });
    (document.fullscreenElement || document.body).appendChild(splitEl);
  }

  // one-shot amber advisory plate (integrated-GPU hint etc.), positioned above warn
  let adviseEl = null, adviseUntil = 0;
  function hideWarnings() {
    overSince = 0;
    adviseUntil = 0;
    if (warnEl) {
      warnEl.style.opacity = '0';
      warnEl.style.transform = 'translateY(-6px)';
    }
    if (adviseEl) adviseEl.style.opacity = '0';
  }
  function advise(text, ms) {
    if (!cfg.showWarnings) return;
    if (!adviseEl) {
      adviseEl = document.createElement('div');
      adviseEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
        + 'background:rgba(60,45,10,.92); color:#ffd88a;'
        + 'border:1px solid rgba(255,200,90,.35); border-radius:12px; padding:8px 14px;'
        + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4); max-width:70vw;'
        + 'opacity:0; transition:opacity .25s;';
      document.body.appendChild(adviseEl);
    }
    adviseEl.textContent = text;
    adviseUntil = performance.now() + ms;
  }
  function updateAdvise(now, vr) {
    if (!adviseEl) return;
    if (!cfg.showWarnings) {
      adviseEl.style.opacity = '0';
      return;
    }
    const show = now < adviseUntil;
    if (show) {
      adviseEl.style.left = Math.max(8, vr.left + vr.width / 2 - adviseEl.offsetWidth / 2) + 'px';
      adviseEl.style.top = (vr.top + 52) + 'px';
    }
    adviseEl.style.opacity = show ? '1' : '0';
  }

  // overload plate: fixed output choices remain visible while the runtime fails safe
  function ensureWarn() {
    if (warnEl) return;
    warnEl = document.createElement('div');
    warnEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
      + 'background:rgba(60,16,16,.9); color:#ffb4a8;'
      + 'border:1px solid rgba(255,120,100,.35); border-radius:12px; padding:8px 14px;'
      + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4);'
      + 'opacity:0; transform:translateY(-6px); transition:opacity .25s, transform .25s;';
    warnEl.textContent = '⚠ Load too high - lower the factor or switch to auto';
    document.body.appendChild(warnEl);
  }
  function updateWarn(now, vr) {
    if (!cfg.showWarnings) {
      hideWarnings();
      return;
    }
    ensureWarn();
    const load = uniqueIntervalMs > 1 ? msAvg * Math.max(0, effN - 1) / uniqueIntervalMs : 0;
    const ratePlan = stableOutputRatePlan(currentOutputRatePlan());
    // fixed factor: over budget OR visibly dropping. auto: even 2x is being skipped
    const dropRate = fpsWin.length ? (dropWin.length / 2) / fpsWin.length : 0; // drops vs shown, per sec
    const dropping = cfg.fg && dropRate > 0.12;
    const fixedOver = cfg.fg && cfg.factor !== 'auto' && (load > 1.02 || dropping);
    const autoOver = cfg.fg && cfg.factor === 'auto' && autoSkipT && now - autoSkipT < 1200;
    const rateClamped = cfg.fg && !!ratePlan.warning && ratePlan.state !== 'measuring';
    if (fixedOver || autoOver || rateClamped) {
      if (!overSince) overSince = now;
      warnEl.textContent = rateClamped
        ? `⚠ ${ratePlan.warning}`
        : fixedOver
          ? '⚠ Frames are dropping - lower the output rate/quality or switch to auto'
          : '⚠ GPU cannot keep up even at 2x - set quality to eco';
    }
    if (!fixedOver && !autoOver && !rateClamped && load < 0.92) overSince = 0;
    const show = overSince && now - overSince > 1500; // sustained, not a warmup blip
    if (show) { // offsetWidth forces layout - only pay it while the warning is up
      warnEl.style.left = (vr.left + vr.width / 2 - warnEl.offsetWidth / 2) + 'px';
      warnEl.style.top = (vr.top + 12) + 'px';
    }
    warnEl.style.opacity = show ? '1' : '0';
    warnEl.style.transform = show ? 'translateY(0)' : 'translateY(-6px)';
  }

  function pump(now, epoch) {
    diag.rafCalls++;
    if (!running || epoch !== playbackLoopEpoch) return;
    // SPA navigation can replace the <video> element entirely: rVFC dies with it
    // and the canvas would keep showing the dead stream's frames forever.
    // Feeds recycle player elements - carry the FC intent to the replacement.
    if (videoEl && !videoEl.isConnected) { stop(); if (inFeed()) reattach(); return; }
    const workStartedAt = performance.now();
    try { pumpBody(now); } catch (e) { log('pump', e); }
    finally {
      if (benchTelemetry) pushBenchSample(benchTelemetry.pumpWorkMs,
        performance.now() - workStartedAt);
      armPumpLoop(epoch);
    }
  }
  function pumpBody(now) {
    recordBenchRaf(now);
    if (lastPumpT) {
      const d = now - lastPumpT;
      // pessimist estimator: believe slowdowns fast (40%), speedups slowly (3%) -
      // auto must not re-inflate on every momentary lull
      if (d > 1 && d < 100) {
        rafMs = rafMs ? (d > rafMs ? rafMs * 0.6 + d * 0.4 : rafMs * 0.97 + d * 0.03) : d;
        Cadence.updateDisplayInterval(refreshEstimate, d);
        rafFloor = refreshEstimate.floorMs;
        const agrees = Math.abs(d - rafFloor) / rafFloor <= 0.06;
        refreshEstimate.stableSamples = agrees
          ? Math.min(32, refreshEstimate.stableSamples + 1)
          : 0;
      }
      if (d >= 100) refreshEstimate.stableSamples = 0;
    }
    lastPumpT = now;
    // UI geometry is independent of display refresh. A fixed 15 Hz service rate
    // keeps controls as responsive as they were on a 60 Hz display without
    // multiplying forced layout/style work on 144-240 Hz displays.
    if (now >= nextUiUpdateAt) {
      nextUiUpdateAt = now + UI_UPDATE_INTERVAL_MS;
    { // our control bar floats above the video bottom, HUD in the top-right corner
      const vr = videoEl.getBoundingClientRect(); // read ONCE per tick, shared with positionOverlay
      lastVr = vr; lastVrT = now; // shared with pointer/UI placement to avoid extra layout reads
      positionOverlay(vr);
      // our bar everywhere except sites whose own controls verifiably sit above
      // the overlay (see SITE_CONTROLS_OK)
      if (cfg.hoverReveal && (videoEl.controls || !SITE_CONTROLS_OK)) {
        const showBar = now < revealUntil;
        const m = Math.max(10, Math.min(16, vr.width * 0.02));
        bar.style.display = 'flex';
        // bar content is static - measure once; the read-after-write forced a
        // synchronous layout every 4th tick
        if (!barH) barH = bar.offsetHeight;
        bar.style.left = (vr.left + m) + 'px';
        bar.style.width = (vr.width - 2 * m) + 'px';
        bar.style.top = (vr.bottom - barH - m) + 'px';
        bar.style.opacity = showBar ? '1' : '0';
        bar.style.transform = showBar ? 'translateY(0)' : 'translateY(8px)';
        bar.style.pointerEvents = showBar ? 'auto' : 'none';
      } else {
        bar.style.display = 'none';
      }
      // HUD: single line along the top-left of the video
      hud.style.display = (cfg.debug || cfg.showFps) ? 'block' : 'none';
      hud.style.left = (vr.left + 8) + 'px';
      hud.style.top = (vr.top + 8) + 'px';
      hud.style.maxWidth = Math.max(120, vr.width - 16) + 'px';
      // Enabled by default; the full settings page can hide it.
      wm.style.display = cfg.showWatermark ? 'block' : 'none';
      wm.style.left = (vr.left + 10) + 'px';
      wm.style.top = (vr.bottom - 26) + 'px';
      if (btn.style.display !== 'none') placeSideButtons(vr); // stay pinned while running
      updateWarn(now, vr);
      updateAdvise(now, vr);
      if (cfg.compare) {
        ensureSplit();
        splitEl.style.display = 'block';
        splitEl.style.left = (vr.left + splitX * vr.width) + 'px';
        splitEl.style.top = vr.top + 'px';
        splitEl.style.height = vr.height + 'px';
      } else {
        if (splitEl) splitEl.style.display = 'none';
      }
    }
    updateBar();
    } // end of throttled UI block
    if (queue.length > 1) queue.sort((a, b) => a.at - b.at);
    const dueSelection = Cadence.selectDuePresentation(queue, now, {
      targetHz: usesExactCadence()
        ? stableOutputRatePlan(currentOutputRatePlan()).outputHz || 0
        : 0,
      displayCapacityHz: Cadence.measureDisplayHz(rafFloor).capacityHz,
    });
    const due = dueSelection.presentIndex;
    // drop pressure: leaky integrator (tau 300ms) - a burst of drops is visible in
    // milliseconds instead of averaging out over seconds
    dropPressure *= Math.exp((lastPressureT - now) / 300);
    lastPressureT = now;
    if (due >= 0) {
      dropped += due;
      dropPressure += due;
      for (let i = 0; i < due; i++) dropWin.push(now);
      // presentation lateness (frames arriving PAST their slot without dropping -
      // external GPU bursts look exactly like this): learn fast, forget slowly,
      // feeds back into the delay target so the buffer grows to absorb bursts
      const late = now - queue[due].at;
      lateAvg = late > lateAvg ? lateAvg * 0.7 + late * 0.3 : lateAvg * 0.985 + late * 0.015;
      present(queue[due].tex, queue[due].mid);
      recordBenchPresentationBatch(due, now);
      queue.splice(0, due + 1); // drop in place - slice would allocate per presented frame
    }
    // Queue the current canvas blit before future inference so an urgent
    // presentation is never trapped behind the next pair's compute work.
    driveJob(now);
    while (dropWin.length && dropWin[0] < now - 2000) dropWin.shift();
    // AIMD controller, evaluated EVERY frame: aggressive decrease on pressure,
    // additive recovery after a long clean stretch
    if (cfg.factor === 'auto') {
      // compositor saturation (frames late by a vsync, not yet dropped) feeds the
      // same controller: rAF stretching 1.7x past the true vsync = strain
      if (rafFloor < 90 && rafMs > rafFloor * 1.7) dropPressure += 0.02;
      if (dropPressure > 1.2 && autoPenalty < 3 && now - penaltyT > 500) {
        autoPenalty = Math.min(3, autoPenalty + (dropPressure > 3 ? 2 : 1));
        penaltyT = now;
        dropPressure = 0; // consumed by the step
      } else if (autoPenalty > 0 && dropPressure < 0.15 && now - penaltyT > 6000) {
        autoPenalty--; penaltyT = now;
      }
    }
    if (now - statsTimer > 400) {
      statsTimer = now;
      const srcFps = intervalMs > 1 ? (1000 / intervalMs) : 0;
      if (cfg.debug) {
        const ds = diagnosticSnapshot(now);
        const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
        const mode = cfg.factor === 'auto' && autoPenalty && cfg.fpsLimit === null
          ? `Auto-${autoPenalty}` : outputRateLabel();
        hud.textContent = [
          `${videoEl.videoWidth}x${videoEl.videoHeight}@${srcFps.toFixed(0)} → ${fpsWin.length}fps ×${effN} (${mode})`,
          `${msAvg.toFixed(1)}ms@${cfg.res}p`,
          `buf ${delayMs.toFixed(0)}`,
          `GPU ${load.toFixed(0)}%`,
          `raf ${rafMs.toFixed(1)}/${rafFloor.toFixed(1)}`,
          `late ${lateAvg.toFixed(1)}`,
          `drop ${dropped} (${dropPressure.toFixed(1)})`,
          `dup ${dups} cut ${cuts}`,
          `motion ${motionAvg.toFixed(0)}`,
          `diff ${lastStat ? lastStat.mean.toFixed(1) : '-'}/${lastStat ? lastStat.max : '-'}${lastStat && lastStat.max === 0 ? ' DRM?' : ''}`,
          `src#${ds.sourceVideoId} t${ds.currentTime?.toFixed(2) ?? '-'} m${ds.mediaTime?.toFixed(2) ?? '-'} pf${ds.presentedFrames ?? '-'} ${ds.paused ? 'paused' : 'play'} rs${ds.readyState ?? '-'} x${ds.playbackRate ?? '-'}`,
          `rate rvfc ${ds.rates.rvfc} raf ${ds.rates.raf} prep ${ds.rates.prep} infer ${ds.rates.inference} present ${ds.rates.present}`,
          `skip dup ${ds.skips.duplicate} cut ${ds.skips.cut} repeat ${ds.skips.repeatedMediaCallbacks} · loop starts ${ds.loops.pumpStarts}/${ds.loops.sourceStarts} stops ${ds.loops.stops}`,
          `canvas ${ds.canvas.backing?.join('x') ?? '-'} css ${ds.canvas.css?.join('x') ?? '-'} @${ds.canvas.dpr}`,
        ].join('  ·  ');
      } else {
        hud.textContent = `FG ${fpsWin.length}fps · ${outputRateLabel()} · ${msAvg.toFixed(0)}ms`;
      }
      if (panel && panel.style.display === 'block') updateStatus();
    }
  }

  // ---------- interpolation (lazy per-mid submission) ----------
  // prepPair runs once per frame pair; each mid's compute is submitted just-in-time
  // for its display slot, so present blits interleave with computes on the GPU
  // queue instead of the first mid waiting for the whole batch.
  let curJob = null;
  // queue is tiny (a handful of entries): a linear scan beats allocating a Set,
  // and this runs up to ~19x per source pair in hz mode
  function texQueued(t) {
    for (let i = 0; i < queue.length; i++) if (queue[i].tex === t) return true;
    return false;
  }
  function texInCompareRing(t) {
    for (let i = 0; i < cmpRing.length; i++) if (cmpRing[i].tex === t) return true;
    return false;
  }
  function retainClassifyTexture(t) {
    classifyTextureUse.set(t, (classifyTextureUse.get(t) || 0) + 1);
  }
  function releaseClassifyTexture(t) {
    const count = classifyTextureUse.get(t) || 0;
    if (count <= 1) classifyTextureUse.delete(t);
    else classifyTextureUse.set(t, count - 1);
  }
  function texInClassification(t) {
    return classifyTextureUse.has(t);
  }
  function texInCurrentJob(t) {
    return !!curJob && (curJob.previous === t || curJob.current === t);
  }
  function submitMid() {
    const k = curJob.next;
    const timingGeneration = rtGeneration;
    const gpuProbe = curJob.gpuProbe === true;
    const disp = curJob.ats ? curJob.ats[k] : curJob.at + curJob.ts[k] * curJob.intervalMs;
    let guard = midTexs.length; // don't clobber queued mids
    while (guard-- > 0 && texQueued(midTexs[midIdx])) midIdx = (midIdx + 1) % midTexs.length;
    if (texQueued(midTexs[midIdx])) {
      if (benchTelemetry) benchTelemetry.midPoolExhausted++;
      const anchor = curJob.ts[k] < 0.5 ? curJob.previous : curJob.current;
      schedulePresentation(anchor, disp, false);
      curJob.next++;
      if (curJob.next >= curJob.ts.length) curJob = null;
      return false;
    }
    const out = midTexs[midIdx];
    midIdx = (midIdx + 1) % midTexs.length;
    const t0 = performance.now();
    try { rt.runT(curJob.ts[k], out); diag.inferenceCalls++; } catch (e) { log('runT', e); curJob = null; return; }
    // sample every 4th mid STARTING AT k=1: mid 0's drain wait also swallows the
    // trunk prep still executing ahead of it, so sampling it inflated msAvg by the
    // trunk share and the auto controller under-committed. Single-mid jobs (x2)
    // have no k=1 - sample k=0 there, prep has usually drained by its slot.
    if ((k & 3) === 1 || curJob.ts.length === 1) { // a drain-probe promise per submit adds up at high factors
      device.queue.onSubmittedWorkDone()
        .then(() => {
          if (timingGeneration !== rtGeneration) return;
          const elapsedMs = performance.now() - t0;
          // queue completion is observed on the main thread and includes prior
          // queue backlog. Use the lower quartile of a rolling window as the
          // achievable clean per-mid service time; a sustained slowdown still
          // replaces the full window and lowers admission within ~250ms.
          midCostSamples.push(elapsedMs);
          if (midCostSamples.length > 16) midCostSamples.shift();
          const orderedCosts = [...midCostSamples].sort((a, b) => a - b);
          const cleanCostMs = orderedCosts[Math.floor((orderedCosts.length - 1) * 0.25)];
          msAvg = msAvg ? msAvg * 0.8 + cleanCostMs * 0.2 : cleanCostMs;
          lastMidCostAt = performance.now();
          if (gpuProbe) {
            gpuProbeActive = false;
            outputRatePlanKey = '';
          }
        })
        .catch(() => {
          if (timingGeneration !== rtGeneration) return;
          if (gpuProbe) {
            gpuProbeActive = false;
            outputRatePlanKey = '';
          }
        });
    }
    schedulePresentation(out, disp, true);
    curJob.next++;
    if (curJob.next >= curJob.ts.length) curJob = null;
    return true;
  }
  function flushJob() {
    while (curJob && curJob.next < curJob.ts.length) submitMid();
    curJob = null;
  }
  function driveJob(now) {
    if (!curJob || switching) return;
    const lead = 2 * (msAvg || 10) + 8; // submit when the display slot is one compute away
    while (curJob && curJob.next < curJob.ts.length) {
      const disp = curJob.ats ? curJob.ats[curJob.next]
        : curJob.at + curJob.ts[curJob.next] * curJob.intervalMs;
      if (disp - now > lead) break;
      submitMid();
    }
  }

  async function onFrame(_callbackNow, metadata, epoch) {
    diag.rvfcCalls++;
    if (!running || epoch !== playbackLoopEpoch) return;
    armVideoFrameLoop(epoch);
    if (metadata) {
      if (metadata.mediaTime === diag.mediaTime) diag.repeatedMediaCallbacks++;
      diag.mediaTime = metadata.mediaTime;
      diag.presentedFrames = metadata.presentedFrames;
    }
    if (videoEl.videoWidth !== overlaySourceWidth || videoEl.videoHeight !== overlaySourceHeight) {
      positionOverlay();
    }
    if (!overlayFitSupported) return;
    const sourceCallbackAt = performance.now();
    recordBenchSource(sourceCallbackAt, metadata);
    if (processingFrame) {
      if (benchTelemetry) benchTelemetry.sourceBusySkipped++;
      return;
    }
    if (benchTelemetry) benchTelemetry.sourceProcessed++;
    processingFrame = true;
    const workStartedAt = performance.now();
    try {
      const arrival = sourceCallbackAt;
      const dt = arrival - lastArrival;
      if (dt > 0.5 && dt < 500) intervalMs = intervalMs * 0.9 + dt * 0.1;
      const decodedFrameDelta = updateDecodedSourceCadence(metadata, dt);
      if (decodedFrameDelta > 1) {
        // rVFC is best-effort: Chromium may coalesce source callbacks under
        // pressure. Never interpolate across that missing anchor as though the
        // current frame followed the previous one at the normal source cadence.
        curJob = null;
        pairSeq++;
        pairDecisionChain = Promise.resolve();
        lastTex = null;
        lastUniqueTs = 0;
        hzNext = 0;
        hzPhaseMs = 0;
        schedT = 0;
        if (benchTelemetry) benchTelemetry.sourceGapHistoryBreaks++;
      }
      lastArrival = arrival;
      // PLL-smoothed schedule clock: decode jitter must not shake presentation.
      // Track the arrival rhythm softly (8%), resync hard on seeks/stalls (>80ms off)
      const expected = schedT + intervalMs;
      schedT = (!schedT || Math.abs(arrival - expected) > 80)
        ? arrival : expected + 0.08 * (arrival - expected);
      if (!videoEl.videoWidth || !videoEl.videoHeight) return;
      const [vw, vh] = poolDims();
      // presentation delay must cover the batch compute time (own + the previous
      // batch draining), or high factors drop their early mids as already-stale.
      // The source pool grows only when the buffered history actually needs it.
      const burstPad = Math.min(60, Math.max(0, (lateAvg - 4) * 2));
      const floorMs = (msAvg && msAvg < 6 && lateAvg < 3 && effN <= 3) ? 42 : 60;
      const playbackRate = Math.max(0.01, Math.abs(Number(videoEl.playbackRate) || 1));
      const wallPairMs = decodedIntervalMs / playbackRate;
      const cadenceConfigured = usesExactCadence() && cfg.fg;
      const dTarget = Cadence.computePresentationDelayMs({
        cadenceMode: cadenceConfigured,
        sourceIntervalMs: wallPairMs,
        midCostMs: msAvg || 10,
        burstPadMs: burstPad,
        floorMs,
        maxDelayMs: cadenceConfigured ? 2500 : 180,
      });
      const riseLimit = cadenceConfigured ? Math.max(2, wallPairMs) : 2;
      let nextDelayMs = delayMs
        + Math.max(-2, Math.min(riseLimit, dTarget - delayMs));
      const sourcePoolCount = requiredFrameTextureCount(nextDelayMs, wallPairMs);
      const historyBudgetMs = (sourcePoolCount - 2) * wallPairMs;
      if (nextDelayMs > historyBudgetMs && delayMs > historyBudgetMs) {
        resetOutputCadence(true);
        lastTex = null;
      }
      nextDelayMs = Math.min(nextDelayMs, historyBudgetMs);
      ensureFrameTextures(vw, vh, sourcePoolCount);
      delayMs = nextDelayMs;
      // note on importExternalTexture (evaluated, rejected): interpolation needs the
      // PREVIOUS frame too, and external textures expire with the video frame - the
      // copy is unavoidable for history and for presenting source frames. Prep/dedup
      // reads scale with MODEL resolution, not source, so reading the external
      // texture instead of the copy saves nothing measurable.
      // NEVER overwrite a texture that is still queued for presentation or needed
      // as an interpolation input - reuse of live textures = timeline soup
      let guard = frameTex.length;
      while (guard-- > 0 && (frameTex[frameIdx] === lastTex
          || texQueued(frameTex[frameIdx]) || texInCompareRing(frameTex[frameIdx])
          || texInClassification(frameTex[frameIdx]) || texInCurrentJob(frameTex[frameIdx]))) {
        frameIdx = (frameIdx + 1) % frameTex.length;
      }
      if (frameTex[frameIdx] === lastTex
          || texQueued(frameTex[frameIdx]) || texInCompareRing(frameTex[frameIdx])
          || texInClassification(frameTex[frameIdx]) || texInCurrentJob(frameTex[frameIdx])) {
        if (benchTelemetry) benchTelemetry.sourcePoolExhausted++;
        resetOutputCadence(true);
        lastTex = null;
        return;
      }
      const tex = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      captureFrame(tex, vw, vh);
      const srcAt = schedT + delayMs;
      const schedulingIntervalMs = cadenceConfigured ? wallPairMs : intervalMs;
      const pairTiming = {
        startAt: schedT - schedulingIntervalMs + delayMs,
        intervalMs: schedulingIntervalMs,
        sourceIntervalMs: wallPairMs,
        cadenceIntervalMs: wallPairMs,
      };
      let ratePlan = cadenceConfigured ? currentOutputRatePlan({ startGpuProbe: true }) : null;
      if (ratePlan) ratePlan = syncOutputRatePlan(ratePlan);
      pairTiming.ratePlan = ratePlan;
      if (cfg.compare) {
        cmpRing.push({ tex, at: srcAt });
        const compareDepth = Math.min(frameTex.length - 2,
          Math.max(6, Math.ceil(delayMs / Math.max(1, wallPairMs)) + 2));
        while (cmpRing.length > compareDepth) cmpRing.shift();
      }
      const cadenceMode = cadenceConfigured && ratePlan.interpolationAllowed;
      if (!cadenceMode) schedulePresentation(tex, srcAt, false);
      const prev = lastTex;
      if (!cfg.fg) { // frame generation off: passthrough (SR-only if enabled)
        effN = 1;
        lastUniqueTs = arrival;
      } else if (prev) {
        // Start readbacks immediately so classification overlaps across source
        // frames, but apply their decisions in source order. Per-pair
        // supersession dropped valid work whenever the next source frame arrived
        // before the previous GPU readback completed.
        const seq = pairSeq;
        retainClassifyTexture(prev);
        retainClassifyTexture(tex);
        const classification = classifyPair(prev, tex);
        pairDecisionChain = pairDecisionChain
          .then(() => classification)
          .then((r) => {
            if (!running || seq !== pairSeq) {
              if (benchTelemetry) benchTelemetry.classificationSuperseded++;
              return;
            }
            try { decidePair(r, prev, tex, arrival, srcAt, cadenceMode, cadenceConfigured, pairTiming); }
            catch (e) { log('decide', e); }
          })
          .catch((e) => {
            if (seq === pairSeq) log('classify', e);
          })
          .finally(() => {
            releaseClassifyTexture(prev);
            releaseClassifyTexture(tex);
          });
      } else {
        lastUniqueTs = arrival;
        if (cadenceMode) schedulePresentation(tex, srcAt, false);
      }
      lastTex = tex;
    } catch (e) {
      if (e.name === 'OperationError') {
        log('frame skipped (decoder gap)'); // transient: no decoded frame this tick
      } else {
        log('frame error', e);
        stop();
        hud.style.display = 'block';
        hud.textContent = 'FG error: ' + (e.message || e);
      }
    } finally {
      processingFrame = false;
      if (benchTelemetry) pushBenchSample(benchTelemetry.sourceWorkMs,
        performance.now() - workStartedAt);
    }
  }

  // everything from "is this pair worth interpolating" to prepPair/curJob:
  // runs when the dedup readback lands
  function planPairCadence(interpolate, ms, timing) {
    const ratePlan = timing.ratePlan;
    const outputHz = ratePlan.outputHz;
    const srcFps = 1000 / timing.sourceIntervalMs;
    // A target at or below the decoded unique-frame rate is a decimation job,
    // not frame generation. Keep exact ticks but fill them from source anchors.
    const shouldInterpolate = interpolate && Cadence.targetNeedsInterpolation(srcFps, outputHz);
    const startAt = timing.startAt;
    const cadence = Cadence.planSourceCadencePresentations({ nextAt: hzNext, phaseMs: hzPhaseMs,
      startAt, sourceIntervalMs: timing.cadenceIntervalMs, outputHz,
      interpolate: shouldInterpolate });
    hzNext = cadence.nextAt;
    hzPhaseMs = cadence.nextPhaseMs;
    return { ...cadence, startAt, intervalMs: timing.cadenceIntervalMs,
      sourceIntervalMs: timing.sourceIntervalMs };
  }

  function scheduleCadenceAnchors(presentations, prev, tex) {
    for (const presentation of presentations) {
      if (presentation.kind === 'interpolate') continue;
      schedulePresentation(presentation.kind === 'current' ? tex : prev, presentation.at, false);
    }
  }

  function decidePair({ dup, cut }, prev, tex, arrival, srcAt, cadenceMode, cadenceConfigured, pairTiming) {
    const noInterpolation = cut || (cfg.anime && dup);
    if (cut) { cuts++; diag.cutSkips++; lastUniqueTs = arrival; }
    else if (cfg.anime && dup) { dups++; diag.duplicateSkips++; }
    else {
      const du = arrival - lastUniqueTs;
      if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
      lastUniqueTs = arrival;
    }

    const ms = pairTiming.ratePlan?.admissionCostMs || msAvg || 10;
    if (cadenceConfigured && !cadenceMode) {
      effN = 1;
      recordBenchPairPlan(1, false, 0);
      return;
    }
    if (cadenceMode) {
      // Every source interval advances one target clock, including cuts, anime
      // duplicates and intervals that need decimation. Exactly one presentation
      // is scheduled per target tick; there is no extra source-frame fallback.
      const cadence = planPairCadence(!noInterpolation, ms, pairTiming);
      const mids = cadence.presentations.filter(item => item.kind === 'interpolate');
      scheduleCadenceAnchors(cadence.presentations, prev, tex);
      const n = Math.max(1, cadence.presentations.length);
      effN = n;

      if (cadence.overflowed) {
        schedulePresentation(tex, srcAt, false); // pathological gap: resync safely
        recordBenchPairPlan(n, false, 0);
        return;
      }
      if (noInterpolation || mids.length === 0) {
        recordBenchPairPlan(n, false, 0);
        return;
      }

      const canRun = !switching && mids.length * ms <= cadence.sourceIntervalMs * 0.9;
      recordBenchPairPlan(n, canRun, canRun ? mids.length : 0);
      if (!canRun) {
        scheduleCadenceAnchors(Cadence.fallbackCadencePresentations(mids), prev, tex);
        return;
      }

      flushJob(); // leftovers of the previous pair go out before the new prep
      try {
        diag.prepCalls++;
        rt.prepPair(prev, tex);
        curJob = { ts: mids.map(item => item.t), next: 0, at: cadence.startAt,
          intervalMs: cadence.intervalMs, ats: mids.map(item => item.at),
          gpuProbe: pairTiming.ratePlan?.gpuProbe === true, previous: prev, current: tex };
      } catch (e) {
        log('prep', e);
        scheduleCadenceAnchors(Cadence.fallbackCadencePresentations(mids), prev, tex);
      }
      return;
    }

    if (noInterpolation) return;
    let n, run = true;
    if (cfg.factor === 'auto') {
      // smart auto: as much as fits the unique-frame budget, display service,
      // drop feedback and current motion. Capped Auto reuses the same policy.
      const policy = autoPolicyFactor(ms);
      n = policy.factor;
      run = policy.runnable;
      if (!run) autoSkipT = arrival;
    } else {
      // fixed by the user = a CEILING: under sustained overload we step down
      // to what actually fits the frame budget (the overload plate explains),
      // because piling up the queue looks far worse than a lower factor
      n = cfg.factor;
      while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 0.9) n--;
      if ((n - 1) * ms > uniqueIntervalMs * 1.15) run = false; // even 2x won't fit
    }
    effN = n;
    recordBenchPairPlan(n, run && !switching, run ? Math.max(0, n - 1) : 0);
    if (run && !switching) {
      flushJob(); // leftovers of the previous pair go out before the new prep
      try {
        diag.prepCalls++;
        rt.prepPair(prev, tex);
        const ts = [];
        for (let k = 1; k < n; k++) ts.push(k / n);
        curJob = { ts, next: 0, at: schedT - intervalMs + delayMs, intervalMs,
          previous: prev, current: tex };
      } catch (e) { log('prep', e); }
    }
  }

  // ---------- lifecycle / UI ----------
  function invalidatePlaybackLoops() {
    playbackLoopEpoch++;
    if (videoFrameCallbackId !== null && videoFrameCallbackVideo) {
      try { videoFrameCallbackVideo.cancelVideoFrameCallback(videoFrameCallbackId); } catch {}
    }
    videoFrameCallbackId = null;
    videoFrameCallbackVideo = null;
    if (pumpRafId !== null) {
      try { cancelAnimationFrame(pumpRafId); } catch {}
    }
    pumpRafId = null;
    return playbackLoopEpoch;
  }

  function armVideoFrameLoop(epoch) {
    const armedVideo = videoEl;
    if (!running || epoch !== playbackLoopEpoch || !armedVideo) return;
    let callbackId = null;
    callbackId = armedVideo.requestVideoFrameCallback((now, metadata) => {
      if (videoFrameCallbackId === callbackId) {
        videoFrameCallbackId = null;
        videoFrameCallbackVideo = null;
      }
      if (!running || epoch !== playbackLoopEpoch || videoEl !== armedVideo) return;
      onFrame(now, metadata, epoch);
    });
    videoFrameCallbackId = callbackId;
    videoFrameCallbackVideo = armedVideo;
  }

  function armPumpLoop(epoch) {
    if (!running || epoch !== playbackLoopEpoch) return;
    let callbackId = null;
    callbackId = requestAnimationFrame((now) => {
      if (pumpRafId === callbackId) pumpRafId = null;
      if (!running || epoch !== playbackLoopEpoch) return;
      pump(now, epoch);
    });
    pumpRafId = callbackId;
  }

  function resetOutputCadence(clearQueue = false) {
    hzNext = 0;
    hzPhaseMs = 0;
    outputRatePlanKey = '';
    outputRatePlanIdentity = null;
    outputRatePlanSnapshot = null;
    outputRatePlanCandidate = null;
    outputRatePlanCandidateSamples = 0;
    curJob = null;
    pairSeq++;
    pairDecisionChain = Promise.resolve();
    overSince = 0;
    if (!clearQueue) return;
    if (benchTelemetry) {
      for (const entry of queue) {
        if (entry.benchEpoch !== benchTelemetry.epoch) continue;
        benchTelemetry.queued--;
        benchTelemetry.dropped++;
        if (entry.mid) benchTelemetry.droppedMid++;
        else benchTelemetry.droppedSource++;
      }
    }
    queue = [];
    cmpRing = [];
    fpsWin = [];
  }

  function setOutputRate(value, persist = true) {
    const legacyTarget = value === 'fps60' ? 60 : value === 'fps120' ? 120 : null;
    const next = Cadence.sanitizeOutputRate(value);
    if (legacyTarget !== null) cfg.targetFps = legacyTarget;
    if (cfg.factor === next) {
      syncPanel();
      if (persist) saveCfg();
      return;
    }
    cfg.factor = next;
    delayMs = DELAY_MS;
    resetOutputCadence(true);
    syncPanel();
    if (persist) saveCfg();
  }

  function setTargetFps(value, persist = true) {
    const next = Cadence.sanitizeTargetFps(value, cfg.targetFps);
    if (cfg.targetFps === next) {
      syncPanel();
      return;
    }
    cfg.targetFps = next;
    delayMs = DELAY_MS;
    resetOutputCadence(true);
    syncPanel();
    if (persist) saveCfg();
  }

  function setFpsLimit(value, persist = true) {
    const next = canonicalFpsLimit(value);
    if (cfg.fpsLimit === next) {
      syncPanel();
      return;
    }
    cfg.fpsLimit = next;
    delayMs = DELAY_MS;
    resetOutputCadence(true);
    syncPanel();
    if (persist) saveCfg();
  }

  function setFrameGeneration(enabled, persist = true) {
    const next = !!enabled;
    if (cfg.fg === next) return;
    cfg.fg = next;
    delayMs = DELAY_MS;
    resetOutputCadence(true);
    overSince = 0;
    syncPanel();
    if (persist) saveCfg();
  }

  // cross-origin video taints the pixel path (SecurityError on copy). Our DNR rule
  // injects ACAO:* on media responses, so reloading the element in CORS mode makes
  // it readable - one reload, playback position preserved, reverted on failure.
  async function makeReadable(v) {
    if (v.crossOrigin === 'anonymous') throw new Error('video unreadable even with CORS');
    const t = v.currentTime, playing = !v.paused;
    v.crossOrigin = 'anonymous';
    v.load();
    try {
      await new Promise((res, rej) => {
        const ok = () => { cleanup(); res(); };
        const bad = () => { cleanup(); rej(new Error('CDN refuses CORS for this video')); };
        const timer = setTimeout(bad, 8000);
        const cleanup = () => {
          clearTimeout(timer);
          v.removeEventListener('loadeddata', ok);
          v.removeEventListener('error', bad);
        };
        v.addEventListener('loadeddata', ok);
        v.addEventListener('error', bad);
      });
    } catch (e) {
      v.removeAttribute('crossorigin'); // put the player back the way it was
      v.load();
      v.currentTime = t;
      if (playing) v.play().catch(() => {});
      throw e;
    }
    v.currentTime = t;
    if (playing) v.play().catch(() => {});
    // loadeddata != decoded frame: copying right away throws "no back resource".
    // Wait for a real presented frame (rVFC), with a timeout so a stalled decoder
    // can't wedge the start path.
    await new Promise((res) => {
      let done = false;
      const fin = () => { if (!done) { done = true; res(); } };
      v.requestVideoFrameCallback(() => fin());
      setTimeout(fin, 1500);
    });
    log('video reloaded with CORS - pixels readable now');
  }

  function onSrcChange() {
    resetOutputCadence(true);
    delayMs = DELAY_MS;
    resetFramePoolOnNextCapture = true;
    lastTex = null; schedT = 0; lastArrival = 0; lastUniqueTs = 0;
    resetDecodedSourceCadence();
    lastPumpT = 0;
    refreshEstimate.stableSamples = 0;
    overlayFitSupported = false;
    overlaySourceWidth = 0;
    overlaySourceHeight = 0;
    if (overlay) {
      // hide INSTANTLY: a fade would blend the dead stream's last frame over the
      // new one for 250ms. present() restores the transition on the next real frame
      overlay.style.transition = 'none';
      overlay.style.opacity = '0';
      overlay.style.visibility = 'hidden';
      overlay.style.pointerEvents = 'none';
    }
  }
  function onPlaybackRateChange() {
    resetOutputCadence(true);
    delayMs = DELAY_MS;
    resetFramePoolOnNextCapture = true;
    lastTex = null;
    schedT = 0;
    lastArrival = 0;
    lastUniqueTs = 0;
    intervalMs = decodedIntervalMs
      / Math.max(0.01, Math.abs(Number(videoEl?.playbackRate) || 1));
    uniqueIntervalMs = intervalMs;
  }
  let srcWatchEl = null;
  async function start(v) {
    if (running && videoEl === v) return; // re-entry insurance: never double-arm the rVFC/rAF loops
    running = false;
    const startEpoch = invalidatePlaybackLoops();
    videoEl = v;
    trackSourceVideo(v);
    delayMs = DELAY_MS;
    resetFramePoolOnNextCapture = true;
    if (srcWatchEl !== v) {
      if (srcWatchEl) {
        srcWatchEl.removeEventListener('emptied', onSrcChange);
        srcWatchEl.removeEventListener('seeking', onSrcChange);
        srcWatchEl.removeEventListener('ratechange', onPlaybackRateChange);
      }
      srcWatchEl = v;
      v.addEventListener('emptied', onSrcChange);
      v.addEventListener('seeking', onSrcChange);
      v.addEventListener('ratechange', onPlaybackRateChange);
    }
    await ensureRuntime();
    if (startEpoch !== playbackLoopEpoch || videoEl !== v) return;
    ensureOverlay();
    positionOverlay();
    overlay.style.opacity = '0';
    overlay.style.display = 'block';
    // native players: we own clicks (play/pause + our fullscreen). Sites with DOM
    // controls (YouTube etc.) keep the overlay transparent to the pointer.
    overlay.style.pointerEvents = overlayFitSupported && videoEl.controls ? 'auto' : 'none';
    // seed the canvas with the current video frame so the reveal is seamless -
    // no black flash while the first interpolated frames are still in flight
    const [vw, vh] = videoEl.videoWidth && videoEl.videoHeight ? poolDims() : [0, 0];
    if (vw && vh) {
      ensureFrameTextures(vw, vh);
      const seed = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      try {
        captureFrame(seed, vw, vh);
        present(seed, false);
      } catch (e) {
        if (e.name === 'SecurityError' || String(e).includes('cross-origin')) {
          hud.style.display = 'block';
          hud.textContent = 'FC: video lacks CORS - reloading…';
          await makeReadable(videoEl); // throws a friendly error if the CDN refuses
          if (startEpoch !== playbackLoopEpoch || videoEl !== v) return;
          // seed is cosmetic (seamless fade-in): if the decoder still has no frame,
          // skip it - the rVFC pipeline below presents the first real frame anyway
          try {
            captureFrame(seed, vw, vh);
            present(seed, false);
          } catch (e2) { log('seed skipped', e2.name); }
        } else if (e.name === 'OperationError') {
          log('seed skipped', e.name); // no decoded frame yet - rVFC will deliver
        } else throw e;
      }
    }
    if (cfg.sr) ensureSR().catch(e => log('sr', e));
    resetOutputCadence(true);
    lastTex = null; schedT = 0; lastArrival = 0; lastUniqueTs = 0;
    resetDecodedSourceCadence();
    lastPumpT = 0;
    refreshEstimate.stableSamples = 0;
    lastVr = null; // the cached rect belongs to the previous video element
    dropped = 0; dups = 0; cuts = 0;
    if (startEpoch !== playbackLoopEpoch || videoEl !== v) return;
    running = true;
    hud.style.display = 'block';
    if (sys.integrated) {
      advise('⚠ Chrome is running on the integrated GPU (' + sys.gpu + '). For full speed: '
        + 'Windows Settings → Display → Graphics → Chrome → High performance, '
        + 'then restart Chrome.', 14000);
    }
    diag.pumpStarts++;
    diag.sourceLoopStarts++;
    armVideoFrameLoop(startEpoch);
    armPumpLoop(startEpoch);
    btn.style.background = 'rgba(25,195,125,.9)';
  }
  function stop() {
    running = false;
    diag.loopStops++;
    invalidatePlaybackLoops();
    const stopEpoch = playbackLoopEpoch;
    if (overlay) { // fade out, then release - the raw video underneath is identical
      overlay.style.opacity = '0';
      setTimeout(() => {
        if (!running && stopEpoch === playbackLoopEpoch && overlay) {
          overlay.style.display = 'none';
        }
      }, 260);
    }
    hud.style.display = 'none';
    if (wm) wm.style.display = 'none';
    if (bar) bar.style.display = 'none';
    hideWarnings();
    if (splitEl) splitEl.style.display = 'none';
    resetOutputCadence(true);
    delayMs = DELAY_MS;
    lastTex = null; schedT = 0; lastArrival = 0; lastUniqueTs = 0;
    resetDecodedSourceCadence();
    btn.style.background = '';
  }

  function biggestVideo() {
    // rank by VISIBLE area in the viewport, not raw size: virtualized feeds
    // (TikTok) keep several same-sized players mounted and rotate them through
    // the viewport - an off-screen one must never win. Playing beats paused.
    let best = null, score = 0;
    for (const v of document.querySelectorAll('video')) {
      if (v.readyState < 2) continue;
      const r = v.getBoundingClientRect();
      const vis = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0))
                * Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const s = vis * (v.paused ? 0.5 : 1);
      if (s > score) { score = s; best = v; }
    }
    return best;
  }

  // live system/status readout in the settings panel: adapter, f16, fps, cost,
  // our estimated GPU load (interp time vs the per-unique-frame budget), VRAM
  function updateStatus() {
    const st = panel && panel.querySelector('#fcStatus');
    if (!st) return;
    const ratePlan = stableOutputRatePlan(currentOutputRatePlan());
    const label = outputRateLabel();
    const rateState = !ratePlan.interpolationAllowed && usesExactCadence()
      ? `${label} · ${ratePlan.warning}`
      : ratePlan.clamped
        ? `${label} → ${Number(ratePlan.outputHz.toFixed(2))} FPS (${ratePlan.clampReason})`
        : ratePlan.outputHz
          ? `${label} (~${Number(ratePlan.outputHz.toFixed(2))} FPS)`
          : label;
    const srState = cfg.sr ? (!sys.f16 ? 'unavailable (no f16)' : (sr ? 'on x2' : 'loading…')) : 'off';
    const lines = [`GPU: ${sys.gpu}${sys.integrated ? ' ⚠ INTEGRATED' : ''}`,
      `f16: ${sys.f16 ? 'yes' : 'NO (slow path)'} · model: ${rtModel ? MODELS[rtModel] : MODELS[cfg.model] || cfg.model}`,
      `FG: ${cfg.fg ? 'on' : 'OFF'} · SR: ${srState}`,
      `output rate: ${rateState}`,
      `HDR: ${!sys.hdrOk ? 'display not HDR' : (cfg.hdr ? (sys.hdrOn ? 'on (ITM)' : 'failed, SDR') : 'off')}`,
      `status: ${running ? 'running' : 'stopped'}`];
    if (running) {
      const [mw, mh] = SIZES[cfg.res];
      const vramMB = (texW * texH * 4 * frameTex.length + mw * mh * 4 * midTexs.length) / 1048576;
      const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
      lines.push(
        `out: ${fpsWin.length}fps · effective pair x${effN}`,
        `display: ~${rafMs > 1 ? (1000 / rafMs).toFixed(0) : '-'}Hz`,
        `mid: ${msAvg.toFixed(1)}ms @ ${cfg.res}p`,
        `GPU load (ours, est.): ~${load.toFixed(0)}%`,
        `VRAM textures: ~${vramMB.toFixed(0)}MB · queue ${queue.length}`);
    }
    st.textContent = lines.join('\n');
  }

  // hot-swap the runtime on quality change: stop/start reseeded the canvas with a
  // LIVE frame while the pipeline serves ~delayMs-old ones - time visibly jumped
  // forward and snapped back. Instead: drain the in-flight batch, drop queued mids
  // (source frames keep presenting), rebuild rt at the new size, relearn timing.
  async function switchRes() {
    switching = true; // gates onFrame/runPair: NO new mids while textures are being replaced
    try {
      // loop: the user may flip the select again mid-rebuild - converge on the latest.
      // model counts too: a model-only change used to skip the loop entirely and
      // silently keep the old weights until the next res change or restart
      while (!runtimeMatches()) {
        curJob = null; // abandon un-submitted mids of the old pair
        queue = queue.filter((it) => !it.mid);
        msAvg = 0; lastMidCostAt = 0; midCostSamples.length = 0; // new size, relearn
        await ensureRuntime();
        queue = queue.filter((it) => !it.mid); // stragglers that slipped in mid-rebuild
      }
    } finally { switching = false; }
  }

  // Keep the quick panel inside the viewport on small and embedded players.
  function clampPanel() {
    if (!panel || panel.style.display !== 'block') return;
    const r = panel.getBoundingClientRect();
    if (r.bottom > innerHeight - 10) panel.style.top = Math.max(10, innerHeight - r.height - 10) + 'px';
    if (r.right > innerWidth - 10) panel.style.left = Math.max(10, innerWidth - r.width - 10) + 'px';
  }

  async function openAdvancedSettings() {
    const button = panel?.querySelector('#fcOpenSettings');
    if (button) button.disabled = true;
    try {
      const response = await chrome.runtime.sendMessage({ type: 'fcOpenOptions' });
      if (!response?.ok) throw new Error(response?.error || 'Could not open settings');
    } catch (messageError) {
      try {
        await chrome.runtime.openOptionsPage();
      } catch (directError) {
        log('open settings', directError || messageError);
      }
    } finally {
      if (button) button.disabled = false;
    }
  }

  function buildPanel() {
    panel = document.createElement('div');
    panel.className = 'fc-panel';
    panel.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'background:#111315; color:#ddd; border:1px solid #303338;'
      + 'border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.4);'
      + 'padding:14px 16px 15px; font:12px/1.5 system-ui; display:none; width:332px; box-sizing:border-box;'
      + 'max-height:calc(100vh - 20px); overflow-y:auto; overscroll-behavior:contain;';
    panel.innerHTML = `
      <div class="fc-panel-head">
        <div class="fc-brand">
          <span class="fc-brand-dot" aria-hidden="true"></span>
          <strong>Framegen</strong>
          <span class="fc-version">v${VERSION}</span>
        </div>
        <a class="fc-icon-link" href="https://github.com/MONZikWasTaken/Framegen"
          target="_blank" rel="noopener noreferrer" title="Open Framegen on GitHub"
          aria-label="Open Framegen on GitHub">
          <svg width="17" height="17" viewBox="0 0 24 24" aria-hidden="true">
            <path fill="currentColor" d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.38.97.1-.75.4-1.27.74-1.56-2.57-.3-5.27-1.29-5.27-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.16 1.18a10.96 10.96 0 0 1 5.75 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.41-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"/>
          </svg>
        </a>
      </div>
      <label class="fc-row fc-profile-row"><span>Profile<small>Apply a saved setup</small></span>
        <select class="fc-sel fc-profile" id="fcProfile" disabled>
          <option>Loading profiles…</option>
        </select></label>
      <div class="fc-divider"></div>
      <label class="fc-row"><span>Frame generation<small>Create smoother motion</small></span>
        <input class="fc-sw" type="checkbox" id="fcFG"></label>
      <label class="fc-row"><span>Output rate<small>Choose how playback is paced</small></span>
        <select class="fc-sel" id="fcFactor">
          <option value="auto">Auto · recommended</option>
          <option value="hz">Match display</option>
          <option value="target">Custom FPS</option>
          <option value="2">2× source</option><option value="3">3× source</option>
          <option value="4">4× source</option><option value="5">5× source</option>
          <option value="6">6× source</option>
        </select></label>
      <label class="fc-target-control">
        <span class="fc-target-head"><span><span id="fcRateSliderTitle">FPS limit</span>
          <small id="fcRateSliderHint">Common rates · Auto may run lower</small></span>
          <output id="fcTargetFpsValue" for="fcTargetFps">Unlimited</output></span>
        <input class="fc-target-slider" id="fcTargetFps" type="range" min="0" max="17"
          step="1" value="17" aria-label="FPS limit">
        <span class="fc-target-scale" id="fcRateSliderScale" aria-hidden="true">
          <span>15</span><span>60</span><span>120</span><span>240</span><span>∞</span>
        </span>
      </label>
      <label class="fc-row"><span>Upscale<small>2× neural resolution boost</small></span>
        <input class="fc-sw" type="checkbox" id="fcSR"></label>
      <label class="fc-row"><span>HDR<small>Brighter highlights on HDR displays</small></span>
        <input class="fc-sw" type="checkbox" id="fcHDR"></label>
      <label class="fc-row"><span>Quality<small>Balance detail and GPU load</small></span>
        <select class="fc-sel" id="fcRes">
          <option value="288">Low power</option><option value="360">Efficient</option>
          <option value="480">Balanced</option>
          <option value="720">High</option>
          <option value="1080">Ultra</option>
        </select></label>
      <label class="fc-row"><span>FPS counter<small>Show frame rate and render time</small></span>
        <input class="fc-sw" type="checkbox" id="fcShowFps"></label>
      <label class="fc-row"><span>Watermark<small>Show the Framegen label on video</small></span>
        <input class="fc-sw" type="checkbox" id="fcWatermark"></label>
      <label class="fc-row"><span>Warnings<small>Show non-critical notices over video</small></span>
        <input class="fc-sw" type="checkbox" id="fcWarnings"></label>
      <button class="fc-open-settings" id="fcOpenSettings" type="button">
        <span>Advanced settings</span>
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
          <path fill="currentColor" d="m6 3 5 5-5 5-1.2-1.2L8.6 8 4.8 4.2 6 3Z"/>
        </svg>
      </button>`;
    document.body.appendChild(panel);
    const F = panel.querySelector('#fcFactor'), R = panel.querySelector('#fcRes');
    syncPanel();
    loadPanelProfiles().catch(e => log('profiles', e));
    panel.querySelector('#fcOpenSettings').onclick = openAdvancedSettings;
    panel.querySelector('#fcProfile').onchange = async event => {
      const select = event.currentTarget;
      select.disabled = true;
      try { await applyPanelProfile(select.value); }
      catch (e) { log('apply profile', e); }
      finally { select.disabled = false; syncPanelProfileSelection(); }
    };
    F.onchange = () => setOutputRate(F.value);
    const Tf = panel.querySelector('#fcTargetFps');
    Tf.oninput = () => updateRateSliderPresentation(Tf);
    Tf.onchange = () => cfg.factor === 'auto'
      ? setFpsLimit(fpsLimitFromSlider(Tf))
      : setTargetFps(Tf.value);
    const Fg = panel.querySelector('#fcFG'), Sr = panel.querySelector('#fcSR');
    Fg.onchange = () => setFrameGeneration(Fg.checked);
    Sr.onchange = () => {
      cfg.sr = Sr.checked; saveCfg();
      if (cfg.sr && device) ensureSR().catch(e => log('sr', e));
    };
    const Hd = panel.querySelector('#fcHDR');
    Hd.onchange = () => { cfg.hdr = Hd.checked; saveCfg(); configureOverlay(); };
    panel.querySelector('#fcShowFps').onchange = event => {
      cfg.showFps = event.currentTarget.checked;
      saveCfg();
      syncHudVisibility();
    };
    panel.querySelector('#fcWatermark').onchange = event => {
      cfg.showWatermark = event.currentTarget.checked;
      saveCfg();
      if (!cfg.showWatermark && wm) wm.style.display = 'none';
    };
    panel.querySelector('#fcWarnings').onchange = event => {
      cfg.showWarnings = event.currentTarget.checked;
      saveCfg();
      if (!cfg.showWarnings) hideWarnings();
    };
    R.onchange = async () => {
      cfg.res = +R.value; saveCfg();
      if (running && !toggling) { // hot-swap, no visible restart
        toggling = true;
        try { await switchRes(); }
        catch (e) { log('res switch', e); }
        finally { toggling = false; }
      }
    };
  }

  async function toggleFC() {
    if (toggling) return; // start/stop in flight - spam-proof
    if (!btn) injectUI(); // popup can toggle before the in-page UI ever booted
    toggling = true;
    try {
      if (running) { stop(); return; }
      const v = biggestVideo();
      if (!v) { hud.style.display = 'block'; hud.textContent = 'FC: no video found'; return; }
      try { await start(v); } catch (e) { hud.style.display = 'block'; hud.textContent = 'FG error: ' + (e.message || e); log(e); }
    } finally { toggling = false; }
  }

  function injectUI() {
    const css = document.createElement('style');
    css.textContent = `
      .fc-btn{background:none;border:none;color:#fff;font:13px/1 system-ui;cursor:pointer;
        padding:4px 6px;opacity:.85;transition:opacity .15s,transform .15s;
        display:inline-flex;align-items:center;justify-content:center}
      .fc-btn:hover{opacity:1;transform:scale(1.15)}
      .fc-side svg{display:block;margin:auto}
      .fc-range{-webkit-appearance:none;appearance:none;height:3px;border-radius:3px;margin:0;
        background:rgba(255,255,255,.22);outline:none;cursor:pointer}
      .fc-range::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;
        border-radius:50%;background:#fff;transition:transform .15s}
      .fc-range:hover::-webkit-slider-thumb{transform:scale(1.35);background:#19c37d}
      /* NO backdrop-filter on anything hovering over the RUNNING video: the
         compositor re-blurs the region every frame of a 100+fps canvas - that
         alone janks playback exactly while the cursor summons the UI */
      .fc-side{position:fixed;z-index:2147483647;
        width:38px!important;height:38px!important;
        min-width:38px!important;min-height:38px!important;
        max-width:38px!important;max-height:38px!important;
        padding:0!important;margin:0!important;box-sizing:border-box!important;
        border-radius:50%;
        border:none;background:rgba(18,18,20,.88);color:#fff;cursor:pointer;display:none;
        font:600 12px/1 system-ui;box-shadow:0 2px 12px rgba(0,0,0,.4);
        transition:background .15s,transform .15s}
      .fc-side:hover{transform:scale(1.1);background:rgba(45,45,45,.85)}
      .fc-panel-head{height:30px;display:flex;align-items:center;justify-content:space-between;
        margin:0 0 8px;padding:0;border:0}
      .fc-brand{display:flex;align-items:center;gap:7px;color:#f4f5f5;font:12px/1 system-ui}
      .fc-brand strong{font:650 14px/1 system-ui;color:#f4f5f5}
      .fc-brand-dot{width:8px;height:8px;border-radius:50%;background:#19c37d}
      .fc-version{color:#747980;font:400 10px/1 system-ui}
      .fc-icon-link{width:28px;height:28px;display:flex;align-items:center;justify-content:center;
        border-radius:7px;color:#92979e;text-decoration:none;outline:none;
        transition:background .15s,color .15s}
      .fc-icon-link:hover{background:#202327;color:#f2f3f3}
      .fc-icon-link:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
      .fc-divider{height:1px;background:#292c30;margin:7px 0}
      .fc-row{display:flex;justify-content:space-between;align-items:center;gap:14px;
        padding:8px 0;margin:0;border:0;width:auto;cursor:default;
        font:12px/1.4 system-ui;color:#e8e8e8;text-align:left}
      .fc-row[hidden]{display:none}
      .fc-row>span{display:block;flex:1 1 auto;min-width:0;margin:0;padding:0;
        font:12px/1.4 system-ui;color:#e8e8e8;text-align:left;
        letter-spacing:normal;text-transform:none;white-space:normal}
      .fc-row small{display:block;color:#8a8f98;font:400 10px/1.3 system-ui;
        margin:1px 0 0;padding:0;letter-spacing:normal;text-transform:none}
      .fc-profile-row{padding-top:5px;padding-bottom:7px}
      .fc-sw{appearance:none;-webkit-appearance:none;width:36px;height:20px;border-radius:20px;
        background:#3d4148;position:relative;cursor:pointer;outline:none;margin:0;
        transition:background .2s;flex:none}
      .fc-sw::after{content:'';position:absolute;width:16px;height:16px;border-radius:50%;
        background:#fff;top:2px;left:2px;transition:left .2s;box-shadow:0 1px 3px rgba(0,0,0,.4)}
      .fc-sw:checked{background:#19c37d}
      .fc-sw:checked::after{left:18px}
      .fc-sw:focus-visible{outline:2px solid #19c37d;outline-offset:2px}
      /* customizable select (Chrome base-select): button + popup in the same glass */
      .fc-sel, .fc-sel::picker(select){appearance:base-select}
      .fc-sel{background:#1b1e21;color:#eee;border:1px solid #383c41;
        border-radius:8px;padding:6px 10px;font:12px system-ui;outline:none;cursor:pointer;
        flex:none;min-width:154px;max-width:174px;display:flex;align-items:center;justify-content:space-between;
        gap:8px;transition:background .15s,border-color .15s}
      .fc-sel:hover{background:#22262a;border-color:#50555c}
      .fc-sel:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
      .fc-sel:open{border-color:rgba(25,195,125,.6)}
      .fc-sel::picker-icon{color:#8a8f98;font-size:9px;transition:rotate .15s}
      .fc-sel:open::picker-icon{rotate:180deg}
      .fc-sel::picker(select){background:rgba(20,22,26,.95);backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.14);border-radius:10px;padding:4px;margin-top:4px;
        box-shadow:0 8px 28px rgba(0,0,0,.55)}
      .fc-sel option{padding:5px 10px;border-radius:7px;font:12px system-ui;color:#ddd;
        background:transparent;cursor:pointer}
      .fc-sel option:hover{background:rgba(255,255,255,.09)}
      .fc-sel option:checked{background:rgba(25,195,125,.16);color:#8ee7bd}
      .fc-sel option::checkmark{color:#19c37d}
      .fc-target-control{display:block;padding:2px 0 8px;margin:0;cursor:pointer}
      .fc-target-control[hidden]{display:none}
      .fc-target-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
        margin:0 0 4px;color:#e8e8e8;font:12px/1.35 system-ui}
      .fc-target-head>span{display:block}
      .fc-target-head small{display:block;margin-top:1px;color:#8a8f98;font:400 10px/1.3 system-ui}
      .fc-target-head output{min-width:72px;color:#7fe0b5;font:600 11px/1.35 system-ui;
        font-variant-numeric:tabular-nums;text-align:right}
      .fc-target-slider{--fc-fill:12%;appearance:none;-webkit-appearance:none;display:block;
        width:100%;height:20px;margin:0;background:transparent;cursor:pointer;outline:none}
      .fc-target-slider::-webkit-slider-runnable-track{height:3px;border-radius:3px;
        background:linear-gradient(to right,#19c37d 0 var(--fc-fill),#3d4148 var(--fc-fill) 100%)}
      .fc-target-slider::-webkit-slider-thumb{appearance:none;-webkit-appearance:none;width:15px;height:15px;
        margin-top:-6px;border:2px solid #111315;border-radius:50%;background:#f2f4f3;
        box-shadow:0 0 0 1px #62686f}
      .fc-target-slider:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
      .fc-target-scale{position:relative;display:block;height:10px;margin-top:-2px;
        color:#656b73;font:500 9px/1 system-ui;font-variant-numeric:tabular-nums;pointer-events:none}
      .fc-target-scale>span{position:absolute;
        left:calc(var(--fc-mark-position) + var(--fc-mark-offset));top:0;
        transform:translateX(-50%);white-space:nowrap}
      .fc-open-settings{width:100%;height:35px;display:flex;align-items:center;justify-content:space-between;
        margin-top:10px;padding:0 11px;border-radius:8px;border:1px solid #383c41;background:#1b1e21;
        color:#e2e4e5;font:600 11px system-ui;cursor:pointer;outline:none;
        transition:background .15s,border-color .15s}
      .fc-open-settings:hover{background:#22262a;border-color:#50555c}
      .fc-open-settings:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
      .fc-open-settings:disabled{opacity:.55;cursor:wait}`;
    (document.head || document.documentElement).appendChild(css);
    btn = document.createElement('button');
    btn.textContent = 'FG';
    btn.className = 'fc-side';
    gear = document.createElement('button');
    gear.innerHTML = svgIcon('gear', 17);
    gear.className = 'fc-side';
    hud = document.createElement('div');
    // single line, pinned to the video's TOP-LEFT; plain dark bar, white text
    hud.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'color:#fff; font:11px/1.5 ui-monospace,monospace; background:rgba(0,0,0,.72);'
      + 'padding:3px 9px; white-space:normal; pointer-events:none; display:none;';
    wm = document.createElement('div');
    // permanent brand mark: bottom-left inside the player, bare white text
    wm.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483645;'
      + 'color:#fff; font:600 12px system-ui; opacity:.75; pointer-events:none;'
      + 'text-shadow:0 1px 3px rgba(0,0,0,.8); display:none;';
    wm.textContent = 'Framegen';
    document.body.appendChild(wm);
    buildPanel();
    ensureBar();
    btn.onclick = toggleFC;
    gear.onclick = () => {
      const open = panel.style.display === 'none';
      panel.style.display = open ? 'block' : 'none';
      if (open) { // dock next to the gear, clamped to the viewport
        const g = gear.getBoundingClientRect();
        panel.style.left = Math.min(g.right + 10, innerWidth - panel.offsetWidth - 10) + 'px';
        panel.style.top = Math.max(10, Math.min(g.top - panel.offsetHeight / 2, innerHeight - panel.offsetHeight - 10)) + 'px';
        updateStatus();
        clampPanel();
      }
    };
    document.body.appendChild(btn);
    document.body.appendChild(gear);
    document.body.appendChild(hud);
  }

  // toolbar popup protocol: status snapshot + remote toggle. With all_frames every
  // frame gets the message; the RUNNING frame answers instantly, a frame that merely
  // has a video answers after 120ms, video-less frames after 250ms - first response
  // wins, so the most relevant frame speaks for the tab.
  const VERSION = '1.4.2';
  try {
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if (msg && msg.type === 'fcStatus') {
        const v = videoEl || biggestVideo();
        const respond = async () => { await probeAdapter();
          const ratePlan = stableOutputRatePlan(currentOutputRatePlan());
          try { sendResponse({
          version: VERSION, gpu: sys.gpu, integrated: sys.integrated, f16: sys.f16,
          hasVideo: !!v, running, fps: fpsWin.length, effN,
          ms: +(msAvg || 0).toFixed(1), res: cfg.res, factor: cfg.factor,
          targetFps: cfg.targetFps, fpsLimit: cfg.fpsLimit,
          rateLabel: outputRateLabel(), targetHz: ratePlan.requestedHz,
          effectiveTargetHz: ratePlan.outputHz, rateClamped: ratePlan.clamped,
          targetState: ratePlan.state, targetReason: ratePlan.clampReason,
          targetWarning: ratePlan.warning,
          drops: dropped, model: rtModel ? MODELS[rtModel] : MODELS[cfg.model] || cfg.model,
        }); } catch {} };
        if (running) respond();
        else setTimeout(respond, v ? 120 : 250);
        return true;
      }
      if (msg && msg.type === 'fcToggle') {
        const v = videoEl || biggestVideo();
        if (!running && !v) return undefined; // let a frame that HAS video take it
        toggleFC().then(() => { try { sendResponse({ running }); } catch {} });
        return true;
      }
    });
  } catch { /* messaging unavailable in some frames */ }

  function installProductBenchBridge() {
    if (!PRODUCT_BENCH) return;
    const reply = (id, ok, value, error) => window.postMessage({
      source: 'framegen-product-bench-extension',
      token: PRODUCT_BENCH.token,
      id,
      ok,
      value,
      error,
    }, location.origin);
    window.addEventListener('message', event => {
      if (event.source !== window) return;
      const message = event.data;
      if (!message || message.source !== 'framegen-product-bench-fixture'
          || message.token !== PRODUCT_BENCH.token || !Number.isInteger(message.id)) return;
      (async () => {
        switch (message.command) {
          case 'ping':
            return { bridgeVersion: 2, extensionVersion: VERSION };
          case 'configure': {
            if (running) throw new Error('configure requires a stopped product path');
            const requestedFactor = message.payload?.factor;
            const numericFactor = Number(requestedFactor);
            const factor = numericFactor === 3 || numericFactor === 4
              ? numericFactor : Cadence.sanitizeOutputRate(requestedFactor);
            const targetFps = factor === 'target'
              ? Cadence.sanitizeTargetFps(message.payload?.targetFps)
              : cfg.targetFps;
            const requestedResolution = Number(message.payload?.resolution ?? 720);
            if (![3, 4, 'target', 'hz'].includes(factor) || (factor === 'target' && targetFps === null)) {
              throw new Error('factor must be 3, 4, hz or target with a positive targetFps');
            }
            if (!SIZES[requestedResolution]) throw new Error('resolution is not supported');
            Object.assign(cfg, {
              factor,
              targetFps,
              anime: false,
              debug: false,
              res: requestedResolution,
              hoverReveal: false,
              compare: false,
              fg: true,
              sr: false,
              hdr: false,
              showFps: false,
              showWatermark: false,
              guard: true,
              model: 'v7s',
            });
            sanitizeCfg();
            syncPanel();
            benchAppliedTune = null;
            return { factor: cfg.factor, targetFps: cfg.targetFps,
              resolution: cfg.res, model: cfg.model };
          }
          case 'prepare': {
            if (running) throw new Error('prepare requires a stopped product path');
            await ensureRuntime();
            let convTune = await loadConvTune();
            if (!convTune) {
              const rtMod = await import(chrome.runtime.getURL('rt/rt.js'));
              await calibrateConvTune(rtMod);
              convTune = await loadConvTune();
              if (!convTune) throw new Error('conv autotune did not produce a persisted result');
              rtRes = 0;
              await ensureRuntime();
            }
            clearTimeout(tuneTimer);
            benchAppliedTune = convTune;
            return { convTune: JSON.parse(JSON.stringify(convTune)), gpu: sys.gpu,
              deviceFeatures: [...device.features].sort() };
          }
          case 'start': {
            const video = biggestVideo();
            if (!video) throw new Error('no ready benchmark video');
            await start(video);
            return { running, width: video.videoWidth, height: video.videoHeight };
          }
          case 'reset':
            if (!running) throw new Error('cannot reset telemetry while stopped');
            return resetBenchTelemetry();
          case 'snapshot':
            if (!benchTelemetry) throw new Error('telemetry has not been reset');
            return snapshotBenchTelemetry();
          case 'stop':
            stop();
            return { running };
          default:
            throw new Error('unknown product benchmark command');
        }
      })().then(value => reply(message.id, true, value, null))
        .catch(error => reply(message.id, false, null, error.stack || error.message || String(error)));
    });
    window.postMessage({
      source: 'framegen-product-bench-extension',
      token: PRODUCT_BENCH.token,
      id: 0,
      ok: true,
      value: { bridgeVersion: 2, extensionVersion: VERSION },
      error: null,
    }, location.origin);
  }

  const bootObs = new MutationObserver(() => boot());
  const boot = () => {
    if (btn) { bootObs.disconnect(); return; } // UI exists - stop watching the whole DOM
    if (document.querySelector('video')) { injectUI(); bootObs.disconnect(); }
  };
  boot();
  if (!btn) bootObs.observe(document.documentElement, { childList: true, subtree: true });
  installProductBenchBridge();
})();
