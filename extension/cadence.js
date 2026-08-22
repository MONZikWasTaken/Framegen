(function installFramegenCadence(root) {
  'use strict';

  const DISPLAY_RATES = Object.freeze([360, 240, 180, 165, 144, 120, 100, 90, 75, 72, 60, 50]);
  const COMMON_VIDEO_RATES = Object.freeze([
    24000 / 1001, 24, 25, 30000 / 1001, 30, 48000 / 1001, 48, 50,
    60000 / 1001, 60, 72, 75, 90, 100, 120000 / 1001, 120,
  ]);
  const LEGACY_TARGETS = Object.freeze({ fps60: 60, fps120: 120 });
  const DEFAULT_TARGET_FPS = 120;
  const MIN_TARGET_FPS = 2;
  const MAX_TARGET_FPS = 1000;
  const MAX_TICKS_PER_INTERVAL = 64;
  const MAX_PENDING_PRESENTATIONS = 24;
  const MAX_MIDS_PER_PAIR = MAX_PENDING_PRESENTATIONS - 1;
  const MAX_RECOVERY_PRESENTATIONS = 3;
  const DISPLAY_CLAMP_HEADROOM = 0.97;
  const REFRESH_TRANSITION_SAMPLES = 10;
  const NOMINAL_RATE_TOLERANCE = 0.03;
  const VIDEO_RATE_NORMALIZE_TOLERANCE = 0.015;
  const VIDEO_RATE_HOLD_TOLERANCE = 0.03;
  const VIDEO_RATE_MATCH_TOLERANCE = 0.005;
  const DISPLAY_SAMPLE_RELATIVE_TOLERANCE = 0.08;
  const DISPLAY_SAMPLE_QUANTIZATION_MS = 1.05;
  const DISPLAY_STABILITY_CV = 0.12;
  const DISPLAY_SERVICE_WINDOW = 32;
  const DISPLAY_FASTER_MISS_BUDGET = 3;
  const SOURCE_INTERVAL_WINDOW = 32;
  const SOURCE_TRANSITION_TOLERANCE = 0.12;
  const SOURCE_TRANSITION_SAMPLES = 4;
  const AUTO_FAST_SIGNAL_WINDOW = 4;
  const AUTO_PROBE_INITIAL_DELAY_MS = 2500;
  const AUTO_PROBE_MAX_DELAY_MS = 20000;
  const AUTO_PROBE_MAX_ATTEMPTS = 4;

  function sanitizeOutputRate(value) {
    if (value === 'auto' || value === 'hz' || value === 'target') return value;
    if (Object.hasOwn(LEGACY_TARGETS, value)) return 'target';
    const factor = Number(value);
    return Number.isInteger(factor) && factor >= 2 && factor <= 6 ? factor : 'auto';
  }

  function isCadenceMode(value) {
    const mode = sanitizeOutputRate(value);
    return mode === 'hz' || mode === 'target';
  }

  function sanitizeTargetFps(value, fallback = null) {
    const numeric = typeof value === 'string' && value.trim() === '' ? NaN : Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return fallback;
    return Math.min(MAX_TARGET_FPS, Math.max(MIN_TARGET_FPS,
      Math.round(numeric * 100) / 100));
  }

  function normalizeVideoRate(value, tolerance = VIDEO_RATE_NORMALIZE_TOLERANCE) {
    if (!Number.isFinite(value) || value <= 0) throw new RangeError('video rate must be positive');
    if (!Number.isFinite(tolerance) || tolerance < 0) throw new RangeError('video rate tolerance is invalid');
    let nearest = COMMON_VIDEO_RATES[0];
    for (const rate of COMMON_VIDEO_RATES) {
      if (Math.abs(value - rate) / rate < Math.abs(value - nearest) / nearest) nearest = rate;
    }
    return Math.abs(value - nearest) / nearest <= tolerance ? nearest : value;
  }

  function estimateSourceCadence(intervalSamples, fallbackIntervalMs = 1000 / 24) {
    if (!Array.isArray(intervalSamples)) throw new TypeError('source cadence samples must be an array');
    if (!Number.isFinite(fallbackIntervalMs) || fallbackIntervalMs <= 0) {
      throw new RangeError('source cadence fallback must be positive');
    }
    const valid = intervalSamples.filter(value => Number.isFinite(value) && value > 0.5 && value < 2000)
      .sort((a, b) => a - b);
    if (!valid.length) {
      const sourceHz = normalizeVideoRate(1000 / fallbackIntervalMs);
      return { intervalMs: 1000 / sourceHz, sourceHz, rawHz: 1000 / fallbackIntervalMs,
        sampleCount: 0, normalized: sourceHz !== 1000 / fallbackIntervalMs };
    }
    const middle = valid.length >> 1;
    const medianMs = valid.length & 1 ? valid[middle] : (valid[middle - 1] + valid[middle]) / 2;
    // Media timestamps can jitter in quantized short/long pairs. A median follows
    // whichever side occupies the middle slot; a symmetric trimmed mean cancels
    // those pairs and rejects isolated doubled frames or stalls.
    const trimCount = valid.length >= 8 ? Math.max(1, Math.floor(valid.length * 0.1)) : 0;
    const robustSamples = trimCount > 0 && trimCount * 2 < valid.length
      ? valid.slice(trimCount, valid.length - trimCount)
      : valid;
    const robustIntervalMs = robustSamples.reduce((sum, value) => sum + value, 0) / robustSamples.length;
    const rawHz = 1000 / robustIntervalMs;
    const measuredHz = normalizeVideoRate(rawHz);
    const previousHz = 1000 / fallbackIntervalMs;
    let previousNominalHz = COMMON_VIDEO_RATES[0];
    for (const rate of COMMON_VIDEO_RATES) {
      if (Math.abs(previousHz - rate) / rate
          < Math.abs(previousHz - previousNominalHz) / previousNominalHz) {
        previousNominalHz = rate;
      }
    }
    const previousIsNominal = Math.abs(previousHz - previousNominalHz) / previousNominalHz < 1e-9;
    // Once a common decoded cadence is established, hold it through bounded
    // timestamp noise. A sustained real rate change still moves the rolling
    // median outside this band and acquires a new cadence.
    const holdPrevious = valid.length >= 8 && previousIsNominal
      && Math.abs(rawHz - previousNominalHz) / previousNominalHz <= VIDEO_RATE_HOLD_TOLERANCE;
    const sourceHz = holdPrevious ? previousNominalHz : measuredHz;
    return { intervalMs: 1000 / sourceHz, sourceHz, rawHz,
      sampleCount: valid.length, normalized: sourceHz !== rawHz };
  }

  function resetSourceTransition(transition) {
    transition.intervalMs = 0;
    transition.samples = 0;
    transition.direction = 0;
  }

  // Keep the robust window unbiased even while deciding whether an abrupt
  // interval change is a real source-rate transition. Dropping the first three
  // outliers made alternating short/long media timestamps converge on only the
  // long half of the pattern and permanently understated the decoded rate.
  function updateSourceInterval(state, sampleMs) {
    if (!state || typeof state !== 'object') throw new TypeError('source cadence state must be an object');
    if (!Number.isFinite(state.intervalMs) || state.intervalMs <= 0) {
      throw new RangeError('source cadence interval must be positive');
    }
    if (!Number.isFinite(sampleMs) || sampleMs <= 0.5 || sampleMs >= 2000) {
      throw new RangeError('source cadence sample is invalid');
    }
    if (!Array.isArray(state.samples)) state.samples = [];
    if (!state.transition || typeof state.transition !== 'object') state.transition = {};
    const transition = state.transition;
    if (!Number.isFinite(transition.intervalMs) || transition.intervalMs < 0
        || !Number.isInteger(transition.samples) || transition.samples < 0
        || ![-1, 0, 1].includes(transition.direction)) {
      resetSourceTransition(transition);
    }

    let acceptedSampleMs = sampleMs;
    let transitioned = false;
    const outsideCurrentBand = state.samples.length >= 8
      && Math.abs(sampleMs - state.intervalMs) / state.intervalMs > SOURCE_TRANSITION_TOLERANCE;
    if (outsideCurrentBand) {
      const direction = sampleMs > state.intervalMs ? 1 : -1;
      if (transition.samples > 0 && transition.direction === direction) {
        transition.samples += 1;
        transition.intervalMs += (sampleMs - transition.intervalMs) / transition.samples;
      } else {
        transition.intervalMs = sampleMs;
        transition.samples = 1;
        transition.direction = direction;
      }
      if (transition.samples >= SOURCE_TRANSITION_SAMPLES) {
        acceptedSampleMs = transition.intervalMs;
        state.samples.length = 0;
        state.intervalMs = acceptedSampleMs;
        resetSourceTransition(transition);
        transitioned = true;
      }
    } else {
      resetSourceTransition(transition);
    }

    state.samples.push(acceptedSampleMs);
    if (state.samples.length > SOURCE_INTERVAL_WINDOW) state.samples.shift();
    state.intervalMs = estimateSourceCadence(state.samples, state.intervalMs).intervalMs;
    return { intervalMs: state.intervalMs, transitioned };
  }

  function targetNeedsInterpolation(sourceHz, targetHz) {
    if (![sourceHz, targetHz].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('source and target rates must be positive');
    }
    const source = normalizeVideoRate(sourceHz);
    const target = normalizeVideoRate(targetHz);
    return target > source && (target - source) / source > VIDEO_RATE_MATCH_TOLERANCE;
  }

  function estimateAutoSourceHz({ decodedIntervalMs, decodedSamples = [],
    wallIntervalMs, playbackRate = 1 } = {}) {
    if (!Array.isArray(decodedSamples)) throw new TypeError('decoded samples must be an array');
    if (!Number.isFinite(playbackRate) || playbackRate <= 0) {
      throw new RangeError('playback rate must be positive');
    }
    const validDecoded = decodedSamples
      .filter(value => Number.isFinite(value) && value > 0.5 && value < 2000);
    const recentDecoded = validDecoded.slice(-AUTO_FAST_SIGNAL_WINDOW).sort((a, b) => a - b);
    // Before the decoded cadence is established, the first real sample replaces
    // the 24 FPS startup fallback. Afterwards require two recent fast samples, so
    // one timestamp outlier cannot suppress Auto for the full 32-sample window.
    const recentFastIntervalMs = validDecoded.length < 8
      ? recentDecoded[0]
      : recentDecoded.length >= 2 ? recentDecoded[1] : null;
    const wallMediaIntervalMs = validDecoded.length < 2 && Number.isFinite(wallIntervalMs)
      ? wallIntervalMs * playbackRate
      : null;
    const intervals = [decodedIntervalMs, recentFastIntervalMs, wallMediaIntervalMs]
      .filter(value => Number.isFinite(value) && value > 0.5 && value < 2000);
    if (!intervals.length) return 24 * playbackRate;
    return playbackRate * 1000 / Math.min(...intervals);
  }

  function mixedPresentationHz(sourceHz, uniqueHz, factor) {
    if (![sourceHz, uniqueHz].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('source and unique rates must be positive');
    }
    if (!Number.isInteger(factor) || factor < 1 || factor > 6) {
      throw new RangeError('auto factor must be an integer from 1 through 6');
    }
    const boundedUniqueHz = Math.min(sourceHz, uniqueHz);
    return sourceHz + (factor - 1) * boundedUniqueHz;
  }

  function capAutoFactorForDisplay(maxFactor, { sourceHz, uniqueHz, displayHz,
    tolerance = VIDEO_RATE_MATCH_TOLERANCE } = {}) {
    if (![sourceHz, uniqueHz, displayHz].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('auto cadence rates must be positive');
    }
    if (!Number.isInteger(maxFactor) || maxFactor < 1 || maxFactor > 6) {
      throw new RangeError('maximum auto factor must be an integer from 1 through 6');
    }
    if (!Number.isFinite(tolerance) || tolerance < 0) {
      throw new RangeError('auto cadence tolerance must be non-negative');
    }
    let factor = maxFactor;
    while (factor > 1
        && mixedPresentationHz(sourceHz, uniqueHz, factor) > displayHz * (1 + tolerance)) {
      factor--;
    }
    return factor;
  }

  function autoDisplayBudgetHz(sourceHz, displayHz, {
    headroom = DISPLAY_CLAMP_HEADROOM, tolerance = VIDEO_RATE_MATCH_TOLERANCE,
  } = {}) {
    if (![sourceHz, displayHz].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('Auto display budget rates must be positive');
    }
    if (!Number.isFinite(headroom) || headroom <= 0 || headroom > 1) {
      throw new RangeError('Auto display headroom must be in (0, 1]');
    }
    if (!Number.isFinite(tolerance) || tolerance < 0) {
      throw new RangeError('Auto display tolerance must be non-negative');
    }
    const reservedHz = displayHz * headroom;
    const minimumGeneratedHz = sourceHz * 2;
    const reservedFits2x = minimumGeneratedHz <= reservedHz * (1 + tolerance);
    const fullDisplayFits2x = minimumGeneratedHz <= displayHz * (1 + tolerance);
    return !reservedFits2x && fullDisplayFits2x ? displayHz : reservedHz;
  }

  function autoProbeDelayMs(attempts, {
    initialMs = AUTO_PROBE_INITIAL_DELAY_MS,
    maximumMs = AUTO_PROBE_MAX_DELAY_MS,
    maxAttempts = AUTO_PROBE_MAX_ATTEMPTS,
  } = {}) {
    if (!Number.isInteger(attempts) || attempts < 0) {
      throw new RangeError('Auto probe attempts must be a non-negative integer');
    }
    if (!Number.isFinite(initialMs) || initialMs <= 0
        || !Number.isFinite(maximumMs) || maximumMs < initialMs
        || !Number.isInteger(maxAttempts) || maxAttempts < 1) {
      throw new RangeError('Auto probe backoff configuration is invalid');
    }
    if (attempts >= maxAttempts) return null;
    return Math.min(maximumMs, initialMs * (2 ** attempts));
  }

  function rafStrainPressure(sampleMs, floorMs, {
    threshold = 1.7, timeConstantMs = 300, sustainedPressure = 2,
  } = {}) {
    if (![sampleMs, floorMs].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('rAF strain intervals must be positive');
    }
    if (!Number.isFinite(threshold) || threshold <= 1
        || !Number.isFinite(timeConstantMs) || timeConstantMs <= 0
        || !Number.isFinite(sustainedPressure) || sustainedPressure <= 0) {
      throw new RangeError('rAF strain configuration is invalid');
    }
    if (sampleMs <= floorMs * threshold) return 0;
    // The controller decays pressure with the same time constant. Charging by
    // elapsed raw service time makes sustained strain converge near the chosen
    // pressure on every refresh rate, while one long callback is counted once.
    return sustainedPressure * sampleMs / timeConstantMs;
  }

  function updateUniqueInterval(intervalMs, sampleMs, {
    minimumMs = 0, growthAlpha = 0.15,
  } = {}) {
    if (!Number.isFinite(sampleMs) || sampleMs <= 0) {
      throw new RangeError('unique interval sample must be positive');
    }
    if (!Number.isFinite(intervalMs) || intervalMs <= 0) return sampleMs;
    if (!Number.isFinite(minimumMs) || minimumMs < 0) {
      throw new RangeError('minimum unique interval must be non-negative');
    }
    if (!Number.isFinite(growthAlpha) || growthAlpha <= 0 || growthAlpha > 1) {
      throw new RangeError('unique interval growth alpha must be in (0, 1]');
    }
    const boundedSampleMs = Math.max(sampleMs, minimumMs);
    // More frequent unique frames increase presentation and GPU load, so react
    // immediately. Longer duplicate-heavy intervals may unlock work gradually.
    if (boundedSampleMs < intervalMs) return boundedSampleMs;
    return intervalMs * (1 - growthAlpha) + boundedSampleMs * growthAlpha;
  }

  function measureDisplayHz(rafFloorMs) {
    const measured = Number.isFinite(rafFloorMs) && rafFloorMs > 2 && rafFloorMs < 90;
    if (!measured) return { measured: false, rawHz: null, displayHz: 60, capacityHz: 60 };
    const rawHz = 1000 / rafFloorMs;
    let nearest = DISPLAY_RATES[0];
    for (const rate of DISPLAY_RATES) {
      if (Math.abs(rawHz - rate) / rate < Math.abs(rawHz - nearest) / nearest) nearest = rate;
    }
    const displayHz = Math.abs(rawHz - nearest) / nearest <= NOMINAL_RATE_TOLERANCE
      ? nearest
      : Math.max(1, Math.round(rawHz * 100) / 100);
    // Keep the friendly nominal rate separate from scheduling capacity. Snapping
    // 110Hz up to a 120Hz label must never authorize 120 presentations/second.
    const capacityHz = Math.max(1, Math.round(Math.min(rawHz, displayHz) * 100) / 100);
    return { measured: true, rawHz, displayHz, capacityHz };
  }

  function displaySampleTolerance(intervalMs) {
    return Math.max(DISPLAY_SAMPLE_QUANTIZATION_MS,
      Math.abs(intervalMs) * DISPLAY_SAMPLE_RELATIVE_TOLERANCE);
  }

  function displaySamplesAgree(leftMs, rightMs) {
    return Math.abs(leftMs - rightMs)
      <= Math.max(displaySampleTolerance(leftMs), displaySampleTolerance(rightMs));
  }

  function displaySampleSummary(samples) {
    const meanMs = samples.reduce((sum, value) => sum + value, 0) / samples.length;
    let variance = 0;
    let minimumMs = Infinity;
    let maximumMs = -Infinity;
    for (const value of samples) {
      variance += (value - meanMs) ** 2;
      minimumMs = Math.min(minimumMs, value);
      maximumMs = Math.max(maximumMs, value);
    }
    variance /= samples.length;
    const coefficientOfVariation = meanMs > 0 ? Math.sqrt(variance) / meanMs : Infinity;
    // A finite quantized window can contain one fewer long interval than the
    // underlying service pattern (for example 4/4/5 ms). Add one range/window
    // unit so the capacity estimate is conservative rather than faster than the
    // service actually observed over time.
    const serviceIntervalMs = meanMs + (maximumMs - minimumMs) / samples.length;
    return { meanMs, serviceIntervalMs, coefficientOfVariation };
  }

  function syncDisplayCandidateState(state, samples) {
    const summary = samples.length ? displaySampleSummary(samples) : null;
    state.candidateMs = summary ? summary.meanMs : 0;
    state.candidateSamples = samples.length;
    // Deprecated aliases remain synchronized for callers created before the
    // estimator learned to handle faster and slower transitions symmetrically.
    state.slowCandidateMs = state.candidateMs;
    state.slowSamples = state.candidateSamples;
  }

  function resetDisplayCandidates(state) {
    state._displayFasterSamples = [];
    state._displayFasterMisses = 0;
    state._displaySlowerSamples = [];
    state._displayCandidateSamples = [];
  }

  function syncDirectionalCandidateState(state) {
    const candidates = state._displayFasterSamples.length
      ? state._displayFasterSamples
      : state._displaySlowerSamples;
    state._displayCandidateSamples = candidates;
    syncDisplayCandidateState(state, candidates);
  }

  function ageFasterCandidate(state) {
    if (!state._displayFasterSamples.length) return;
    state._displayFasterMisses += 1;
    if (state._displayFasterMisses > DISPLAY_FASTER_MISS_BUDGET) {
      state._displayFasterSamples = [];
      state._displayFasterMisses = 0;
    }
  }

  function confirmDisplayService(state, samples) {
    const summary = displaySampleSummary(samples);
    const previous = state.floorMs;
    state.floorMs = summary.serviceIntervalMs;
    state._displayConfirmed = true;
    state._displayServiceSamples = samples.slice(-DISPLAY_SERVICE_WINDOW);
    resetDisplayCandidates(state);
    state.stableSamples = Math.max(REFRESH_TRANSITION_SAMPLES,
      state._displayServiceSamples.length);
    state.ready = true;
    syncDisplayCandidateState(state, state._displayCandidateSamples);
    return !Number.isFinite(previous) || Math.abs(state.floorMs - previous) > 1e-9;
  }

  // Track a sustained move in either direction. A single short callback must
  // not authorize a faster target, and a single long callback must not relabel
  // the monitor. Quantized callback patterns are summarized as service time,
  // never as their fastest individual sample.
  function updateDisplayInterval(state, sampleMs) {
    if (!state || typeof state !== 'object') throw new TypeError('refresh state must be an object');
    if (!Number.isFinite(state.floorMs) || state.floorMs <= 0) state.floorMs = 100;
    if (!Number.isFinite(sampleMs) || sampleMs <= 1 || sampleMs >= 100) return false;

    if (!Array.isArray(state._displayServiceSamples)) state._displayServiceSamples = [];
    if (!Array.isArray(state._displayCandidateSamples)) state._displayCandidateSamples = [];
    if (!Array.isArray(state._displayFasterSamples)) state._displayFasterSamples = [];
    if (!Array.isArray(state._displaySlowerSamples)) state._displaySlowerSamples = [];
    if (!Number.isInteger(state._displayFasterMisses) || state._displayFasterMisses < 0) {
      state._displayFasterMisses = 0;
    }
    if (state._displayConfirmed !== true) {
      state._displayConfirmed = state.ready === true
        && Number.isFinite(state.floorMs) && state.floorMs > 1 && state.floorMs < 100;
    }
    if (!Number.isInteger(state.stableSamples) || state.stableSamples < 0) {
      state.stableSamples = state._displayConfirmed ? REFRESH_TRANSITION_SAMPLES : 0;
    }

    if (state._displayConfirmed && displaySamplesAgree(sampleMs, state.floorMs)) {
      ageFasterCandidate(state);
      state._displaySlowerSamples = [];
      state._displayServiceSamples.push(sampleMs);
      if (state._displayServiceSamples.length > DISPLAY_SERVICE_WINDOW) {
        state._displayServiceSamples.shift();
      }
      const previous = state.floorMs;
      const summary = displaySampleSummary(state._displayServiceSamples);
      if (summary.coefficientOfVariation <= DISPLAY_STABILITY_CV) {
        state.floorMs = summary.serviceIntervalMs;
      }
      state.stableSamples = Math.min(DISPLAY_SERVICE_WINDOW,
        Math.max(REFRESH_TRANSITION_SAMPLES, state.stableSamples + 1));
      state.ready = true;
      syncDirectionalCandidateState(state);
      return Math.abs(state.floorMs - previous) > 1e-9;
    }

    // Delayed callbacks cannot disprove faster service. Keep up to ten agreeing
    // faster observations through at most three intervening misses so a missed
    // callback at a harmonic interval cannot permanently latch a slower rate.
    // The bounded miss budget prevents isolated fast outliers accumulating into a
    // false transition over an unbounded amount of time.
    const faster = sampleMs < state.floorMs;
    const candidates = faster ? state._displayFasterSamples : state._displaySlowerSamples;
    if (faster) state._displaySlowerSamples = [];
    else ageFasterCandidate(state);
    if (candidates.length) {
      const candidateMeanMs = displaySampleSummary(candidates).meanMs;
      if (!displaySamplesAgree(sampleMs, candidateMeanMs)) {
        candidates.length = 0;
        if (faster) state._displayFasterMisses = 0;
      }
    }
    candidates.push(sampleMs);
    if (candidates.length > REFRESH_TRANSITION_SAMPLES) candidates.shift();
    syncDirectionalCandidateState(state);

    // One or two long callbacks are ordinary OS/browser scheduling stalls. Keep
    // an already confirmed display usable through those outliers, but fail safe
    // once three agreeing slow samples indicate a real refresh-rate transition.
    const confirmedSlowdown = state._displayConfirmed
      && !faster
      && candidates.length >= 3;
    if (!state._displayConfirmed || confirmedSlowdown) {
      state.ready = false;
      state.stableSamples = 0;
    }
    if (candidates.length < REFRESH_TRANSITION_SAMPLES) return false;
    const summary = displaySampleSummary(candidates);
    if (summary.coefficientOfVariation > DISPLAY_STABILITY_CV) return false;
    return confirmDisplayService(state, candidates);
  }

  function resolveOutputRate(mode, rafFloorMs, {
    targetFps = DEFAULT_TARGET_FPS,
    sourceHz = null,
    sourceReady = Number.isFinite(sourceHz) && sourceHz > 0,
    displayReady = true,
    midCostMs = null,
    pairCostMs = 0,
    presentationCostMs = 0,
    strictCeiling = false,
  } = {}) {
    const safeMode = sanitizeOutputRate(mode);
    const display = measureDisplayHz(rafFloorMs);
    if (!isCadenceMode(safeMode)) {
      return {
        mode: safeMode, state: 'factor', ...display, requestedHz: null,
        minimumHz: null, computeCapacityHz: null, runtimeCapacityHz: null, outputHz: null,
        interpolationAllowed: false, clamped: false, clampReason: null, warning: null,
      };
    }

    const requestedHz = safeMode === 'target'
      ? sanitizeTargetFps(Object.hasOwn(LEGACY_TARGETS, mode) ? LEGACY_TARGETS[mode] : targetFps)
      : display.capacityHz;
    // Callers pass the already playback-adjusted source cadence. The encoded
    // cadence estimator performs nominal-rate normalization before playbackRate
    // is applied; snapping again here would turn 60fps at 1.01x (60.6Hz) back
    // into 60Hz and violate the strict 2x floor.
    const normalizedSourceHz = sourceReady && Number.isFinite(sourceHz) && sourceHz > 0
      ? sourceHz
      : null;
    const minimumHz = normalizedSourceHz === null ? null : normalizedSourceHz * 2;
    const base = {
      mode: safeMode, ...display, requestedHz, minimumHz, computeCapacityHz: null,
      runtimeCapacityHz: null,
      outputHz: null, interpolationAllowed: false, clamped: false,
      clampReason: null, warning: null,
    };

    if (safeMode === 'target' && requestedHz === null) {
      return { ...base, state: 'invalid-target',
        warning: 'Target FPS must be a positive number' };
    }
    if (!sourceReady || normalizedSourceHz === null || !displayReady || !display.measured) {
      return { ...base, state: 'measuring',
        warning: 'Measuring source FPS and display refresh rate' };
    }

    const runtimeCapacityHz = normalizedSourceHz * (MAX_MIDS_PER_PAIR + 1);
    let computeCapacityHz = Infinity;
    const safeMidCostMs = Number.isFinite(midCostMs) && midCostMs > 0 ? midCostMs : 0;
    const safePairCostMs = Number.isFinite(pairCostMs) && pairCostMs > 0 ? pairCostMs : 0;
    const safePresentationCostMs = Number.isFinite(presentationCostMs)
      && presentationCostMs > 0 ? presentationCostMs : 0;
    if (safeMidCostMs > 0 || safePairCostMs > 0 || safePresentationCostMs > 0) {
      const sourceIntervalMs = 1000 / normalizedSourceHz;
      const computeBudgetMs = sourceIntervalMs * 0.9;
      let maxMidsPerPair = 0;
      for (let mids = 1; mids < MAX_TICKS_PER_INTERVAL; mids++) {
        const pairCostMs = safePairCostMs + mids * safeMidCostMs
          + (mids + 1) * safePresentationCostMs;
        if (pairCostMs > computeBudgetMs) break;
        maxMidsPerPair = mids;
      }
      computeCapacityHz = normalizedSourceHz * (maxMidsPerPair + 1);
      if (safePresentationCostMs > 0) {
        // Strict ceilings can decimate below the source rate without any mids.
        // Their capacity is bounded by the output-frame cost itself, not by the
        // decoded source cadence used by the pair model above.
        computeCapacityHz = Math.min(computeCapacityHz, 900 / safePresentationCostMs);
      }
    }
    const toleranceHz = Math.max(0.01, minimumHz * VIDEO_RATE_MATCH_TOLERANCE);
    if (!strictCeiling && display.capacityHz + toleranceHz < minimumHz) {
      return {
        ...base, state: 'no-2x-display-range', computeCapacityHz,
        runtimeCapacityHz,
        warning: `Needs at least ${formatRate(minimumHz)} Hz; display is ~${formatRate(display.capacityHz)} Hz`,
      };
    }
    if (!strictCeiling && computeCapacityHz + toleranceHz < minimumHz) {
      return {
        ...base, state: 'no-2x-gpu-range', computeCapacityHz, runtimeCapacityHz,
        warning: `GPU cannot sustain the ${formatRate(minimumHz)} FPS minimum at this quality`,
      };
    }

    const desiredHz = strictCeiling
      ? requestedHz
      : safeMode === 'target'
      ? Math.max(requestedHz, minimumHz)
      : display.capacityHz;
    // When an explicit request is above the display ceiling, reserve a small
    // amount of real presentation service for bounded catch-up after an rAF
    // hitch. Never let that reserve violate the strict 2x source floor.
    const headroomDisplayHz = display.capacityHz * DISPLAY_CLAMP_HEADROOM;
    const targetsDisplayCeiling = requestedHz >= display.capacityHz;
    const useDisplayHeadroom = targetsDisplayCeiling
      && (strictCeiling || headroomDisplayHz + toleranceHz >= minimumHz);
    const displayLimitHz = useDisplayHeadroom ? headroomDisplayHz : display.capacityHz;
    // Display Hz asks for the panel ceiling itself. Use the headroom-adjusted
    // service target so ordinary rAF hitches can be recovered without reporting
    // a misleading display clamp. Explicit targets retain their existing clamp
    // semantics, and strict Auto caps remain strict.
    const adjustedDesiredHz = !strictCeiling && safeMode === 'hz'
      ? displayLimitHz
      : desiredHz;
    const maximumHz = Math.min(displayLimitHz, computeCapacityHz, runtimeCapacityHz);
    const outputHz = Math.min(adjustedDesiredHz, maximumHz);
    let clampReason = null;
    if (adjustedDesiredHz > maximumHz + toleranceHz) {
      if (maximumHz === displayLimitHz) clampReason = 'display';
      else if (maximumHz === computeCapacityHz) clampReason = 'gpu';
      else clampReason = 'runtime';
    } else if (!strictCeiling && safeMode === 'target' && requestedHz + toleranceHz < minimumHz) {
      clampReason = 'minimum';
    }
    let warning = null;
    if (clampReason === 'minimum') {
      warning = `${formatRate(requestedHz)} FPS raised to ${formatRate(outputHz)} FPS: minimum is 2x the source`;
    } else if (clampReason === 'display') {
      warning = `${formatRate(requestedHz)} FPS capped to ${formatRate(outputHz)} FPS by the display`;
    } else if (clampReason === 'gpu') {
      warning = `${formatRate(requestedHz)} FPS capped to ${formatRate(outputHz)} FPS at this quality`;
    } else if (clampReason === 'runtime') {
      warning = `${formatRate(requestedHz)} FPS capped to ${formatRate(outputHz)} FPS by the scheduler`;
    }
    return {
      ...base,
      state: 'active',
      computeCapacityHz,
      runtimeCapacityHz,
      outputHz,
      interpolationAllowed: true,
      clamped: clampReason !== null,
      clampReason,
      warning,
    };
  }

  function formatRate(value) {
    if (!Number.isFinite(value)) return '-';
    return Number(value.toFixed(2)).toString();
  }

  function computePresentationDelayMs({
    cadenceMode = false,
    sourceIntervalMs = 0,
    midCostMs = 10,
    pairCostMs = 0,
    burstPadMs = 0,
    floorMs = 60,
    maxDelayMs = 180,
  } = {}) {
    for (const [label, value] of Object.entries({
      sourceIntervalMs, midCostMs, pairCostMs, burstPadMs, floorMs, maxDelayMs,
    })) {
      if (!Number.isFinite(value) || value < 0) {
        throw new RangeError(`${label} must be finite and non-negative`);
      }
    }
    if (cadenceMode && sourceIntervalMs <= 0) {
      throw new RangeError('cadence source interval must be positive');
    }
    if (maxDelayMs < floorMs) {
      throw new RangeError('maximum presentation delay must cover its floor');
    }
    const pairLookaheadMs = cadenceMode ? sourceIntervalMs : 0;
    const requiredMs = pairLookaheadMs + pairCostMs + 2 * midCostMs + 25 + burstPadMs;
    return Math.min(maxDelayMs, Math.max(floorMs, requiredMs));
  }

  function planCadenceInterval({ nextAt = 0, startAt, endAt, outputHz }) {
    if (![nextAt, startAt, endAt, outputHz].every(Number.isFinite)) {
      throw new TypeError('cadence inputs must be finite numbers');
    }
    if (endAt <= startAt || outputHz <= 0) throw new RangeError('cadence interval and outputHz must be positive');
    const stepMs = 1000 / outputHz;
    let cursor = nextAt;
    let resynced = false;
    if (cursor <= 0 || cursor < startAt - stepMs || cursor > endAt + stepMs) {
      cursor = startAt + stepMs;
      resynced = true;
    }
    const ticks = [];
    // Partition the exact target clock into half-open decoded intervals. Keeping
    // ticks near the right edge in the interval is required when target is only
    // slightly faster than source (50/55 -> 60): otherwise one interval can get
    // no target tick and the next one two, so no one-anchor-per-source split is
    // possible even though the total cadence is numerically correct.
    const stopBefore = endAt - Math.max(1e-7, stepMs * 1e-9);
    while (cursor < stopBefore && ticks.length < MAX_TICKS_PER_INTERVAL) {
      ticks.push({ at: cursor, t: (cursor - startAt) / (endAt - startAt) });
      cursor += stepMs;
    }
    const overflowed = cursor < stopBefore;
    return {
      nextAt: overflowed ? endAt + stepMs : cursor,
      ticks: overflowed ? [] : ticks,
      stepMs,
      resynced,
      overflowed,
    };
  }

  function assignPresentationKinds(ticks, interpolate) {
    const presentations = ticks.map(tick => ({
      ...tick,
      kind: interpolate ? 'interpolate' : (tick.t < 0.5 ? 'previous' : 'current'),
    }));
    if (interpolate && presentations.length) {
      let anchorIndex = 0;
      let anchorDistance = Math.min(Math.abs(presentations[0].t), Math.abs(1 - presentations[0].t));
      for (let index = 1; index < presentations.length; index += 1) {
        const distance = Math.min(Math.abs(presentations[index].t), Math.abs(1 - presentations[index].t));
        if (distance < anchorDistance) {
          anchorIndex = index;
          anchorDistance = distance;
        }
      }
      const anchor = presentations[anchorIndex];
      anchor.kind = anchor.t <= 0.5 ? 'previous' : 'current';
    }
    return presentations;
  }

  // This is the product-level contract on top of the raw target clock. Every
  // target tick becomes exactly one presentation: an endpoint anchor or a model
  // interpolation. Callers may disable interpolation for cuts/duplicates while
  // still advancing and filling the exact same target grid.
  function planCadencePresentations({ nextAt = 0, startAt, endAt, outputHz, interpolate = true }) {
    const cadence = planCadenceInterval({ nextAt, startAt, endAt, outputHz });
    const presentations = assignPresentationKinds(cadence.ticks, interpolate);
    return { ...cadence, presentations };
  }

  // Decoded media time decides how many exact target ticks belong to a source
  // pair; wall time only anchors the absolute output grid. Keeping these phases
  // separate prevents callback jitter from turning nominal 60->120 into 1/3-tick
  // bursts while every absolute deadline remains exactly one target step apart.
  function planSourceCadencePresentations({ nextAt = 0, phaseMs = 0, startAt,
    sourceIntervalMs, outputHz, interpolate = true }) {
    if (![nextAt, phaseMs, startAt, sourceIntervalMs, outputHz].every(Number.isFinite)) {
      throw new TypeError('source cadence inputs must be finite numbers');
    }
    if (phaseMs < 0 || sourceIntervalMs <= 0 || outputHz <= 0) {
      throw new RangeError('source cadence interval, phase and outputHz are invalid');
    }
    const stepMs = 1000 / outputHz;
    let cursorPhase = phaseMs;
    let cursorAt = nextAt;
    let resynced = false;
    if (cursorAt <= 0) {
      cursorPhase = stepMs;
      cursorAt = startAt + stepMs;
      resynced = true;
    } else {
      // Keep the absolute clock close to the wall-time location of this media
      // phase. A tab/decoder stall may move startAt by many target periods; move
      // the cursor only by whole periods so the original target grid and media
      // phase survive, but no stale deadline backlog is emitted.
      const desiredAt = startAt + cursorPhase;
      const epsilonMs = Math.max(1e-7, stepMs * 1e-9);
      if (desiredAt - cursorAt > stepMs + epsilonMs) {
        const shiftSteps = Math.ceil((desiredAt - cursorAt) / stepMs - 1e-10);
        cursorAt += shiftSteps * stepMs;
        resynced = true;
      } else if (cursorAt - desiredAt > stepMs + epsilonMs) {
        // Never rewind the absolute target clock: doing so emits a deadline that
        // was already scheduled. Carry the clock's lead as media phase instead.
        cursorPhase = Math.max(0, cursorAt - startAt);
        resynced = true;
      }
    }
    const ticks = [];
    const stopBefore = sourceIntervalMs - Math.max(1e-7, stepMs * 1e-9);
    while (cursorPhase < stopBefore && ticks.length < MAX_TICKS_PER_INTERVAL) {
      ticks.push({ at: cursorAt, t: cursorPhase / sourceIntervalMs });
      cursorPhase += stepMs;
      cursorAt += stepMs;
    }
    const overflowed = cursorPhase < stopBefore;
    return {
      nextAt: overflowed ? startAt + sourceIntervalMs + stepMs : cursorAt,
      nextPhaseMs: overflowed ? stepMs : Math.max(0, cursorPhase - sourceIntervalMs),
      ticks: overflowed ? [] : ticks,
      presentations: overflowed ? [] : assignPresentationKinds(ticks, interpolate),
      stepMs,
      resynced,
      overflowed,
    };
  }

  // If the GPU cannot afford a requested interpolation, preserve cadence and
  // fail safe to the nearest decoded anchor instead of adding an off-grid frame.
  function fallbackCadencePresentations(presentations) {
    if (!Array.isArray(presentations)) throw new TypeError('cadence presentations must be an array');
    return presentations.map((presentation) => presentation.kind === 'interpolate'
      ? { ...presentation, kind: presentation.t < 0.5 ? 'previous' : 'current' }
      : { ...presentation });
  }

  function outputRateLabel(mode, targetFps = DEFAULT_TARGET_FPS) {
    if (Object.hasOwn(LEGACY_TARGETS, mode)) return `${LEGACY_TARGETS[mode]} FPS`;
    const safeMode = sanitizeOutputRate(mode);
    if (safeMode === 'auto') return 'Auto';
    if (safeMode === 'hz') return 'Display Hz';
    if (safeMode === 'target') return `${formatRate(sanitizeTargetFps(targetFps, DEFAULT_TARGET_FPS))} FPS`;
    return `${safeMode}x source`;
  }

  function enqueuePresentation(queue, entry) {
    if (!Array.isArray(queue)) throw new TypeError('presentation queue must be an array');
    queue.push(entry);
    if (queue.length <= MAX_PENDING_PRESENTATIONS) return null;
    let oldestIndex = 0;
    for (let index = 1; index < queue.length; index += 1) {
      if (queue[index].at < queue[oldestIndex].at) oldestIndex = index;
    }
    return queue.splice(oldestIndex, 1)[0];
  }

  // Select one due entry without mutating the queue. Exact targets may recover
  // up to three missed slots oldest-first when confirmed display service has
  // fractional headroom. Larger/no-headroom backlogs keep the low-latency
  // newest-due policy and explicitly drop superseded slots.
  function selectDuePresentation(queue, now, {
    targetHz = 0,
    displayCapacityHz = 0,
  } = {}) {
    if (!Array.isArray(queue)) throw new TypeError('presentation queue must be an array');
    if (!Number.isFinite(now)) throw new TypeError('presentation time must be finite');
    if (!Number.isFinite(displayCapacityHz) || displayCapacityHz < 0) {
      throw new RangeError('display capacity must be non-negative');
    }
    let newestDueIndex = -1;
    let previousAt = -Infinity;
    for (let index = 0; index < queue.length; index += 1) {
      const at = queue[index]?.at;
      if (!Number.isFinite(at)) throw new TypeError('presentation deadline must be finite');
      if (at < previousAt) throw new RangeError('presentation queue must be ordered by deadline');
      previousAt = at;
      if (at <= now) newestDueIndex = index;
    }
    if (newestDueIndex < 0) {
      return { presentIndex: -1, dueCount: 0, dropCount: 0, recovering: false, recoveryCapacity: 1 };
    }

    if (!Number.isFinite(targetHz) || targetHz < 0) {
      throw new RangeError('target rate must be non-negative');
    }
    const hasRecoveryHeadroom = targetHz > 0
      && displayCapacityHz > targetHz * (1 + VIDEO_RATE_MATCH_TOLERANCE);
    const serviceSlots = hasRecoveryHeadroom ? MAX_RECOVERY_PRESENTATIONS : 1;
    const dueCount = newestDueIndex + 1;
    const recovering = serviceSlots >= 2 && dueCount > 1 && dueCount <= serviceSlots;
    const presentIndex = recovering ? 0 : newestDueIndex;
    return {
      presentIndex,
      dueCount,
      dropCount: presentIndex,
      recovering,
      recoveryCapacity: serviceSlots,
    };
  }

  const api = Object.freeze({
    DISPLAY_RATES,
    COMMON_VIDEO_RATES,
    LEGACY_TARGETS,
    DEFAULT_TARGET_FPS,
    MIN_TARGET_FPS,
    MAX_TARGET_FPS,
    MAX_TICKS_PER_INTERVAL,
    MAX_PENDING_PRESENTATIONS,
    MAX_MIDS_PER_PAIR,
    MAX_RECOVERY_PRESENTATIONS,
    DISPLAY_CLAMP_HEADROOM,
    REFRESH_TRANSITION_SAMPLES,
    VIDEO_RATE_NORMALIZE_TOLERANCE,
    VIDEO_RATE_HOLD_TOLERANCE,
    VIDEO_RATE_MATCH_TOLERANCE,
    sanitizeOutputRate,
    sanitizeTargetFps,
    isCadenceMode,
    normalizeVideoRate,
    estimateSourceCadence,
    updateSourceInterval,
    targetNeedsInterpolation,
    estimateAutoSourceHz,
    mixedPresentationHz,
    capAutoFactorForDisplay,
    autoDisplayBudgetHz,
    autoProbeDelayMs,
    rafStrainPressure,
    updateUniqueInterval,
    measureDisplayHz,
    updateDisplayInterval,
    resolveOutputRate,
    computePresentationDelayMs,
    planCadenceInterval,
    planCadencePresentations,
    planSourceCadencePresentations,
    fallbackCadencePresentations,
    outputRateLabel,
    enqueuePresentation,
    selectDuePresentation,
  });
  root.FramegenCadence = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(globalThis);
