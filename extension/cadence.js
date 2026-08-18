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
    const previousNominalHz = normalizeVideoRate(previousHz);
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

  function targetNeedsInterpolation(sourceHz, targetHz) {
    if (![sourceHz, targetHz].every(value => Number.isFinite(value) && value > 0)) {
      throw new RangeError('source and target rates must be positive');
    }
    const source = normalizeVideoRate(sourceHz);
    const target = normalizeVideoRate(targetHz);
    return target > source && (target - source) / source > VIDEO_RATE_MATCH_TOLERANCE;
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

  function confirmDisplayService(state, samples) {
    const summary = displaySampleSummary(samples);
    const previous = state.floorMs;
    state.floorMs = summary.serviceIntervalMs;
    state._displayConfirmed = true;
    state._displayServiceSamples = samples.slice(-DISPLAY_SERVICE_WINDOW);
    state._displayCandidateSamples = [];
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
    if (state._displayConfirmed !== true) {
      state._displayConfirmed = state.ready === true
        && Number.isFinite(state.floorMs) && state.floorMs > 1 && state.floorMs < 100;
    }
    if (!Number.isInteger(state.stableSamples) || state.stableSamples < 0) {
      state.stableSamples = state._displayConfirmed ? REFRESH_TRANSITION_SAMPLES : 0;
    }

    if (state._displayConfirmed && displaySamplesAgree(sampleMs, state.floorMs)) {
      state._displayCandidateSamples = [];
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
      syncDisplayCandidateState(state, state._displayCandidateSamples);
      return Math.abs(state.floorMs - previous) > 1e-9;
    }

    const candidates = state._displayCandidateSamples;
    if (candidates.length) {
      const candidateMeanMs = displaySampleSummary(candidates).meanMs;
      if (!displaySamplesAgree(sampleMs, candidateMeanMs)) candidates.length = 0;
    }
    candidates.push(sampleMs);
    if (candidates.length > REFRESH_TRANSITION_SAMPLES) candidates.shift();
    syncDisplayCandidateState(state, candidates);

    // One or two long callbacks are ordinary OS/browser scheduling stalls. Keep
    // an already confirmed display usable through those outliers, but fail safe
    // once three agreeing slow samples indicate a real refresh-rate transition.
    const confirmedSlowdown = state._displayConfirmed
      && sampleMs > state.floorMs
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
    if (Number.isFinite(midCostMs) && midCostMs > 0) {
      const sourceIntervalMs = 1000 / normalizedSourceHz;
      const maxMidsPerPair = Math.max(0, Math.min(MAX_TICKS_PER_INTERVAL - 1,
        Math.floor(sourceIntervalMs * 0.9 / midCostMs)));
      computeCapacityHz = normalizedSourceHz * (maxMidsPerPair + 1);
    }
    const toleranceHz = Math.max(0.01, minimumHz * VIDEO_RATE_MATCH_TOLERANCE);
    if (display.capacityHz + toleranceHz < minimumHz) {
      return {
        ...base, state: 'no-2x-display-range', computeCapacityHz,
        runtimeCapacityHz,
        warning: `Needs at least ${formatRate(minimumHz)} Hz; display is ~${formatRate(display.capacityHz)} Hz`,
      };
    }
    if (computeCapacityHz + toleranceHz < minimumHz) {
      return {
        ...base, state: 'no-2x-gpu-range', computeCapacityHz, runtimeCapacityHz,
        warning: `GPU cannot sustain the ${formatRate(minimumHz)} FPS minimum at this quality`,
      };
    }

    const desiredHz = safeMode === 'target'
      ? Math.max(requestedHz, minimumHz)
      : display.capacityHz;
    // When an explicit request is above the display ceiling, reserve a small
    // amount of real presentation service for bounded catch-up after an rAF
    // hitch. Never let that reserve violate the strict 2x source floor.
    const headroomDisplayHz = display.capacityHz * DISPLAY_CLAMP_HEADROOM;
    const targetsDisplayCeiling = safeMode === 'target'
      && requestedHz >= display.capacityHz;
    const useDisplayHeadroom = targetsDisplayCeiling
      && headroomDisplayHz + toleranceHz >= minimumHz;
    const displayLimitHz = useDisplayHeadroom ? headroomDisplayHz : display.capacityHz;
    const maximumHz = Math.min(displayLimitHz, computeCapacityHz, runtimeCapacityHz);
    const outputHz = Math.min(desiredHz, maximumHz);
    let clampReason = null;
    if (desiredHz > maximumHz + toleranceHz) {
      if (maximumHz === displayLimitHz) clampReason = 'display';
      else if (maximumHz === computeCapacityHz) clampReason = 'gpu';
      else clampReason = 'runtime';
    } else if (safeMode === 'target' && requestedHz + toleranceHz < minimumHz) {
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
    burstPadMs = 0,
    floorMs = 60,
    maxDelayMs = 180,
  } = {}) {
    for (const [label, value] of Object.entries({
      sourceIntervalMs, midCostMs, burstPadMs, floorMs, maxDelayMs,
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
    const requiredMs = pairLookaheadMs + 2 * midCostMs + 25 + burstPadMs;
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
      if (Math.abs(cursorAt - desiredAt) > stepMs + epsilonMs) {
        const shiftSteps = Math.ceil((desiredAt - cursorAt) / stepMs - 1e-10);
        cursorAt += shiftSteps * stepMs;
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
    targetNeedsInterpolation,
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
