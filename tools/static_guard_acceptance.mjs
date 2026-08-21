export const STATIC_GUARD_CONTRACT = Object.freeze({
  benchmarkId: 'framecast-static-guard-text-over-motion-v1',
  schemaVersion: 1,
  patternId: 'pixel-glyph-over-moving-checker-v1',
  model: 'v7s',
  width: 512,
  height: 288,
  timesteps: Object.freeze([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]),
  thresholds: Object.freeze({
    textCandidateMaeLsb: 0.75,
    textCandidateMaxAbsLsb: 1.5,
    anchorStepMaxErrorLsb: 1.25,
    temporalSecondDifferenceMaxLsb: 2,
    textMinimumContrastLsb: 150,
    ordinaryMotionMaeLsb: 0.05,
    ordinaryMotionMaxAbsLsb: 1,
  }),
});

function finiteNumber(value, name) {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}

function assertFrame(frame, size, name) {
  if (!(frame instanceof Uint8Array) || frame.length !== size) {
    throw new Error(`${name} must be a ${size}-byte Uint8Array`);
  }
}

function assertMask(mask, pixels, name) {
  if (!(mask instanceof Uint8Array) || mask.length !== pixels) {
    throw new Error(`${name} must be a ${pixels}-byte Uint8Array`);
  }
  if (!mask.some(Boolean)) throw new Error(`${name} must select at least one pixel`);
}

function sameNumbers(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

export function analyzeStaticGuardFrames({
  width,
  height,
  timesteps,
  anchorA,
  anchorB,
  guardFrames,
  defaultGuardOffFrames,
  explicitGuardOffFrames,
  textCoreMask,
  textPlateMask,
  ordinaryMotionMask,
}) {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error('width and height must be positive integers');
  }
  if (!Array.isArray(timesteps) || timesteps.length < 3
      || timesteps.some((value, index) => !Number.isFinite(value) || value <= 0 || value >= 1
        || (index > 0 && value <= timesteps[index - 1]))) {
    throw new Error('timesteps must be a strictly increasing array inside (0, 1)');
  }
  for (let index = 1; index < timesteps.length; index++) {
    const previousStep = timesteps[index - 1] - (index === 1 ? 0 : timesteps[index - 2]);
    const currentStep = timesteps[index] - timesteps[index - 1];
    if (Math.abs(previousStep - currentStep) > 1e-9) {
      throw new Error('timesteps must use a uniform grid');
    }
  }

  const pixels = width * height;
  const bytes = pixels * 4;
  assertFrame(anchorA, bytes, 'anchorA');
  assertFrame(anchorB, bytes, 'anchorB');
  for (const [name, frames] of Object.entries({
    guardFrames,
    defaultGuardOffFrames,
    explicitGuardOffFrames,
  })) {
    if (!Array.isArray(frames) || frames.length !== timesteps.length) {
      throw new Error(`${name} must contain one frame per timestep`);
    }
    frames.forEach((frame, index) => assertFrame(frame, bytes, `${name}[${index}]`));
  }
  assertMask(textCoreMask, pixels, 'textCoreMask');
  assertMask(textPlateMask, pixels, 'textPlateMask');
  assertMask(ordinaryMotionMask, pixels, 'ordinaryMotionMask');

  let parityMismatchValues = 0;
  let parityMaxAbsLsb = 0;
  for (let frameIndex = 0; frameIndex < timesteps.length; frameIndex++) {
    const implicit = defaultGuardOffFrames[frameIndex];
    const explicit = explicitGuardOffFrames[frameIndex];
    for (let index = 0; index < bytes; index++) {
      const delta = Math.abs(implicit[index] - explicit[index]);
      if (delta) parityMismatchValues++;
      parityMaxAbsLsb = Math.max(parityMaxAbsLsb, delta);
    }
  }

  let candidateErrorSum = 0;
  let candidateMaxAbsLsb = 0;
  let candidateSamples = 0;
  let minimumContrastLsb = Infinity;
  for (let frameIndex = 0; frameIndex < timesteps.length; frameIndex++) {
    const t = timesteps[frameIndex];
    const frame = guardFrames[frameIndex];
    let textLuma = 0;
    let textPixels = 0;
    let plateLuma = 0;
    let platePixels = 0;
    for (let pixel = 0; pixel < pixels; pixel++) {
      const offset = pixel * 4;
      if (textCoreMask[pixel]) {
        for (let channel = 0; channel < 3; channel++) {
          const expected = anchorA[offset + channel]
            + (anchorB[offset + channel] - anchorA[offset + channel]) * t;
          const error = Math.abs(frame[offset + channel] - expected);
          candidateErrorSum += error;
          candidateMaxAbsLsb = Math.max(candidateMaxAbsLsb, error);
          candidateSamples++;
        }
        textLuma += 0.2126 * frame[offset] + 0.7152 * frame[offset + 1] + 0.0722 * frame[offset + 2];
        textPixels++;
      }
      if (textPlateMask[pixel]) {
        plateLuma += 0.2126 * frame[offset] + 0.7152 * frame[offset + 1] + 0.0722 * frame[offset + 2];
        platePixels++;
      }
    }
    minimumContrastLsb = Math.min(minimumContrastLsb,
      textLuma / textPixels - plateLuma / platePixels);
  }

  const sequence = [anchorA, ...guardFrames, anchorB];
  let temporalSecondDifferenceMaxLsb = 0;
  let monotonicViolations = 0;
  let anchorStepMaxErrorLsb = 0;
  const step = timesteps[0];
  for (let pixel = 0; pixel < pixels; pixel++) {
    if (!textCoreMask[pixel]) continue;
    const offset = pixel * 4;
    for (let channel = 0; channel < 3; channel++) {
      const direction = Math.sign(anchorB[offset + channel] - anchorA[offset + channel]);
      const expectedStep = (anchorB[offset + channel] - anchorA[offset + channel]) * step;
      anchorStepMaxErrorLsb = Math.max(anchorStepMaxErrorLsb,
        Math.abs((sequence[1][offset + channel] - sequence[0][offset + channel]) - expectedStep),
        Math.abs((sequence.at(-1)[offset + channel] - sequence.at(-2)[offset + channel]) - expectedStep));
      for (let frameIndex = 1; frameIndex < sequence.length; frameIndex++) {
        const delta = sequence[frameIndex][offset + channel] - sequence[frameIndex - 1][offset + channel];
        if ((direction > 0 && delta < 0) || (direction < 0 && delta > 0)) monotonicViolations++;
      }
      for (let frameIndex = 1; frameIndex < sequence.length - 1; frameIndex++) {
        temporalSecondDifferenceMaxLsb = Math.max(temporalSecondDifferenceMaxLsb,
          Math.abs(sequence[frameIndex - 1][offset + channel]
            - 2 * sequence[frameIndex][offset + channel]
            + sequence[frameIndex + 1][offset + channel]));
      }
    }
  }

  let motionErrorSum = 0;
  let motionMaxAbsLsb = 0;
  let motionSamples = 0;
  for (let frameIndex = 0; frameIndex < timesteps.length; frameIndex++) {
    const guarded = guardFrames[frameIndex];
    const unguarded = defaultGuardOffFrames[frameIndex];
    for (let pixel = 0; pixel < pixels; pixel++) {
      if (!ordinaryMotionMask[pixel]) continue;
      const offset = pixel * 4;
      for (let channel = 0; channel < 3; channel++) {
        const delta = Math.abs(guarded[offset + channel] - unguarded[offset + channel]);
        motionErrorSum += delta;
        motionMaxAbsLsb = Math.max(motionMaxAbsLsb, delta);
        motionSamples++;
      }
    }
  }

  return {
    guardOffParity: {
      mismatchValues: parityMismatchValues,
      maxAbsLsb: parityMaxAbsLsb,
      sampleValues: bytes * timesteps.length,
    },
    text: {
      candidateMaeLsb: candidateErrorSum / candidateSamples,
      candidateMaxAbsLsb,
      anchorStepMaxErrorLsb,
      temporalSecondDifferenceMaxLsb,
      monotonicViolations,
      minimumContrastLsb,
      sampleValues: candidateSamples,
    },
    ordinaryMotion: {
      guardVsOffMaeLsb: motionErrorSum / motionSamples,
      guardVsOffMaxAbsLsb: motionMaxAbsLsb,
      sampleValues: motionSamples,
    },
  };
}

export function evaluateStaticGuardReport(report) {
  const contract = STATIC_GUARD_CONTRACT;
  const checks = [];
  const check = (name, passed, actual, expected) => checks.push({ name, passed: Boolean(passed), actual, expected });
  check('schema-version', report?.schemaVersion === contract.schemaVersion,
    report?.schemaVersion, contract.schemaVersion);
  check('benchmark-id', report?.benchmarkId === contract.benchmarkId,
    report?.benchmarkId, contract.benchmarkId);
  check('status', report?.status === 'complete', report?.status, 'complete');
  check('pattern', report?.workload?.patternId === contract.patternId,
    report?.workload?.patternId, contract.patternId);
  check('model', report?.workload?.model === contract.model, report?.workload?.model, contract.model);
  check('resolution', report?.workload?.width === contract.width && report?.workload?.height === contract.height,
    `${report?.workload?.width}x${report?.workload?.height}`, `${contract.width}x${contract.height}`);
  check('timesteps', sameNumbers(report?.workload?.timesteps, contract.timesteps),
    report?.workload?.timesteps, contract.timesteps);

  const parity = report?.metrics?.guardOffParity ?? {};
  const text = report?.metrics?.text ?? {};
  const motion = report?.metrics?.ordinaryMotion ?? {};
  check('guard-off-byte-parity', parity.mismatchValues === 0 && parity.maxAbsLsb === 0,
    { mismatchValues: parity.mismatchValues, maxAbsLsb: parity.maxAbsLsb }, { mismatchValues: 0, maxAbsLsb: 0 });
  check('text-candidate-mae', finiteNumber(text.candidateMaeLsb, 'text.candidateMaeLsb') <= contract.thresholds.textCandidateMaeLsb,
    text.candidateMaeLsb, `<= ${contract.thresholds.textCandidateMaeLsb}`);
  check('text-candidate-max', finiteNumber(text.candidateMaxAbsLsb, 'text.candidateMaxAbsLsb') <= contract.thresholds.textCandidateMaxAbsLsb,
    text.candidateMaxAbsLsb, `<= ${contract.thresholds.textCandidateMaxAbsLsb}`);
  check('anchor-step-continuity', finiteNumber(text.anchorStepMaxErrorLsb, 'text.anchorStepMaxErrorLsb') <= contract.thresholds.anchorStepMaxErrorLsb,
    text.anchorStepMaxErrorLsb, `<= ${contract.thresholds.anchorStepMaxErrorLsb}`);
  check('temporal-linearity', finiteNumber(text.temporalSecondDifferenceMaxLsb, 'text.temporalSecondDifferenceMaxLsb') <= contract.thresholds.temporalSecondDifferenceMaxLsb,
    text.temporalSecondDifferenceMaxLsb, `<= ${contract.thresholds.temporalSecondDifferenceMaxLsb}`);
  check('text-monotonicity', text.monotonicViolations === 0, text.monotonicViolations, 0);
  check('text-contrast', finiteNumber(text.minimumContrastLsb, 'text.minimumContrastLsb') >= contract.thresholds.textMinimumContrastLsb,
    text.minimumContrastLsb, `>= ${contract.thresholds.textMinimumContrastLsb}`);
  check('ordinary-motion-mae', finiteNumber(motion.guardVsOffMaeLsb, 'ordinaryMotion.guardVsOffMaeLsb') <= contract.thresholds.ordinaryMotionMaeLsb,
    motion.guardVsOffMaeLsb, `<= ${contract.thresholds.ordinaryMotionMaeLsb}`);
  check('ordinary-motion-max', finiteNumber(motion.guardVsOffMaxAbsLsb, 'ordinaryMotion.guardVsOffMaxAbsLsb') <= contract.thresholds.ordinaryMotionMaxAbsLsb,
    motion.guardVsOffMaxAbsLsb, `<= ${contract.thresholds.ordinaryMotionMaxAbsLsb}`);
  check('sample-coverage', parity.sampleValues > 0 && text.sampleValues > 0 && motion.sampleValues > 0,
    { parity: parity.sampleValues, text: text.sampleValues, motion: motion.sampleValues }, 'all > 0');

  return { passed: checks.every(item => item.passed), checks };
}

export function finalizeStaticGuardReport(report) {
  const validation = evaluateStaticGuardReport(report);
  return { ...report, validation, passed: validation.passed };
}
