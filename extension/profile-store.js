(function installFramegenProfiles(root) {
  'use strict';

  const SCHEMA_VERSION = 3;
  const PREVIOUS_SCHEMA_VERSION = 2;
  const LEGACY_SCHEMA_VERSION = 1;
  const STORE_KEY = 'fcProfileStore';
  const SETTINGS_KEYS = Object.freeze([
    'factor', 'targetFps', 'fpsLimit', 'anime', 'debug', 'res', 'hoverReveal', 'compare',
    'fg', 'sr', 'hdr', 'sharpness', 'showFps', 'showWatermark', 'showWarnings', 'guard', 'model',
  ]);
  const OUTPUT_RATES = Object.freeze(['auto', 'hz', 'target', 2, 3, 4, 5, 6]);
  const RESOLUTIONS = Object.freeze([288, 360, 480, 720, 1080]);
  const MODELS = Object.freeze(['v6', 'v7s']);
  const TARGET_FPS_MIN = 2;
  const TARGET_FPS_MAX = 1000;
  const FPS_LIMIT_PRESETS = Object.freeze([
    15, 24, 25, 30, 48, 50, 60, 72, 75,
    90, 100, 120, 144, 165, 180, 240, 360,
    null,
  ]);
  const DEFAULT_SETTINGS = Object.freeze({
    factor: 'auto',
    targetFps: 120,
    fpsLimit: null,
    anime: true,
    debug: false,
    res: 480,
    hoverReveal: true,
    compare: false,
    fg: true,
    sr: false,
    hdr: false,
    sharpness: 0,
    showFps: true,
    showWatermark: true,
    showWarnings: true,
    guard: true,
    model: 'v7s',
  });
  const LEGACY_CURRENT_PROFILE = Object.freeze({
    id: 'current-settings',
    name: 'Current settings',
    description: 'Settings preserved from before profiles were added.',
  });

  class ProfileStoreError extends Error {
    constructor(message) {
      super(message);
      this.name = 'ProfileStoreError';
    }
  }

  class UnsupportedProfileStoreVersionError extends ProfileStoreError {
    constructor(version) {
      super(`profile store schema ${String(version)} is not supported`);
      this.name = 'UnsupportedProfileStoreVersionError';
      this.schemaVersion = version;
    }
  }

  const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  function booleanValue(value, fallback) {
    return typeof value === 'boolean' ? value : fallback;
  }

  function sanitizeTargetFps(value, fallback = DEFAULT_SETTINGS.targetFps) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return fallback;
    const clamped = Math.min(TARGET_FPS_MAX, Math.max(TARGET_FPS_MIN, numeric));
    return Math.round((clamped + Number.EPSILON) * 100) / 100;
  }

  function fpsLimitPresetIndex(value) {
    if (value === null || value === undefined
        || (typeof value === 'string' && value.trim() === '')) {
      return FPS_LIMIT_PRESETS.length - 1;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return FPS_LIMIT_PRESETS.length - 1;
    let nearest = 0;
    for (let index = 1; index < FPS_LIMIT_PRESETS.length - 1; index++) {
      if (Math.abs(FPS_LIMIT_PRESETS[index] - numeric)
          < Math.abs(FPS_LIMIT_PRESETS[nearest] - numeric)) {
        nearest = index;
      }
    }
    return nearest;
  }

  function sanitizeFpsLimit(value, fallback = DEFAULT_SETTINGS.fpsLimit) {
    if (value === null || value === undefined
        || (typeof value === 'string' && value.trim() === '')) return fallback;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return fallback;
    return FPS_LIMIT_PRESETS[fpsLimitPresetIndex(numeric)];
  }

  function sanitizeFactor(value) {
    if (value === 'auto' || value === 'hz' || value === 'target') return value;
    if (value === 'fps60' || value === 'fps120') return 'target';
    const numeric = Number(value);
    return Number.isInteger(numeric) && numeric >= 2 && numeric <= 6
      ? numeric
      : DEFAULT_SETTINGS.factor;
  }

  function sanitizeSettings(value = {}) {
    const source = value && typeof value === 'object' ? value : {};
    const numericResolution = Number(source.res);
    const targetFps = source.factor === 'fps60'
      ? 60
      : source.factor === 'fps120'
        ? 120
        : sanitizeTargetFps(source.targetFps);
    return {
      factor: sanitizeFactor(source.factor),
      targetFps,
      fpsLimit: sanitizeFpsLimit(source.fpsLimit),
      anime: booleanValue(source.anime, DEFAULT_SETTINGS.anime),
      debug: booleanValue(source.debug, DEFAULT_SETTINGS.debug),
      res: RESOLUTIONS.includes(numericResolution) ? numericResolution : DEFAULT_SETTINGS.res,
      hoverReveal: booleanValue(source.hoverReveal, DEFAULT_SETTINGS.hoverReveal),
      compare: booleanValue(source.compare, DEFAULT_SETTINGS.compare),
      fg: booleanValue(source.fg, DEFAULT_SETTINGS.fg),
      sr: booleanValue(source.sr, DEFAULT_SETTINGS.sr),
      hdr: booleanValue(source.hdr, DEFAULT_SETTINGS.hdr),
      sharpness: [0, 1, 2, 3].includes(Number(source.sharpness)) ? Number(source.sharpness) : DEFAULT_SETTINGS.sharpness,
      showFps: booleanValue(source.showFps, DEFAULT_SETTINGS.showFps),
      showWatermark: booleanValue(source.showWatermark, DEFAULT_SETTINGS.showWatermark),
      showWarnings: booleanValue(source.showWarnings, DEFAULT_SETTINGS.showWarnings),
      guard: booleanValue(source.guard, DEFAULT_SETTINGS.guard),
      model: MODELS.includes(source.model) ? source.model : DEFAULT_SETTINGS.model,
    };
  }

  function pickFlatSettings(storage = {}) {
    const source = storage && typeof storage === 'object' ? storage : {};
    const picked = {};
    for (const key of SETTINGS_KEYS) {
      if (hasOwn(source, key)) picked[key] = source[key];
    }
    return sanitizeSettings(picked);
  }

  function settingsEqual(left, right) {
    const a = sanitizeSettings(left);
    const b = sanitizeSettings(right);
    return SETTINGS_KEYS.every(key => a[key] === b[key]);
  }

  function normalizeText(value, fallback, maxLength) {
    const raw = typeof value === 'string' ? value : '';
    const clean = raw
      .replace(/[\u0000-\u001f\u007f-\u009f]/g, ' ')
      .trim()
      .replace(/\s+/g, ' ');
    return (clean || fallback).slice(0, maxLength);
  }

  function normalizeName(value, fallback = 'Custom profile') {
    return normalizeText(value, fallback, 48);
  }

  function normalizeDescription(value) {
    return normalizeText(value, '', 160);
  }

  function sanitizeProfileId(value) {
    if (typeof value !== 'string') return null;
    const id = value.trim().toLowerCase();
    return /^[a-z0-9][a-z0-9_-]{2,63}$/.test(id) ? id : null;
  }

  function settingsWithCompatibilityDefaults(value, compatibilitySettings = DEFAULT_SETTINGS) {
    const source = value && typeof value === 'object' ? value : {};
    const fallback = sanitizeSettings(compatibilitySettings);
    return {
      ...source,
      fpsLimit: hasOwn(source, 'fpsLimit') ? source.fpsLimit : fallback.fpsLimit,
      showWarnings: hasOwn(source, 'showWarnings') ? source.showWarnings : fallback.showWarnings,
    };
  }

  function normalizeProfiles(rawProfiles, label, compatibilitySettings = DEFAULT_SETTINGS) {
    if (!Array.isArray(rawProfiles)) {
      throw new ProfileStoreError(`${label}.profiles must be an array`);
    }
    const ids = new Set();
    return rawProfiles.map((candidate, index) => {
      if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
        throw new ProfileStoreError(`${label}.profiles[${index}] must be an object`);
      }
      const id = sanitizeProfileId(candidate.id);
      if (!id) throw new ProfileStoreError(`${label}.profiles[${index}].id is invalid`);
      if (ids.has(id)) throw new ProfileStoreError(`${label}.profiles contains duplicate id ${id}`);
      ids.add(id);
      return {
        id,
        name: normalizeName(candidate.name),
        description: normalizeDescription(candidate.description),
        settings: sanitizeSettings(settingsWithCompatibilityDefaults(
          candidate.settings,
          compatibilitySettings,
        )),
      };
    });
  }

  function emptyStore() {
    return {
      schemaVersion: SCHEMA_VERSION,
      lastAppliedProfileId: null,
      profiles: [],
    };
  }

  function normalizeStore(rawStore, compatibilitySettings = DEFAULT_SETTINGS) {
    const profiles = normalizeProfiles(
      rawStore.profiles,
      'profile store',
      compatibilitySettings,
    );
    const requestedId = rawStore.lastAppliedProfileId == null
      ? null
      : sanitizeProfileId(rawStore.lastAppliedProfileId);
    const lastAppliedProfileId = requestedId && profiles.some(profile => profile.id === requestedId)
      ? requestedId
      : null;
    return {
      schemaVersion: SCHEMA_VERSION,
      lastAppliedProfileId,
      profiles,
    };
  }

  function isUntouchedLegacyCurrentProfile(rawStore, candidate, currentSettings) {
    return rawStore.migratedFromFlat === true
      && candidate?.builtIn === false
      && candidate.id === LEGACY_CURRENT_PROFILE.id
      && candidate.name === LEGACY_CURRENT_PROFILE.name
      && candidate.description === LEGACY_CURRENT_PROFILE.description
      && settingsEqual(
        settingsWithCompatibilityDefaults(candidate.settings, currentSettings),
        currentSettings,
      );
  }

  function migrateV1Store(rawStore, currentSettings) {
    if (!Array.isArray(rawStore.customProfiles)) {
      throw new ProfileStoreError('schema v1 customProfiles must be an array');
    }
    const retained = rawStore.customProfiles.filter(
      candidate => !isUntouchedLegacyCurrentProfile(rawStore, candidate, currentSettings),
    );
    const profiles = normalizeProfiles(retained, 'schema v1', currentSettings);
    const activeId = sanitizeProfileId(rawStore.activeProfileId);
    const lastAppliedProfileId = activeId && profiles.some(profile => profile.id === activeId)
      ? activeId
      : null;
    return {
      schemaVersion: SCHEMA_VERSION,
      lastAppliedProfileId,
      profiles,
    };
  }

  function loadStore(rawStore, currentSettings = DEFAULT_SETTINGS) {
    const cleanCurrent = sanitizeSettings(currentSettings);
    if (rawStore == null) {
      return { store: emptyStore(), needsWrite: true, sourceSchemaVersion: null };
    }
    if (typeof rawStore !== 'object' || Array.isArray(rawStore)) {
      throw new ProfileStoreError('profile store must be an object');
    }
    if (rawStore.schemaVersion === SCHEMA_VERSION) {
      const normalized = normalizeStore(rawStore);
      return {
        store: normalized,
        needsWrite: JSON.stringify(normalized) !== JSON.stringify(rawStore),
        sourceSchemaVersion: SCHEMA_VERSION,
      };
    }
    if (rawStore.schemaVersion === PREVIOUS_SCHEMA_VERSION) {
      return {
        store: normalizeStore(rawStore, cleanCurrent),
        needsWrite: true,
        sourceSchemaVersion: PREVIOUS_SCHEMA_VERSION,
      };
    }
    if (rawStore.schemaVersion === LEGACY_SCHEMA_VERSION) {
      return {
        store: migrateV1Store(rawStore, cleanCurrent),
        needsWrite: true,
        sourceSchemaVersion: LEGACY_SCHEMA_VERSION,
      };
    }
    throw new UnsupportedProfileStoreVersionError(rawStore.schemaVersion);
  }

  function requireStore(store) {
    if (!store || typeof store !== 'object' || store.schemaVersion !== SCHEMA_VERSION) {
      throw new ProfileStoreError(`schema v${SCHEMA_VERSION} profile store is required`);
    }
    return normalizeStore(store);
  }

  function profileList(store) {
    return requireStore(store).profiles;
  }

  function getProfile(store, id) {
    const normalizedId = sanitizeProfileId(id);
    if (!normalizedId) return null;
    return profileList(store).find(profile => profile.id === normalizedId) || null;
  }

  function findMatchingProfileId(store, currentSettings) {
    const match = profileList(store).find(profile => settingsEqual(profile.settings, currentSettings));
    return match ? match.id : null;
  }

  function resolveProfileSelection(store, preferredProfileId, currentSettings = DEFAULT_SETTINGS) {
    const normalized = requireStore(store);
    const profile = getProfile(normalized, preferredProfileId);
    if (!profile) {
      return {
        selectedProfileId: null,
        source: 'current',
        settings: sanitizeSettings(currentSettings),
      };
    }
    return {
      selectedProfileId: profile.id,
      source: 'profile',
      settings: sanitizeSettings(profile.settings),
    };
  }

  function isProfileModified(store, id, currentSettings) {
    const profile = getProfile(store, id);
    return !profile || !settingsEqual(profile.settings, currentSettings);
  }

  function setLastAppliedProfile(store, id) {
    const normalized = requireStore(store);
    if (id == null) return { ...normalized, lastAppliedProfileId: null };
    const profile = getProfile(normalized, id);
    if (!profile) throw new TypeError('profile does not exist');
    return { ...normalized, lastAppliedProfileId: profile.id };
  }

  function upsertProfile(store, candidate) {
    const normalized = requireStore(store);
    const profile = normalizeProfiles([candidate], 'candidate')[0];
    const profiles = normalized.profiles.filter(item => item.id !== profile.id);
    profiles.push(profile);
    return { ...normalized, profiles };
  }

  function renameProfile(store, id, name) {
    const normalized = requireStore(store);
    const profile = getProfile(normalized, id);
    if (!profile) throw new TypeError('profile does not exist');
    return {
      ...normalized,
      profiles: normalized.profiles.map(item => (
        item.id === profile.id ? { ...item, name: normalizeName(name, item.name) } : item
      )),
    };
  }

  function deleteProfile(store, id) {
    const normalizedId = sanitizeProfileId(id);
    const normalized = requireStore(store);
    if (!normalizedId || !normalized.profiles.some(profile => profile.id === normalizedId)) {
      return normalized;
    }
    return {
      ...normalized,
      lastAppliedProfileId: normalized.lastAppliedProfileId === normalizedId
        ? null
        : normalized.lastAppliedProfileId,
      profiles: normalized.profiles.filter(profile => profile.id !== normalizedId),
    };
  }

  function toStoragePayload(store, settings) {
    return {
      ...sanitizeSettings(settings),
      [STORE_KEY]: requireStore(store),
    };
  }

  const api = Object.freeze({
    SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    STORE_KEY,
    SETTINGS_KEYS,
    OUTPUT_RATES,
    RESOLUTIONS,
    MODELS,
    TARGET_FPS_MIN,
    TARGET_FPS_MAX,
    FPS_LIMIT_PRESETS,
    DEFAULT_SETTINGS,
    ProfileStoreError,
    UnsupportedProfileStoreVersionError,
    sanitizeSettings,
    sanitizeFpsLimit,
    fpsLimitPresetIndex,
    pickFlatSettings,
    settingsEqual,
    normalizeName,
    sanitizeProfileId,
    emptyStore,
    loadStore,
    profileList,
    getProfile,
    findMatchingProfileId,
    resolveProfileSelection,
    isProfileModified,
    setLastAppliedProfile,
    upsertProfile,
    renameProfile,
    deleteProfile,
    toStoragePayload,
  });
  root.FramegenProfiles = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(globalThis);
