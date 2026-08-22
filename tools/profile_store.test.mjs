import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const Profiles = require('../extension/profile-store.js');
const contentJs = readFileSync(new URL('../extension/content.js', import.meta.url), 'utf8');
const optionsHtml = readFileSync(new URL('../extension/options.html', import.meta.url), 'utf8');
const optionsCss = readFileSync(new URL('../extension/options.css', import.meta.url), 'utf8');
const popupHtml = readFileSync(new URL('../extension/popup.html', import.meta.url), 'utf8');
const popupJs = readFileSync(new URL('../extension/popup.js', import.meta.url), 'utf8');
const optionsJs = readFileSync(new URL('../extension/options.js', import.meta.url), 'utf8');
const backgroundJs = readFileSync(new URL('../extension/background.js', import.meta.url), 'utf8');
const manifest = JSON.parse(readFileSync(new URL('../extension/manifest.json', import.meta.url), 'utf8'));

const customSettings = Object.freeze({
  factor: 'target',
  targetFps: 137.25,
  fpsLimit: 60,
  anime: false,
  debug: true,
  res: 1080,
  hoverReveal: false,
  compare: true,
  fg: false,
  sr: true,
  hdr: true,
  showFps: false,
  showWatermark: false,
  showWarnings: false,
  guard: false,
  model: 'v6',
});

function v1Profile(overrides = {}) {
  return {
    id: 'custom-one',
    name: 'Living room',
    description: 'Exact setup',
    builtIn: false,
    settings: customSettings,
    ...overrides,
  };
}

test('missing store initializes empty schema v3 without materializing current settings', () => {
  const storage = {
    ...customSettings,
    fcTune: { adapter: 'must stay outside profiles' },
    gpuMs: 2.7,
  };
  const current = Profiles.pickFlatSettings(storage);
  const loaded = Profiles.loadStore(undefined, current);

  assert.deepEqual(current, customSettings);
  assert.equal(loaded.needsWrite, true);
  assert.equal(loaded.sourceSchemaVersion, null);
  assert.deepEqual(loaded.store, {
    schemaVersion: 3,
    lastAppliedProfileId: null,
    profiles: [],
  });
});

test('legacy fixed output rates migrate to target mode and exact FPS', () => {
  const fps60 = Profiles.sanitizeSettings({ ...Profiles.DEFAULT_SETTINGS, factor: 'fps60' });
  const fps120 = Profiles.sanitizeSettings({ ...Profiles.DEFAULT_SETTINGS, factor: 'fps120' });

  assert.equal(fps60.factor, 'target');
  assert.equal(fps60.targetFps, 60);
  assert.equal(fps120.factor, 'target');
  assert.equal(fps120.targetFps, 120);
});

test('target FPS accepts decimals, clamps its range, and rounds to two places', () => {
  const clean = Profiles.sanitizeSettings({
    factor: 'target',
    targetFps: '143.5',
    res: '1080',
    model: 'v8-does-not-exist',
    anime: 'false',
    debug: 1,
    showWatermark: null,
    fcTune: { leak: true },
  });

  assert.equal(clean.factor, 'target');
  assert.equal(clean.targetFps, 143.5);
  assert.equal(clean.res, 1080);
  assert.equal(clean.model, 'v7s');
  assert.equal(clean.anime, true);
  assert.equal(clean.debug, false);
  assert.equal(clean.showWatermark, true);
  assert.equal(clean.fpsLimit, null);
  assert.equal(clean.showWarnings, true);
  assert.deepEqual(Object.keys(clean), Profiles.SETTINGS_KEYS);
  assert.equal('fcTune' in clean, false);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 1 }).targetFps, 2);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 1001 }).targetFps, 1000);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 59.944 }).targetFps, 59.94);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: -4 }).targetFps, 120);
  assert.equal(Profiles.sanitizeSettings({ factor: '6', targetFps: 240 }).factor, 6);
  assert.equal(Profiles.sanitizeFpsLimit(137), 144);
  assert.equal(Profiles.sanitizeFpsLimit(247), 240);
  assert.equal(Profiles.sanitizeFpsLimit(null), null);
});

test('schema v2 profiles inherit the current FPS limit and warning preference', () => {
  const legacySettings = { ...customSettings };
  delete legacySettings.fpsLimit;
  delete legacySettings.showWarnings;
  const rawStore = {
    schemaVersion: 2,
    lastAppliedProfileId: 'custom-one',
    profiles: [{
      id: 'custom-one',
      name: 'Living room',
      description: 'Exact setup',
      settings: legacySettings,
    }],
  };
  const snapshot = structuredClone(rawStore);
  const loaded = Profiles.loadStore(rawStore, customSettings);

  assert.equal(loaded.needsWrite, true);
  assert.equal(loaded.sourceSchemaVersion, 2);
  assert.equal(loaded.store.schemaVersion, 3);
  assert.equal(loaded.store.lastAppliedProfileId, 'custom-one');
  assert.equal(loaded.store.profiles[0].settings.fpsLimit, 60);
  assert.equal(loaded.store.profiles[0].settings.showWarnings, false);
  assert.deepEqual(rawStore, snapshot);
  assert.equal(Profiles.loadStore(loaded.store, customSettings).needsWrite, false);
});

test('schema v1 built-in selection disappears while custom profiles and flat settings survive', () => {
  const current = { ...customSettings };
  const loaded = Profiles.loadStore({
    schemaVersion: 1,
    activeProfileId: 'balanced',
    customProfiles: [v1Profile()],
    migratedFromFlat: false,
  }, current);

  assert.equal(loaded.needsWrite, true);
  assert.equal(loaded.sourceSchemaVersion, 1);
  assert.equal(loaded.store.lastAppliedProfileId, null);
  assert.equal(loaded.store.profiles.length, 1);
  assert.deepEqual(loaded.store.profiles[0], {
    id: 'custom-one',
    name: 'Living room',
    description: 'Exact setup',
    settings: customSettings,
  });
  assert.deepEqual(current, customSettings);
  assert.equal(JSON.stringify(loaded.store).includes('builtIn'), false);
});

test('schema v1 custom selection preserves order, identities, metadata, and provenance', () => {
  const secondSettings = { ...customSettings, targetFps: 222.5, res: 720 };
  const loaded = Profiles.loadStore({
    schemaVersion: 1,
    activeProfileId: 'SECOND-PROFILE',
    customProfiles: [
      v1Profile(),
      v1Profile({
        id: 'SECOND-PROFILE',
        name: '  Desk\u0000 setup  ',
        description: '  High   refresh  ',
        settings: secondSettings,
      }),
    ],
    migratedFromFlat: false,
  }, customSettings);

  assert.deepEqual(loaded.store.profiles.map(profile => profile.id), ['custom-one', 'second-profile']);
  assert.equal(loaded.store.profiles[1].name, 'Desk setup');
  assert.equal(loaded.store.profiles[1].description, 'High refresh');
  assert.deepEqual(loaded.store.profiles[1].settings, secondSettings);
  assert.equal(loaded.store.lastAppliedProfileId, 'second-profile');
});

test('v1 applied profile snapshot and quick-control-modified current settings remain distinct', () => {
  const profileSettings = { ...customSettings, res: 720 };
  const current = { ...customSettings, res: 360 };
  const loaded = Profiles.loadStore({
    schemaVersion: 1,
    activeProfileId: 'custom-one',
    customProfiles: [v1Profile({ settings: profileSettings })],
    migratedFromFlat: false,
  }, current);

  assert.equal(loaded.store.lastAppliedProfileId, 'custom-one');
  assert.deepEqual(loaded.store.profiles[0].settings, profileSettings);
  assert.deepEqual(current, { ...customSettings, res: 360 });
  assert.equal(Profiles.isProfileModified(loaded.store, 'custom-one', current), true);
});

test('untouched generated v1 current-settings duplicate collapses into virtual Current settings', () => {
  const current = Profiles.sanitizeSettings({
    ...Profiles.DEFAULT_SETTINGS,
    factor: 'fps60',
  });
  const loaded = Profiles.loadStore({
    schemaVersion: 1,
    activeProfileId: 'current-settings',
    migratedFromFlat: true,
    customProfiles: [{
      id: 'current-settings',
      name: 'Current settings',
      description: 'Settings preserved from before profiles were added.',
      builtIn: false,
      settings: { ...Profiles.DEFAULT_SETTINGS, factor: 'fps60' },
    }],
  }, current);

  assert.deepEqual(loaded.store.profiles, []);
  assert.equal(loaded.store.lastAppliedProfileId, null);
});

test('edited or divergent v1 current-settings profile is retained as user data', () => {
  const current = { ...customSettings, targetFps: 144 };
  const loaded = Profiles.loadStore({
    schemaVersion: 1,
    activeProfileId: 'current-settings',
    migratedFromFlat: true,
    customProfiles: [{
      id: 'current-settings',
      name: 'Current settings',
      description: 'Settings preserved from before profiles were added.',
      builtIn: false,
      settings: { ...customSettings, targetFps: 165.5 },
    }],
  }, current);

  assert.equal(loaded.store.profiles.length, 1);
  assert.equal(loaded.store.profiles[0].settings.targetFps, 165.5);
  assert.equal(loaded.store.lastAppliedProfileId, 'current-settings');
});

test('future and malformed stores fail closed instead of being overwritten', () => {
  const future = {
    schemaVersion: 999,
    profiles: [v1Profile({ builtIn: undefined })],
  };
  const snapshot = structuredClone(future);

  assert.throws(
    () => Profiles.loadStore(future, customSettings),
    Profiles.UnsupportedProfileStoreVersionError,
  );
  assert.deepEqual(future, snapshot);
  assert.throws(
    () => Profiles.loadStore({
      schemaVersion: 1,
      activeProfileId: null,
      customProfiles: 'not-an-array',
    }, customSettings),
    Profiles.ProfileStoreError,
  );
});

test('schema v2 rejects duplicate profile ids without dropping either profile silently', () => {
  assert.throws(
    () => Profiles.loadStore({
      schemaVersion: 2,
      lastAppliedProfileId: null,
      profiles: [
        v1Profile({ builtIn: undefined }),
        v1Profile({ id: 'CUSTOM-ONE', name: 'Duplicate', builtIn: undefined }),
      ],
    }, customSettings),
    /duplicate id/,
  );
});

test('profile roundtrip preserves every canonical setting and excludes transient storage', () => {
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, {
    id: 'custom-roundtrip',
    name: 'Roundtrip',
    settings: { ...customSettings, fcTune: { mustNotPersist: true } },
  });
  store = Profiles.setLastAppliedProfile(store, 'custom-roundtrip');

  const payload = Profiles.toStoragePayload(store, customSettings);
  const reloadedSettings = Profiles.pickFlatSettings(payload);
  const reloaded = Profiles.loadStore(payload[Profiles.STORE_KEY], reloadedSettings);

  assert.deepEqual(reloadedSettings, customSettings);
  assert.deepEqual(Profiles.getProfile(reloaded.store, 'custom-roundtrip').settings, customSettings);
  assert.equal(reloaded.store.lastAppliedProfileId, 'custom-roundtrip');
  assert.equal('fcTune' in payload, false);
  assert.equal(JSON.stringify(payload).includes('fcTune'), false);
  assert.deepEqual(Object.keys(payload).sort(), [...Profiles.SETTINGS_KEYS, Profiles.STORE_KEY].sort());
});

test('Current settings and saved profile previews resolve without mutating the store', () => {
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, {
    id: 'saved-preview',
    name: 'Saved preview',
    settings: customSettings,
  });
  const before = structuredClone(store);
  const current = { ...customSettings, targetFps: 143.5 };

  const currentSelection = Profiles.resolveProfileSelection(store, null, current);
  const profileSelection = Profiles.resolveProfileSelection(store, 'saved-preview', current);

  assert.deepEqual(currentSelection, {
    selectedProfileId: null,
    source: 'current',
    settings: current,
  });
  assert.equal(profileSelection.selectedProfileId, 'saved-preview');
  assert.equal(profileSelection.source, 'profile');
  assert.deepEqual(profileSelection.settings, customSettings);
  profileSelection.settings.res = 288;
  assert.equal(Profiles.getProfile(store, 'saved-preview').settings.res, 1080);
  assert.deepEqual(store, before);
});

test('save changes only the profile snapshot while apply publishes flat runtime settings', () => {
  const current = { ...customSettings, targetFps: 120 };
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, {
    id: 'saved-profile',
    name: 'Saved',
    settings: customSettings,
  });

  assert.deepEqual(current, { ...customSettings, targetFps: 120 });
  assert.equal(store.lastAppliedProfileId, null);

  store = Profiles.setLastAppliedProfile(store, 'saved-profile');
  const applied = Profiles.toStoragePayload(store, customSettings);
  assert.equal(applied.lastAppliedProfileId, undefined);
  assert.equal(applied[Profiles.STORE_KEY].lastAppliedProfileId, 'saved-profile');
  assert.deepEqual(Profiles.pickFlatSettings(applied), customSettings);

  const currentApplied = Profiles.toStoragePayload(
    Profiles.setLastAppliedProfile(store, null),
    current,
  );
  assert.equal(currentApplied[Profiles.STORE_KEY].lastAppliedProfileId, null);
  assert.deepEqual(Profiles.pickFlatSettings(currentApplied), current);
});

test('quick-control changes preserve profile snapshots and last-applied provenance', () => {
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, {
    id: 'last-applied',
    name: 'Last applied',
    settings: customSettings,
  });
  store = Profiles.setLastAppliedProfile(store, 'last-applied');
  const before = structuredClone(store);
  const changedInPlayer = { ...customSettings, res: 720 };

  assert.equal(Profiles.isProfileModified(store, 'last-applied', changedInPlayer), true);
  assert.equal(Profiles.findMatchingProfileId(store, changedInPlayer), null);
  assert.deepEqual(store, before);
});

test('deleting selected or last-applied profile clears provenance without changing flat settings', () => {
  const current = { ...customSettings };
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, {
    id: 'active-custom',
    name: 'Active custom',
    settings: customSettings,
  });
  store = Profiles.setLastAppliedProfile(store, 'active-custom');
  const deleted = Profiles.deleteProfile(store, 'active-custom');
  const selection = Profiles.resolveProfileSelection(deleted, 'active-custom', current);

  assert.equal(deleted.lastAppliedProfileId, null);
  assert.deepEqual(deleted.profiles, []);
  assert.equal(selection.selectedProfileId, null);
  assert.equal(selection.source, 'current');
  assert.deepEqual(selection.settings, current);
});

test('advanced page keeps compact toolbar and exposes exact target FPS controls', () => {
  for (const key of Profiles.SETTINGS_KEYS) {
    const matches = optionsHtml.match(new RegExp(`id=["']${key}["']`, 'g')) || [];
    assert.equal(matches.length, 1, `expected exactly one #${key} control`);
  }

  const factorOptions = [...optionsHtml.matchAll(/<option value="(auto|hz|target|[2-6])">/g)]
    .map(match => match[1]);
  assert.deepEqual(factorOptions, ['auto', 'hz', 'target', '2', '3', '4', '5', '6']);
  assert.match(optionsHtml, /class="profile-toolbar"/);
  assert.match(optionsHtml, /id="profileSelect"/);
  assert.match(optionsHtml, /id="targetFps"[^>]*type="number"/);
  assert.match(optionsHtml, /min="2" max="1000" step="0\.01" inputmode="decimal"/);
  assert.equal(/profile-sidebar|Built-in|builtInProfiles|fps60|fps120|60 FPS|120 FPS/.test(optionsHtml), false);
  assert.match(optionsHtml, /v6 · legacy/);
  assert.match(optionsHtml, /id="showWatermark"/);
});

test('advanced page keeps actions visible and reserves scroll space for the fixed bar', () => {
  const actionBarRule = optionsCss.match(/\.action-bar\s*\{([^}]*)\}/)?.[1] || '';
  const workspaceRule = optionsCss.match(/\.workspace\s*\{([^}]*)\}/)?.[1] || '';
  assert.match(actionBarRule, /\bposition\s*:\s*fixed\b/);
  assert.match(actionBarRule, /\bbottom\s*:\s*0\b/);
  assert.match(workspaceRule, /padding\s*:\s*30px 0 104px/);
  assert.match(optionsCss, /@media \(max-width: 640px\)[\s\S]*?padding-bottom:\s*104px/);
});

test('options controller exposes virtual Current first and keeps preview, save, and apply separate', () => {
  assert.match(optionsJs, /current\.textContent = 'Current settings'/);
  assert.match(optionsJs, /group\.label = 'My profiles'/);
  assert.match(optionsJs, /selectedProfileId = null/);
  assert.match(optionsJs, /function loadCurrentDraft\(\)/);
  assert.match(optionsJs, /Profiles\.loadStore\(/);
  assert.match(optionsJs, /chrome\.storage\.local\.set\(\{ \[Profiles\.STORE_KEY\]: nextStore \}\)/);
  assert.match(optionsJs, /Profiles\.toStoragePayload\(nextStore, draftSettings\)/);
  assert.match(optionsJs, /Profiles\.setLastAppliedProfile\(store, selected\?\.id \|\| null\)/);
  assert.match(optionsJs, /updateRateControlVisibility\(draftSettings\.factor\)/);
  assert.match(optionsJs, /'showFps', 'showWatermark', 'showWarnings'/);
  assert.match(optionsJs, /fpsLimit:\s*fpsLimitFromSlider\(\)/);
  assert.match(optionsJs, /let profileStoreReadOnly = false/);
  assert.match(optionsJs, /enterProfileStoreReadOnly\(error\)/);
  assert.match(optionsJs, /if \(profileStoreReadOnly\) return/);
  assert.match(optionsJs, /for \(const id of STORE_MUTATION_CONTROLS\) \$\(id\)\.disabled = true/);
  assert.equal(/BUILT_IN_PROFILES|builtIn|activeProfileId|profile\.description/.test(optionsJs), false);
});

test('extension exposes the full-page configurator without replacing popup status and help', () => {
  assert.equal(manifest.version, '1.4.5');
  assert.match(contentJs, /const VERSION = '1\.4\.5'/);
  assert.match(optionsHtml, /<span class="version">v1\.4\.5<\/span>/);
  assert.deepEqual(manifest.options_ui, { page: 'options.html', open_in_tab: true });
  assert.deepEqual(manifest.content_scripts[0].js, ['cadence.js', 'profile-store.js', 'content.js']);
  assert.deepEqual(manifest.background, { service_worker: 'background.js' });
  assert.match(optionsHtml, /id="runtimeStatus"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(popupHtml, /id="tabStatus"/);
  assert.match(popupHtml, /id="tabHelp"/);
  assert.match(popupHtml, /id="fullSettings"/);
  assert.match(popupHtml, /github\.com\/MONZikWasTaken\/Framegen/);
  assert.match(popupJs, /chrome\.runtime\.openOptionsPage\(\)/);
  assert.match(backgroundJs, /message\?\.type !== 'fcOpenOptions'/);
  assert.match(backgroundJs, /chrome\.tabs\.create\(createOptions\)/);
});

test('runtime shares the profile contract while continuing to consume flat settings', () => {
  assert.match(contentJs, /const Profiles = globalThis\.FramegenProfiles/);
  assert.match(contentJs, /Profiles\.STORE_KEY/);
  assert.match(contentJs, /Profiles\.toStoragePayload\(nextStore, profile\.settings\)/);
  assert.match(contentJs, /chrome\.storage\.local\.set\(cfg\)/);
  assert.match(contentJs, /showWatermark:\s*true/);
  assert.match(contentJs, /showWarnings:\s*true/);
  assert.match(contentJs, /fpsLimit:\s*null/);
  assert.match(contentJs, /wm\.style\.display = cfg\.showWatermark \? 'block' : 'none'/);
  assert.match(contentJs, /k === 'res' \|\| k === 'model' \|\| k === 'guard'/);
  assert.match(contentJs, /rtGuard === guard/);
  assert.match(contentJs, /while \(!runtimeMatches\(\)\)/);
  assert.match(contentJs, /await settingsReady/);
  assert.match(contentJs, /staticGuard: buildGuard/);
  assert.match(contentJs, /chrome\.runtime\.sendMessage\(\{ type: 'fcOpenOptions' \}\)/);
  assert.doesNotMatch(contentJs, /id="fcModel"/);
  assert.match(optionsHtml, /<option value="v6">v6 · legacy<\/option>/);
});
