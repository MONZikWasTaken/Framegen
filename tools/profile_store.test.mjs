import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { Profiles } from '../extension/src/profiles.js';

const manifest = JSON.parse(readFileSync(new URL('../extension/manifest.json', import.meta.url), 'utf8'));
const optionsHtml = readFileSync(new URL('../extension/options.html', import.meta.url), 'utf8');

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
  sharpness: 0,
  showFps: false,
  showWatermark: false,
  showWarnings: false,
  subtitleOverlay: true,
  guard: false,
  fillDisplay: false,
  limitToDisplay: false,
  motionAdaptive: false,
  periodicRepair: false,
  cadenceSmoothing: false,
  renderCubic: false,
});

test('the schema is the one the options page edits', () => {
  for (const key of Profiles.SETTINGS_KEYS) {
    assert.match(optionsHtml, new RegExp(`id="${key}"`), `options.html has no control for ${key}`);
  }
  assert.doesNotMatch(optionsHtml, /id="model"|id="flowUpsample"/);
  assert.equal(manifest.version, '2.0.0');
  assert.deepEqual(manifest.content_scripts[0].js, ['content.js']);
});

test('a missing store initializes empty without materializing the current settings', () => {
  const current = Profiles.pickFlatSettings({ ...customSettings, fcTune: { adapter: 'stays outside profiles' } });
  const loaded = Profiles.loadStore(undefined, current);
  assert.deepEqual(current, customSettings);
  assert.equal(loaded.needsWrite, true);
  assert.equal(loaded.sourceSchemaVersion, null);
  assert.deepEqual(loaded.store, { schemaVersion: Profiles.SCHEMA_VERSION, lastAppliedProfileId: null, profiles: [] });
});

test('legacy fixed output rates migrate to target mode and exact FPS', () => {
  const fps60 = Profiles.sanitizeSettings({ ...Profiles.DEFAULT_SETTINGS, factor: 'fps60' });
  const fps120 = Profiles.sanitizeSettings({ ...Profiles.DEFAULT_SETTINGS, factor: 'fps120' });
  assert.equal(fps60.factor, 'target');
  assert.equal(fps60.targetFps, 60);
  assert.equal(fps120.factor, 'target');
  assert.equal(fps120.targetFps, 120);
});

test('sanitizing clamps, rounds and drops unknown keys', () => {
  const clean = Profiles.sanitizeSettings({
    factor: 'target', targetFps: '143.5', res: '1080', model: 'v7s', flowUpsample: 'gate',
    anime: 'false', debug: 1, showWatermark: null, fcTune: { leak: true },
  });
  assert.equal(clean.factor, 'target');
  assert.equal(clean.targetFps, 143.5);
  assert.equal(clean.res, 1080);
  assert.equal(clean.anime, true);
  assert.equal(clean.debug, false);
  assert.equal(clean.showWatermark, true);
  assert.equal(clean.fpsLimit, null);
  assert.deepEqual(Object.keys(clean), Profiles.SETTINGS_KEYS);
  assert.equal('model' in clean, false);
  assert.equal('fcTune' in clean, false);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 1 }).targetFps, 2);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 1001 }).targetFps, 1000);
  assert.equal(Profiles.sanitizeSettings({ factor: 'target', targetFps: 59.944 }).targetFps, 59.94);
  assert.equal(Profiles.sanitizeSettings({ factor: '6', targetFps: 240 }).factor, 6);
  assert.equal(Profiles.sanitizeFpsLimit(137), 144);
  assert.equal(Profiles.sanitizeFpsLimit(247), 240);
  assert.equal(Profiles.sanitizeFpsLimit(null), null);
  assert.equal(Profiles.sanitizeSettings({ sharpness: '3' }).sharpness, 3);
  assert.equal(Profiles.sanitizeSettings({ sharpness: 4 }).sharpness, 0);
});

test('schema 4 stores load, lose the removed keys and are written back as schema 5', () => {
  const v4 = {
    schemaVersion: 4,
    lastAppliedProfileId: 'living-room',
    profiles: [{ id: 'living-room', name: 'Living room', description: '',
      settings: { ...customSettings, model: 'v6', flowUpsample: 'bilinear' } }],
  };
  const loaded = Profiles.loadStore(v4, Profiles.DEFAULT_SETTINGS);
  assert.equal(loaded.needsWrite, true);
  assert.equal(loaded.sourceSchemaVersion, 4);
  assert.equal(loaded.store.schemaVersion, 5);
  assert.equal(loaded.store.lastAppliedProfileId, 'living-room');
  assert.deepEqual(loaded.store.profiles[0].settings, customSettings);
  assert.equal(Profiles.findMatchingProfileId(loaded.store, customSettings), 'living-room');
});

test('schema 1 stores drop the untouched legacy "current settings" profile', () => {
  const v1 = {
    schemaVersion: 1,
    migratedFromFlat: true,
    activeProfileId: 'current-settings',
    customProfiles: [
      { id: 'current-settings', name: 'Current settings', builtIn: false,
        description: 'Settings preserved from before profiles were added.', settings: customSettings },
      { id: 'mine', name: 'Mine', builtIn: false, settings: { ...customSettings, res: 720 } },
    ],
  };
  const loaded = Profiles.loadStore(v1, customSettings);
  assert.deepEqual(loaded.store.profiles.map(p => p.id), ['mine']);
  assert.equal(loaded.store.lastAppliedProfileId, null);
});

test('unknown schemas and malformed stores are refused', () => {
  assert.throws(() => Profiles.loadStore({ schemaVersion: 99, profiles: [] }), Profiles.UnsupportedProfileStoreVersionError);
  assert.throws(() => Profiles.loadStore([]), Profiles.ProfileStoreError);
  assert.throws(() => Profiles.loadStore({ schemaVersion: 5, profiles: [{ id: 'bad id', settings: {} }] }), Profiles.ProfileStoreError);
});

test('profile edits keep the store normalized', () => {
  let store = Profiles.emptyStore();
  store = Profiles.upsertProfile(store, { id: 'Night', name: '  Night   mode ', description: 'x'.repeat(200), settings: customSettings });
  assert.equal(store.profiles[0].id, 'night');
  assert.equal(store.profiles[0].name, 'Night mode');
  assert.equal(store.profiles[0].description.length, 160);
  store = Profiles.renameProfile(store, 'night', 'Evening');
  assert.equal(store.profiles[0].name, 'Evening');
  store = Profiles.setLastAppliedProfile(store, 'night');
  assert.equal(store.lastAppliedProfileId, 'night');
  const payload = Profiles.toStoragePayload(store, customSettings);
  assert.equal(payload[Profiles.STORE_KEY].lastAppliedProfileId, 'night');
  assert.equal(payload.res, 1080);
  store = Profiles.deleteProfile(store, 'night');
  assert.deepEqual(store.profiles, []);
  assert.equal(store.lastAppliedProfileId, null);
  assert.throws(() => Profiles.setLastAppliedProfile(store, 'night'), TypeError);
});
