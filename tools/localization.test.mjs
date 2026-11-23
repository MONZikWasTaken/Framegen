import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import test from 'node:test';
import { chooseLanguage, DEFAULT_LANGUAGE, getLanguage, initLanguage, LANGUAGES, LANGUAGE_KEY, onLanguageChange, t } from '../extension/src/i18n.js';

const extension = new URL('../extension/', import.meta.url);
const localeRoot = new URL('_locales/', extension);
const locales = readdirSync(localeRoot);
const read = path => readFileSync(new URL(path, extension), 'utf8');
const catalog = locale => JSON.parse(read(`_locales/${locale}/messages.json`));
const english = catalog(DEFAULT_LANGUAGE);
const placeholders = message => [...message.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort();

test('browser languages, regional variants and explicit overrides select the intended locale', () => {
  assert.equal(chooseLanguage('auto', ['ru-RU', 'en-US']), 'ru');
  assert.equal(chooseLanguage('auto', ['en-GB', 'ru']), 'en');
  assert.equal(chooseLanguage('auto', ['de-DE', 'ru-RU']), 'ru');
  assert.equal(chooseLanguage('auto', ['de-DE']), 'en');
  assert.equal(chooseLanguage('ru', ['en-US']), 'ru');
  assert.equal(chooseLanguage('en', ['ru']), 'en');
  assert.equal(chooseLanguage('../en', ['ru']), 'ru');
  assert.equal(chooseLanguage('auto', ['ja-JP', 'en-US']), 'ja');
  for (const variant of ['zh-CN', 'zh_CN', 'zh-Hans', 'zh-Hans-CN', 'zh-SG', 'zh']) {
    assert.equal(chooseLanguage('auto', [variant, 'en-US']), 'zh_CN', variant);
  }
  for (const variant of ['zh-TW', 'zh-HK', 'zh-MO', 'zh-Hant', 'zh-Hant-CN']) {
    assert.equal(chooseLanguage('auto', [variant, 'en-US']), 'en', variant);
  }
  assert.equal(chooseLanguage('auto', ['zh-Hant', 'ja-JP']), 'ja');
  assert.equal(chooseLanguage('ja', ['ru']), 'ja');
  assert.equal(chooseLanguage('zh_CN', ['ja-JP']), 'zh_CN');
});

test('each packaged locale has the English keys and preserves placeholders', () => {
  assert.deepEqual(locales.sort(), Object.keys(LANGUAGES).sort());
  const manifest = JSON.parse(read('manifest.json'));
  assert.equal(manifest.default_locale, DEFAULT_LANGUAGE);
  assert.ok(manifest.web_accessible_resources.some(entry => entry.resources.includes('_locales/*')));
  for (const file of ['manifest.json', 'options.html', 'popup.html', 'src/ui/controls.js', 'src/ui/bar.js']) {
    for (const [, attribute, manifestKey] of read(file).matchAll(/data-i18n(?:-aria-[a-z]+)?="(\w+)"|__MSG_(\w+)__/g)) {
      const key = attribute || manifestKey;
      assert.ok(english[key], `${file}: ${key}`);
    }
  }
  for (const locale of locales) {
    const messages = catalog(locale);
    assert.deepEqual(Object.keys(messages).sort(), Object.keys(english).sort());
    for (const [key, { message }] of Object.entries(messages)) {
      assert.ok(message.trim(), `${locale}: ${key}`);
      assert.deepEqual(placeholders(message), placeholders(english[key].message), `${locale}: ${key}`);
    }
  }
});

test('settings changes switch catalogs, interpolate text and use English for a missing message', async context => {
  let storageChanged;
  globalThis.chrome = {
    runtime: { getURL: path => path },
    i18n: { getUILanguage: () => 'en-US' },
    storage: {
      local: { get: async () => ({ [LANGUAGE_KEY]: 'en' }) },
      onChanged: { addListener: listener => { storageChanged = listener; } },
    },
  };
  const fetchCatalog = globalThis.fetch;
  globalThis.fetch = async url => {
    const messages = JSON.parse(readFileSync(url, 'utf8'));
    if (url.pathname.includes('/ru/')) delete messages.debug;
    return { ok: true, json: async () => messages };
  };
  try {
    await initLanguage();
    assert.equal(getLanguage(), 'en');
    assert.equal(t('settings'), 'Settings');
    const changed = new Promise(resolve => onLanguageChange(resolve));
    storageChanged({ [LANGUAGE_KEY]: { newValue: 'ru' } }, 'local');
    await changed;
    assert.equal(getLanguage(), 'ru');
    assert.equal(t('previewing', { name: '<Film> & TV' }), 'Предпросмотр «<Film> & TV»');
    assert.equal(t('debug'), 'Debug overlay');
    await context.test('technical player diagnostics stay in English in the Russian interface', async () => {
      const { Player } = await import('../extension/src/player.js');
      const cfg = { fg: true, factor: 'auto', fpsLimit: 120, sharpness: 0, res: 480 };
      const video = { videoWidth: 2560, videoHeight: 1440 };
      const player = new Player({ settings: { cfg }, hooks: { candidateVideo: () => video } });
      player.sys.gpu = 'nvidia lovelace';
      player.backend = { class: 'portable-f16' };
      assert.deepEqual(player.statusLines(), [
        'GPU: nvidia lovelace',
        'engine: portable-f16',
        'FG: on · upscale: off',
        'source: 2560×1440',
        'output rate: Auto · max 120 FPS',
        'HDR: display not HDR',
        'sharpness: off',
        'status: stopped',
      ]);
      assert.match(player.hudText(), /Авто/);

      cfg.debug = true;
      player.running = true;
      player.video = video;
      player.pacer = { report: () => ({
        plan: { state: 'active', clamp: 'display', interpolation_allowed: true, output_hz: 60 },
        pair_cost_ms: 2, mid_cost_ms: 1, present_cost_ms: 0.5, delay_ms: 42, late_avg_ms: 0,
        source_hz: 30, display: { display_hz: 60, measured: true },
        pressure: 0, penalty: 0, strain_penalty: 0, counters: { cuts: 0, holds: 0 }, motion_avg: 0, queued: 0,
      }) };
      assert.match(player.hudText(), /\(Auto · max 120 FPS\).*pair 2\.0.*queue 0/);
      assert.match(player.statusLines().join('\n'), /output rate: Auto · max 120 FPS \(~60 FPS\) · browser callback limit/);
      assert.match(player.status().rateState, /Авто.*лимит браузера/);
    });
    for (const [locale, settings, preview] of [
      ['ja', '設定', '「<Film> & TV」をプレビュー中'],
      ['zh_CN', '设置', '正在预览“<Film> & TV”'],
      ['en', 'Settings', 'Previewing <Film> & TV'],
      ['ru', 'Настройки', 'Предпросмотр «<Film> & TV»'],
    ]) {
      const switched = new Promise(resolve => onLanguageChange(resolve));
      storageChanged({ [LANGUAGE_KEY]: { newValue: locale } }, 'local');
      await switched;
      assert.equal(getLanguage(), locale);
      assert.equal(t('settings'), settings);
      assert.equal(t('previewing', { name: '<Film> & TV' }), preview);
    }
  } finally {
    globalThis.fetch = fetchCatalog;
    delete globalThis.chrome;
  }
});
