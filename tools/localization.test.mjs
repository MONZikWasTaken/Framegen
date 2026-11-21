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

test('settings changes switch catalogs, interpolate text and use English for a missing message', async () => {
  let storageChanged;
  globalThis.chrome = {
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
  } finally {
    globalThis.fetch = fetchCatalog;
    delete globalThis.chrome;
  }
});
