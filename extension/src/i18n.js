export const DEFAULT_LANGUAGE = 'en';
export const LANGUAGES = Object.freeze({ en: 'English', ru: 'Русский' });
export const LANGUAGE_KEY = 'fcLanguage';

export function chooseLanguage(preference, browserLanguages) {
  if (Object.hasOwn(LANGUAGES, preference)) return preference;
  for (const language of browserLanguages) {
    const parts = language.toLowerCase().replaceAll('_', '-').split('-');
    while (parts.length) {
      const match = Object.keys(LANGUAGES).find(code => code.toLowerCase().replaceAll('_', '-') === parts.join('-'));
      if (match) return match;
      parts.pop();
    }
  }
  return DEFAULT_LANGUAGE;
}

let preference = 'auto';
let language = DEFAULT_LANGUAGE;
let english = {};
let messages = {};
let loading;
let revision = 0;
const catalogs = new Map();
const listeners = new Set();

export function getLanguage() { return language; }
export function getLanguagePreference() { return preference; }
export function getAutomaticLanguage() {
  const uiLanguage = chrome.i18n?.getUILanguage?.();
  return chooseLanguage('auto', [uiLanguage, ...(navigator.languages || [])].filter(Boolean));
}

function loadCatalog(code) {
  if (!catalogs.has(code)) {
    const url = new URL(`../_locales/${code}/messages.json`, import.meta.url);
    const pending = fetch(url).then(response => {
      if (!response.ok) throw new Error(`Could not load Framegen locale ${code}`);
      return response.json();
    }).catch(() => {
      catalogs.delete(code);
      return null;
    });
    catalogs.set(code, pending);
  }
  return catalogs.get(code);
}

async function applyPreference(value) {
  const current = ++revision;
  const nextPreference = Object.hasOwn(LANGUAGES, value) ? value : 'auto';
  const selected = nextPreference === 'auto' ? getAutomaticLanguage() : nextPreference;
  const [fallback, catalog] = await Promise.all([loadCatalog(DEFAULT_LANGUAGE), loadCatalog(selected)]);
  if (current !== revision) return;
  preference = nextPreference;
  language = catalog ? selected : DEFAULT_LANGUAGE;
  english = fallback || {};
  messages = catalog || english;
  for (const listener of listeners) listener();
}

export function initLanguage() {
  if (loading) return loading;
  let applied;
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && Object.hasOwn(changes, LANGUAGE_KEY)) {
      applied = applyPreference(changes[LANGUAGE_KEY].newValue);
    }
  });
  loading = (async () => {
    const saved = await chrome.storage.local.get(LANGUAGE_KEY).catch(() => ({}));
    applied ||= applyPreference(saved[LANGUAGE_KEY]);
    let pending;
    do { pending = applied; await pending; } while (pending !== applied);
  })();
  return loading;
}

export function onLanguageChange(listener) { listeners.add(listener); }

export function t(key, values = {}) {
  const message = messages[key]?.message || english[key]?.message
    || globalThis.chrome?.i18n?.getMessage?.(key) || key;
  return interpolate(message, values);
}

export function tEnglish(key, values = {}) {
  return interpolate(english[key]?.message || key, values);
}

function interpolate(message, values) {
  return message.replace(/\{([a-z_][a-z0-9_]*)\}/gi, (token, name) => Object.hasOwn(values, name) ? String(values[name]) : token);
}

const textAttributes = ['aria-label', 'aria-valuetext', 'label'];
const selector = ['[data-i18n]', ...textAttributes.map(attr => `[data-i18n-${attr}]`)].join(',');

export function translate(root) {
  for (const element of root.querySelectorAll(selector)) {
    if (element.dataset.i18n) element.textContent = t(element.dataset.i18n);
    for (const attribute of textAttributes) {
      const key = element.getAttribute(`data-i18n-${attribute}`);
      if (key) element.setAttribute(attribute, t(key));
    }
  }
}
