// The live settings of this page: the flat keys of chrome.storage.local, sanitized through the schema, kept in sync
// with every other tab and the options page, and announced to the host as sets of changed keys.
import { Profiles } from './profiles.js';

export function createSettings({ live = true, log = () => {} } = {}) {
  const cfg = { ...Profiles.DEFAULT_SETTINGS };
  const listeners = new Set();
  const profileListeners = new Set();
  let settle = () => {};
  const ready = live
    ? new Promise(resolve => { settle = resolve; })
    : Promise.resolve();

  function assign(values) {
    const previous = { ...cfg };
    const clean = Profiles.sanitizeSettings({ ...cfg, ...values });
    const changed = new Set();
    for (const key of Profiles.SETTINGS_KEYS) {
      if (cfg[key] !== clean[key]) changed.add(key);
      cfg[key] = clean[key];
    }
    return { changed, previous };
  }

  function notify(change) {
    if (!change.changed.size) return;
    for (const listener of listeners) {
      try { listener(change); } catch (e) { log('settings listener', e); }
    }
  }

  function persist(keys) {
    const payload = {};
    for (const key of keys) payload[key] = cfg[key];
    try { chrome.storage.local.set(payload); } catch { /* storage unavailable in this frame */ }
  }

  if (live) {
    try {
      chrome.storage.local.get(Profiles.SETTINGS_KEYS, stored => {
        try {
          const storedFpsLimit = stored.fpsLimit;
          notify(assign(stored));
          if (storedFpsLimit !== undefined && storedFpsLimit !== cfg.fpsLimit) persist(['fpsLimit']);
        } finally { settle(); }
      });
      chrome.storage.onChanged.addListener((changes, area) => {
        if (area !== 'local') return;
        const incoming = {};
        for (const key of Profiles.SETTINGS_KEYS) {
          if (Object.hasOwn(changes, key)) incoming[key] = changes[key].newValue;
        }
        if (Object.keys(incoming).length) {
          const change = assign(incoming);
          notify(change);
          if (Object.hasOwn(incoming, 'fpsLimit') && incoming.fpsLimit !== cfg.fpsLimit) persist(['fpsLimit']);
        }
        if (Object.hasOwn(changes, Profiles.STORE_KEY)) {
          for (const listener of profileListeners) {
            try { listener(changes[Profiles.STORE_KEY].newValue); } catch (e) { log('profile listener', e); }
          }
        }
      });
    } catch { settle(); }
  }

  return {
    cfg,
    ready,
    onChange(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    onProfilesChange(listener) { profileListeners.add(listener); return () => profileListeners.delete(listener); },
    set(values, { persist: write = true } = {}) {
      const change = assign(values);
      notify(change);
      if (write && change.changed.size) persist(change.changed);
      return change;
    },
  };
}
