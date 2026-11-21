import { Profiles } from './src/profiles.js';
import { PICTURE_SWATCH_FILTERS } from './src/picture-modes.js';
import { bindLanguageSelect, getLanguage, initLanguage, onLanguageChange, t, translate } from './src/i18n.js';

(function framegenOptions() {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const BOOLEAN_FIELDS = Object.freeze([
    'fg', 'sr', 'hdr', 'anime', 'guard', 'fillDisplay', 'limitToDisplay', 'motionAdaptive', 'periodicRepair', 'cadenceSmoothing', 'renderCubic', 'hoverReveal',
    'showFps', 'showWatermark', 'showWarnings', 'subtitleOverlay', 'compare', 'debug',
  ]);
  const FPS_LIMIT_PRESETS = Profiles.FPS_LIMIT_PRESETS;
  const STORE_MUTATION_CONTROLS = Object.freeze([
    'profileSelect', 'newProfile', 'duplicateProfile', 'renameProfile',
    'deleteProfile', 'saveProfile', 'resetDraft', 'applySettings',
  ]);
  let storageSnapshot = {};
  let store = null;
  let runtimeSettings = null;
  let selectedProfileId = null;
  let draftSettings = null;
  let draftOrigin = null;
  let draftSource = 'current';
  let dialogMode = null;
  let externalChangeNotice = false;
  let lastMessage = '';
  let profileStoreReadOnly = false;
  let profileStoreStatus = '';
  let unsupportedSchema;

  function enterProfileStoreReadOnly(error, status = 'profile_read_only') {
    profileStoreReadOnly = true;
    profileStoreStatus = status;
    unsupportedSchema = error instanceof Profiles.UnsupportedProfileStoreVersionError ? error.schemaVersion : undefined;
    console.warn('Framegen profile storage:', error);
    renderStatus();
    for (const id of ['profileDialog', 'deleteDialog']) {
      const dialog = $(id);
      if (dialog.open) dialog.close();
    }
  }

  function leaveProfileStoreReadOnly() {
    profileStoreReadOnly = false;
    profileStoreStatus = '';
    unsupportedSchema = undefined;
    for (const id of STORE_MUTATION_CONTROLS) $(id).disabled = false;
  }

  function settingsCopy(settings) {
    return { ...Profiles.sanitizeSettings(settings) };
  }

  function currentProfile() {
    return selectedProfileId ? Profiles.getProfile(store, selectedProfileId) : null;
  }

  function loadCurrentDraft() {
    selectedProfileId = null;
    draftSettings = settingsCopy(runtimeSettings);
    draftOrigin = settingsCopy(runtimeSettings);
    draftSource = 'current';
  }

  function loadProfileDraft(profileId) {
    const selection = Profiles.resolveProfileSelection(store, profileId, runtimeSettings);
    if (selection.source === 'current') {
      loadCurrentDraft();
      return;
    }
    selectedProfileId = selection.selectedProfileId;
    draftSettings = settingsCopy(selection.settings);
    draftOrigin = settingsCopy(selection.settings);
    draftSource = 'profile';
  }

  function createProfileId() {
    const token = globalThis.crypto?.randomUUID
      ? globalThis.crypto.randomUUID().replace(/-/g, '')
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
    return `custom-${token}`.slice(0, 63);
  }

  function fpsLimitFromSlider() {
    const index = Math.max(0, Math.min(
      FPS_LIMIT_PRESETS.length - 1,
      Math.round(Number($('fpsLimit').value) || 0),
    ));
    return FPS_LIMIT_PRESETS[index];
  }

  function updateFpsLimitPresentation() {
    const input = $('fpsLimit');
    const value = fpsLimitFromSlider();
    const label = value === null ? t('unlimited') : t('fps_value', { rate: value });
    const ratio = Number(input.value) / Math.max(1, FPS_LIMIT_PRESETS.length - 1);
    input.style.setProperty('--fill', `${ratio * 100}%`);
    input.setAttribute('aria-valuetext', label);
    $('fpsLimitValue').value = label;
    $('fpsLimitValue').textContent = label;
  }

  function renderFpsLimitScale() {
    const values = [15, 60, 120, 240, null];
    const maxIndex = FPS_LIMIT_PRESETS.length - 1;
    $('fpsLimitScale').replaceChildren(...values.map(value => {
      const mark = document.createElement('span');
      const position = Profiles.fpsLimitPresetIndex(value) / maxIndex * 100;
      const offset = Number((7.5 * (1 - 2 * position / 100)).toFixed(3));
      mark.textContent = value === null ? '∞' : String(value);
      mark.style.setProperty('--position', `${position}%`);
      mark.style.setProperty('--offset', `${offset}px`);
      return mark;
    }));
  }

  function updateRateControlVisibility(factor) {
    const custom = factor === 'target';
    const auto = factor === 'auto';
    $('targetFpsRow').hidden = !custom;
    $('targetFps').disabled = !custom;
    $('targetFps').required = custom;
    $('fpsLimitRow').hidden = !auto;
    $('fpsLimit').disabled = !auto;
    $('fillDisplayRow').hidden = factor !== 'hz' && factor !== 'target';
  }

  function validateVisibleRateControl() {
    if (draftSettings.factor !== 'target') return true;
    const input = $('targetFps');
    input.setCustomValidity('');
    if (input.checkValidity()) return true;
    lastMessage = 'custom_fps_range';
    input.setCustomValidity(t(lastMessage));
    input.reportValidity();
    renderStatus();
    return false;
  }

  // Tiles and segments that stand in for a select: the select stays the form's value, these show and set it.
  function buildChoices() {
    for (const group of document.querySelectorAll('[data-choice]')) {
      const select = $(group.dataset.choice);
      const look = group.classList.contains('look-grid');
      group.replaceChildren(...[...select.options].map(option => {
        const button = document.createElement('button');
        button.type = 'button';
        button.setAttribute('role', 'radio');
        button.dataset.value = option.value;
        if (look) {
          button.className = 'look';
          const swatch = document.createElement('span');
          swatch.className = 'look-swatch';
          swatch.style.filter = PICTURE_SWATCH_FILTERS[option.value] || 'none';
          const name = document.createElement('span');
          name.className = 'look-name';
          name.textContent = option.textContent;
          button.append(swatch, name);
        } else {
          button.textContent = option.textContent;
        }
        return button;
      }));
      group.addEventListener('click', event => {
        const button = event.target.closest('button');
        if (!button || select.disabled) return;
        select.value = button.dataset.value;
        syncChoices();
        select.dispatchEvent(new Event('input', { bubbles: true }));
      });
    }
  }

  function syncChoices() {
    for (const group of document.querySelectorAll('[data-choice]')) {
      const value = $(group.dataset.choice).value;
      for (const button of group.children) button.setAttribute('aria-checked', String(button.dataset.value === value));
    }
  }

  function watchSections() {
    const links = [...document.querySelectorAll('[data-nav]')];
    const mark = id => { for (const link of links) link.setAttribute('aria-current', String(link.hash === '#' + id)); };
    const visible = new Set();
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target.id);
        else visible.delete(entry.target.id);
      }
      const first = links.find(link => visible.has(link.hash.slice(1)));
      if (first && !atEnd()) mark(first.hash.slice(1));
    }, { rootMargin: '-80px 0px -55% 0px' });
    // the last sections are too short to reach the top of the window
    const atEnd = () => innerHeight + scrollY >= document.documentElement.scrollHeight - 2;
    addEventListener('scroll', () => { if (atEnd()) mark(links.at(-1).hash.slice(1)); }, { passive: true });
    for (const link of links) observer.observe($(link.hash.slice(1)));
    mark(links[0].hash.slice(1));
  }

  function readForm() {
    const candidate = {
      factor: $('factor').value,
      targetFps: Number($('targetFps').value),
      fpsLimit: fpsLimitFromSlider(),
      res: Number($('res').value),
      sharpness: Number($('sharpness').value),
      pictureMode: $('pictureMode').value,
    };
    for (const key of BOOLEAN_FIELDS) candidate[key] = $(key).checked;
    draftSettings = settingsCopy(candidate);
    updateRateControlVisibility(draftSettings.factor);
  }

  function writeForm(settings) {
    const clean = Profiles.sanitizeSettings(settings);
    $('factor').value = String(clean.factor);
    $('targetFps').value = String(clean.targetFps);
    $('fpsLimit').value = String(Profiles.fpsLimitPresetIndex(clean.fpsLimit));
    updateFpsLimitPresentation();
    $('res').value = String(clean.res);
    $('sharpness').value = String(clean.sharpness);
    $('pictureMode').value = clean.pictureMode;
    for (const key of BOOLEAN_FIELDS) $(key).checked = clean[key];
    syncChoices();
    updateRateControlVisibility(clean.factor);
  }

  function renderProfileSelect() {
    const current = document.createElement('option');
    current.value = '';
    current.textContent = t('current_settings');
    const children = [current];
    const profiles = Profiles.profileList(store);
    if (profiles.length > 0) {
      const group = document.createElement('optgroup');
      group.label = t('my_profiles');
      for (const profile of profiles) {
        const option = document.createElement('option');
        option.value = profile.id;
        option.textContent = profile.name;
        group.append(option);
      }
      children.push(group);
    }
    $('profileSelect').replaceChildren(...children);
    $('profileSelect').value = selectedProfileId || '';
  }

  function renderRuntimeStatus(key, values, emphasized) {
    const strong = document.createElement('strong');
    strong.textContent = values[emphasized];
    const parts = t(key).split(new RegExp(`(\\{${emphasized}\\})`, 'g'));
    $('runtimeStatus').replaceChildren(...parts.map((part, index) => index % 2 ? strong.cloneNode(true) : (
      part.replaceAll('{name}', () => values.name)
    )));
  }

  function renderStatus() {
    if (profileStoreReadOnly) {
      $('runtimeStatus').textContent = t(profileStoreStatus);
      $('editState').textContent = unsupportedSchema === undefined
        ? t('profile_read_only') : t('profile_schema_unsupported', { version: unsupportedSchema });
      for (const id of STORE_MUTATION_CONTROLS) $(id).disabled = true;
      return;
    }
    const selected = currentProfile();
    const lastApplied = Profiles.getProfile(store, store.lastAppliedProfileId);
    const lastAppliedMatches = lastApplied
      && Profiles.settingsEqual(lastApplied.settings, runtimeSettings);

    if (lastApplied && !lastAppliedMatches) {
      renderRuntimeStatus('current_after', { current: t('current_settings'), name: lastApplied.name }, 'current');
    } else {
      renderRuntimeStatus('current_profile', { name: lastApplied?.name || t('custom_settings') }, 'name');
    }

    const hasDraftEdits = !Profiles.settingsEqual(draftSettings, draftOrigin);
    const pendingRuntimeChange = !Profiles.settingsEqual(draftSettings, runtimeSettings);
    const pendingProfileChange = !!selected
      && selected.id !== store.lastAppliedProfileId;
    const pendingApply = pendingRuntimeChange || pendingProfileChange;
    const selectedModified = selected
      ? !Profiles.settingsEqual(draftSettings, selected.settings)
      : pendingRuntimeChange;

    if (externalChangeNotice) {
      $('editState').textContent = t('changed_another_tab');
    } else if (lastMessage) {
      $('editState').textContent = t(lastMessage);
    } else if (hasDraftEdits || selectedModified) {
      $('editState').textContent = t('unsaved_preview');
    } else if (selected && pendingApply) {
      $('editState').textContent = t('previewing', { name: selected.name });
    } else {
      $('editState').textContent = t('up_to_date');
    }

    $('resetDraft').disabled = !externalChangeNotice && !hasDraftEdits;
    $('applySettings').disabled = !pendingApply;
    $('saveProfile').hidden = !selected;
    $('saveProfile').disabled = !selected || !selectedModified;
    for (const id of ['renameProfile', 'deleteProfile', 'duplicateProfile']) {
      $(id).disabled = !selected;
      $(id).hidden = !selected;
    }
  }

  function renderAll({ write = false } = {}) {
    if (write) writeForm(draftSettings);
    renderProfileSelect();
    renderStatus();
  }

  function selectProfile(id) {
    if (!id) loadCurrentDraft();
    else loadProfileDraft(id);
    externalChangeNotice = false;
    lastMessage = '';
    renderAll({ write: true });
  }

  async function persistStore(nextStore = store) {
    if (profileStoreReadOnly) throw new Profiles.ProfileStoreError('Profiles are read-only');
    await chrome.storage.local.set({ [Profiles.STORE_KEY]: nextStore });
    store = nextStore;
    storageSnapshot[Profiles.STORE_KEY] = nextStore;
    return nextStore;
  }

  async function applyDraft() {
    if (profileStoreReadOnly) return;
    if (!validateVisibleRateControl()) return;
    const selected = currentProfile();
    const pendingRuntimeChange = !Profiles.settingsEqual(draftSettings, runtimeSettings);
    const pendingProfileChange = !!selected
      && selected.id !== store.lastAppliedProfileId;
    if (!pendingRuntimeChange && !pendingProfileChange) return;
    try {
      const nextStore = Profiles.setLastAppliedProfile(store, selected?.id || null);
      const payload = Profiles.toStoragePayload(nextStore, draftSettings);
      await chrome.storage.local.set(payload);
      Object.assign(storageSnapshot, payload);
      runtimeSettings = settingsCopy(draftSettings);
      store = payload[Profiles.STORE_KEY];
      draftOrigin = selected
        ? settingsCopy(selected.settings)
        : settingsCopy(draftSettings);
      draftSource = selected ? 'profile' : 'current';
      externalChangeNotice = false;
      lastMessage = 'settings_applied';
      renderAll();
    } catch {
      lastMessage = 'apply_failed';
      renderStatus();
    }
  }

  function updateProfileDialogLabels() {
    const { title, action } = {
      new: { title: 'new_profile', action: 'create' },
      duplicate: { title: 'duplicate_profile', action: 'create_copy' },
      rename: { title: 'rename_profile', action: 'rename' },
    }[dialogMode];
    $('profileDialogTitle').textContent = t(title);
    $('confirmProfile').textContent = t(action);
  }

  function openProfileDialog(mode) {
    if (profileStoreReadOnly) return;
    if ((mode === 'new' || mode === 'duplicate') && !validateVisibleRateControl()) return;
    const selected = currentProfile();
    dialogMode = mode;
    const names = {
      new: t('custom_profile'),
      duplicate: selected ? t('profile_copy_named', { name: selected.name }) : t('profile_copy'),
      rename: selected?.name || t('custom_profile'),
    };
    updateProfileDialogLabels();
    $('profileName').value = names[mode].slice(0, 48);
    $('profileName').setCustomValidity('');
    $('profileDialog').showModal();
    $('profileName').select();
  }

  async function submitProfileDialog(event) {
    event.preventDefault();
    if (profileStoreReadOnly) return;
    const name = Profiles.normalizeName($('profileName').value, '');
    if (!name) {
      $('profileName').setCustomValidity(t('enter_profile_name'));
      $('profileName').reportValidity();
      return;
    }
    const selected = currentProfile();
    try {
      let nextStore;
      let createdProfileId = null;
      if (dialogMode === 'rename') {
        if (!selected) throw new TypeError('profile does not exist');
        nextStore = Profiles.renameProfile(store, selected.id, name);
      } else {
        const id = createProfileId();
        nextStore = Profiles.upsertProfile(store, {
          id,
          name,
          description: '',
          settings: draftSettings,
        });
        createdProfileId = id;
      }
      await persistStore(nextStore);
      if (createdProfileId) {
        selectedProfileId = createdProfileId;
        draftOrigin = settingsCopy(draftSettings);
        draftSource = 'profile';
      }
      $('profileDialog').close();
      lastMessage = dialogMode === 'rename' ? 'profile_renamed' : 'profile_saved';
      externalChangeNotice = false;
      renderAll({ write: true });
      $('profileSelect').focus();
    } catch (error) {
      console.warn('Framegen profile save:', error);
      $('profileName').setCustomValidity(t('profile_save_failed'));
      $('profileName').reportValidity();
    }
  }

  async function saveSelectedProfile() {
    if (profileStoreReadOnly) return;
    if (!validateVisibleRateControl()) return;
    const selected = currentProfile();
    if (!selected) {
      openProfileDialog('new');
      return;
    }
    try {
      const nextStore = Profiles.upsertProfile(store, {
        ...selected,
        settings: draftSettings,
      });
      await persistStore(nextStore);
      draftOrigin = settingsCopy(draftSettings);
      draftSource = 'profile';
      externalChangeNotice = false;
      lastMessage = 'profile_saved';
      renderAll();
    } catch {
      lastMessage = 'profile_save_failed';
      renderStatus();
    }
  }

  function openDeleteDialog() {
    if (profileStoreReadOnly) return;
    const selected = currentProfile();
    if (!selected) return;
    $('deleteDialogCopy').textContent = t('delete_profile_named', { name: selected.name });
    $('deleteDialog').showModal();
    $('cancelDelete').focus();
  }

  async function deleteSelectedProfile(event) {
    event.preventDefault();
    if (profileStoreReadOnly) return;
    const selected = currentProfile();
    if (!selected) return;
    try {
      const nextStore = Profiles.deleteProfile(store, selected.id);
      await persistStore(nextStore);
      loadCurrentDraft();
      $('deleteDialog').close();
      externalChangeNotice = false;
      lastMessage = 'profile_deleted';
      renderAll({ write: true });
      $('profileSelect').focus();
    } catch {
      lastMessage = 'profile_delete_failed';
      renderStatus();
    }
  }

  function resetDraft() {
    if (selectedProfileId) loadProfileDraft(selectedProfileId);
    else loadCurrentDraft();
    externalChangeNotice = false;
    lastMessage = '';
    renderAll({ write: true });
  }

  function storageChanged(changes, areaName) {
    if (areaName !== 'local') return;
    const storeChanged = Profiles.STORE_KEY in changes;
    const settingsChanged = Profiles.SETTINGS_KEYS.some(key => key in changes);
    if (!storeChanged && !settingsChanged) return;
    for (const [key, change] of Object.entries(changes)) {
      if (typeof change.newValue === 'undefined') delete storageSnapshot[key];
      else storageSnapshot[key] = change.newValue;
    }

    const hadLocalEdits = !Profiles.settingsEqual(draftSettings, draftOrigin);
    runtimeSettings = Profiles.pickFlatSettings(storageSnapshot);
    if (storeChanged) {
      try {
        const loaded = Profiles.loadStore(
          storageSnapshot[Profiles.STORE_KEY],
          runtimeSettings,
        );
        store = loaded.store;
        leaveProfileStoreReadOnly();
        if (loaded.needsWrite) {
          const migratedStore = loaded.store;
          chrome.storage.local.set({ [Profiles.STORE_KEY]: migratedStore })
            .catch(error => enterProfileStoreReadOnly(error, 'profile_update_failed'));
        }
      } catch (error) {
        enterProfileStoreReadOnly(error);
        return;
      }
    }

    let writeFormAfterChange = false;
    const selected = currentProfile();
    if (selectedProfileId && !selected) {
      loadCurrentDraft();
      externalChangeNotice = false;
      writeFormAfterChange = true;
    } else if (storeChanged && selected && draftSource === 'profile'
      && !Profiles.settingsEqual(selected.settings, draftOrigin)) {
      if (hadLocalEdits) {
        externalChangeNotice = true;
      } else {
        loadProfileDraft(selected.id);
        externalChangeNotice = false;
        writeFormAfterChange = true;
      }
    }

    if (settingsChanged) {
      if (selectedProfileId === null && !hadLocalEdits) {
        loadCurrentDraft();
        externalChangeNotice = false;
        writeFormAfterChange = true;
      } else if (!Profiles.settingsEqual(draftSettings, runtimeSettings)) {
        externalChangeNotice = true;
      }
    }
    lastMessage = '';
    renderAll({ write: writeFormAfterChange });
  }

  function bindEvents() {
    $('fpsLimit').addEventListener('input', updateFpsLimitPresentation);
    $('settingsForm').addEventListener('input', () => {
      $('targetFps').setCustomValidity('');
      readForm();
      externalChangeNotice = false;
      lastMessage = '';
      renderStatus();
    });
    $('profileSelect').addEventListener('change', event => selectProfile(event.target.value));
    $('newProfile').addEventListener('click', () => openProfileDialog('new'));
    $('duplicateProfile').addEventListener('click', () => openProfileDialog('duplicate'));
    $('renameProfile').addEventListener('click', () => openProfileDialog('rename'));
    $('deleteProfile').addEventListener('click', openDeleteDialog);
    $('saveProfile').addEventListener('click', saveSelectedProfile);
    $('resetDraft').addEventListener('click', resetDraft);
    $('applySettings').addEventListener('click', applyDraft);
    $('profileDialogForm').addEventListener('submit', submitProfileDialog);
    $('profileName').addEventListener('input', event => event.target.setCustomValidity(''));
    $('confirmDelete').addEventListener('click', deleteSelectedProfile);
    for (const closeButton of document.querySelectorAll('[data-close-dialog]')) {
      closeButton.addEventListener('click', () => closeButton.closest('dialog').close());
    }
    chrome.storage.onChanged.addListener(storageChanged);
  }

  function localize() {
    document.documentElement.lang = getLanguage().replaceAll('_', '-');
    translate(document);
    for (const group of document.querySelectorAll('[data-choice]')) {
      const options = $(group.dataset.choice).options;
      for (const [index, button] of [...group.children].entries()) {
        (button.querySelector('.look-name') || button).textContent = options[index].textContent;
      }
    }
    if (draftSettings) updateFpsLimitPresentation();
    if ($('profileDialog').open) updateProfileDialogLabels();
    if ($('deleteDialog').open) {
      $('deleteDialogCopy').textContent = t('delete_profile_named', { name: currentProfile()?.name || '' });
    }
    if (store || profileStoreReadOnly) renderStatus();
    if (store) renderProfileSelect();
  }

  async function initialize() {
    $('version').textContent = 'v' + chrome.runtime.getManifest().version;
    await initLanguage();
    bindLanguageSelect($('interfaceLanguage'), $('languageError'));
    localize();
    onLanguageChange(localize);
    try {
      storageSnapshot = await chrome.storage.local.get(null);
      runtimeSettings = Profiles.pickFlatSettings(storageSnapshot);
      const loaded = Profiles.loadStore(storageSnapshot[Profiles.STORE_KEY], runtimeSettings);
      store = loaded.store;
      loadCurrentDraft();
      renderFpsLimitScale();
      buildChoices();
      watchSections();
      writeForm(draftSettings);
      bindEvents();
      renderAll();
      if (loaded.needsWrite) await persistStore();
    } catch (error) {
      enterProfileStoreReadOnly(error, 'settings_unavailable');
    }
  }

  initialize();
})();
