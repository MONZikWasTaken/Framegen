(function framegenOptions() {
  'use strict';

  const Profiles = globalThis.FramegenProfiles;
  if (!Profiles) throw new Error('Framegen profile store is unavailable');

  const $ = (id) => document.getElementById(id);
  const BOOLEAN_FIELDS = Object.freeze([
    'fg', 'sr', 'hdr', 'anime', 'guard', 'hoverReveal',
    'showFps', 'showWatermark', 'showWarnings', 'compare', 'debug',
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
  let profileStoreError = '';

  function enterProfileStoreReadOnly(error, status = 'Profile storage is read-only') {
    profileStoreReadOnly = true;
    profileStoreError = error?.message || 'Profiles are read-only';
    $('runtimeStatus').textContent = status;
    $('editState').textContent = profileStoreError;
    for (const id of STORE_MUTATION_CONTROLS) $(id).disabled = true;
    for (const id of ['profileDialog', 'deleteDialog']) {
      const dialog = $(id);
      if (dialog.open) dialog.close();
    }
  }

  function leaveProfileStoreReadOnly() {
    profileStoreReadOnly = false;
    profileStoreError = '';
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
    const label = value === null ? 'Unlimited' : `${value} FPS`;
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
  }

  function validateVisibleRateControl() {
    if (draftSettings.factor !== 'target') return true;
    if ($('targetFps').checkValidity()) return true;
    $('targetFps').reportValidity();
    lastMessage = 'Enter a Custom FPS from 2 to 1000';
    renderStatus();
    return false;
  }

  function readForm() {
    const candidate = {
      factor: $('factor').value,
      targetFps: Number($('targetFps').value),
      fpsLimit: fpsLimitFromSlider(),
      res: Number($('res').value),
      model: $('model').value,
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
    $('model').value = clean.model;
    for (const key of BOOLEAN_FIELDS) $(key).checked = clean[key];
    updateRateControlVisibility(clean.factor);
  }

  function renderProfileSelect() {
    const current = document.createElement('option');
    current.value = '';
    current.textContent = 'Current settings';
    const children = [current];
    const profiles = Profiles.profileList(store);
    if (profiles.length > 0) {
      const group = document.createElement('optgroup');
      group.label = 'My profiles';
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

  function renderStatus() {
    if (profileStoreReadOnly) {
      $('runtimeStatus').textContent = 'Profile storage is read-only';
      $('editState').textContent = profileStoreError || 'Profiles are read-only';
      for (const id of STORE_MUTATION_CONTROLS) $(id).disabled = true;
      return;
    }
    const selected = currentProfile();
    const lastApplied = Profiles.getProfile(store, store.lastAppliedProfileId);
    const lastAppliedMatches = lastApplied
      && Profiles.settingsEqual(lastApplied.settings, runtimeSettings);

    if (lastAppliedMatches) {
      $('runtimeStatus').innerHTML = `Current: <strong>${escapeHtml(lastApplied.name)}</strong>`;
    } else if (lastApplied) {
      $('runtimeStatus').innerHTML = `<strong>Current settings</strong> · changed after ${escapeHtml(lastApplied.name)}`;
    } else {
      $('runtimeStatus').innerHTML = 'Current: <strong>Custom settings</strong>';
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
      $('editState').textContent = 'Changed in another tab';
    } else if (lastMessage) {
      $('editState').textContent = lastMessage;
    } else if (hasDraftEdits || selectedModified) {
      $('editState').textContent = 'Unsaved preview';
    } else if (selected && pendingApply) {
      $('editState').textContent = `Previewing ${selected.name}`;
    } else {
      $('editState').textContent = 'Up to date';
    }

    $('resetDraft').disabled = !externalChangeNotice && !hasDraftEdits;
    $('applySettings').disabled = !pendingApply;
    $('saveProfile').hidden = !selected;
    $('saveProfile').disabled = !selected || !selectedModified;
    $('saveProfile').textContent = 'Save profile';
    for (const id of ['renameProfile', 'deleteProfile', 'duplicateProfile']) {
      $(id).disabled = !selected;
      $(id).hidden = !selected;
    }
  }

  function escapeHtml(value) {
    const span = document.createElement('span');
    span.textContent = String(value);
    return span.innerHTML;
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
      lastMessage = 'Settings applied';
      renderAll();
    } catch {
      lastMessage = 'Could not apply settings';
      renderStatus();
    }
  }

  function openProfileDialog(mode) {
    if (profileStoreReadOnly) return;
    if ((mode === 'new' || mode === 'duplicate') && !validateVisibleRateControl()) return;
    const selected = currentProfile();
    dialogMode = mode;
    const titles = {
      new: ['New profile', 'Create', 'Custom profile'],
      duplicate: ['Duplicate profile', 'Create copy', selected ? `${selected.name} copy` : 'Profile copy'],
      rename: ['Rename profile', 'Rename', selected?.name || 'Custom profile'],
    };
    const [title, action, value] = titles[mode];
    $('profileDialogTitle').textContent = title;
    $('confirmProfile').textContent = action;
    $('profileName').value = value.slice(0, 48);
    $('profileName').setCustomValidity('');
    $('profileDialog').showModal();
    $('profileName').select();
  }

  async function submitProfileDialog(event) {
    event.preventDefault();
    if (profileStoreReadOnly) return;
    const name = Profiles.normalizeName($('profileName').value, '');
    if (!name) {
      $('profileName').setCustomValidity('Enter a profile name.');
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
      lastMessage = dialogMode === 'rename' ? 'Profile renamed' : 'Profile saved';
      externalChangeNotice = false;
      renderAll({ write: true });
      $('profileSelect').focus();
    } catch (error) {
      $('profileName').setCustomValidity(error?.message || 'Could not save this profile.');
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
      lastMessage = 'Profile saved';
      renderAll();
    } catch {
      lastMessage = 'Could not save profile';
      renderStatus();
    }
  }

  function openDeleteDialog() {
    if (profileStoreReadOnly) return;
    const selected = currentProfile();
    if (!selected) return;
    $('deleteDialogCopy').textContent = `“${selected.name}” will be removed. Current video settings will stay unchanged.`;
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
      lastMessage = 'Profile deleted';
      renderAll({ write: true });
      $('profileSelect').focus();
    } catch {
      lastMessage = 'Could not delete profile';
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
    let settingsChanged = false;
    const storeChanged = Profiles.STORE_KEY in changes;
    for (const [key, change] of Object.entries(changes)) {
      if (typeof change.newValue === 'undefined') delete storageSnapshot[key];
      else storageSnapshot[key] = change.newValue;
      if (Profiles.SETTINGS_KEYS.includes(key)) settingsChanged = true;
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
            .catch(error => enterProfileStoreReadOnly(error, 'Could not update profiles'));
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
    $('confirmDelete').addEventListener('click', deleteSelectedProfile);
    for (const closeButton of document.querySelectorAll('[data-close-dialog]')) {
      closeButton.addEventListener('click', () => closeButton.closest('dialog').close());
    }
    chrome.storage.onChanged.addListener(storageChanged);
  }

  async function initialize() {
    try {
      storageSnapshot = await chrome.storage.local.get(null);
      runtimeSettings = Profiles.pickFlatSettings(storageSnapshot);
      const loaded = Profiles.loadStore(storageSnapshot[Profiles.STORE_KEY], runtimeSettings);
      store = loaded.store;
      loadCurrentDraft();
      renderFpsLimitScale();
      writeForm(draftSettings);
      bindEvents();
      renderAll();
      if (loaded.needsWrite) await persistStore();
    } catch (error) {
      enterProfileStoreReadOnly(error, 'Settings unavailable');
    }
  }

  initialize();
})();
