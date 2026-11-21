// The popup reports the active tab's state. Playback controls live in the player.
import { getLanguage, initLanguage, onLanguageChange, t, translate } from './src/i18n.js';

const $ = id => document.getElementById(id);
let lastStatus;

function writeText(id, text) {
  const element = $(id);
  if (element.textContent !== text) element.textContent = text;
}

function readout(status) {
  if (status === undefined) return ['idle', 'popup_checking'];
  if (!status) return ['idle', 'popup_tab_unavailable', 'popup_no_video_hint'];
  if (status.blocked) return ['warning', 'popup_blocked', 'blocked_wasm'];
  if (status.running) return ['ok', 'popup_running'];
  if (!status.hasAdapter) return ['warning', 'popup_gpu_missing', 'popup_gpu_missing_hint'];
  return ['ok', 'popup_ready', status.hasVideo ? 'hover_video' : 'popup_no_video_hint'];
}

function render(status) {
  lastStatus = status;
  const [state, heading, hint] = readout(status);
  $('playerStatus').dataset.state = state;
  writeText('state', t(heading));
  $('stateHint').hidden = !hint;
  if (hint) writeText('stateHint', t(hint));

  $('details').hidden = !status?.hasAdapter;
  $('gpuNotice').hidden = !status?.hasAdapter || !status.integrated;
  if (status?.hasAdapter) {
    writeText('gpu', status.gpu && status.gpu !== '-' ? status.gpu : t('popup_gpu_name_unavailable'));
  }
}

function showHelp(open) {
  $('status').hidden = open;
  $('help').hidden = !open;
  $('openHelp').hidden = open;
  $('closeHelp').hidden = !open;
  $('tabHelp').setAttribute('aria-expanded', String(open));
  if (open) $('help').scrollTop = 0;
}

$('tabHelp').onclick = () => showHelp($('help').hidden);
$('showGpuHelp').onclick = () => {
  showHelp(true);
  $('gpuHelp').open = true;
  $('gpuHelp').querySelector('summary').focus();
};
$('fullSettings').onclick = async () => {
  try {
    await chrome.runtime.openOptionsPage();
    window.close();
  } catch { /* keep the popup open if the browser rejects the request */ }
};

async function refreshStatus(tabId) {
  let status = null;
  try { status = await chrome.tabs.sendMessage(tabId, { type: 'fcStatus' }) ?? null; }
  catch { /* this tab has no content script */ }
  render(status);
  setTimeout(() => refreshStatus(tabId), 1000);
}

(async () => {
  writeText('ver', 'v' + chrome.runtime.getManifest().version);
  await initLanguage();
  const localize = () => {
    document.documentElement.lang = getLanguage().replaceAll('_', '-');
    translate(document);
    render(lastStatus);
  };
  localize();
  onLanguageChange(localize);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id == null) render(null);
  else refreshStatus(tab.id);
})();
