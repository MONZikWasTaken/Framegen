// The popup reports the active tab's state. Playback controls live in the player.
import { getLanguage, initLanguage, onLanguageChange, t, translate } from './src/i18n.js';

const $ = id => document.getElementById(id);
let lastStatus;

function writeText(id, text) {
  const element = $(id);
  if (element.textContent !== text) element.textContent = text;
}

function render(status) {
  lastStatus = status;
  const running = !!status?.running;
  const blocked = !running && !!status?.blocked;
  const ready = !running && !blocked && !!status?.hasVideo;
  const hasGpu = !!status?.gpu && status.gpu !== '-';
  const state = running ? 'running' : blocked ? 'blocked' : ready ? 'ready' : 'idle';
  $('statusCard').dataset.state = state;
  const heading = {
    running: 'popup_running', blocked: 'popup_blocked', ready: 'popup_ready',
    idle: status === undefined ? 'popup_checking' : 'popup_no_video',
  }[state];
  writeText('state', t(heading));
  const hint = blocked ? 'blocked_wasm' : ready ? 'hover_video' : status === undefined || running ? '' : 'popup_no_video_hint';
  $('stateHint').hidden = !hint;
  if (hint) writeText('stateHint', t(hint));

  $('metrics').hidden = !running;
  $('details').hidden = !running && !hasGpu;
  $('rateRow').hidden = !running;
  $('engineRow').hidden = !running;
  $('dropRow').hidden = !running;
  $('gpuRow').hidden = !hasGpu;
  $('gpuNotice').hidden = !status?.integrated;
  if (running) {
    writeText('fps', String(status.fps));
    writeText('factor', `×${status.factor}`);
    writeText('renderTime', t('popup_time_value', { time: status.midMs }));
    writeText('resolution', `${status.res}p`);
    writeText('rate', status.rateState);
    writeText('engine', status.engine || '–');
    writeText('drops', String(status.drops));
  }
  if (hasGpu) writeText('gpu', status.gpu + (status.f16 ? '' : t('no_f16_suffix')));
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
