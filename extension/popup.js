// Framegen toolbar popup: read-only status for the active tab. The controls live in the player.
import { getLanguage, initLanguage, onLanguageChange, t, translate } from './src/i18n.js';

const $ = (id) => document.getElementById(id);

let tabId = null;
let lastStatus = null;

async function askStatus() {
  if (tabId == null) return null;
  try {
    return await chrome.tabs.sendMessage(tabId, { type: 'fcStatus' });
  } catch { return null; }
}

function render(s) {
  lastStatus = s;
  const dot = $('dot'), state = $('state'), stat = $('stat'), sys = $('sys');
  if (!s) {
    dot.classList.remove('on');
    state.textContent = t('off');
    state.className = '';
    stat.textContent = '';
    sys.textContent = '';
    return;
  }
  $('ver').textContent = 'v' + s.version;
  dot.classList.toggle('on', s.running);
  state.textContent = t(s.running ? 'running' : 'off');
  state.className = s.running ? 'on' : '';
  stat.textContent = s.running
    ? [
        t('popup_output', { fps: s.fps, factor: s.factor }),
        t('popup_rate', { rate: s.rateState }),
        t('popup_render', { time: s.midMs, resolution: s.res, drops: s.drops }),
        t('popup_engine', { engine: s.engine || '-' }),
      ].join('\n')
    : (s.blocked ? t('blocked_wasm') : '');
  sys.textContent = t('gpu_line', { gpu: s.gpu || t('unknown_gpu') })
    + (s.f16 ? '' : t('no_f16_suffix'))
    + (s.integrated ? '\n' + t('popup_integrated') : '');
  sys.className = s.integrated ? 'warn' : '';
}

const showTab = (help) => {
  $('status').style.display = help ? 'none' : 'block';
  $('help').style.display = help ? 'block' : 'none';
  $('tabStatus').classList.toggle('act', !help);
  $('tabHelp').classList.toggle('act', help);
};
$('tabStatus').onclick = () => showTab(false);
$('tabHelp').onclick = () => showTab(true);
$('fullSettings').onclick = async () => {
  try {
    await chrome.runtime.openOptionsPage();
    window.close();
  } catch { /* leave the popup open if the browser rejects the request */ }
};

(async () => {
  await initLanguage();
  const localize = () => {
    document.documentElement.lang = getLanguage().replaceAll('_', '-');
    translate(document);
    render(lastStatus);
  };
  localize();
  onLanguageChange(localize);
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = tab ? tab.id : null;
  const first = await askStatus();
  render(first);
  if (first && first.integrated) showTab(true);
  setInterval(async () => render(await askStatus()), 1000);
})();
