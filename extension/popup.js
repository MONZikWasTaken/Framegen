// Framegen toolbar popup: read-only status for the active tab. The controls live in the player.
const $ = (id) => document.getElementById(id);

let tabId = null;

async function askStatus() {
  if (tabId == null) return null;
  try {
    return await chrome.tabs.sendMessage(tabId, { type: 'fcStatus' });
  } catch { return null; }
}

function render(s) {
  const dot = $('dot'), state = $('state'), stat = $('stat'), sys = $('sys');
  if (!s) {
    dot.classList.remove('on');
    state.textContent = 'Off';
    state.className = '';
    stat.textContent = '';
    sys.textContent = '';
    return;
  }
  $('ver').textContent = 'v' + s.version;
  dot.classList.toggle('on', s.running);
  state.textContent = s.running ? 'Running' : 'Off';
  state.className = s.running ? 'on' : '';
  stat.textContent = s.running
    ? `Output   ${s.fps} fps · ×${s.factor}\n`
      + `Rate     ${s.rateState}\n`
      + `Render   ${s.midMs} ms at ${s.res}p · ${s.drops} dropped\n`
      + `Engine   ${s.engine || '-'}`
    : (s.blocked || '');
  sys.textContent = `GPU: ${s.gpu || '-'}${s.f16 ? '' : ' · no half-float support'}`
    + (s.integrated ? '\nIntegrated GPU in use. Assign the browser to the discrete one: Windows Settings → Display → Graphics.' : '');
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
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = tab ? tab.id : null;
  const first = await askStatus();
  render(first);
  if (first && first.integrated) showTab(true);
  setInterval(async () => render(await askStatus()), 1000);
})();
