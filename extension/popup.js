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
    state.textContent = 'off';
    state.className = '';
    stat.textContent = '';
    sys.textContent = '';
    return;
  }
  $('ver').textContent = 'v' + s.version;
  dot.classList.toggle('on', s.running);
  state.textContent = s.running ? 'running' : 'off';
  state.className = s.running ? 'on' : '';
  stat.textContent = s.running
    ? `out: ${s.fps} fps · ×${s.factor}\n`
      + `rate: ${s.rateState}\n`
      + `mid: ${s.midMs} ms @ ${s.res}p · drops: ${s.drops}\n`
      + `engine: ${s.engine || '-'}`
    : (s.blocked || '');
  sys.textContent = `GPU: ${s.gpu || '-'}${s.integrated ? '\n⚠ INTEGRATED - assign the browser to the discrete GPU\n(Settings → Display → Graphics)' : ''}\nf16: ${s.f16 ? 'yes' : 'no'}`;
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
