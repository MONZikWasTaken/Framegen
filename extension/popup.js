// Framecast toolbar popup: READ-ONLY status/metadata for the active tab.
// All controls live in the in-player UI by design.
const $ = (id) => document.getElementById(id);

let tabId = null;

async function askStatus() {
  if (tabId == null) return null;
  try {
    return await chrome.tabs.sendMessage(tabId, { type: 'fcStatus' });
  } catch { return null; } // no content script (chrome:// pages, store, etc.)
}

function render(s) {
  const dot = $('dot'), state = $('state'), stat = $('stat'), sys = $('sys');
  if (!s) {
    dot.classList.remove('on');
    state.textContent = 'выключено';
    state.className = '';
    stat.textContent = '';
    sys.textContent = '';
    return;
  }
  $('ver').textContent = 'v' + s.version;
  dot.classList.toggle('on', s.running);
  state.textContent = s.running ? 'работает' : 'выключено';
  state.className = s.running ? 'on' : '';
  stat.textContent = s.running
    ? `выход: ${s.fps} fps · множитель ×${s.effN}${s.factor === 'auto' ? ' (авто)' : ''}\n`
      + `вставка: ${s.ms} ms @ ${s.res}p\n`
      + `дропы: ${s.drops} · модель: ${s.model}`
    : '';
  sys.textContent = `GPU: ${s.gpu || '—'}${s.integrated ? '\n⚠ ВСТРОЙКА — назначь Chrome дискретную видеокарту\n(Параметры → Дисплей → Графика)' : ''}\nf16: ${s.f16 ? 'да' : 'нет'}`;
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

(async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  tabId = tab ? tab.id : null;
  const first = await askStatus();
  render(first);
  if (first && first.integrated) showTab(true); // dual-GPU case: open the fix guide
  setInterval(async () => render(await askStatus()), 1000);
})();
