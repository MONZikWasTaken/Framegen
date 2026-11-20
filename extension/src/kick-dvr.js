// Kick: keep the place in a rewound live stream. Kick's player rebuilds its media source when the quality changes
// (and on its own quality switches), and for a rewound live stream it then plays the recording from 0: the viewer
// lands hours back, at the start of the stream. The player never does this to a recording on purpose, so when a
// rebuilt source of the same recording starts at 0, the position from before the rebuild is put back.

const NEAR_START_S = 5;       // where a rebuilt source starts when the player lost the position
const MIN_POSITION_S = 30;    // below this there is nothing worth restoring
const MAX_GROWTH_S = 900;     // a live recording grows while it is rebuilt; another video does not match
const RESTORE_WINDOW_MS = 10_000;

/** The position to put back once a rebuilt source plays, or null. `before` is the state when the source was torn
 * down, `after` the state of the new one: { url, duration, position }. */
export function positionToRestore(before, after) {
  if (!before || !after || before.url !== after.url) return null;
  if (!Number.isFinite(before.duration) || !Number.isFinite(after.duration)) return null;   // live on either side
  if (before.position < MIN_POSITION_S || before.position > before.duration - 2) return null; // the start, or the end
  if (after.position >= NEAR_START_S) return null;                                           // the player kept the place
  const growth = after.duration - before.duration;
  if (growth < -1 || growth > MAX_GROWTH_S) return null;                                     // not the same recording
  return before.position;
}

const clock = s => {
  const t = Math.max(0, Math.floor(s)), h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), sec = t % 60;
  return (h ? h + ':' + String(m).padStart(2, '0') : String(m)) + ':' + String(sec).padStart(2, '0');
};

function toast(text) {
  const el = document.createElement('div');
  el.textContent = text;
  el.style.cssText = 'position:fixed; left:50%; top:72px; transform:translateX(-50%); z-index:2147483646; pointer-events:none;'
    + 'background:rgba(60,45,10,.92); color:#ffd88a; border:1px solid rgba(255,200,90,.35); border-radius:12px;'
    + 'padding:8px 14px; font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4); transition:opacity .4s;';
  document.documentElement.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; }, 4500);
  setTimeout(() => el.remove(), 5000);
}

export function watchKickDvr({ log = () => {} } = {}) {
  let last = null;      // the playing recording: { video, url, duration, position }
  let pending = null;   // `last` at the moment its source was torn down, with an expiry
  const main = v => v instanceof HTMLVideoElement && v.offsetWidth > 100;
  const snapshot = video => ({ video, url: location.href, duration: video.duration, position: video.currentTime });
  const discardStale = video => {
    if (pending && (pending.video !== video || pending.url !== location.href || performance.now() >= pending.expires)) {
      pending = null;
    }
  };
  const note = e => {
    const v = e.target;
    if (!main(v)) return;
    discardStale(v);
    if (!pending) last = Number.isFinite(v.duration) && v.duration > 0 ? snapshot(v) : null;
  };
  const settle = e => {
    const v = e.target;
    if (!main(v)) return;
    discardStale(v);
    if (!pending || v.readyState < 1) return;
    const to = positionToRestore(pending, snapshot(v));
    pending = null;
    if (to === null) return;
    v.currentTime = to;
    log('kick: the player restarted the recording at 0, back to', to.toFixed(1));
    toast(`Kick restarted the stream from the beginning: back at ${clock(to)}`);
  };
  for (const type of ['timeupdate', 'seeking']) document.addEventListener(type, note, true);
  document.addEventListener('emptied', e => {
    if (e.target !== last?.video) return;
    if (last.url === location.href) pending = { ...last, expires: performance.now() + RESTORE_WINDOW_MS };
    last = null;
  }, true);
  for (const type of ['seeked', 'playing']) document.addEventListener(type, settle, true);
}
