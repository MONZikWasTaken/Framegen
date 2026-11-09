// Our own control bar on top of the canvas: native controls render inside the video element and can never show above
// it, so instead of revealing the raw video we drive the <video> ourselves (play, seek, volume, fullscreen) as regular
// DOM above everything. Interpolation is never interrupted.
import { svgIcon } from './icons.js';

const formatTime = (s) => {
  s = Math.max(0, Math.floor(s || 0));
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  return (h ? h + ':' + String(m % 60).padStart(2, '0') : m) + ':' + String(s % 60).padStart(2, '0');
};

const rangeFill = (el, p, color) => {
  el.style.background = `linear-gradient(to right, ${color} ${p}%, rgba(255,255,255,.22) ${p}%)`;
};

const cooldown = (ms) => { let t = 0; return () => {
  const n = performance.now(); if (n - t < ms) return false; t = n; return true; }; };

export class ControlBar {
  // `video()` returns the video the bar drives; `onReveal(ms)` keeps the UI shown while the pointer is on the bar.
  constructor({ video, onReveal, onFlash, log }) {
    this.video = video;
    this.onReveal = onReveal;
    this.onFlash = onFlash;
    this.log = log;
    this.el = null;
    this.els = null;
    this.height = 0;
    this.seeking = false;
    this.last = { play: '', mute: '', vol: -1, cur: '', dur: '', seek: '', fill: -1 };
    this.fullscreenByUs = false;
    this.savedStyle = '';
    document.addEventListener('fullscreenchange', () => this.onFullscreenChange());
  }

  ensure() {
    if (this.el) return this.el;
    const bar = document.createElement('div');
    bar.style.cssText = 'position:fixed; z-index:2147483646; display:none; align-items:center; gap:10px;'
      + 'background:rgba(16,17,20,.88);'
      + 'border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:8px 14px;'
      + 'color:#fff; font:11px system-ui; box-sizing:border-box; user-select:none;'
      + 'box-shadow:0 4px 20px rgba(0,0,0,.4); opacity:0; transform:translateY(8px);'
      + 'transition:opacity .18s, transform .18s; pointer-events:none;';
    bar.innerHTML = `
      <button id="fcPlay" class="fc-btn">${svgIcon('play', 19)}</button>
      <span id="fcCur" style="min-width:34px; text-align:right">0:00</span>
      <input id="fcSeek" class="fc-range" type="range" min="0" max="1000" value="0" style="flex:1">
      <span id="fcDur" style="min-width:34px; color:rgba(255,255,255,.55)">0:00</span>
      <button id="fcMute" class="fc-btn">${svgIcon('vol')}</button>
      <input id="fcVol" class="fc-range" type="range" min="0" max="100" value="100" style="width:60px">
      <button id="fcFull" class="fc-btn">${svgIcon('full')}</button>`;
    document.body.appendChild(bar);
    const q = (id) => bar.querySelector(id);
    // per-button cooldowns: hammering the buttons must never wedge the player
    const gPlay = cooldown(180), gMute = cooldown(120), gFull = cooldown(400);
    q('#fcPlay').onclick = () => { if (gPlay()) this.togglePlay(); };
    q('#fcMute').onclick = () => { const v = this.video(); if (v && gMute()) { v.muted = !v.muted; this.update(); } };
    q('#fcVol').oninput = (e) => { const v = this.video(); if (v) { v.volume = e.target.value / 100; v.muted = false; } };
    q('#fcSeek').addEventListener('pointerdown', () => { this.seeking = true; });
    q('#fcSeek').addEventListener('pointerup', () => { this.seeking = false; });
    q('#fcSeek').oninput = (e) => {
      const v = this.video();
      if (v && v.duration) v.currentTime = e.target.value / 1000 * v.duration;
    };
    q('#fcFull').onclick = () => { if (gFull()) this.toggleFullscreen(); };
    bar.addEventListener('mousemove', () => this.onReveal(2000), { passive: true });
    this.el = bar;
    this.els = { play: q('#fcPlay'), mute: q('#fcMute'), vol: q('#fcVol'), cur: q('#fcCur'), dur: q('#fcDur'), seek: q('#fcSeek') };
    return bar;
  }

  togglePlay() {
    const v = this.video();
    if (!v) return;
    if (v.paused) v.play().catch(() => {}); else v.pause();
    this.onFlash(svgIcon(v.paused ? 'pause' : 'play', 30));
    this.update();
  }

  // Fullscreen the parent so the canvas comes along, and stretch the video to fill the screen: the container alone
  // keeps the video at its layout size, which looks like fullscreen not working.
  toggleFullscreen() {
    const v = this.video();
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (v) {
      this.fullscreenByUs = true;
      (v.parentElement || v).requestFullscreen().catch(e => { this.fullscreenByUs = false; this.log('fullscreen', e); });
    }
  }

  onFullscreenChange() {
    const v = this.video();
    if (!v || !this.fullscreenByUs) return;
    if (document.fullscreenElement) {
      this.savedStyle = v.style.cssText;
      v.style.cssText += ';position:fixed;left:0;top:0;width:100vw;height:100vh;'
        + 'max-width:none;max-height:none;object-fit:contain;background:#000;z-index:1;';
    } else {
      this.fullscreenByUs = false;
      v.style.cssText = this.savedStyle;
    }
  }

  mount(host) {
    if (this.el && this.el.parentElement !== host) host.appendChild(this.el);
  }

  layout(rect, show, now = performance.now()) {
    const bar = this.ensure();
    if (show) this.lastShown = now;
    // Out of the layout once it has faded: a fixed layer over the canvas costs the compositor even at opacity 0. In
    // Firefox it stalled the display loop 50 to 100 ms two or three times a second, which read as drops, and Auto
    // sat at x1 below the video's own rate.
    if (!show && this.height && now - (this.lastShown ?? -Infinity) > 250) { this.hide(); return; }
    const m = Math.max(10, Math.min(16, rect.width * 0.02));
    const wasOut = bar.style.display === 'none';
    bar.style.display = 'flex';
    if (!this.height) this.height = bar.offsetHeight;
    bar.style.left = (rect.left + m) + 'px';
    bar.style.width = (rect.width - 2 * m) + 'px';
    bar.style.top = (rect.bottom - this.height - m) + 'px';
    // back in the layout at opacity 0 for one tick first, so the fade-in has something to start from
    const visible = show && !wasOut;
    bar.style.opacity = visible ? '1' : '0';
    bar.style.transform = visible ? 'translateY(0)' : 'translateY(8px)';
    bar.style.pointerEvents = visible ? 'auto' : 'none';
  }

  hide() {
    if (this.el) this.el.style.display = 'none';
  }

  // Element refs and last-written values are cached: this runs every UI tick, and unconditional style writes force a
  // style recalc for nothing.
  update() {
    const v = this.video();
    if (!this.el || this.el.style.display === 'none' || !v) return;
    const e = this.els, last = this.last;
    const playIcon = v.paused ? 'play' : 'pause';
    if (playIcon !== last.play) { last.play = playIcon; e.play.innerHTML = svgIcon(playIcon, 19); }
    const muteIcon = (v.muted || v.volume === 0) ? 'volX' : 'vol';
    if (muteIcon !== last.mute) { last.mute = muteIcon; e.mute.innerHTML = svgIcon(muteIcon); }
    const vol = Math.round((v.muted ? 0 : v.volume) * 100);
    if (vol !== last.vol) { last.vol = vol; e.vol.value = String(vol); rangeFill(e.vol, vol, '#fff'); }
    const d = v.duration || 0, c = v.currentTime || 0;
    const cur = formatTime(c), dur = formatTime(d);
    if (cur !== last.cur) { last.cur = cur; e.cur.textContent = cur; }
    if (dur !== last.dur) { last.dur = dur; e.dur.textContent = dur; }
    const p = d ? c / d * 100 : 0;
    if (!this.seeking && d) {
      const sv = String(Math.round(p * 10));
      if (sv !== last.seek) { last.seek = sv; e.seek.value = sv; }
    }
    const fill = Math.round(p * 10);
    if (fill !== last.fill) { last.fill = fill; rangeFill(e.seek, p, '#19c37d'); }
  }
}

// Compare mode: a draggable divider; the original shows left of it, generated frames right of it.
export class CompareSplit {
  constructor({ video }) {
    this.video = video;
    this.el = null;
    this.x = 0.5;
  }

  ensure() {
    if (this.el) return this.el;
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed; z-index:2147483645; width:18px; margin-left:-9px;'
      + 'cursor:ew-resize; touch-action:none; display:none;';
    el.innerHTML = `
      <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; margin-left:-1px;
        background:rgba(255,255,255,.85); box-shadow:0 0 10px rgba(0,0,0,.7)"></div>
      <div style="position:absolute; right:14px; bottom:10px; color:#fff; font:10px system-ui;
        background:rgba(15,15,15,.6); border-radius:6px; padding:2px 6px; white-space:nowrap">orig.</div>
      <div style="position:absolute; left:14px; bottom:10px; color:#fff; font:10px system-ui;
        background:rgba(25,150,100,.65); border-radius:6px; padding:2px 6px">FG</div>`;
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      const move = (ev) => {
        const v = this.video();
        if (!v) return;
        const r = v.getBoundingClientRect();
        this.x = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width));
      };
      move(e);
      el.onpointermove = move;
      el.onpointerup = () => { el.onpointermove = null; el.onpointerup = null; };
      e.preventDefault();
    });
    (document.fullscreenElement || document.body).appendChild(el);
    this.el = el;
    return el;
  }

  mount(host) {
    if (this.el && this.el.parentElement !== host) host.appendChild(this.el);
  }

  layout(rect, visible) {
    if (!visible) { if (this.el) this.el.style.display = 'none'; return; }
    const el = this.ensure();
    el.style.display = 'block';
    el.style.left = (rect.left + this.x * rect.width) + 'px';
    el.style.top = rect.top + 'px';
    el.style.height = rect.height + 'px';
  }

  hide() {
    if (this.el) this.el.style.display = 'none';
  }
}
