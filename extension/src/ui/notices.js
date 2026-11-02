// The small plates over the video: the HUD line (fps or the debug readout), the watermark, the overload warning, the
// one-shot amber advisory, and the play/pause splash.
const Z_TOP = 2147483647;

export class Notices {
  constructor() {
    this.hud = document.createElement('div');
    this.hud.style.cssText = `position:fixed; left:0; top:0; z-index:${Z_TOP};`
      + 'color:#fff; font:11px/1.5 ui-monospace,monospace; background:rgba(0,0,0,.72);'
      + 'padding:3px 9px; white-space:normal; pointer-events:none; display:none;';
    this.watermark = document.createElement('div');
    this.watermark.style.cssText = `position:fixed; left:0; top:0; z-index:${Z_TOP - 2};`
      + 'color:#fff; font:600 12px system-ui; opacity:.75; pointer-events:none;'
      + 'text-shadow:0 1px 3px rgba(0,0,0,.8); display:none;';
    this.watermark.textContent = 'Framegen';
    this.warnEl = null;
    this.adviseEl = null;
    this.flashEl = null;
    this.warnSince = 0;
    this.warnText = '';
    this.adviseUntil = 0;
    this.critical = false;
  }

  elements() {
    return [this.hud, this.watermark, this.warnEl, this.adviseEl, this.flashEl].filter(Boolean);
  }

  mount(host) {
    for (const el of this.elements()) if (el.parentElement !== host) host.appendChild(el);
  }

  // A message the viewer must see (an error, a recovery): stays until the next status text replaces it.
  showCritical(text) {
    this.critical = true;
    this.hud.textContent = text;
    this.hud.style.display = 'block';
  }

  setHud(text, visible) {
    this.critical = false;
    this.hud.textContent = text;
    this.hud.style.display = visible ? 'block' : 'none';
  }

  hudVisible(visible) {
    if (this.critical) return;
    this.hud.style.display = visible ? 'block' : 'none';
  }

  layout(rect, { hud, watermark }) {
    this.hud.style.left = (rect.left + 8) + 'px';
    this.hud.style.top = (rect.top + 8) + 'px';
    this.hud.style.maxWidth = Math.max(120, rect.width - 16) + 'px';
    if (!this.critical) this.hud.style.display = hud ? 'block' : 'none';
    this.watermark.style.display = watermark ? 'block' : 'none';
    this.watermark.style.left = (rect.left + 10) + 'px';
    this.watermark.style.top = (rect.bottom - 26) + 'px';
  }

  hideAll() {
    if (!this.critical) this.hud.style.display = 'none';
    this.watermark.style.display = 'none';
    this.hideWarnings();
  }

  hideWarnings() {
    this.warnSince = 0;
    this.adviseUntil = 0;
    if (this.warnEl) {
      this.warnEl.style.opacity = '0';
      this.warnEl.style.transform = 'translateY(-6px)';
    }
    if (this.adviseEl) this.adviseEl.style.opacity = '0';
  }

  advise(text, ms) {
    if (!this.adviseEl) {
      this.adviseEl = document.createElement('div');
      this.adviseEl.style.cssText = `position:fixed; z-index:${Z_TOP - 1}; pointer-events:none;`
        + 'background:rgba(60,45,10,.92); color:#ffd88a;'
        + 'border:1px solid rgba(255,200,90,.35); border-radius:12px; padding:8px 14px;'
        + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4); max-width:70vw;'
        + 'opacity:0; transition:opacity .25s;';
      (this.hud.parentElement || document.body).appendChild(this.adviseEl);
    }
    this.adviseEl.textContent = text;
    this.adviseUntil = performance.now() + ms;
  }

  // `text` is the problem the viewer can act on, or null. Shown after it has lasted 1.5 s, not on a warm-up blip.
  warn(text, now, rect, enabled) {
    if (!enabled) { this.hideWarnings(); return; }
    if (!this.warnEl) {
      this.warnEl = document.createElement('div');
      this.warnEl.style.cssText = `position:fixed; z-index:${Z_TOP - 1}; pointer-events:none;`
        + 'background:rgba(60,16,16,.9); color:#ffb4a8;'
        + 'border:1px solid rgba(255,120,100,.35); border-radius:12px; padding:8px 14px;'
        + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4);'
        + 'opacity:0; transform:translateY(-6px); transition:opacity .25s, transform .25s;';
      (this.hud.parentElement || document.body).appendChild(this.warnEl);
    }
    if (text) {
      if (!this.warnSince) this.warnSince = now;
      this.warnText = text;
      this.warnEl.textContent = text;
    } else {
      this.warnSince = 0;
    }
    const show = this.warnSince && now - this.warnSince > 1500;
    if (show) {
      this.warnEl.style.left = (rect.left + rect.width / 2 - this.warnEl.offsetWidth / 2) + 'px';
      this.warnEl.style.top = (rect.top + 12) + 'px';
    }
    this.warnEl.style.opacity = show ? '1' : '0';
    this.warnEl.style.transform = show ? 'translateY(0)' : 'translateY(-6px)';
    if (this.adviseEl) {
      const advising = now < this.adviseUntil;
      if (advising) {
        this.adviseEl.style.left = Math.max(8, rect.left + rect.width / 2 - this.adviseEl.offsetWidth / 2) + 'px';
        this.adviseEl.style.top = (rect.top + 52) + 'px';
      }
      this.adviseEl.style.opacity = advising ? '1' : '0';
    }
  }

  // The centred play/pause splash: fades out while scaling up.
  flash(symbol, rect) {
    if (!this.flashEl) {
      this.flashEl = document.createElement('div');
      this.flashEl.style.cssText = `position:fixed; z-index:${Z_TOP - 1}; pointer-events:none;`
        + 'color:#fff; font:600 26px system-ui; background:rgba(16,17,20,.85);'
        + 'border-radius:50%; width:72px; height:72px;'
        + 'display:flex; align-items:center; justify-content:center; opacity:0;';
      (this.hud.parentElement || document.body).appendChild(this.flashEl);
    }
    const el = this.flashEl;
    el.innerHTML = symbol;
    el.style.left = (rect.left + rect.width / 2 - 36) + 'px';
    el.style.top = (rect.top + rect.height / 2 - 36) + 'px';
    el.style.transition = 'none';
    el.style.opacity = '0.95';
    el.style.transform = 'scale(0.8)';
    requestAnimationFrame(() => {
      el.style.transition = 'opacity .5s ease-out, transform .5s ease-out';
      el.style.opacity = '0';
      el.style.transform = 'scale(1.4)';
    });
  }
}
