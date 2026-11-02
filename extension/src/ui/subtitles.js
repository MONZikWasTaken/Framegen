// Subtitles above the canvas: the video's text tracks rendered into our own layer, and the site's caption elements
// raised over the canvas where the page draws them itself.
const SITE_SUBTITLE_SELECTORS = [
  '[class*="subtitle" i]', '[id*="subtitle" i]', '[class*="caption" i]', '[id*="caption" i]',
  '[class*="text-track" i]', '[id*="text-track" i]', '.vjs-text-track-display', '.plyr__captions',
  '.jw-text-track-display', '.shaka-text-container', '.dplayer-subtitle', '.art-subtitle',
  '.vds-captions', '.ytp-caption-window-container', '.player-timedtext',
].join(',');

export class Subtitles {
  constructor() {
    this.layer = null;
    this.cueRoot = null;
    this.trackVideo = null;
    this.trackList = null;
    this.trackListeners = new Map();
    this.domOverrides = new Map();
    this.domScanAt = 0;
    this.onTrackListChange = () => this.bindTracks();
  }

  ensureLayer(canvas) {
    if (this.layer) return;
    this.layer = document.createElement('div');
    this.layer.style.cssText = 'position:absolute!important; z-index:2147483647!important; pointer-events:none!important; overflow:hidden!important; display:none; contain:layout style paint;';
    const shadow = this.layer.attachShadow({ mode: 'closed' });
    const style = document.createElement('style');
    style.textContent = `
      :host { all: initial; }
      #cues { position:absolute; inset:0; overflow:hidden; container-type:size; color:white;
        font:700 clamp(14px, 4cqw, 48px)/1.2 Arial,sans-serif; text-align:center; }
      .cue { position:absolute; box-sizing:border-box; padding:.08em .22em; border-radius:.12em;
        color:#fff; background:rgba(0,0,0,.72); text-shadow:0 1px 2px #000;
        white-space:pre-line; overflow-wrap:anywhere; }
      .cue b { font-weight:900; }
      .cue i { font-style:italic; }
      .cue u { text-decoration:underline; }
    `;
    this.cueRoot = document.createElement('div');
    this.cueRoot.id = 'cues';
    shadow.append(style, this.cueRoot);
    if (canvas?.parentElement) canvas.parentElement.insertBefore(this.layer, canvas.nextSibling);
  }

  position(canvas, forceOutsideFullscreen = false) {
    if (!this.layer || !canvas) return;
    const fullscreenHost = !forceOutsideFullscreen && document.fullscreenElement?.tagName !== 'VIDEO'
      ? document.fullscreenElement : null;
    if (fullscreenHost) {
      if (this.layer.parentElement !== fullscreenHost) fullscreenHost.appendChild(this.layer);
      const rect = canvas.getBoundingClientRect();
      this.layer.style.setProperty('position', 'fixed', 'important');
      this.layer.style.setProperty('left', `${rect.left}px`, 'important');
      this.layer.style.setProperty('top', `${rect.top}px`, 'important');
      this.layer.style.setProperty('width', `${rect.width}px`, 'important');
      this.layer.style.setProperty('height', `${rect.height}px`, 'important');
      return;
    }
    if (canvas.parentElement
        && (this.layer.parentElement !== canvas.parentElement || canvas.nextSibling !== this.layer)) {
      canvas.parentElement.insertBefore(this.layer, canvas.nextSibling);
    }
    this.layer.style.setProperty('position', 'absolute', 'important');
    for (const property of ['left', 'top', 'width', 'height']) {
      this.layer.style.setProperty(property, canvas.style[property], 'important');
    }
  }

  renderCues() {
    if (!this.cueRoot || !this.trackVideo) return;
    const cues = [];
    for (const track of Array.from(this.trackVideo.textTracks || [])) {
      if (track.mode === 'disabled' || !track.activeCues) continue;
      for (const cue of Array.from(track.activeCues)) cues.push(cue);
    }
    this.cueRoot.replaceChildren();
    cues.forEach((cue, index) => {
      const box = document.createElement('div');
      box.className = 'cue';
      const size = Number.isFinite(cue.size) ? Math.max(1, Math.min(100, cue.size)) : 100;
      const position = Number.isFinite(cue.position) ? Math.max(0, Math.min(100, cue.position)) : 50;
      const align = ['start', 'left'].includes(cue.align) ? 'left'
        : ['end', 'right'].includes(cue.align) ? 'right' : 'center';
      const positionAlign = cue.positionAlign || 'auto';
      const translateX = ['line-left', 'left'].includes(positionAlign) ? '0'
        : ['line-right', 'right'].includes(positionAlign) ? '-100%'
          : align === 'left' ? '0' : align === 'right' ? '-100%' : '-50%';
      box.style.left = `${position}%`;
      box.style.width = `${size}%`;
      box.style.textAlign = align;
      box.style.transform = `translate(${translateX}, var(--fg-cue-y, 0))`;
      if (cue.vertical === 'rl' || cue.vertical === 'lr') {
        box.style.writingMode = cue.vertical === 'rl' ? 'vertical-rl' : 'vertical-lr';
        box.style.width = 'auto';
        box.style.height = `${size}%`;
      }
      const line = cue.line;
      if (typeof line === 'number' && Number.isFinite(line)) {
        if (cue.snapToLines === false) {
          box.style.top = `${Math.max(0, Math.min(100, line))}%`;
          const lineAlign = cue.lineAlign || 'start';
          box.style.setProperty('--fg-cue-y', lineAlign === 'end' ? '-100%' : lineAlign === 'center' ? '-50%' : '0');
        } else if (line < 0) {
          box.style.bottom = `${Math.abs(line) * 1.2}em`;
        } else {
          box.style.top = `${line * 1.2}em`;
        }
      } else {
        box.style.bottom = `${(cues.length - index - 1) * 1.35 + .35}em`;
      }
      try {
        if (typeof cue.getCueAsHTML === 'function') box.append(cue.getCueAsHTML());
        else box.textContent = cue.text || '';
      } catch { box.textContent = cue.text || ''; }
      this.cueRoot.append(box);
    });
  }

  clearTrackBindings() {
    for (const [track, listener] of this.trackListeners) track.removeEventListener('cuechange', listener);
    this.trackListeners.clear();
    if (this.trackList) {
      this.trackList.removeEventListener('change', this.onTrackListChange);
      this.trackList.removeEventListener('addtrack', this.onTrackListChange);
      this.trackList.removeEventListener('removetrack', this.onTrackListChange);
    }
    this.trackList = null;
    this.trackVideo = null;
  }

  bindTracks() {
    if (!this.trackVideo) return;
    const tracks = new Set(Array.from(this.trackVideo.textTracks || []));
    for (const [track, listener] of this.trackListeners) {
      if (!tracks.has(track)) {
        track.removeEventListener('cuechange', listener);
        this.trackListeners.delete(track);
      }
    }
    for (const track of tracks) {
      if (this.trackListeners.has(track)) continue;
      const listener = () => this.renderCues();
      track.addEventListener('cuechange', listener);
      this.trackListeners.set(track, listener);
    }
    this.renderCues();
  }

  restoreDomOverrides() {
    for (const [element, original] of this.domOverrides) restoreStyle(element, original);
    this.domOverrides.clear();
    this.domScanAt = 0;
  }

  raiseSiteLayers(video, canvas) {
    const now = performance.now();
    if (now - this.domScanAt < 250 || !video?.isConnected || !canvas?.parentElement) return;
    this.domScanAt = now;
    const videoRect = video.getBoundingClientRect();
    const canvasBranches = new Map();
    let node = canvas, branch = canvas;
    while (node) {
      canvasBranches.set(node, branch);
      branch = node.parentElement;
      node = node.parentElement;
    }
    const active = new Set();
    for (const candidate of document.querySelectorAll(SITE_SUBTITLE_SELECTORS)) {
      if (!(candidate instanceof HTMLElement) || candidate === canvas || candidate === this.layer
          || candidate.closest('.fc-controls-root, .fc-hud, .fc-watermark')) continue;
      const text = (candidate.innerText || candidate.textContent || '').trim();
      if (!text || text.length > 500) continue;
      const style = getComputedStyle(candidate);
      if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || 1) === 0) continue;
      const rect = candidate.getBoundingClientRect();
      if (rect.width < 8 || rect.height < 8 || rect.right <= videoRect.left || rect.left >= videoRect.right
          || rect.bottom <= videoRect.top || rect.top >= videoRect.bottom) continue;
      let own = candidate;
      let common = own.parentElement;
      while (common && !canvasBranches.has(common)) {
        own = common;
        common = common.parentElement;
      }
      if (!common || own === video || own === canvasBranches.get(common)) continue;
      active.add(own);
      if (!this.domOverrides.has(own)) {
        this.domOverrides.set(own, {
          position: { value: own.style.getPropertyValue('position'), priority: own.style.getPropertyPriority('position') },
          zIndex: { value: own.style.getPropertyValue('z-index'), priority: own.style.getPropertyPriority('z-index') },
        });
      }
      if (getComputedStyle(own).position === 'static') own.style.setProperty('position', 'relative', 'important');
      own.style.setProperty('z-index', '2147483647', 'important');
    }
    for (const [element, original] of this.domOverrides) {
      if (active.has(element)) continue;
      restoreStyle(element, original);
      this.domOverrides.delete(element);
    }
  }

  // `active`: the setting is on, playback runs through the canvas, and there is a video.
  sync({ active, video, canvas, canvasVisible }) {
    if (!active) {
      if (this.trackVideo || this.trackList) this.clearTrackBindings();
      if (this.cueRoot?.childNodes.length) this.cueRoot.replaceChildren();
      this.position(canvas, true);
      if (this.layer && this.layer.style.display !== 'none') this.layer.style.display = 'none';
      this.restoreDomOverrides();
      return;
    }
    this.ensureLayer(canvas);
    this.position(canvas);
    this.raiseSiteLayers(video, canvas);
    if (this.trackVideo !== video) {
      this.clearTrackBindings();
      this.trackVideo = video;
      this.trackList = video.textTracks;
      this.trackList?.addEventListener('change', this.onTrackListChange);
      this.trackList?.addEventListener('addtrack', this.onTrackListChange);
      this.trackList?.addEventListener('removetrack', this.onTrackListChange);
      this.bindTracks();
    }
    const display = canvasVisible ? 'block' : 'none';
    if (this.layer.style.display !== display) this.layer.style.display = display;
  }
}

function restoreStyle(element, original) {
  if (original.position.value) element.style.setProperty('position', original.position.value, original.position.priority);
  else element.style.removeProperty('position');
  if (original.zIndex.value) element.style.setProperty('z-index', original.zIndex.value, original.zIndex.priority);
  else element.style.removeProperty('z-index');
}
