// The overlay canvas: a sibling of the <video> with a modest z-index (above the video, below the site's controls), sized
// and placed like the video and mirroring its object-fit, so the browser applies the same fit and crop to our frame.
export class Overlay {
  constructor({ log }) {
    this.log = log;
    this.canvas = null;
    this.fit = 'fill';
    this.fitRequested = 'fill';
    this.fitSupported = true;
    this.sourceWidth = 0;
    this.sourceHeight = 0;
    this.onClick = null;
    this.onDblClick = null;
  }

  get width() { return this.canvas ? this.canvas.width : 0; }
  get height() { return this.canvas ? this.canvas.height : 0; }
  get visible() { return !!this.canvas && this.canvas.style.opacity === '1' && this.canvas.style.visibility !== 'hidden'; }

  ensure() {
    if (this.canvas) return this.canvas;
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:absolute; pointer-events:none; z-index:2; opacity:0; transition:opacity .25s;';
    // on native-controls players the canvas owns the pointer: a dblclick must never reach the <video>, whose own
    // handler would fullscreen the bare element, where no canvas can exist
    canvas.addEventListener('click', () => this.onClick?.());
    canvas.addEventListener('dblclick', e => { e.preventDefault(); this.onDblClick?.(); });
    this.canvas = canvas;
    return canvas;
  }

  // Returns whether the fit, its support or the source size changed: frames queued for the old geometry are stale.
  position(video, rectIn, { needsCanvas, debug }) {
    const canvas = this.canvas;
    if (!canvas || !video?.isConnected || !video.parentElement) return false;
    if (canvas.parentElement !== video.parentElement) video.parentElement.insertBefore(canvas, video.nextSibling);
    const r = rectIn || video.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return false;
    const style = getComputedStyle(video);
    const sourceWidth = video.videoWidth, sourceHeight = video.videoHeight;
    const sourceChanged = sourceWidth !== this.sourceWidth || sourceHeight !== this.sourceHeight;
    this.sourceWidth = sourceWidth;
    this.sourceHeight = sourceHeight;
    this.fitRequested = style.objectFit || 'fill';
    let fit = this.fitRequested;
    if (!['fill', 'contain', 'cover', 'none', 'scale-down'].includes(fit)) fit = 'fill';
    // scale-down is the smaller of none and contain; resolved here because the canvas's own intrinsic size would
    // otherwise decide which one the browser picks
    // Ultrawide stretch and zoom extensions leave object-fit:contain on the video and stretch it with
    // scale/translate/transform. The bounding rect is the post-transform box, so fitting into it letterboxes where
    // the video is stretched (issue #24). Use the same layout box and the same transforms instead.
    const transformed = style.transform !== 'none' || style.scale !== 'none'
      || style.translate !== 'none' || style.rotate !== 'none';
    const boxW = transformed ? parseFloat(style.width) || r.width : r.width;
    const boxH = transformed ? parseFloat(style.height) || r.height : r.height;
    if (fit === 'scale-down' && sourceWidth && sourceHeight) {
      fit = sourceWidth <= boxW && sourceHeight <= boxH ? 'none' : 'contain';
    }
    canvas.style.objectFit = fit;
    canvas.style.objectPosition = style.objectPosition;
    canvas.style.transform = style.transform;
    canvas.style.scale = style.scale;
    canvas.style.translate = style.translate;
    canvas.style.rotate = style.rotate;
    canvas.style.transformOrigin = style.transformOrigin;
    // size before measuring: with a transform the rect depends on it
    canvas.style.width = boxW + 'px';
    canvas.style.height = boxH + 'px';
    // self-calibrating placement: measure where the canvas landed and nudge by the delta, whatever containing block
    // and margins the site uses
    const cur = canvas.getBoundingClientRect();
    const dx = r.left - cur.left, dy = r.top - cur.top;
    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      canvas.style.left = ((parseFloat(canvas.style.left) || 0) + dx) + 'px';
      canvas.style.top = ((parseFloat(canvas.style.top) || 0) + dy) + 'px';
    }
    canvas.style.outline = debug ? '3px solid #19c37d' : 'none';
    let bw = r.width * devicePixelRatio, bh = r.height * devicePixelRatio;
    let supported = fit === 'fill' || !!(sourceWidth && sourceHeight);
    if (fit === 'none' && supported) {
      // object-fit:none depends on the intrinsic CSS pixels; an unscaled source above the FHD cap cannot obey both,
      // so the raw video stays visible there
      bw = sourceWidth;
      bh = sourceHeight;
      supported = bw <= 1920 && bh <= 1080;
    } else if (fit !== 'fill' && supported) {
      const sx = bw / sourceWidth, sy = bh / sourceHeight;
      const scale = fit === 'cover' ? Math.max(sx, sy) : Math.min(sx, sy);
      bw = sourceWidth * scale;
      bh = sourceHeight * scale;
    }
    const changed = supported !== this.fitSupported || fit !== this.fit || sourceChanged;
    if (changed) {
      this.fitSupported = supported;
      this.fit = fit;
      canvas.style.opacity = '0';
      if (debug && !supported) this.log('object-fit fallback to the raw video', this.fitRequested, bw, bh);
    }
    canvas.style.visibility = supported && needsCanvas ? 'visible' : 'hidden';
    canvas.style.pointerEvents = supported && needsCanvas && video.controls ? 'auto' : 'none';
    if (supported && needsCanvas) {
      const cap = Math.min(1, 1920 / bw, 1080 / bh);
      bw = Math.round(bw * cap);
      bh = Math.round(bh * cap);
      if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    }
    return changed;
  }

  show() {
    if (!this.canvas) return;
    this.canvas.style.opacity = '0';
    this.canvas.style.display = 'block';
  }

  // Revealed only once pixels exist, so there is no black flash, and at once: the first frame drawn is the one the
  // video shows, while a fade would blend the video (ahead by the presentation delay) with the frames under way.
  reveal() {
    const c = this.canvas;
    if (!c || c.style.opacity === '1') return;
    c.style.transition = 'none';
    c.style.visibility = 'visible';
    c.style.opacity = '1';
  }

  // No fade: a fade would blend the previous stream's last frame over the new one for 250 ms.
  hideInstant() {
    const c = this.canvas;
    if (!c) return;
    c.style.transition = 'none';
    c.style.opacity = '0';
    c.style.visibility = 'hidden';
    c.style.pointerEvents = 'none';
  }

  // Native passthrough: the browser's own compositor shows the video; the canvas stays in the DOM, invisible.
  passthrough() {
    const c = this.canvas;
    if (!c) return;
    c.style.opacity = '0';
    c.style.visibility = 'hidden';
    c.style.pointerEvents = 'none';
  }

  // Fade out, then release the layer; the raw video underneath is identical.
  fadeOut(stillStopped) {
    const c = this.canvas;
    if (!c) return;
    c.style.opacity = '0';
    setTimeout(() => { if (stillStopped() && this.canvas) this.canvas.style.display = 'none'; }, 260);
  }

  resetGeometry() {
    this.fitSupported = false;
    this.sourceWidth = 0;
    this.sourceHeight = 0;
  }
}
