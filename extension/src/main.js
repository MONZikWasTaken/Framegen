// The content script's module: settings, the player and the in-page UI wired together, the popup protocol, page
// navigation and the hover behaviour. content.js imports this once per frame.
import { createSettings } from './settings.js';
import { log } from './log.js';
import { Player } from './player.js';
import { Overlay } from './overlay.js';
import { Notices } from './ui/notices.js';
import { Subtitles } from './ui/subtitles.js';
import { ControlBar, CompareSplit } from './ui/bar.js';
import { Controls, pointInsideRect } from './ui/controls.js';
import { svgIcon } from './ui/icons.js';
import { SITE_CONTROLS_OK, inFeed, keepTwitchControlsAboveOverlay, biggestVideo } from './sites.js';
import { installBenchBridge, benchToken } from './bench.js';

export const VERSION = chrome.runtime.getManifest().version;

export function boot() {
  const bench = benchToken();
  const settings = createSettings({ live: !bench, log });
  const overlay = new Overlay({ log });
  const notices = new Notices();
  const subtitles = new Subtitles();
  let controls = null;
  let bar = null;
  let split = null;
  let toggling = false;
  let hoverVideo = null, hoverScanAt = -1e9, mouseMoveAt = -1e9, preloadFailedAt = -1e9;

  const activeVideo = () => (player.isRunning() ? player.video : hoverVideo);

  const player = new Player({
    settings, overlay, subtitles, notices, benchMode: !!bench,
    hooks: {
      version: VERSION,
      splitX: () => (split ? split.x : 0.5),
      candidateVideo: () => hoverVideo || biggestVideo(),
      onStateChange: running => controls?.setActive(running),
      onPreparing: on => controls?.setPreparing(on),
      onStatus: () => { if (controls?.isPanelOpen()) controls.updateStatus(); },
      onVideoGone: () => { if (inFeed()) reattach(); },
      onUiTick: (now, rect) => uiTick(now, rect),
    },
  });

  // ---------- the UI tick (15 Hz from the player's display loop) ----------
  function uiTick(now, rect) {
    const cfg = settings.cfg;
    const video = player.video;
    if (!controls) return;
    if (cfg.hoverReveal && (video.controls || !SITE_CONTROLS_OK)) {
      bar.layout(rect, now < controls.revealUntil, now);
    } else {
      bar.hide();
    }
    bar.update();
    notices.layout(rect, { hud: cfg.debug || cfg.showFps, watermark: cfg.showWatermark });
    notices.warn(player.warningText(player.report()), now, rect, cfg.showWarnings);
    if (controls.isVisible()) controls.place(rect, video);
    split.layout(rect, cfg.compare);
  }

  // ---------- the in-page UI ----------
  function injectUI() {
    if (controls) return;
    keepTwitchControlsAboveOverlay();
    controls = new Controls({
      version: VERSION, settings, log,
      host: {
        onToggle: () => toggle(),
        onOpenAdvanced: () => openAdvancedSettings(),
        activeVideo,
        hdrSupported: () => player.sys.hdrOk,
        statusLines: () => player.statusLines(),
      },
    });
    controls.inject();
    bar = new ControlBar({ video: () => player.video, log,
      onReveal: ms => controls.revealFor(ms),
      onFlash: symbol => { if (player.video) notices.flash(symbol, player.video.getBoundingClientRect()); } });
    bar.ensure();
    split = new CompareSplit({ video: () => player.video });
    overlay.onClick = () => { if (player.video?.controls) bar.togglePlay(); };
    overlay.onDblClick = () => { if (player.video?.controls) bar.toggleFullscreen(); };
    notices.mount(document.body);
    settings.onProfilesChange(store => controls.loadProfiles(store).catch(e => log('profile sync', e)));
  }

  async function openAdvancedSettings() {
    // after the extension was updated or reloaded, the script already in a page has lost it: nothing it asks for can
    // be answered until the page is loaded again, so say that instead of doing nothing
    if (!chrome.runtime?.id) { controls?.setAdvancedLabel('Reload the page: Framegen was updated'); return; }
    try {
      const response = await chrome.runtime.sendMessage({ type: 'fcOpenOptions' });
      if (!response?.ok) throw new Error(response?.error || 'Could not open settings');
    } catch (messageError) {
      try { await chrome.runtime.openOptionsPage(); }
      catch (directError) {
        log('open settings', directError || messageError);
        controls?.setAdvancedLabel(chrome.runtime?.id ? 'Open settings from the toolbar icon' : 'Reload the page: Framegen was updated');
      }
    }
  }

  async function toggle() {
    if (toggling) return;
    if (!controls) injectUI();
    toggling = true;
    try {
      if (player.isRunning()) { player.stop(); return; }
      const video = biggestVideo();
      if (!video) { notices.showCritical('Framegen: no video found'); return; }
      try { await player.start(video); }
      catch (e) { player.preparing(false); notices.showCritical('Framegen error: ' + (e?.message || e)); if (!player.blocked) log(e); }
    } finally { toggling = false; }
  }

  // Fullscreen renders in the browser's top layer: anything not inside the fullscreen element is invisible there.
  // The whole UI moves in and back out, from the site's own fullscreen button too.
  function reparentUI() {
    const host = document.fullscreenElement || document.body;
    if (host.tagName === 'VIDEO') return;
    controls?.mount(host);
    notices.mount(host);
    bar?.mount(host);
    split?.mount(host);
  }

  document.addEventListener('fullscreenchange', () => {
    reparentUI();
    controls?.resetGeometry();
    controls?.setPanelOpen(false);
    controls?.setVisible(false);
    hoverScanAt = 0;
    player.onFullscreenChange();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const v = activeVideo();
      if (v && controls && performance.now() < controls.revealUntil) controls.place(v.getBoundingClientRect(), v);
    }));
  });

  // ---------- hover: the controls appear, the backend preloads ----------
  document.addEventListener('mousemove', (e) => {
    const now = performance.now();
    // gaming mice fire at up to 1000 Hz and the rect read below forces layout
    if (now - mouseMoveAt < 33) return;
    mouseMoveAt = now;
    if (!player.isRunning() && now - hoverScanAt > 300) { hoverScanAt = now; hoverVideo = biggestVideo(); }
    const v = activeVideo();
    if (!v || !controls) return;
    const r = (player.isRunning() && player.lastRect && now - player.nextUiAt < 250) ? player.lastRect : v.getBoundingClientRect();
    const insideVideo = pointInsideRect(e.clientX, e.clientY, r);
    const insideControls = controls.pointInsideControls(e.clientX, e.clientY, r);
    if (insideVideo || insideControls) {
      controls.revealFor(2000);
      controls.place(r, v);
      // hovering a video signals intent: the weights and the kernels load now, so the click lands instantly
      if (insideVideo && !player.backend && !player.backendLoading && !player.blocked && now - preloadFailedAt > 5000) {
        player.preload().catch(err => { preloadFailedAt = performance.now(); if (!player.blocked) log('preload', err); });
      }
    } else {
      controls.limitReveal(250);
    }
  }, { passive: true });

  // ---------- navigation ----------
  let reattachSeq = 0;
  function reattach() {
    const seq = ++reattachSeq, t0 = performance.now();
    const tick = async () => {
      if (seq !== reattachSeq || player.isRunning() || !inFeed()) return;
      if (performance.now() - t0 > 12000) return;
      const v = biggestVideo();
      if (!v || !v.videoWidth || toggling) { setTimeout(tick, 150); return; }
      toggling = true;
      try { await player.start(v); }
      catch (e) { player.preparing(false); log('feed reattach', e); }
      finally { toggling = false; }
    };
    setTimeout(tick, 120);
  }

  let pageHref = location.href;
  setInterval(() => {
    // SPA navigation: the old stream is dead and its frames on the new page are nonsense. Off; the viewer re-enables,
    // except inside feeds, where the enable carries to the next clip.
    if (location.href !== pageHref) {
      pageHref = location.href;
      if (player.isRunning()) {
        player.stop();
        if (inFeed()) reattach();
      }
    }
    if (!controls) return;
    if (controls.isPanelOpen()) {
      controls.revealFor(2000);
      controls.schedulePlacement();
      return;
    }
    if (controls.isVisible()) controls.schedulePlacement();
    if (performance.now() > controls.revealUntil) {
      controls.blurFocus();
      controls.setPanelOpen(false);
      controls.setVisible(false);
    }
  }, 300);

  document.addEventListener('scroll', () => {
    if (!controls?.isVisible()) return;
    const v = activeVideo();
    if (v) controls.place(v.getBoundingClientRect(), v);
  }, { passive: true, capture: true });
  window.addEventListener('resize', () => controls?.schedulePlacement(), { passive: true });
  window.visualViewport?.addEventListener('resize', () => controls?.schedulePlacement(), { passive: true });

  // ---------- settings ----------
  settings.onChange(({ changed }) => {
    controls?.syncPanel();
    player.applySettings(changed);
  });

  // ---------- the toolbar popup ----------
  // With all_frames every frame gets the message; the running frame answers at once, a frame with a video after
  // 120 ms, video-less frames after 250 ms: the first response wins, so the most relevant frame speaks for the tab.
  try {
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if (msg?.type === 'fcStatus') {
        const v = player.video || biggestVideo();
        const respond = async () => {
          await player.probeAdapter();
          try { sendResponse({ version: VERSION, hasVideo: !!v, ...player.status() }); } catch { /* popup closed */ }
        };
        if (player.isRunning()) respond();
        else setTimeout(respond, v ? 120 : 250);
        return true;
      }
      if (msg?.type === 'fcToggle') {
        const v = player.video || biggestVideo();
        if (!player.isRunning() && !v) return undefined;
        toggle().then(() => { try { sendResponse({ running: player.isRunning() }); } catch { /* popup closed */ } });
        return true;
      }
      return undefined;
    });
  } catch { /* messaging unavailable in some frames */ }

  // ---------- boot ----------
  const observer = new MutationObserver(() => mount());
  const mount = () => {
    if (controls) { observer.disconnect(); return; }
    if (document.querySelector('video')) { injectUI(); observer.disconnect(); }
  };
  mount();
  if (!controls) observer.observe(document.documentElement, { childList: true, subtree: true });
  if (bench) installBenchBridge({ token: bench, settings, player, version: VERSION, start: toggle, biggestVideo });
  window.__framegenDiagnostics = () => player.status();
  window.addEventListener('pagehide', () => { if (player.isRunning()) player.stop(); });
}

export { svgIcon };
