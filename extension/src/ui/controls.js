// The in-player controls: the stage clipped to the video box with the FG and settings buttons on its rail, and the quick
// settings panel that morphs out of it. Settings changes go through the settings object; the host reacts to them.
import { Profiles } from '../profiles.js';
import { PICTURE_MODES, PICTURE_SWATCH_FILTERS } from '../picture-modes.js';
import { svgIcon, GITHUB_PATH, FRAMEGEN_MARK } from './icons.js';

const INLINE_SETTINGS_MIN_WIDTH = 300, INLINE_SETTINGS_MIN_HEIGHT = 180;
const ISLAND_HALF = 57;
const VIEW_HEIGHT = { settings: 626, picture: 436 };
const SHARPNESS_LABELS = ['Off', 'Low', 'Medium', 'High'];
const clampUnit = value => Math.min(1, Math.max(0, value));
const ease = value => { const t = clampUnit(value); return t * t * (3 - 2 * t); };
const mix = (from, to, progress) => from + (to - from) * progress;
const coord = value => Number(value.toFixed(3));

const CSS = `
  .fc-btn{background:none;border:none;color:#fff;font:13px/1 system-ui;cursor:pointer;
    padding:4px 6px;opacity:.85;transition:opacity .15s,transform .15s;
    display:inline-flex;align-items:center;justify-content:center}
  .fc-btn:hover{opacity:1;transform:scale(1.15)}
  .fc-range{-webkit-appearance:none;appearance:none;height:3px;border-radius:3px;margin:0;
    background:rgba(255,255,255,.22);outline:none;cursor:pointer}
  .fc-range::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;
    border-radius:50%;background:#fff;transition:transform .15s}
  .fc-range:hover::-webkit-slider-thumb{transform:scale(1.35);background:#19c37d}
  .fc-controls-root{--fc-green:#19c37d;--fc-rail-radius:5px;
    position:fixed!important;z-index:2147483647!important;left:0;top:0;
    width:376px;height:626px;min-width:0!important;max-width:none!important;
    margin:0!important;padding:0!important;overflow:hidden!important;box-sizing:border-box!important;
    border:0!important;background:transparent!important;pointer-events:none!important}
  .fc-controls-stage{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;
    opacity:0;transform:translateX(-43px);
    transition:transform .26s cubic-bezier(.45,0,.55,1),opacity 0s linear .26s;
    will-change:transform;pointer-events:none!important}
  .fc-controls-root[data-visible="true"] .fc-controls-stage{opacity:1;transform:translateX(0);
    transition:transform .18s cubic-bezier(.22,.75,.25,1),opacity 0s linear}
  .fc-controls-root:focus-within .fc-controls-stage{opacity:1;transform:translateX(0);
    transition:transform .18s cubic-bezier(.22,.75,.25,1),opacity 0s linear}
  .fc-control-surface{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;
    overflow:visible!important;pointer-events:none!important;filter:drop-shadow(0 2px 4px rgba(0,0,0,.28))}
  .fc-surface-shape{fill:rgba(13,14,15,.97);stroke:rgba(255,255,255,.13);stroke-width:1;
    vector-effect:non-scaling-stroke}
  .fc-settings-clip{position:absolute!important;z-index:1!important;inset:0!important;
    width:100%!important;height:100%!important;pointer-events:none;will-change:clip-path}
  .fc-controls-root[data-panel-open="true"] .fc-settings-clip{pointer-events:auto}
  .fc-control-rail{position:absolute!important;z-index:2!important;left:5px!important;
    top:var(--fc-island-center)!important;width:34px!important;display:grid!important;gap:2px!important;
    transform:translateY(-50%);pointer-events:none}
  .fc-controls-root[data-visible="true"] .fc-control-rail,
  .fc-controls-root:focus-within .fc-control-rail{pointer-events:auto}
  .fc-side{position:relative!important;width:34px!important;height:34px!important;
    min-width:34px!important;min-height:34px!important;max-width:34px!important;max-height:34px!important;
    display:grid!important;place-items:center!important;padding:0!important;margin:0!important;
    box-sizing:border-box!important;border:0!important;border-radius:var(--fc-rail-radius)!important;
    background:transparent!important;color:#fff!important;cursor:pointer!important;font:600 12px/1 system-ui!important;
    box-shadow:none!important;opacity:1!important;transform:none!important;
    transition:background-color .12s ease,color .12s ease!important}
  .fc-side:hover{background:rgba(255,255,255,.09)!important;color:#fff!important;transform:none!important}
  .fc-side:focus-visible{outline:1px solid rgba(255,255,255,.72)!important;outline-offset:-2px!important}
  .fc-side[data-active="true"]{background:var(--fc-green)!important;color:#fff!important}
  .fc-side[data-active="true"]:hover{background:#20cd85!important}
  .fc-side[data-preparing="true"]{animation:fc-prep 1.1s ease-in-out infinite!important}
  @keyframes fc-prep{0%,100%{opacity:1}50%{opacity:.45}}
  .fc-side[data-open="true"]{background:rgba(255,255,255,.11)!important;color:#fff!important}
  .fc-side svg{display:block!important;margin:auto!important}
  .fc-mark{width:19px!important;height:14px!important}
  .fc-settings-icon{width:17px!important;height:17px!important}
  .fc-panel{position:absolute!important;left:43px!important;top:var(--fc-open-top,0px)!important;
    width:calc(100% - 44px)!important;height:var(--fc-open-height,100%)!important;min-width:0!important;max-width:none!important;padding:14px 16px 15px!important;
    margin:0!important;box-sizing:border-box!important;border:0!important;border-radius:0!important;
    background:transparent!important;box-shadow:none!important;color:#ddd!important;
    font:12px/1.5 system-ui,sans-serif!important;overflow-y:auto!important;overscroll-behavior:contain;
    opacity:0;visibility:hidden;pointer-events:none;transform:translateX(-3px);transform-origin:left center;
    transition:transform .07s cubic-bezier(.4,0,1,1),opacity .07s linear,visibility 0s linear .07s;
    will-change:transform,opacity}
  .fc-panel{scrollbar-width:thin;scrollbar-color:#555a61 transparent}
  .fc-panel::-webkit-scrollbar{width:6px}
  .fc-panel::-webkit-scrollbar-track{background:transparent}
  .fc-panel::-webkit-scrollbar-thumb{border-radius:3px;background:#555a61}
  .fc-controls-root[data-panel-open="true"] .fc-panel{opacity:1;visibility:visible;pointer-events:auto;
    transform:translateX(0);transition:transform .11s cubic-bezier(.2,.8,.25,1) .15s,
    opacity .11s linear .15s,visibility 0s linear}
  .fc-panel-head{height:30px;display:flex;align-items:center;justify-content:space-between;
    margin:0 0 8px;padding:0;border:0}
  .fc-brand{display:flex;align-items:center;gap:6px;color:#f4f5f5;font:12px/1 system-ui}
  .fc-brand strong{font:650 14px/1 system-ui;color:#f4f5f5}
  .fc-panel[data-view="settings"] .fc-view[data-view="picture"],
  .fc-panel[data-view="picture"] .fc-view[data-view="settings"],
  .fc-panel[data-view="picture"] .fc-version,
  .fc-panel[data-view="picture"] .fc-icon-link{display:none!important}
  .fc-looks{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:2px 0 4px;padding:0}
  .fc-look{display:flex;align-items:center;gap:9px;min-width:0;height:44px;margin:0;padding:0 10px 0 6px;
    border:1px solid #2b2e32;border-radius:7px;background:#17191c;color:#d8dadd;font:500 12px/1.2 system-ui;
    text-align:left;cursor:pointer;outline:0;transition:background-color .12s ease,border-color .12s ease}
  .fc-look:hover{background:#1e2124;border-color:#44484e;color:#f4f5f5}
  .fc-look:focus-visible{outline:2px solid var(--fc-green);outline-offset:1px}
  .fc-look[aria-checked="true"]{border-color:var(--fc-green);background:#14211c;color:#fff}
  .fc-look:first-child{grid-column:1 / -1}
  .fc-look-swatch{width:32px;height:32px;flex:none;border-radius:5px;
    background:linear-gradient(135deg,#27357f 0%,#b4405e 48%,#efb84c 100%)}
  .fc-look-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .fc-seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;width:196px;height:31px;flex:none;padding:2px;
    box-sizing:border-box;border:1px solid #34373b;border-radius:var(--fc-rail-radius);background:#181a1d}
  .fc-seg button{min-width:0;margin:0;padding:0;border:0;border-radius:3px;background:transparent;color:#a5aab1;
    font:500 11px/1 system-ui;cursor:pointer;outline:0;transition:background-color .12s ease,color .12s ease}
  .fc-seg button:hover{color:#f4f5f5}
  .fc-seg button:focus-visible{outline:2px solid var(--fc-green);outline-offset:1px}
  .fc-seg button[aria-checked="true"]{background:#33373c;color:#fff}
  .fc-version{color:#747980;font:400 10px/1 system-ui}
  .fc-icon-link{width:28px;height:28px;display:flex;align-items:center;justify-content:center;
    border-radius:7px;color:#92979e;text-decoration:none;outline:none;
    transition:background .15s,color .15s}
  .fc-icon-link:hover{background:#202327;color:#f2f3f3}
  .fc-icon-link:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
  .fc-divider{height:1px;background:#292c30;margin:7px 0}
  .fc-row{display:flex;justify-content:space-between;align-items:center;gap:14px;
    padding:8px 0;margin:0;border:0;width:auto;cursor:default;
    font:12px/1.4 system-ui;color:#e8e8e8;text-align:left}
  .fc-row[hidden]{display:none}
  .fc-row>span{display:block;flex:1 1 auto;min-width:0;margin:0;padding:0;
    font:12px/1.4 system-ui;color:#e8e8e8;text-align:left;
    letter-spacing:normal;text-transform:none;white-space:normal}
  .fc-row small{display:block;color:#8a8f98;font:400 10px/1.3 system-ui;
    margin:1px 0 0;padding:0;letter-spacing:normal;text-transform:none}
  .fc-profile-row{padding-top:5px;padding-bottom:7px}
  .fc-sw{appearance:none;-webkit-appearance:none;width:36px;height:20px;border-radius:20px;
    background:#3d4148;position:relative;cursor:pointer;outline:none;margin:0;
    transition:background .2s;flex:none}
  .fc-sw::after{content:'';position:absolute;width:16px;height:16px;border-radius:50%;
    background:#fff;top:2px;left:2px;transition:left .2s;box-shadow:0 1px 3px rgba(0,0,0,.4)}
  .fc-sw:checked{background:#19c37d}
  .fc-sw:checked::after{left:18px}
  .fc-sw:focus-visible{outline:2px solid #19c37d;outline-offset:2px}
  .fc-sel:not([data-enhanced="true"]){width:154px;height:31px;flex:none;border:1px solid #34373b;
    border-radius:var(--fc-rail-radius);background:#181a1d;color:#eceeef;font:12px/1 system-ui,sans-serif}
  .fc-sel[data-enhanced="true"]{display:none!important}
  .fc-select{position:relative;width:154px;flex:none;font:12px/1 system-ui,sans-serif}
  .fc-select-trigger{width:100%;height:31px;display:flex;align-items:center;justify-content:space-between;
    gap:8px;margin:0;padding:0 9px 0 10px;border:1px solid #34373b;
    border-radius:var(--fc-rail-radius);background:#181a1d;color:#eceeef;font:inherit;
    text-align:left;cursor:pointer;outline:0;transition:background-color .12s ease,border-color .12s ease}
  .fc-select-trigger:hover{background:#202327;border-color:#484c52}
  .fc-select-trigger:focus-visible{outline:2px solid var(--fc-green);outline-offset:1px}
  .fc-select-trigger:disabled{opacity:.55;cursor:wait}
  .fc-select[data-open="true"] .fc-select-trigger{background:#202327;border-color:#5a5f66}
  .fc-select-value{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;line-height:1.5}
  .fc-select-chevron{width:12px;height:12px;flex:none;color:#8d9299;transition:transform .12s ease}
  .fc-select[data-open="true"] .fc-select-chevron{transform:rotate(180deg)}
  .fc-select-menu{position:absolute;z-index:20;top:calc(100% + 4px);right:0;width:164px;
    max-height:250px;overflow-y:auto;padding:4px;border:1px solid #383c41;
    border-radius:var(--fc-rail-radius);background:#151719;box-shadow:0 4px 8px rgba(0,0,0,.28)}
  .fc-select[data-placement="up"] .fc-select-menu{top:auto;bottom:calc(100% + 4px)}
  .fc-select-menu[hidden]{display:none!important}
  .fc-select-options-group{margin:0;padding:0}
  .fc-select-group{padding:7px 8px 4px;color:#737980;font:500 10px/1 system-ui,sans-serif}
  .fc-select-option{width:100%;min-height:30px;display:flex;align-items:center;justify-content:space-between;
    gap:10px;margin:0;padding:0 10px;border:0;border-radius:3px;background:transparent;color:#d8dadd;
    font:inherit;text-align:left;cursor:pointer;outline:0;white-space:nowrap}
  .fc-select-option:hover,.fc-select-option:focus-visible{background:#23262a;color:#f4f5f5}
  .fc-select-option:focus-visible{box-shadow:inset 0 0 0 1px #555a61}
  .fc-select-option[aria-selected="true"]{color:#f4f5f5}
  .fc-select-option::after{content:'';width:7px;height:4px;flex:none;border-left:1.5px solid transparent;
    border-bottom:1.5px solid transparent;transform:rotate(-45deg) translateY(-1px)}
  .fc-select-option[aria-selected="true"]::after{border-color:var(--fc-green)}
  .fc-target-control{display:block;padding:2px 0 8px;margin:0;cursor:pointer}
  .fc-target-control[hidden]{display:none}
  .fc-target-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
    margin:0 0 4px;color:#e8e8e8;font:12px/1.35 system-ui}
  .fc-target-head>span{display:block}
  .fc-target-head small{display:block;margin-top:1px;color:#8a8f98;font:400 10px/1.3 system-ui}
  .fc-target-head output{min-width:72px;color:#7fe0b5;font:600 11px/1.35 system-ui;
    font-variant-numeric:tabular-nums;text-align:right}
  .fc-target-slider{--fc-fill:12%;appearance:none;-webkit-appearance:none;display:block;
    width:100%;height:20px;margin:0;background:transparent;cursor:pointer;outline:none}
  .fc-target-slider::-webkit-slider-runnable-track{height:3px;border-radius:3px;
    background:linear-gradient(to right,#19c37d 0 var(--fc-fill),#3d4148 var(--fc-fill) 100%)}
  .fc-target-slider::-webkit-slider-thumb{appearance:none;-webkit-appearance:none;width:15px;height:15px;
    margin-top:-6px;border:2px solid #111315;border-radius:50%;background:#f2f4f3;
    box-shadow:0 0 0 1px #62686f}
  .fc-target-slider:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
  .fc-target-scale{position:relative;display:block;height:10px;margin-top:-2px;
    color:#656b73;font:500 9px/1 system-ui;font-variant-numeric:tabular-nums;pointer-events:none}
  .fc-target-scale>span{position:absolute;
    left:calc(var(--fc-mark-position) + var(--fc-mark-offset));top:0;
    transform:translateX(-50%);white-space:nowrap}
  .fc-status{margin:8px 0 0;padding:8px 10px;border-radius:6px;background:#15171a;color:#aeb3ba;
    font:10.5px/1.5 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
  .fc-open-settings{width:100%;height:35px;display:flex;align-items:center;justify-content:space-between;
    margin-top:10px;padding:0 11px;border-radius:8px;border:1px solid #383c41;background:#1b1e21;
    color:#e2e4e5;font:600 11px system-ui;cursor:pointer;outline:none;
    transition:background .15s,border-color .15s}
  .fc-open-settings:hover{background:#22262a;border-color:#50555c}
  .fc-open-settings:focus-visible{outline:2px solid #19c37d;outline-offset:1px}
  .fc-open-settings:disabled{opacity:.55;cursor:wait}
  .fc-panel:focus{outline:0}
  .fc-engine-link{display:block;margin:9px 0 1px;text-align:center;color:#70767e;font:500 10.5px/1.4 system-ui;
    text-decoration:none;border-radius:4px;transition:color .12s}
  .fc-engine-link b{color:#a3a8af;font-weight:600;transition:color .12s}
  .fc-engine-link:hover,.fc-engine-link:hover b{color:#f2f3f3}
  .fc-engine-link:focus-visible{outline:2px solid #19c37d;outline-offset:2px}`;

export class Controls {
  constructor({ version, settings, host, log }) {
    this.version = version;
    this.settings = settings;
    this.host = host;
    this.log = log;
    this.root = null; this.stage = null; this.surface = null; this.shape = null; this.clip = null;
    this.btn = null; this.gear = null; this.pic = null; this.panel = null;
    this.view = 'settings';
    this.width = 376; this.height = 626; this.islandCenter = 313;
    this.surfaceProgress = 0; this.surfaceFrame = 0;
    this.panelOpen = false;
    this.placed = { left: -1, top: -1, width: -1, height: -1, center: -1 };
    this.placementFrame = 0;
    this.observedVideo = null;
    this.videoObserver = typeof ResizeObserver === 'function' ? new ResizeObserver(() => this.schedulePlacement()) : null;
    this.customSelects = new Map();
    this.documentHandlers = false;
    this.profileStore = null;
    this.revealUntil = 0;
  }

  setAdvancedLabel(text) {
    const label = this.panel?.querySelector('#fcOpenSettings span');
    if (label) label.textContent = text;
  }

  revealFor(ms) { this.revealUntil = Math.max(this.revealUntil, performance.now() + ms); }
  limitReveal(ms) { const n = performance.now(); if (this.revealUntil > n + ms) this.revealUntil = n + ms; }

  inject() {
    if (this.root) return;
    const css = document.createElement('style');
    css.textContent = CSS;
    (document.head || document.documentElement).appendChild(css);

    const root = document.createElement('div');
    root.className = 'fc-controls-root';
    root.dataset.visible = 'false';
    root.dataset.panelOpen = 'false';
    root.style.setProperty('--fc-island-center', this.islandCenter + 'px');
    root.innerHTML = `
      <div class="fc-controls-stage">
        <svg class="fc-control-surface" viewBox="0 0 376 626" aria-hidden="true">
          <path class="fc-surface-shape"></path>
        </svg>
        <div class="fc-control-rail"></div>
        <div class="fc-settings-clip"></div>
      </div>`;
    this.root = root;
    this.stage = root.querySelector('.fc-controls-stage');
    this.surface = root.querySelector('.fc-control-surface');
    this.shape = root.querySelector('.fc-surface-shape');
    this.clip = root.querySelector('.fc-settings-clip');
    const rail = root.querySelector('.fc-control-rail');

    const btn = document.createElement('button');
    btn.className = 'fc-side';
    btn.type = 'button';
    btn.dataset.active = 'false';
    btn.setAttribute('aria-label', 'Toggle Framegen');
    btn.setAttribute('aria-pressed', 'false');
    btn.innerHTML = FRAMEGEN_MARK;
    const gear = document.createElement('button');
    gear.className = 'fc-side';
    gear.type = 'button';
    gear.dataset.open = 'false';
    gear.setAttribute('aria-label', 'Open Framegen settings');
    gear.setAttribute('aria-controls', 'fcSettingsPanel');
    gear.setAttribute('aria-expanded', 'false');
    gear.innerHTML = `<svg class="fc-settings-icon" viewBox="0 0 24 24" aria-hidden="true">${svgIcon('gear').replace(/^<svg[^>]*>|<\/svg>$/g, '')}</svg>`;
    const pic = document.createElement('button');
    pic.className = 'fc-side';
    pic.type = 'button';
    pic.dataset.open = 'false';
    pic.setAttribute('aria-label', 'Open picture settings');
    pic.setAttribute('aria-controls', 'fcSettingsPanel');
    pic.setAttribute('aria-expanded', 'false');
    pic.innerHTML = `<svg class="fc-settings-icon" viewBox="0 0 24 24" aria-hidden="true">${svgIcon('picture').replace(/^<svg[^>]*>|<\/svg>$/g, '')}</svg>`;
    rail.append(btn, pic, gear);
    this.btn = btn;
    this.gear = gear;
    this.pic = pic;

    this.buildPanel();
    this.applySurfaceProgress(0);
    btn.onclick = () => this.host.onToggle();
    const toggleView = view => {
      if (this.width < INLINE_SETTINGS_MIN_WIDTH || this.height < INLINE_SETTINGS_MIN_HEIGHT) {
        this.setPanelOpen(false);
        this.host.onOpenAdvanced();
        return;
      }
      const same = this.panelOpen && this.view === view;
      this.view = view;
      this.setPanelOpen(!same);
    };
    gear.onclick = () => toggleView('settings');
    pic.onclick = () => toggleView('picture');
    root.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || !this.panelOpen) return;
      event.preventDefault();
      event.stopPropagation();
      this.setPanelOpen(false, true);
    });
    root.addEventListener('pointerdown', event => {
      if (!event.target.closest('.fc-select')) this.closeCustomSelects();
      this.pressedInsideAt = performance.now();
      event.stopPropagation();
    });
    root.addEventListener('click', event => event.stopPropagation());
    root.addEventListener('focusin', () => {
      const video = this.host.activeVideo();
      if (!video) return;
      this.revealFor(2000);
      this.place(video.getBoundingClientRect(), video);
    });
    // Focus leaving the controls closes the panel, but a press on the panel itself is not leaving: a click on a
    // label or on bare background takes the focus off the button that had it and gives it to nothing.
    root.addEventListener('focusout', () => {
      requestAnimationFrame(() => {
        if (this.root?.contains(document.activeElement)) return;
        if (performance.now() - (this.pressedInsideAt ?? -Infinity) < 400) return;
        this.setPanelOpen(false);
      });
    });
    root.addEventListener('mouseleave', event => {
      const video = this.host.activeVideo();
      const rect = video?.getBoundingClientRect();
      if (rect && pointInsideRect(event.clientX, event.clientY, rect)) return;
      this.limitReveal(250);
    });
    window.addEventListener('blur', () => {
      this.blurFocus();
      this.setPanelOpen(false);
      this.setVisible(false);
    });
    document.body.appendChild(root);
  }

  mount(host) {
    if (this.root && this.root.parentElement !== host) host.appendChild(this.root);
  }

  setActive(active) {
    if (!this.btn) return;
    this.btn.dataset.active = String(!!active);
    this.btn.setAttribute('aria-pressed', String(!!active));
  }

  // FG switched on, the first generated frame not shown yet
  setPreparing(on) {
    if (this.btn) this.btn.dataset.preparing = String(!!on);
  }

  // ---------- the morphing surface ----------
  surfacePath(progress) {
    const x = ease(progress / .82);
    const y = ease((progress - .12) / .88);
    const px = (from, to) => coord(mix(from, to, x));
    const py = (from, to) => coord(mix(from, to, y));
    const islandTop = this.islandCenter - ISLAND_HALF;
    const islandBottom = this.islandCenter + ISLAND_HALF;
    const openRight = this.width - 1;
    const openRightInset = this.width - 12;
    const { top, height } = this.openBox();
    const openTop = top + 1;
    const openBottom = top + height - 1;
    return `M${px(-2, 55)} ${py(islandTop, openTop)}H${px(38, openRightInset)}`
      + `Q${px(43, openRight)} ${py(islandTop, openTop)} ${px(43, openRight)} ${py(islandTop + 5, openTop + 11)}`
      + `V${py(islandBottom - 5, openBottom - 11)}`
      + `Q${px(43, openRight)} ${py(islandBottom, openBottom)} ${px(38, openRightInset)} ${py(islandBottom, openBottom)}`
      + `H${px(-2, 55)}Q${px(-2, 43)} ${py(islandBottom, openBottom)} `
      + `${px(-2, 43)} ${py(islandBottom, openBottom - 12)}`
      + `V${islandBottom}H-2V${islandTop}H${px(-2, 43)}V${py(islandTop, openTop + 12)}`
      + `Q${px(-2, 43)} ${py(islandTop, openTop)} ${px(-2, 55)} ${py(islandTop, openTop)}Z`;
  }

  // The open panel is as tall as its view needs, around the rail and inside the stage.
  openBox() {
    const height = Math.min(this.height, Math.max(2 * ISLAND_HALF + 2, VIEW_HEIGHT[this.view]));
    const top = Math.max(0, Math.min(this.height - height, Math.round(this.islandCenter - height / 2)));
    return { top, height };
  }

  applySurfaceProgress(progress) {
    this.surfaceProgress = clampUnit(progress);
    if (!this.shape || !this.clip) return;
    const path = this.surfacePath(this.surfaceProgress);
    const box = this.openBox();
    this.root.style.setProperty('--fc-open-top', box.top + 'px');
    this.root.style.setProperty('--fc-open-height', box.height + 'px');
    this.shape.setAttribute('d', path);
    this.clip.style.clipPath = `path("${path}")`;
  }

  animateSurface(open) {
    const target = open ? 1 : 0;
    cancelAnimationFrame(this.surfaceFrame);
    const start = this.surfaceProgress;
    const distance = Math.abs(target - start);
    if (distance < .001 || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      this.applySurfaceProgress(target);
      return;
    }
    const duration = Math.max(90, 260 * distance);
    const startedAt = performance.now();
    const tick = now => {
      const elapsed = clampUnit((now - startedAt) / duration);
      this.applySurfaceProgress(mix(start, target, ease(elapsed)));
      if (elapsed < 1) this.surfaceFrame = requestAnimationFrame(tick);
    };
    this.surfaceFrame = requestAnimationFrame(tick);
  }

  isPanelOpen() { return this.panelOpen; }

  setPanelOpen(open, restoreFocus = false) {
    this.panelOpen = Boolean(open && this.root);
    if (!this.panelOpen) this.closeCustomSelects();
    if (!this.root) return;
    this.root.dataset.panelOpen = String(this.panelOpen);
    for (const [button, view] of [[this.gear, 'settings'], [this.pic, 'picture']]) {
      const open = this.panelOpen && this.view === view;
      button.dataset.open = String(open);
      button.setAttribute('aria-expanded', String(open));
    }
    this.panel.setAttribute('aria-hidden', String(!this.panelOpen));
    if (this.panelOpen) {
      this.panel.dataset.view = this.view;
      this.panel.setAttribute('aria-label', this.view === 'picture' ? 'Framegen picture' : 'Framegen settings');
      this.panel.querySelector('#fcPanelTitle').textContent = this.view === 'picture' ? 'Picture' : 'Framegen';
      this.panel.scrollTop = 0;
      this.applySurfaceProgress(this.surfaceProgress);
    }
    this.animateSurface(this.panelOpen);
    if (this.panelOpen) {
      this.syncPanel();
      this.updateStatus();
    } else if (restoreFocus) {
      (this.view === 'picture' ? this.pic : this.gear).focus({ preventScroll: true });
    }
  }

  setVisible(visible) {
    if (this.root) this.root.dataset.visible = String(Boolean(visible));
  }

  isVisible() { return this.root?.dataset.visible === 'true'; }

  blurFocus() {
    if (this.root?.contains(document.activeElement)) document.activeElement.blur();
  }

  pointInsideControls(x, y, videoRect) {
    if (!this.root) return false;
    return pointInsideRect(x, y, this.root.getBoundingClientRect()) && pointInsideRect(x, y, videoRect);
  }

  // ---------- placement ----------
  observeVideo(video) {
    if (!video || this.observedVideo === video) return;
    this.videoObserver?.disconnect();
    this.observedVideo = video;
    this.videoObserver?.observe(video);
  }

  schedulePlacement() {
    if (this.placementFrame || !this.isVisible()) return;
    this.placementFrame = requestAnimationFrame(() => {
      this.placementFrame = 0;
      if (!this.isVisible()) return;
      const video = this.host.activeVideo() || this.observedVideo;
      if (video?.isConnected) this.place(video.getBoundingClientRect(), video);
    });
  }

  resetGeometry() {
    this.placed = { left: -1, top: -1, width: -1, height: -1, center: -1 };
  }

  // The stage rests 43 px behind the video's left edge and slides into the picture on hover. Edges are rounded inward
  // on the physical-pixel grid: at a fractional DPR an integer CSS edge can still split a device pixel with the outside.
  place(r, video = null) {
    if (!this.root) return;
    this.observeVideo(video);
    const scale = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
    const ceilRaster = value => Math.ceil(value * scale) / scale;
    const floorRaster = value => Math.floor(value * scale) / scale;
    const left = Math.max(0, ceilRaster(r.left));
    const right = Math.min(innerWidth, floorRaster(r.right));
    const topBound = Math.max(0, ceilRaster(r.top));
    const bottomBound = Math.min(innerHeight, floorRaster(r.bottom));
    const width = Math.min(376, right - left);
    const height = Math.min(626, bottomBound - topBound);
    if (width < 80 || height < 2 * ISLAND_HALF) {
      this.setPanelOpen(false);
      this.setVisible(false);
      return;
    }
    if (this.panelOpen && (width < INLINE_SETTINGS_MIN_WIDTH || height < INLINE_SETTINGS_MIN_HEIGHT)) {
      this.setPanelOpen(false);
    }
    const preferredTop = Math.round(r.top + r.height / 2 - height / 2);
    const top = Math.max(topBound, Math.min(preferredTop, bottomBound - height));
    const center = Math.max(ISLAND_HALF, Math.min(height - ISLAND_HALF, Math.round(r.top + r.height / 2 - top)));
    const p = this.placed;
    if (left !== p.left || top !== p.top || width !== p.width || height !== p.height || center !== p.center) {
      this.placed = { left, top, width, height, center };
      this.width = width; this.height = height; this.islandCenter = center;
      this.root.style.left = left + 'px';
      this.root.style.top = top + 'px';
      this.root.style.width = width + 'px';
      this.root.style.height = height + 'px';
      this.root.style.setProperty('--fc-island-center', center + 'px');
      this.surface.setAttribute('viewBox', `0 0 ${width} ${height}`);
      this.applySurfaceProgress(this.surfaceProgress);
    }
    this.setVisible(true);
  }

  // ---------- the panel ----------
  buildPanel() {
    const panel = document.createElement('section');
    panel.className = 'fc-panel';
    panel.id = 'fcSettingsPanel';
    panel.setAttribute('aria-label', 'Framegen settings');
    panel.setAttribute('aria-hidden', 'true');
    // focusable by a press, so the focus stays inside the controls when the panel's own surface is clicked
    panel.tabIndex = -1;
    panel.dataset.view = this.view;
    panel.innerHTML = `
      <div class="fc-panel-head">
        <div class="fc-brand">
          <strong id="fcPanelTitle">Framegen</strong>
          <span class="fc-version">v${this.version}</span>
        </div>
        <a class="fc-icon-link" href="https://github.com/MONZikWasTaken/Framegen"
          target="_blank" rel="noopener noreferrer"
          aria-label="Open Framegen on GitHub">
          <svg width="17" height="17" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="${GITHUB_PATH}"/></svg>
        </a>
      </div>
      <div class="fc-view" data-view="picture">
      <div class="fc-looks" id="fcLooks" role="radiogroup" aria-label="Picture mode">
        ${Object.entries(PICTURE_MODES).map(([key, mode]) => `<button class="fc-look" type="button" role="radio"
          aria-checked="false" data-mode="${key}"><span class="fc-look-swatch"
          style="filter:${PICTURE_SWATCH_FILTERS[key]}"></span><span class="fc-look-name">${mode.label}</span></button>`).join('')}
      </div>
      <div class="fc-divider"></div>
      <div class="fc-row"><span>Sharpness</span>
        <div class="fc-seg" id="fcSharpness" role="radiogroup" aria-label="Sharpness">
          ${SHARPNESS_LABELS.map((label, value) => `<button type="button" role="radio" aria-checked="false"
            data-value="${value}">${label}</button>`).join('')}
        </div></div>
      <label class="fc-row"><span>Upscale<small>2× neural resolution boost</small></span>
        <input class="fc-sw" type="checkbox" id="fcSR"></label>
      <label class="fc-row"><span>HDR<small>Brighter highlights on HDR displays</small></span>
        <input class="fc-sw" type="checkbox" id="fcHDR"></label>
      </div>
      <div class="fc-view" data-view="settings">
      <div class="fc-row fc-profile-row"><span>Profile<small>Apply a saved setup</small></span>
        <select class="fc-sel fc-profile" id="fcProfile" disabled>
          <option>Loading profiles…</option>
        </select></div>
      <div class="fc-divider"></div>
      <label class="fc-row"><span>Frame generation<small>Create smoother motion</small></span>
        <input class="fc-sw" type="checkbox" id="fcFG"></label>
      <div class="fc-row"><span>Output rate<small>How playback is paced</small></span>
        <select class="fc-sel" id="fcFactor">
          <option value="auto">Auto · recommended</option>
          <option value="hz">Match display</option>
          <option value="target">Custom FPS</option>
          <option value="2">2× source</option><option value="3">3× source</option>
          <option value="4">4× source</option><option value="5">5× source</option>
          <option value="6">6× source</option>
        </select></div>
      <label class="fc-target-control">
        <span class="fc-target-head"><span><span id="fcRateSliderTitle">FPS limit</span>
          <small id="fcRateSliderHint">Common rates · Auto may run lower</small></span>
          <output id="fcTargetFpsValue" for="fcTargetFps">Unlimited</output></span>
        <input class="fc-target-slider" id="fcTargetFps" type="range" min="0" max="17"
          step="1" value="17" aria-label="FPS limit">
        <span class="fc-target-scale" id="fcRateSliderScale" aria-hidden="true">
          <span>15</span><span>60</span><span>120</span><span>240</span><span>∞</span>
        </span>
      </label>
      <div class="fc-row"><span>Quality<small>Detail against GPU load</small></span>
        <select class="fc-sel" id="fcRes">
          <option value="288">Low power</option><option value="360">Efficient</option>
          <option value="480">Balanced</option>
          <option value="720">High</option>
          <option value="1080">Ultra</option>
        </select></div>
      <div class="fc-divider"></div>
      <label class="fc-row"><span>FPS counter<small>Show frame rate and render time</small></span>
        <input class="fc-sw" type="checkbox" id="fcShowFps"></label>
      <label class="fc-row"><span>Watermark<small>Show the Framegen label on video</small></span>
        <input class="fc-sw" type="checkbox" id="fcWatermark"></label>
      <label class="fc-row"><span>Warnings<small>Show non-critical notices over video</small></span>
        <input class="fc-sw" type="checkbox" id="fcWarnings"></label>
      <pre class="fc-status" id="fcStatus"></pre>
      <button class="fc-open-settings" id="fcOpenSettings" type="button">
        <span>Advanced settings</span>
        <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
          <path fill="currentColor" d="m6 3 5 5-5 5-1.2-1.2L8.6 8 4.8 4.2 6 3Z"/>
        </svg>
      </button>
      </div>
      <a class="fc-engine-link" href="https://github.com/getflyhigh/flyhigh-engine" target="_blank" rel="noopener noreferrer">Powered by
        <b>FlyHigh engine</b></a>`;
    this.clip.appendChild(panel);
    this.panel = panel;
    this.enhanceCustomSelects(panel);
    const q = id => panel.querySelector(id);
    const set = values => this.settings.set(values);
    this.syncPanel();
    this.loadProfiles().catch(e => this.log('profiles', e));
    q('#fcOpenSettings').onclick = () => this.host.onOpenAdvanced();
    q('#fcProfile').onchange = async event => {
      const select = event.currentTarget;
      select.disabled = true;
      this.syncCustomSelect(select);
      try { await this.applyProfile(select.value); }
      catch (e) { this.log('apply profile', e); }
      finally {
        select.disabled = false;
        this.syncProfileSelection();
        this.syncCustomSelect(select);
      }
    };
    q('#fcFactor').onchange = e => set({ factor: e.currentTarget.value });
    const slider = q('#fcTargetFps');
    slider.oninput = () => this.updateRateSlider(slider);
    slider.onchange = () => this.settings.cfg.factor === 'auto'
      ? set({ fpsLimit: this.fpsLimitFromSlider(slider) })
      : set({ targetFps: slider.value });
    q('#fcFG').onchange = e => set({ fg: e.currentTarget.checked });
    q('#fcSR').onchange = e => set({ sr: e.currentTarget.checked });
    q('#fcHDR').onchange = e => set({ hdr: e.currentTarget.checked });
    q('#fcLooks').onclick = e => {
      const look = e.target.closest('.fc-look');
      if (look) set({ pictureMode: look.dataset.mode });
    };
    q('#fcSharpness').onclick = e => {
      const option = e.target.closest('button');
      if (option) set({ sharpness: Number(option.dataset.value) });
    };
    q('#fcShowFps').onchange = e => set({ showFps: e.currentTarget.checked });
    q('#fcWatermark').onchange = e => set({ showWatermark: e.currentTarget.checked });
    q('#fcWarnings').onchange = e => set({ showWarnings: e.currentTarget.checked });
    q('#fcRes').onchange = e => set({ res: Number(e.currentTarget.value) });
  }

  syncPanel() {
    const panel = this.panel;
    if (!panel) return;
    const cfg = this.settings.cfg;
    for (const look of panel.querySelectorAll('.fc-look')) {
      look.setAttribute('aria-checked', String(look.dataset.mode === cfg.pictureMode));
    }
    for (const option of panel.querySelectorAll('#fcSharpness button')) {
      option.setAttribute('aria-checked', String(Number(option.dataset.value) === cfg.sharpness));
    }
    const factor = panel.querySelector('#fcFactor');
    const resolution = panel.querySelector('#fcRes');
    factor.value = String(cfg.factor);
    this.syncCustomSelect(factor);
    this.syncRateSlider();
    resolution.value = String(cfg.res);
    this.syncCustomSelect(resolution);
    panel.querySelector('#fcFG').checked = cfg.fg;
    panel.querySelector('#fcSR').checked = cfg.sr;
    panel.querySelector('#fcShowFps').checked = cfg.showFps;
    panel.querySelector('#fcWatermark').checked = cfg.showWatermark;
    panel.querySelector('#fcWarnings').checked = cfg.showWarnings;
    const hdr = panel.querySelector('#fcHDR');
    hdr.checked = cfg.hdr;
    if (!this.host.hdrSupported()) { hdr.disabled = true; hdr.style.opacity = '.35'; }
    this.syncProfileSelection();
  }

  updateStatus() {
    const el = this.panel?.querySelector('#fcStatus');
    if (!el) return;
    el.textContent = this.host.statusLines().join('\n');
  }

  // ---------- the rate slider ----------
  fpsLimitFromSlider(input) {
    const presets = Profiles.FPS_LIMIT_PRESETS;
    const index = Math.max(0, Math.min(presets.length - 1, Math.round(Number(input.value) || 0)));
    return presets[index];
  }

  updateRateSlider(input = this.panel?.querySelector('#fcTargetFps')) {
    if (!input || !this.panel) return;
    const cfg = this.settings.cfg;
    const value = cfg.factor === 'auto'
      ? this.fpsLimitFromSlider(input)
      : Profiles.sanitizeTargetFps(input.value, cfg.targetFps);
    const output = this.panel.querySelector('#fcTargetFpsValue');
    const label = value === null ? 'Unlimited' : `${Number(value.toFixed(2))} FPS`;
    if (output) output.value = label;
    input.setAttribute('aria-valuetext', label);
    const min = Number.isFinite(Number(input.min)) ? Number(input.min) : 2;
    const max = Number.isFinite(Number(input.max)) ? Number(input.max) : 1000;
    const ratio = Math.max(0, Math.min(1, (Number(input.value) - min) / Math.max(1, max - min)));
    const thumbOffset = Number((7.5 * (1 - 2 * ratio)).toFixed(3));
    input.style.setProperty('--fc-fill', `calc(${ratio * 100}% + ${thumbOffset}px)`);
  }

  renderRateScale(scale, marks) {
    scale.replaceChildren(...marks.map(({ label, position }) => {
      const mark = document.createElement('span');
      mark.textContent = label;
      mark.style.setProperty('--fc-mark-position', `${position}%`);
      mark.style.setProperty('--fc-mark-offset', `${Number((7.5 * (1 - 2 * position / 100)).toFixed(3))}px`);
      return mark;
    }));
  }

  syncRateSlider() {
    const input = this.panel?.querySelector('#fcTargetFps');
    if (!input) return;
    const cfg = this.settings.cfg;
    const control = input.closest('.fc-target-control');
    const visible = cfg.factor === 'auto' || cfg.factor === 'target';
    control.hidden = !visible;
    if (!visible) return;
    const title = this.panel.querySelector('#fcRateSliderTitle');
    const hint = this.panel.querySelector('#fcRateSliderHint');
    const scale = this.panel.querySelector('#fcRateSliderScale');
    const presets = Profiles.FPS_LIMIT_PRESETS;
    if (cfg.factor === 'auto') {
      title.textContent = 'FPS limit';
      hint.textContent = 'Common rates · Auto may run lower';
      input.min = '0';
      input.max = String(presets.length - 1);
      input.step = '1';
      input.value = String(Profiles.fpsLimitPresetIndex(cfg.fpsLimit));
      input.setAttribute('aria-label', 'FPS limit');
      this.renderRateScale(scale, [15, 60, 120, 240, null].map(value => ({
        label: value === null ? '∞' : String(value),
        position: Profiles.fpsLimitPresetIndex(value) / (presets.length - 1) * 100,
      })));
    } else {
      title.textContent = 'Custom FPS';
      hint.textContent = 'Exact target · minimum 2× source';
      input.min = '2';
      input.max = '1000';
      input.step = '0.01';
      input.value = String(cfg.targetFps);
      input.setAttribute('aria-label', 'Custom FPS');
      this.renderRateScale(scale, [
        { label: '2', position: 0 },
        { label: '500', position: 49.9 },
        { label: '1000', position: 100 },
      ]);
    }
    this.updateRateSlider(input);
  }

  // ---------- profiles ----------
  syncProfileSelection() {
    const select = this.panel?.querySelector('#fcProfile');
    if (!select || !this.profileStore) return;
    try { select.value = Profiles.findMatchingProfileId(this.profileStore, this.settings.cfg) || ''; }
    catch { select.value = ''; }
    this.syncCustomSelect(select);
  }

  renderProfiles() {
    const select = this.panel?.querySelector('#fcProfile');
    if (!select || !this.profileStore) return;
    const current = document.createElement('option');
    current.value = '';
    current.textContent = 'Current settings';
    const children = [current];
    const profiles = Profiles.profileList(this.profileStore);
    if (profiles.length) {
      const group = document.createElement('optgroup');
      group.label = 'My profiles';
      for (const profile of profiles) {
        const option = document.createElement('option');
        option.value = profile.id;
        option.textContent = profile.name;
        group.append(option);
      }
      children.push(group);
    }
    select.replaceChildren(...children);
    select.disabled = false;
    this.rebuildCustomSelect(select);
    this.syncProfileSelection();
  }

  async loadProfiles(rawStore) {
    if (!this.panel) return;
    const select = this.panel.querySelector('#fcProfile');
    try {
      let candidate = rawStore;
      if (arguments.length === 0) {
        const snapshot = await chrome.storage.local.get(Profiles.STORE_KEY);
        candidate = snapshot[Profiles.STORE_KEY];
      }
      const loaded = Profiles.loadStore(candidate, this.settings.cfg);
      this.profileStore = loaded.store;
      this.renderProfiles();
      if (loaded.needsWrite) await chrome.storage.local.set({ [Profiles.STORE_KEY]: loaded.store });
    } catch (error) {
      this.profileStore = null;
      if (select) {
        const unavailable = document.createElement('option');
        unavailable.textContent = 'Profiles unavailable';
        select.replaceChildren(unavailable);
        select.disabled = true;
        this.rebuildCustomSelect(select);
      }
      this.log('profiles', error);
    }
  }

  async applyProfile(profileId) {
    if (!profileId || !this.profileStore) { this.syncProfileSelection(); return; }
    const profile = Profiles.getProfile(this.profileStore, profileId);
    if (!profile) { this.syncProfileSelection(); return; }
    const nextStore = Profiles.setLastAppliedProfile(this.profileStore, profile.id);
    this.profileStore = nextStore;
    await chrome.storage.local.set(Profiles.toStoragePayload(nextStore, profile.settings));
  }

  // ---------- custom selects (the native popup cannot be styled and escapes the clip) ----------
  closeCustomSelect(component, restoreFocus = false) {
    if (!component || component.root.dataset.open !== 'true') return;
    component.root.dataset.open = 'false';
    component.trigger.setAttribute('aria-expanded', 'false');
    component.menu.hidden = true;
    if (restoreFocus) component.trigger.focus({ preventScroll: true });
  }

  closeCustomSelects(except = null) {
    for (const component of this.customSelects.values()) {
      if (component !== except) this.closeCustomSelect(component);
    }
  }

  focusCustomOption(component, edge = 'selected') {
    const options = [...component.menu.querySelectorAll('.fc-select-option:not(:disabled)')];
    if (!options.length) return;
    const selectedIndex = Math.max(0, options.findIndex(option => option.getAttribute('aria-selected') === 'true'));
    const index = edge === 'first' ? 0 : edge === 'last' ? options.length - 1 : selectedIndex;
    options[index].focus({ preventScroll: true });
  }

  openCustomSelect(component, focusOption = false, edge = 'selected') {
    if (!component || component.trigger.disabled) return;
    this.closeCustomSelects(component);
    component.root.dataset.open = 'true';
    component.trigger.setAttribute('aria-expanded', 'true');
    component.menu.hidden = false;
    const rootRect = component.root.getBoundingClientRect();
    const panelRect = this.panel?.getBoundingClientRect() || document.documentElement.getBoundingClientRect();
    const menuHeight = Math.min(250, component.menu.scrollHeight);
    const below = Math.max(0, panelRect.bottom - rootRect.bottom - 4);
    const above = Math.max(0, rootRect.top - panelRect.top - 4);
    const opensUp = below < menuHeight && above > below;
    component.root.dataset.placement = opensUp ? 'up' : 'down';
    component.menu.style.maxHeight = Math.max(72, Math.min(250, opensUp ? above : below)) + 'px';
    if (focusOption) requestAnimationFrame(() => this.focusCustomOption(component, edge));
  }

  syncCustomSelect(select) {
    const component = this.customSelects.get(select);
    if (!component) return;
    const selected = select.selectedOptions[0] || select.options[0];
    component.value.textContent = selected?.textContent || '';
    component.trigger.disabled = select.disabled;
    component.trigger.setAttribute('aria-disabled', String(select.disabled));
    component.menu.querySelectorAll('.fc-select-option').forEach(item => {
      item.setAttribute('aria-selected', String(item.dataset.value === select.value));
    });
    if (select.disabled) this.closeCustomSelect(component);
  }

  chooseCustomSelectOption(component, item) {
    if (!item || item.disabled || component.select.disabled) return;
    const changed = component.select.value !== item.dataset.value;
    component.select.value = item.dataset.value;
    this.syncCustomSelect(component.select);
    this.closeCustomSelect(component, true);
    if (changed) component.select.dispatchEvent(new Event('change', { bubbles: true }));
  }

  appendCustomSelectOption(component, option, parent, index) {
    const item = document.createElement('button');
    item.className = 'fc-select-option';
    item.type = 'button';
    item.id = `${component.menu.id}Option${index}`;
    item.dataset.value = option.value;
    item.setAttribute('role', 'option');
    item.setAttribute('aria-selected', String(option.selected));
    item.disabled = option.disabled;
    item.tabIndex = -1;
    item.textContent = option.textContent;
    parent.append(item);
  }

  rebuildCustomSelect(select) {
    const component = this.customSelects.get(select);
    if (!component) return;
    this.closeCustomSelect(component);
    component.menu.replaceChildren();
    let optionIndex = 0, groupIndex = 0;
    for (const child of select.children) {
      if (child instanceof HTMLOptGroupElement) {
        const group = document.createElement('div');
        const groupLabel = document.createElement('div');
        group.className = 'fc-select-options-group';
        group.setAttribute('role', 'group');
        groupLabel.className = 'fc-select-group';
        groupLabel.id = `${component.menu.id}Group${groupIndex++}`;
        groupLabel.textContent = child.label;
        group.setAttribute('aria-labelledby', groupLabel.id);
        group.append(groupLabel);
        for (const option of child.children) this.appendCustomSelectOption(component, option, group, optionIndex++);
        component.menu.append(group);
      } else if (child instanceof HTMLOptionElement) {
        this.appendCustomSelectOption(component, child, component.menu, optionIndex++);
      }
    }
    this.syncCustomSelect(select);
  }

  enhanceCustomSelect(select, index) {
    if (this.customSelects.has(select)) return;
    const label = select.closest('.fc-row')?.querySelector(':scope > span')?.childNodes[0]?.textContent?.trim() || 'Choose an option';
    const root = document.createElement('div');
    const trigger = document.createElement('button');
    const value = document.createElement('span');
    const menu = document.createElement('div');
    root.className = 'fc-select';
    root.dataset.open = 'false';
    trigger.className = 'fc-select-trigger';
    trigger.type = 'button';
    trigger.setAttribute('role', 'combobox');
    trigger.setAttribute('aria-label', label);
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');
    const menuId = `__framegenSelectMenu${index}`;
    trigger.setAttribute('aria-controls', menuId);
    value.className = 'fc-select-value';
    trigger.append(value);
    trigger.insertAdjacentHTML('beforeend',
      '<svg class="fc-select-chevron" viewBox="0 0 12 12" aria-hidden="true">'
      + '<path d="m3 4.5 3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.4" '
      + 'stroke-linecap="round" stroke-linejoin="round"/></svg>');
    menu.className = 'fc-select-menu';
    menu.id = menuId;
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-label', label);
    menu.hidden = true;
    select.before(root);
    root.append(select, trigger, menu);
    select.dataset.enhanced = 'true';
    select.setAttribute('aria-hidden', 'true');
    select.tabIndex = -1;
    const component = { root, select, trigger, value, menu };
    this.customSelects.set(select, component);
    this.rebuildCustomSelect(select);

    trigger.addEventListener('click', event => {
      event.stopPropagation();
      if (root.dataset.open === 'true') this.closeCustomSelect(component);
      else this.openCustomSelect(component);
    });
    trigger.addEventListener('keydown', event => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); event.stopPropagation();
        this.openCustomSelect(component, true, event.key === 'ArrowUp' ? 'last' : 'selected');
      } else if (event.key === 'Home' || event.key === 'End') {
        event.preventDefault(); event.stopPropagation();
        this.openCustomSelect(component, true, event.key === 'Home' ? 'first' : 'last');
      } else if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        this.closeCustomSelect(component);
      }
    });
    menu.addEventListener('click', event => {
      event.stopPropagation();
      this.chooseCustomSelectOption(component, event.target.closest('.fc-select-option'));
    });
    menu.addEventListener('keydown', event => {
      const options = [...menu.querySelectorAll('.fc-select-option:not(:disabled)')];
      const current = options.indexOf(document.activeElement);
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault(); event.stopPropagation();
        const step = event.key === 'ArrowDown' ? 1 : -1;
        options[(current + step + options.length) % options.length]?.focus({ preventScroll: true });
      } else if (event.key === 'Home' || event.key === 'End') {
        event.preventDefault(); event.stopPropagation();
        options[event.key === 'Home' ? 0 : options.length - 1]?.focus({ preventScroll: true });
      } else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault(); event.stopPropagation();
        this.chooseCustomSelectOption(component, document.activeElement.closest('.fc-select-option'));
      } else if (event.key === 'Escape') {
        event.preventDefault(); event.stopPropagation();
        this.closeCustomSelect(component, true);
      } else if (event.key === 'Tab') {
        this.closeCustomSelect(component);
      } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
        const query = event.key.toLocaleLowerCase();
        const match = options.find(option => option.textContent.trim().toLocaleLowerCase().startsWith(query));
        if (match) {
          event.preventDefault(); event.stopPropagation();
          match.focus({ preventScroll: true });
        }
      }
    });
    select.addEventListener('change', () => this.syncCustomSelect(select));
  }

  enhanceCustomSelects(root) {
    [...root.querySelectorAll('.fc-sel')].forEach((select, index) => this.enhanceCustomSelect(select, index));
    if (this.documentHandlers) return;
    this.documentHandlers = true;
    document.addEventListener('pointerdown', event => {
      for (const component of this.customSelects.values()) {
        if (!component.root.contains(event.target)) this.closeCustomSelect(component);
      }
    });
  }
}

export function pointInsideRect(x, y, rect) {
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}
