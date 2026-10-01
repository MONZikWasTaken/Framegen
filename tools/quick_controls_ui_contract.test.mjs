import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const [content, optionsHtml, popupHtml] = await Promise.all([
  readFile(new URL('../extension/content.js', import.meta.url), 'utf8'),
  readFile(new URL('../extension/options.html', import.meta.url), 'utf8'),
  readFile(new URL('../extension/popup.html', import.meta.url), 'utf8'),
]);

function functionBody(name, nextName) {
  const start = content.indexOf(`function ${name}(`);
  const end = content.indexOf(`function ${nextName}(`, start + 1);
  assert.notEqual(start, -1, `${name} must exist`);
  assert.notEqual(end, -1, `${nextName} must follow ${name}`);
  return content.slice(start, end);
}

test('quick controls mount as one morphing surface clipped inside the video', () => {
  assert.match(content, /className = 'fc-controls-root'/);
  assert.match(content, /class="fc-controls-stage"[\s\S]*class="fc-control-surface"[\s\S]*class="fc-control-rail"[\s\S]*class="fc-settings-clip"/);
  assert.match(content, /controlsRail\.append\(btn, gear\)/);
  assert.match(content, /controlsClip\.appendChild\(panel\)/);
  assert.match(content, /document\.body\.append\(controlsRoot, wm, hud\)/);
  assert.match(content, /controlsShape\.setAttribute\('d', path\)/);
  assert.match(content, /controlsClip\.style\.clipPath = `path\("\$\{path\}"\)`/);
  assert.match(content, /prefers-reduced-motion: reduce/);
  assert.match(content, /uiScan = -1e9, mmLast = -1e9/);
  assert.match(content, /const ceilRaster = value => Math\.ceil\(value \* rasterScale\) \/ rasterScale/);
  assert.match(content, /const floorRaster = value => Math\.floor\(value \* rasterScale\) \/ rasterScale/);
  assert.match(content, /const left = Math\.max\(0, ceilRaster\(r\.left\)\)/);
  assert.match(content, /const right = Math\.min\(innerWidth, floorRaster\(r\.right\)\)/);
  assert.match(content, /controlsRoot\.style\.left = left \+ 'px'/);
  assert.match(content, /overflow:hidden!important/);
  assert.match(content, /\.fc-controls-stage[\s\S]*transform:translateX\(-43px\)/);
  assert.match(content, /\.fc-controls-root\[data-visible="true"\] \.fc-controls-stage[\s\S]*transform:translateX\(0\)/);
  assert.doesNotMatch(content, /playerDockRect|findPlayerDockElement|playerDockElements/);
  assert.match(content, /function pointInsideControls\(x, y, videoRect\)/);
  assert.match(content, /const insideControls = pointInsideControls\(e\.clientX, e\.clientY, r\)/);
  assert.match(content, /new ResizeObserver\(\(\) => scheduleControlsPlacement\(\)\)/);
  assert.match(content, /window\.addEventListener\('resize', scheduleControlsPlacement/);
  assert.match(content, /window\.visualViewport\?\.addEventListener\('resize', scheduleControlsPlacement/);
  assert.match(content, /panelOpen && \(width < INLINE_SETTINGS_MIN_WIDTH \|\| height < INLINE_SETTINGS_MIN_HEIGHT\)[\s\S]*setPanelOpen\(false\)/);
  assert.doesNotMatch(content, /const insideControls = panelOpen && controlsRoot/);
  assert.doesNotMatch(content, /document\.body\.appendChild\((?:btn|gear|panel)\)/);
  assert.doesNotMatch(content, /(?:btn|gear|panel)\.style\.(?:display|left|top|background)/);
});

test('runtime toggle and settings disclosure retain separate semantics', () => {
  const panelDisclosure = functionBody('setPanelOpen', 'setControlsVisible');
  assert.match(content, /btn\.onclick = toggleFC/);
  assert.match(content, /INLINE_SETTINGS_MIN_WIDTH = 300, INLINE_SETTINGS_MIN_HEIGHT = 180/);
  assert.match(content, /gear\.onclick = \(\) => \{[\s\S]*controlsWidth < INLINE_SETTINGS_MIN_WIDTH[\s\S]*controlsHeight < INLINE_SETTINGS_MIN_HEIGHT[\s\S]*openAdvancedSettings\(\)[\s\S]*setPanelOpen\(!isPanelOpen\(\)\)/);
  assert.doesNotMatch(panelDisclosure, /cfg\.fg|setFrameGeneration|toggleFC|\brunning\b/);
  assert.match(content, /btn\.dataset\.active = 'true'[\s\S]*aria-pressed', 'true'/);
  assert.match(content, /btn\.dataset\.active = 'false'[\s\S]*aria-pressed', 'false'/);
});

test('custom selects keep native values, keyboard support, and dynamic profile sync', () => {
  assert.match(content, /select\.before\(root\)/);
  assert.match(content, /root\.append\(select, trigger, menu\)/);
  assert.match(content, /select\.dataset\.enhanced = 'true'/);
  assert.match(content, /dispatchEvent\(new Event\('change', \{ bubbles: true \}\)\)/);
  for (const key of ['ArrowDown', 'ArrowUp', 'Home', 'End', 'Enter', 'Escape', 'Tab']) {
    assert.match(content, new RegExp(`event\\.key === '${key}'`));
  }
  assert.match(content, /renderPanelProfiles\(\)[\s\S]*rebuildCustomSelect\(select\)/);
  assert.match(content, /syncPanelProfileSelection\(\)[\s\S]*syncCustomSelect\(select\)/);
  assert.match(content, /\.fc-sel\[data-enhanced="true"\]\{display:none!important\}/);
  assert.doesNotMatch(content, /appearance:base-select|::picker\(|backdrop-filter:/);
});

test('twitch player controls stay above the canvas without shifting the bar', () => {
  const rule = functionBody('keepTwitchControlsAboveOverlay', 'ensureSubtitleLayer');
  assert.match(rule, /\(\^\|\\\.\)twitch\\\.tv\$/);
  assert.match(rule, /data-a-target="player-controls"/);
  assert.match(rule, /\.player-controls/);
  assert.match(rule, /\.top-bar/);
  assert.match(rule, /\.video-player__overlay/);
  assert.match(rule, /z-index:\s*100\s*!important/);
  assert.doesNotMatch(rule, /textContent = `[\s\S]*position\s*:/);
  assert.match(content, /keepTwitchControlsAboveOverlay\(\)/);
});

test('quick settings expose no native title tooltips or obsolete brand dot', () => {
  for (const source of [content, optionsHtml, popupHtml]) {
    assert.doesNotMatch(source, /\btitle\s*=\s*["']/);
  }
  assert.doesNotMatch(content, /fc-brand-dot/);
  assert.match(content, /aria-label="Open Framegen on GitHub"/);
});
