// Framegen content script: real-time frame generation for any <video> on the page, on the FlyHigh engine (WebGPU,
// weights bundled). The pipeline lives in src/, loaded as ES modules; this file only guards against a second run.
// DRM (EME) video decodes to black frames in every extension: nothing here can change that.
(() => {
  'use strict';
  if (window.__framegen) return;
  window.__framegen = true;
  import(chrome.runtime.getURL('src/main.js'))
    .then(m => m.boot())
    .catch(e => console.error('[framegen] failed to start', e));
})();
