// Cross-origin video taints the pixel path (a SecurityError on the first read). The extension's DNR rule puts
// Access-Control-Allow-Origin: * on media responses, so reloading the element in CORS mode makes it readable: one
// reload, playback position preserved, reverted when the CDN refuses.
export async function makeReadable(video, log) {
  if (video.crossOrigin === 'anonymous') throw new Error('video unreadable even with CORS');
  const t = video.currentTime, playing = !video.paused;
  video.crossOrigin = 'anonymous';
  video.load();
  try {
    await new Promise((resolve, reject) => {
      const ok = () => { cleanup(); resolve(); };
      const bad = () => { cleanup(); reject(new Error('CDN refuses CORS for this video')); };
      const timer = setTimeout(bad, 8000);
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener('loadeddata', ok);
        video.removeEventListener('error', bad);
      };
      video.addEventListener('loadeddata', ok);
      video.addEventListener('error', bad);
    });
  } catch (e) {
    video.removeAttribute('crossorigin');
    video.load();
    video.currentTime = t;
    if (playing) video.play().catch(() => {});
    throw e;
  }
  video.currentTime = t;
  if (playing) video.play().catch(() => {});
  // loadeddata is not a decoded frame: a copy right away throws "no back resource"
  await new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    video.requestVideoFrameCallback(() => finish());
    setTimeout(finish, 1500);
  });
  log('video reloaded with CORS, pixels readable now');
}

export const isCorsError = e => e?.name === 'SecurityError' || String(e).includes('cross-origin');
