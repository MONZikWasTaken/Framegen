import assert from 'node:assert/strict';
import test from 'node:test';

async function siteControlsOkOn(hostname) {
  globalThis.location = { hostname, href: `https://${hostname}/` };
  const sites = await import(`../extension/src/sites.js?host=${hostname}`);
  return sites.SITE_CONTROLS_OK;
}

test('players whose own controls stay above the canvas get no second control bar', async () => {
  for (const host of ['kick.com', 'www.kick.com', 'www.twitch.tv', 'www.youtube.com', 'youtu.be', 'vimeo.com']) {
    assert.equal(await siteControlsOkOn(host), true, host);
  }
  for (const host of ['notkick.com', 'kick.com.example.org', 'jut.su']) {
    assert.equal(await siteControlsOkOn(host), false, host);
  }
});
