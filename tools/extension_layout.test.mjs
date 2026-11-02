import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const extension = fileURLToPath(new URL('../extension', import.meta.url));
const manifest = JSON.parse(readFileSync(join(extension, 'manifest.json'), 'utf8'));

function modules(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...modules(p));
    else if (name.endsWith('.js')) out.push(p);
  }
  return out;
}

test('every relative import in src resolves to a file', () => {
  for (const file of modules(join(extension, 'src'))) {
    const text = readFileSync(file, 'utf8');
    for (const match of text.matchAll(/from '(\.[^']+)'/g)) {
      assert.ok(existsSync(join(dirname(file), match[1])), `${file} imports a missing ${match[1]}`);
    }
  }
});

test('the content script loads the modules from web-accessible paths', () => {
  const resources = manifest.web_accessible_resources.flatMap(r => r.resources);
  for (const prefix of ['src/*', 'engine/*', 'rt/*', 'assets/*']) assert.ok(resources.includes(prefix), prefix);
  const content = readFileSync(join(extension, 'content.js'), 'utf8');
  assert.match(content, /import\(chrome\.runtime\.getURL\('src\/main\.js'\)\)/);
  for (const url of readFileSync(join(extension, 'src/player.js'), 'utf8').matchAll(/url\('([^']+)'\)/g)) {
    assert.ok(existsSync(join(extension, url[1])), `player.js refers to a missing ${url[1]}`);
  }
});

test('the engine payload is the module, its adapters and one model', () => {
  const names = readdirSync(join(extension, 'engine')).sort();
  assert.deepEqual(names, ['WEIGHTS_LICENSE.md', 'adapters.js', 'flyhigh_web.js', 'flyhigh_web_bg.wasm', 'flyhighfg.bin', 'flyhighfg.json', 'graph.json']);
  const wasm = readFileSync(join(extension, 'engine/flyhigh_web_bg.wasm'));
  assert.deepEqual([...wasm.subarray(0, 4)], [0, 0x61, 0x73, 0x6d]);
  const manifestJson = JSON.parse(readFileSync(join(extension, 'engine/flyhighfg.json'), 'utf8'));
  let end = 0;
  for (const t of Object.values(manifestJson.tensors)) end = Math.max(end, t.offset + t.shape.reduce((a, b) => a * b, 1));
  assert.equal(statSync(join(extension, 'engine/flyhighfg.bin')).size, end * 2);
  assert.match(readFileSync(join(extension, 'engine/flyhigh_web.js'), 'utf8'), /class FhPacer|FhPacer/);
});

test('the upscaler is self-contained: sr.js, its conv kernels and its weights', () => {
  assert.deepEqual(readdirSync(join(extension, 'rt')).sort(), ['conv.js', 'sr.js']);
  assert.match(readFileSync(join(extension, 'rt/sr.js'), 'utf8'), /from '\.\/conv\.js'/);
  assert.deepEqual(readdirSync(join(extension, 'assets')).sort(), ['rt_sr.bin', 'rt_sr.json']);
});
