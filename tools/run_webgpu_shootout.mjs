#!/usr/bin/env node
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const require = createRequire('/Users/nimajafari/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const { chromium } = require('playwright-core');
const root = path.resolve(import.meta.dirname, '..');
const scenes = JSON.parse(await readFile(path.join(root, 'benchmarks/vfi-shootout/scenes.json'), 'utf8'));
const res = process.argv.includes('--res') ? process.argv[process.argv.indexOf('--res') + 1] : '480';
const outputRoot = path.join(root, '.bench/results/framegen-v7s-' + res);
const port = res === '720' ? 4174 : 4173;

await mkdir(outputRoot, { recursive: true });
const server = spawn('/opt/homebrew/bin/python3', ['-m', 'http.server', String(port), '--directory', root], { stdio: 'ignore' });
for (let attempt = 0; attempt < 20; attempt++) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/web/shootout.html`);
    if (response.ok) break;
  } catch {}
  await new Promise((resolve) => setTimeout(resolve, 100));
}
const browser = await chromium.launch({
  executablePath: '/Applications/Comet.app/Contents/MacOS/Comet',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal'],
});

try {
  const records = [];
  for (const scene of scenes) {
    const page = await browser.newPage();
    const i0 = new URL(scene.i0, `http://127.0.0.1:${port}/`).toString();
    const i1 = new URL(scene.i1, `http://127.0.0.1:${port}/`).toString();
    await page.goto(`http://127.0.0.1:${port}/web/shootout.html?res=${res}&i0=${encodeURIComponent(i0)}&i1=${encodeURIComponent(i1)}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__shootoutResult !== undefined, null, { timeout: 120000 });
    const result = await page.evaluate(() => window.__shootoutResult);
    await page.close();
    if (result.error) throw new Error(`${scene.id}: ${result.error}\n${result.stack || ''}`);
    const image = Buffer.from(result.imageData.split(',', 2)[1], 'base64');
    await mkdir(path.join(outputRoot, scene.id), { recursive: true });
    await writeFile(path.join(outputRoot, scene.id, 'framegen-v7s.png'), image);
    delete result.imageData;
    records.push({ scene: scene.id, ...result });
    console.log(`${scene.id}: ${result.ms.toFixed(2)}ms at ${result.rung.join('x')}`);
  }
  await writeFile(path.join(outputRoot, 'timings.json'), JSON.stringify(records, null, 2) + '\n');
} finally {
  await browser.close();
  server.kill();
}
