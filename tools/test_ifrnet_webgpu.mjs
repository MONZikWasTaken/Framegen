#!/usr/bin/env node
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';

const require = createRequire('/Users/nimajafari/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const { chromium } = require('playwright-core');
const root = path.resolve(import.meta.dirname, '..');
const port = 4180;
const image = process.argv[2] || 'demo/I0_0.png';
const image1 = process.argv[3] || image;
const resolution = process.argv.includes('--res') ? process.argv[process.argv.indexOf('--res') + 1] : '848x480';
const server = spawn('/opt/homebrew/bin/python3', ['-m', 'http.server', String(port), '--directory', root], { stdio: 'ignore' });
const browser = await chromium.launch({ executablePath: '/Applications/Comet.app/Contents/MacOS/Comet', headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal'] });
try {
  const page = await browser.newPage();
  const source = new URL(image, `http://127.0.0.1:${port}/`).toString();
  const source1 = new URL(image1, `http://127.0.0.1:${port}/`).toString();
  await page.goto(`http://127.0.0.1:${port}/web/ifrnet_test.html?res=${encodeURIComponent(resolution)}&i0=${encodeURIComponent(source)}&i1=${encodeURIComponent(source1)}`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__ifrnetResult !== undefined, null, { timeout: 600000 });
  const result = await page.evaluate(() => window.__ifrnetResult);
  if (result.error || !result.ok) throw new Error(`${result.error || 'IFRNet output is unexpectedly black'} (${JSON.stringify(result)})\n${result.stack || ''}`);
  await writeFile(path.join(root, '.bench', 'ifrnet-webgpu.png'), Buffer.from(result.imageData.split(',', 2)[1], 'base64'));
  delete result.imageData;
  console.log(JSON.stringify(result));
  console.log('IFRNet WebGPU probe passed');
} finally {
  await browser.close();
  server.kill();
}
