// Runtime sources live in web/rt; release weights live in the tracked extension bundle.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
for (const f of ['rt.js', 'sr.js']) copyFileSync(join(root, 'web', 'rt', f), join(here, f));
mkdirSync(join(here, 'weights'), { recursive: true });
for (const f of ['rt_v7s.bin', 'rt_v7s.json', 'rt_sr.bin', 'rt_sr.json']) {
  copyFileSync(join(root, 'extension', 'assets', f), join(here, 'weights', f));
}
console.log('runtime + weights copied');
