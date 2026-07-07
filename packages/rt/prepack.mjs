// single source of truth: runtime from web/rt, weights from assets - copied at pack time
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
for (const f of ['rt.js', 'sr.js']) copyFileSync(join(root, 'web', 'rt', f), join(here, f));
mkdirSync(join(here, 'weights'), { recursive: true });
for (const f of ['rt_v7s.bin', 'rt_v7s.json', 'rt_sr.bin', 'rt_sr.json']) {
  copyFileSync(join(root, 'assets', f), join(here, 'weights', f));
}
console.log('runtime + weights copied');
