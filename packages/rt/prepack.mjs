// single source of truth is web/rt/ - copy the runtime in at pack time
import { copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', '..', 'web', 'rt');
for (const f of ['rt.js', 'sr.js']) copyFileSync(join(src, f), join(here, f));
console.log('runtime copied from web/rt');
