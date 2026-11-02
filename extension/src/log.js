// Console logging with the prefix every message carries, and the error list the bench bridge reports.
export const errors = [];

export function log(...parts) {
  console.log('[framegen]', ...parts);
  const message = parts.map(v => v instanceof Error ? (v.stack || v.message) : String(v)).join(' ');
  if (parts.some(v => v instanceof Error)
      || /(^|\s)(error|failed|skipped|lost|unavailable)(\s|$)/i.test(message)) {
    if (errors.length < 100) errors.push({ atMs: performance.now(), message });
  }
}
