// Monochrome SVG icons (Feather-style) for the in-page UI.
const ICONS = {
  play: '<path d="M8 5.5v13a.5.5 0 0 0 .77.42l10.2-6.5a.5.5 0 0 0 0-.84L8.77 5.08A.5.5 0 0 0 8 5.5z" fill="currentColor"/>',
  pause: '<rect x="7" y="5" width="3.4" height="14" rx="1" fill="currentColor"/><rect x="13.6" y="5" width="3.4" height="14" rx="1" fill="currentColor"/>',
  vol: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
    + '<path d="M15 8.6a5 5 0 0 1 0 6.8M17.7 6a9 9 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  volX: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
    + '<path d="m15.5 9.5 5 5m0-5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  full: '<path d="M8.5 3.5H5a1.5 1.5 0 0 0-1.5 1.5v3.5m17 0V5A1.5 1.5 0 0 0 19 3.5h-3.5m0 17H19a1.5 1.5 0 0 0 1.5-1.5v-3.5m-17 0V19A1.5 1.5 0 0 0 5 20.5h3.5"'
    + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  gear: '<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1.5 14H7m2-6h6m2.5 8H21"'
    + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
};

export const svgIcon = (name, size = 16) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

export const GITHUB_PATH = 'M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.78 2.72 1.27 3.38.97.1-.75.4-1.27.74-1.56-2.57-.3-5.27-1.29-5.27-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.16 1.18a10.96 10.96 0 0 1 5.75 0c2.19-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.41-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z';

export const FRAMEGEN_MARK = `<svg class="fc-mark" viewBox="0 0 28 20" aria-hidden="true">
  <path d="M3.5 16.5v-13h7.25M3.5 9.75h6M24 6.15c-.9-1.75-2.4-2.65-4.45-2.65-3.6 0-5.85 2.75-5.85 6.5s2.25 6.5 5.9 6.5c1.9 0 3.45-.65 4.5-1.9v-4h-4.65"
    fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="square" stroke-linejoin="miter"/>
</svg>`;
