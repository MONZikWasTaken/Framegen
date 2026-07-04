# Framecast Privacy Policy

Effective date: 2026-07-04

Framecast is a browser extension that performs neural frame interpolation of
videos locally, on your GPU.

## What we collect

Nothing.

Framecast has no servers and no analytics. It does not collect, transmit,
sell, or share any data - no browsing history, no video content, no
identifiers, no telemetry of any kind.

## What stays on your device

- Your settings (quality, interpolation factor, HUD visibility, etc.) are
  saved with the browser's extension storage (`chrome.storage.local`).
- A one-time GPU kernel benchmark result is saved the same way so the
  runtime does not need to re-calibrate on every page load.

This data never leaves your device. Uninstalling the extension removes it.

## Network access

The extension makes no network requests of its own. Its neural network
weights are bundled inside the extension package and loaded from disk. The
`declarativeNetRequest` rules only adjust CORS headers on media requests the
page itself makes, so that video frames can be read into GPU textures; the
rules do not redirect, log, or modify content.

## Video content

Video frames are processed entirely inside your GPU and are never stored or
transmitted. When you close the tab, nothing persists.

## Changes

If this policy ever changes, the new version will be published at the same
URL with an updated effective date. Given what the extension is, "we still
collect nothing" is the expected content of any future version.

## Contact

Questions: open an issue at https://github.com/MONZikWasTaken/Framecast
