"""Clean the AI-generated logo: auto-crop the rounded icon square (drop the
wordmark below), denoise, cut transparent rounded corners, export icon sizes.

Usage: python clean_logo.py <src.png>
Writes extension/icons/icon{16,32,48,128}.png + assets/logo512.png master.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = sys.argv[1]

img = Image.open(src_path).convert("RGB")
a = np.asarray(img).astype(np.int32)
bright = a.max(axis=2)  # per-pixel max channel

if len(sys.argv) >= 6:  # explicit crop: left top right bottom
    box = tuple(int(v) for v in sys.argv[2:6])
    icon = img.crop(box)
    print("crop (explicit):", box, "->", icon.size)
else:
    # auto: content rows (above bg), biggest cluster = the square band
    bg = np.percentile(bright, 5)
    row_hits = (bright > bg + 18).sum(axis=1)
    rows = np.where(row_hits > 8)[0]
    clusters, start = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] - rows[i - 1] > 12:
            clusters.append((start, rows[i - 1]))
            start = rows[i]
    clusters.append((start, rows[-1]))
    top, bottom = max(clusters, key=lambda c: c[1] - c[0])
    band = bright[top:bottom + 1]
    col_hits = (band > bg + 18).sum(axis=0)
    cols = np.where(col_hits > 8)[0]
    left, right = cols[0], cols[-1]
    # trust the horizontal borders; make it square from the top edge down
    side = right - left
    box = (int(left), int(top), int(right), int(top + side))
    icon = img.crop(box)
    print("crop (auto):", box, "->", icon.size)

# denoise: median kills AI speckle, tiny gaussian smooths banding; then master size
icon = icon.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
icon = icon.resize((512, 512), Image.LANCZOS)

# transparent rounded corners (radius like the drawn square, ~22.5%)
mask = Image.new("L", (512, 512), 0)
d = ImageDraw.Draw(mask)
d.rounded_rectangle([0, 0, 511, 511], radius=int(512 * 0.225), fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(1))
out = icon.convert("RGBA")
out.putalpha(mask)

out.save(os.path.join(REPO, "assets", "logo512.png"))
for s in (16, 32, 48, 128):
    out.resize((s, s), Image.LANCZOS).save(
        os.path.join(REPO, "extension", "icons", f"icon{s}.png"))
print("written: assets/logo512.png + extension/icons/icon16/32/48/128.png")
