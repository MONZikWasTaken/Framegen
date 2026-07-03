"""Generate the extension icons: dark rounded square, green FC monogram,
a small 'play' notch — matches the in-page glass buttons. Pure PIL."""
import os

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "extension", "icons")
os.makedirs(OUT, exist_ok=True)

GREEN = (25, 195, 125, 255)
DARK = (18, 20, 24, 255)


def draw(size):
    s = 8  # supersample
    n = size * s
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = n * 0.22
    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=r, fill=DARK,
                        outline=(255, 255, 255, 40), width=max(1, n // 64))
    # play triangle, green, centered-left
    tw = n * 0.34
    x0, y0 = n * 0.30, n * 0.30
    d.polygon([(x0, y0), (x0, n - y0), (x0 + tw, n / 2)], fill=GREEN)
    # speed ticks to the right of the triangle (the "x2" idea without text)
    for i, h in enumerate((0.16, 0.24, 0.32)):
        x = x0 + tw + n * 0.07 + i * n * 0.075
        d.rounded_rectangle([x, n / 2 - n * h / 2, x + n * 0.035, n / 2 + n * h / 2],
                            radius=n * 0.02, fill=(255, 255, 255, 210 - i * 40))
    return img.resize((size, size), Image.LANCZOS)


for size in (16, 32, 48, 128):
    draw(size).save(os.path.join(OUT, f"icon{size}.png"))
print("icons written to", OUT)
