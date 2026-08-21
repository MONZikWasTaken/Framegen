#!/usr/bin/env python3
"""Score VFI shootout outputs and create full-frame plus 4x error-focused comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "benchmarks/vfi-shootout/scenes.json").read_text())
METHODS = (
    ("Framegen v7s 480", ROOT / ".bench/results/framegen-v7s-480", "framegen-v7s.png"),
    ("Framegen v7s 720", ROOT / ".bench/results/framegen-v7s-720", "framegen-v7s.png"),
    ("Practical-RIFE v4.25", ROOT / ".bench/results/reference", "practical-rife-v425.png"),
    ("IFRNet", ROOT / ".bench/results/reference", "ifrnet.png"),
)


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mse = np.mean((actual - expected) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def global_ssim(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual.mean(axis=2)
    expected = expected.mean(axis=2)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mux, muy = actual.mean(), expected.mean()
    vx, vy = actual.var(), expected.var()
    cov = ((actual - mux) * (expected - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def crop_box(predictions: list[np.ndarray], target: np.ndarray) -> tuple[int, int, int, int]:
    error = np.mean(np.stack([np.abs(prediction - target).mean(axis=2) for prediction in predictions]), axis=0)
    h, w = error.shape
    cw, ch = min(192, w), min(144, h)
    stride = max(8, min(cw, ch) // 8)
    best = (0.0, w // 2, h // 2)
    for y in range(ch // 2, h - ch // 2 + 1, stride):
        for x in range(cw // 2, w - cw // 2 + 1, stride):
            value = error[y - ch // 2:y + ch // 2, x - cw // 2:x + cw // 2].mean()
            if value > best[0]:
                best = (value, x, y)
    _, x, y = best
    return x - cw // 2, y - ch // 2, x + cw // 2, y + ch // 2


def tile(label: str, pixels: np.ndarray, box: tuple[int, int, int, int], scale: int) -> Image.Image:
    crop = Image.fromarray(pixels.astype(np.uint8)).crop(box).resize(((box[2] - box[0]) * scale, (box[3] - box[1]) * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (crop.width, crop.height + 22), "#151515")
    canvas.paste(crop, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="white", font=ImageFont.load_default())
    return canvas


def main() -> None:
    out = ROOT / ".bench/results/report"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"methods": [name for name, _, _ in METHODS], "scenes": []}
    rows: list[Image.Image] = []
    for scene in SCENES:
        target = image(ROOT / scene["gt"])
        predictions = [image(base / scene["id"] / filename) for _, base, filename in METHODS]
        if any(prediction.shape != target.shape for prediction in predictions):
            raise ValueError(f"dimension mismatch for {scene['id']}")
        box = crop_box(predictions, target)
        values = [{"method": name, "psnr": round(psnr(prediction, target), 3), "ssim": round(global_ssim(prediction, target), 5)} for (name, _, _), prediction in zip(METHODS, predictions)]
        report["scenes"].append({"id": scene["id"], "dataset": scene["dataset"], "kind": scene["kind"], "crop": box, "metrics": values})
        row = [tile("ground truth", target, box, 4)]
        row.extend(tile(name, prediction, box, 4) for (name, _, _), prediction in zip(METHODS, predictions))
        width = sum(part.width for part in row)
        height = max(part.height for part in row)
        sheet = Image.new("RGB", (width, height), "#151515")
        x = 0
        for part in row:
            sheet.paste(part, (x, 0))
            x += part.width
        sheet.save(out / f"{scene['id']}-crop-4x.png")
        rows.append(sheet)

    width = max(row.width for row in rows)
    height = sum(row.height + 8 for row in rows)
    sheet = Image.new("RGB", (width, height), "#0d0d0d")
    y = 0
    for scene, row in zip(SCENES, rows):
        sheet.paste(row, (0, y))
        ImageDraw.Draw(sheet).text((4, y + 2), scene["id"], fill="#ffcc66", font=ImageFont.load_default())
        y += row.height + 8
    sheet.save(out / "all-crops-4x.png")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "metrics.json")
    print(out / "all-crops-4x.png")


if __name__ == "__main__":
    main()
