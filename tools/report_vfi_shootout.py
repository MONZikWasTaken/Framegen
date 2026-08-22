#!/usr/bin/env python3
"""Score provenance-matched VFI outputs and create error-focused comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENES_PATH = ROOT / "benchmarks/vfi-shootout/scenes.json"
SHOOTOUT_MANIFEST_PATH = ROOT / "benchmarks/vfi-shootout/manifest.json"
DEFAULT_RESULTS_ROOT = ROOT / ".bench/results/report"
SCENE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
FRAMEGEN_SOURCE_KEYS = (
    "runner",
    "shootoutManifest",
    "scenes",
    "playwrightContract",
    "harness",
    "runtime",
    "weights",
    "weightsManifest",
)


def utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.float64)


def luma(pixels: np.ndarray) -> np.ndarray:
    return pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114


def psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    mean_squared_error = np.mean((actual - expected) ** 2)
    return (
        99.0
        if mean_squared_error == 0
        else float(10 * np.log10(255.0**2 / mean_squared_error))
    )


def box_filter(values: np.ndarray, radius: int = 5) -> np.ndarray:
    size = radius * 2 + 1
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    ) / (size * size)


def box_window_ssim(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_luma = luma(actual)
    expected_luma = luma(expected)
    actual_mean = box_filter(actual_luma)
    expected_mean = box_filter(expected_luma)
    actual_variance = np.maximum(
        0.0, box_filter(actual_luma * actual_luma) - actual_mean * actual_mean
    )
    expected_variance = np.maximum(
        0.0, box_filter(expected_luma * expected_luma) - expected_mean * expected_mean
    )
    covariance = box_filter(actual_luma * expected_luma) - actual_mean * expected_mean
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * actual_mean * expected_mean + c1) * (2 * covariance + c2)
    denominator = (actual_mean * actual_mean + expected_mean * expected_mean + c1) * (
        actual_variance + expected_variance + c2
    )
    return float(np.mean(numerator / denominator))


def luma_gradient_mae(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_y, actual_x = np.gradient(luma(actual))
    expected_y, expected_x = np.gradient(luma(expected))
    actual_magnitude = np.hypot(actual_x, actual_y)
    expected_magnitude = np.hypot(expected_x, expected_y)
    return float(np.mean(np.abs(actual_magnitude - expected_magnitude)))


def crop_box(
    predictions: list[np.ndarray], target: np.ndarray
) -> tuple[int, int, int, int]:
    error = np.mean(
        np.stack(
            [np.abs(prediction - target).mean(axis=2) for prediction in predictions]
        ),
        axis=0,
    )
    height, width = error.shape
    crop_width, crop_height = min(192, width), min(144, height)
    stride = max(8, min(crop_width, crop_height) // 8)
    best = (0.0, width // 2, height // 2)
    for y in range(crop_height // 2, height - crop_height // 2 + 1, stride):
        for x in range(crop_width // 2, width - crop_width // 2 + 1, stride):
            value = error[
                y - crop_height // 2 : y + crop_height // 2,
                x - crop_width // 2 : x + crop_width // 2,
            ].mean()
            if value > best[0]:
                best = (value, x, y)
    _, x, y = best
    return (
        x - crop_width // 2,
        y - crop_height // 2,
        x + crop_width // 2,
        y + crop_height // 2,
    )


def tile(
    label: str,
    pixels: np.ndarray,
    box: tuple[int, int, int, int],
    scale: int,
) -> Image.Image:
    crop = (
        Image.fromarray(pixels.astype(np.uint8))
        .crop(box)
        .resize(
            ((box[2] - box[0]) * scale, (box[3] - box[1]) * scale),
            Image.Resampling.NEAREST,
        )
    )
    canvas = Image.new("RGB", (crop.width, crop.height + 22), "#151515")
    canvas.paste(crop, (0, 22))
    ImageDraw.Draw(canvas).text(
        (4, 4), label, fill="white", font=ImageFont.load_default()
    )
    return canvas


def compose_scene_sheet(
    scene_id: str, parts: list[Image.Image], header_height: int = 20
) -> Image.Image:
    if not parts or header_height <= 0:
        raise ValueError("scene sheet requires tiles and a positive header")
    content_width = sum(part.width for part in parts)
    content_height = max(part.height for part in parts)
    sheet = Image.new("RGB", (content_width, content_height + header_height), "#151515")
    ImageDraw.Draw(sheet).text(
        (4, 3), scene_id, fill="#ffcc66", font=ImageFont.load_default()
    )
    x = 0
    for part in parts:
        sheet.paste(part, (x, header_height))
        x += part.width
    return sheet


def load_run(directory: Path, expected_kind: str) -> tuple[Path, dict[str, Any]]:
    run_directory = directory.resolve()
    run_path = run_directory / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run.json is missing: {run_path}")
    report = json.loads(run_path.read_text(encoding="utf-8"))
    if report.get("schemaVersion") != 1 or report.get("kind") != expected_kind:
        raise ValueError(f"unexpected run schema or kind: {run_path}")
    return run_directory, report


def validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 is invalid")
    return value


def stable_file_hash(metadata: Any, label: str) -> str:
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} metadata is missing")
    start = validate_sha256(metadata.get("sha256"), label)
    end = validate_sha256(metadata.get("sha256End"), f"{label} end")
    if start != end:
        raise ValueError(f"{label} changed during its source run")
    return start


def clean_git_identity(metadata: Any, label: str) -> dict[str, Any]:
    commit = metadata.get("commit") if isinstance(metadata, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", commit
    ):
        raise ValueError(f"{label} has no Git commit")
    if metadata.get("dirty") is not False:
        raise ValueError(f"{label} repository was dirty")
    if metadata.get("statusPorcelain") != []:
        raise ValueError(f"{label} repository status was not clean")
    if metadata.get("errors") != []:
        raise ValueError(f"{label} Git provenance contains errors")
    return metadata


def validate_reference_run(
    run: dict[str, Any], shootout_manifest: dict[str, Any]
) -> None:
    if run.get("measurement", {}).get("timestep") != 0.5:
        raise ValueError("reference run timestep must be 0.5")
    expected_methods = shootout_manifest.get("referenceMethods", {})
    models = run.get("models")
    if not isinstance(models, dict):
        raise ValueError("reference model provenance is missing")
    for key in ("practicalRife", "ifrnet"):
        model = models.get(key)
        if not isinstance(model, dict):
            raise ValueError(f"reference {key} provenance is missing")
        repository = model.get("repository", {})
        start_git = clean_git_identity(repository.get("git"), f"reference {key}")
        end_git = clean_git_identity(repository.get("gitEnd"), f"reference {key} end")
        if start_git != end_git:
            raise ValueError(f"reference {key} repository changed during its run")
        checkpoint = model.get("checkpoint")
        implementation = model.get("implementation")
        stable_file_hash(checkpoint, f"reference {key} checkpoint")
        stable_file_hash(implementation, f"reference {key} implementation")
        expected_implementation = (
            "train_log/RIFE_HDv3.py" if key == "practicalRife" else "models/IFRNet.py"
        )
        if implementation.get("path") != expected_implementation:
            raise ValueError(f"reference {key} implementation path is unexpected")
        if key == "practicalRife" and checkpoint.get("path") != "train_log/flownet.pkl":
            raise ValueError("reference Practical-RIFE checkpoint path is unexpected")
        imported_sources = model.get("importedSources")
        if not isinstance(imported_sources, list) or not imported_sources:
            raise ValueError(f"reference {key} imported source provenance is missing")
        imported_paths = []
        for index, source in enumerate(imported_sources):
            stable_file_hash(source, f"reference {key} imported source {index}")
            path = source.get("path") if isinstance(source, dict) else None
            if not isinstance(path, str) or not path:
                raise ValueError(f"reference {key} imported source path is invalid")
            imported_paths.append(path)
        if len(imported_paths) != len(set(imported_paths)):
            raise ValueError(f"reference {key} imported source paths are duplicated")
        if expected_implementation not in imported_paths:
            raise ValueError(
                f"reference {key} implementation is absent from imported sources"
            )
        if model.get("preprocessing") != expected_methods.get(key):
            raise ValueError(f"reference {key} preprocessing contract is unexpected")
    source = run.get("source", {})
    for key in ("runner", "scenes", "shootoutManifest"):
        stable_file_hash(source.get(key), f"reference {key}")


def framegen_browser_identity(run: dict[str, Any]) -> dict[str, Any]:
    browser = run.get("browser")
    if not isinstance(browser, dict):
        raise ValueError("Framegen browser provenance is missing")
    host = run.get("host")
    environment = browser.get("environment")
    if not isinstance(host, dict) or not isinstance(environment, dict):
        raise ValueError("Framegen host or browser environment provenance is missing")
    for key in (
        "channel",
        "playwrightCliPackage",
        "playwrightCliVersion",
    ):
        if not isinstance(browser.get(key), str) or not browser[key]:
            raise ValueError(f"Framegen browser {key} provenance is missing")
    if (
        not isinstance(environment.get("userAgent"), str)
        or not environment["userAgent"]
    ):
        raise ValueError("Framegen browser user agent provenance is missing")
    adapter = browser.get("adapterIdentity")
    if not isinstance(adapter, dict) or not any(
        adapter.get(key) for key in ("vendor", "architecture", "device", "description")
    ):
        raise ValueError("Framegen adapter identity is missing")
    features = browser.get("deviceFeatures")
    if (
        not isinstance(features, list)
        or any(not isinstance(value, str) for value in features)
        or features != sorted(set(features))
    ):
        raise ValueError("Framegen device feature provenance is invalid")
    limits = browser.get("deviceLimits")
    if (
        not isinstance(limits, dict)
        or not limits
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in limits.values()
        )
    ):
        raise ValueError("Framegen device limit provenance is invalid")
    return {
        "host": host,
        "channel": browser.get("channel"),
        "headed": browser.get("headed"),
        "playwrightCliPackage": browser.get("playwrightCliPackage"),
        "playwrightCliVersion": browser.get("playwrightCliVersion"),
        "environment": environment,
        "adapterIdentity": adapter,
        "deviceFeatures": features,
        "deviceLimits": limits,
    }


def validate_framegen_run(
    run: dict[str, Any], resolution: int, shootout_manifest: dict[str, Any]
) -> dict[str, str]:
    workload = run.get("workload", {})
    expected_rung = shootout_manifest.get("inferenceRungs", {}).get(str(resolution))
    expected_scenes = json.loads(SCENES_PATH.read_text(encoding="utf-8"))
    if (
        workload.get("model") != "v7s"
        or workload.get("timestep") != 0.5
        or workload.get("inferenceResolution") != resolution
        or workload.get("inferenceRung") != expected_rung
        or workload.get("scenes") != [scene["id"] for scene in expected_scenes]
    ):
        raise ValueError(f"Framegen {resolution} workload identity is invalid")
    source_files = run.get("source", {}).get("files")
    if not isinstance(source_files, dict):
        raise ValueError(f"Framegen {resolution} source provenance is missing")
    build_identity = {
        key: stable_file_hash(
            source_files.get(key), f"Framegen {resolution} source {key}"
        )
        for key in FRAMEGEN_SOURCE_KEYS
    }
    expected_adapter = framegen_browser_identity(run)["adapterIdentity"]
    expected_features = run["browser"]["deviceFeatures"]
    expected_limits = run["browser"]["deviceLimits"]
    records = run.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Framegen {resolution} records are missing")
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("rung") != expected_rung
            or record.get("adapterIdentity") != expected_adapter
            or record.get("deviceFeatures") != expected_features
            or record.get("deviceLimits") != expected_limits
        ):
            raise ValueError(f"Framegen {resolution} record identity is unstable")
    return build_identity


def validate_framegen_pair(
    framegen_480: dict[str, Any],
    framegen_720: dict[str, Any],
    shootout_manifest: dict[str, Any],
) -> None:
    build_480 = validate_framegen_run(framegen_480, 480, shootout_manifest)
    build_720 = validate_framegen_run(framegen_720, 720, shootout_manifest)
    if build_480 != build_720:
        raise ValueError("Framegen 480 and 720 runs use different source builds")
    if framegen_browser_identity(framegen_480) != framegen_browser_identity(
        framegen_720
    ):
        raise ValueError(
            "Framegen 480 and 720 runs use different host, browser, or GPU identities"
        )


def resolve_run_output(run_directory: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("run output path is invalid")
    output = (run_directory / relative_path).resolve()
    if output == run_directory or run_directory not in output.parents:
        raise ValueError(f"run output escapes its directory: {relative_path}")
    if not output.is_file():
        raise FileNotFoundError(f"run output is missing: {output}")
    return output


def current_dataset(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen_ids: set[str] = set()
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"scenes[{index}] must be an object")
        scene_id = scene.get("id")
        if not isinstance(scene_id, str) or not SCENE_ID_PATTERN.fullmatch(scene_id):
            raise ValueError(f"scenes[{index}].id is invalid")
        if scene_id in seen_ids:
            raise ValueError(f"scenes[{index}].id is duplicated")
        seen_ids.add(scene_id)
        files = {}
        for field in ("i0", "gt", "i1"):
            file_path = (ROOT / scene[field]).resolve()
            if file_path == ROOT or ROOT not in file_path.parents:
                raise ValueError(f"{scene_id} {field} escapes the repository")
            if not file_path.is_file():
                raise FileNotFoundError(f"{scene_id} {field} is missing: {file_path}")
            files[field] = {
                "path": file_path.relative_to(ROOT).as_posix(),
                "sizeBytes": file_path.stat().st_size,
                "sha256": sha256(file_path),
            }
        result.append(
            {
                "id": scene_id,
                "dataset": scene["dataset"],
                "kind": scene["kind"],
                "files": files,
            }
        )
    return result


def validate_dataset_identity(
    run: dict[str, Any], expected_dataset: list[dict[str, Any]], label: str
) -> None:
    actual_rows = run.get("source", {}).get("dataset")
    if not isinstance(actual_rows, list):
        raise ValueError(f"{label} has no dataset provenance")
    actual = {row.get("id"): row for row in actual_rows if isinstance(row, dict)}
    if set(actual) != {row["id"] for row in expected_dataset}:
        raise ValueError(f"{label} scene set does not match the current manifest")
    for expected in expected_dataset:
        row = actual[expected["id"]]
        for field in ("i0", "gt", "i1"):
            if (
                row.get("files", {}).get(field, {}).get("sha256")
                != expected["files"][field]["sha256"]
            ):
                raise ValueError(
                    f"{label} {expected['id']} {field} hash does not match"
                )


def record_map(run: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = run.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{label} records are missing")
    mapped = {
        record.get("scene"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("scene"), str)
    }
    if len(mapped) != len(records):
        raise ValueError(f"{label} records contain invalid or duplicate scene ids")
    return mapped


def verified_output(
    run_directory: Path, relative_path: str, expected_hash: str, label: str
) -> Path:
    expected_hash = validate_sha256(expected_hash, f"{label} output")
    output = resolve_run_output(run_directory, relative_path)
    actual_hash = sha256(output)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} output hash mismatch: {output}")
    return output


def run_provenance(directory: Path, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "directoryName": directory.name,
        "runJsonSha256": sha256(directory / "run.json"),
        "createdAt": run.get("createdAt"),
        "source": run.get("source"),
        "host": run.get("host"),
        "browser": run.get("browser"),
        "models": run.get("models"),
        "workload": run.get("workload"),
        "measurement": run.get("measurement"),
        "limitations": run.get("limitations"),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a quality-only report from explicit, provenance-matched runs."
    )
    parser.add_argument("--framegen-480", type=Path, required=True)
    parser.add_argument("--framegen-720", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_path = (
        arguments.out.resolve()
        if arguments.out is not None
        else DEFAULT_RESULTS_ROOT / utc_label()
    )
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    scenes = json.loads(SCENES_PATH.read_text(encoding="utf-8"))
    shootout_manifest = json.loads(SHOOTOUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scene manifest must contain a non-empty array")
    if (
        shootout_manifest.get("schemaVersion") != 1
        or shootout_manifest.get("id") != "framegen-v7s-vfi-shootout-v1"
    ):
        raise ValueError("unexpected shootout manifest")
    dataset = current_dataset(scenes)

    framegen_480_directory, framegen_480 = load_run(
        arguments.framegen_480, "framegen-webgpu-vfi-shootout"
    )
    framegen_720_directory, framegen_720 = load_run(
        arguments.framegen_720, "framegen-webgpu-vfi-shootout"
    )
    reference_directory, reference = load_run(
        arguments.reference, "pytorch-vfi-reference-shootout"
    )
    validate_framegen_pair(framegen_480, framegen_720, shootout_manifest)
    validate_reference_run(reference, shootout_manifest)
    for label, run in (
        ("framegen-480", framegen_480),
        ("framegen-720", framegen_720),
        ("reference", reference),
    ):
        validate_dataset_identity(run, dataset, label)

    framegen_480_records = record_map(framegen_480, "framegen-480")
    framegen_720_records = record_map(framegen_720, "framegen-720")
    reference_records = record_map(reference, "reference")
    expected_ids = {scene["id"] for scene in scenes}
    if any(
        set(records) != expected_ids
        for records in (
            framegen_480_records,
            framegen_720_records,
            reference_records,
        )
    ):
        raise ValueError("one or more runs do not contain the complete scene set")

    reference_models = reference.get("models", {})
    methods = (
        ("Framegen v7s 480", framegen_480_directory, framegen_480_records, None),
        ("Framegen v7s 720", framegen_720_directory, framegen_720_records, None),
        (
            reference_models.get("practicalRife", {}).get(
                "label", "Practical-RIFE reference"
            ),
            reference_directory,
            reference_records,
            "practicalRife",
        ),
        (
            reference_models.get("ifrnet", {}).get("label", "IFRNet reference"),
            reference_directory,
            reference_records,
            "ifrnet",
        ),
    )
    method_names = [name for name, _, _, _ in methods]
    if any(not isinstance(name, str) or not name.strip() for name in method_names):
        raise ValueError("method labels must be non-empty strings")
    if len(set(method_names)) != len(method_names):
        raise ValueError("method labels must be unique")
    aggregate: dict[str, list[dict[str, float]]] = {
        name: [] for name, _, _, _ in methods
    }
    scene_reports: list[dict[str, Any]] = []
    rows: list[Image.Image] = []

    with tempfile.TemporaryDirectory(prefix="framegen-vfi-report-") as temporary:
        staged_output = Path(temporary) / "report"
        staged_output.mkdir()
        for scene in scenes:
            scene_id = scene["id"]
            target = image((ROOT / scene["gt"]).resolve())
            predictions: list[np.ndarray] = []
            names: list[str] = []
            for name, directory, records, reference_key in methods:
                record = records[scene_id]
                method_record = (
                    record
                    if reference_key is None
                    else record.get("methods", {}).get(reference_key, {})
                )
                output_path_for_method = verified_output(
                    directory,
                    method_record.get("output"),
                    method_record.get("outputSha256"),
                    f"{name} {scene_id}",
                )
                prediction = image(output_path_for_method)
                if prediction.shape != target.shape:
                    raise ValueError(f"dimension mismatch for {name} {scene_id}")
                names.append(name)
                predictions.append(prediction)

            box = crop_box(predictions, target)
            metrics = []
            for name, prediction in zip(names, predictions):
                values = {
                    "psnrDb": round(psnr(prediction, target), 4),
                    "boxWindowSsim": round(box_window_ssim(prediction, target), 7),
                    "lumaGradientMae": round(luma_gradient_mae(prediction, target), 7),
                }
                metrics.append({"method": name, **values})
                aggregate[name].append(values)
            scene_reports.append(
                {
                    "id": scene_id,
                    "dataset": scene["dataset"],
                    "kind": scene["kind"],
                    "crop": box,
                    "metrics": metrics,
                }
            )

            parts = [tile("ground truth", target, box, 4)]
            parts.extend(
                tile(name, prediction, box, 4)
                for name, prediction in zip(names, predictions)
            )
            sheet = compose_scene_sheet(scene_id, parts)
            sheet.save(staged_output / f"{scene_id}-crop-4x.png")
            rows.append(sheet)

        width = max(row.width for row in rows)
        height = sum(row.height + 8 for row in rows)
        combined = Image.new("RGB", (width, height), "#0d0d0d")
        y = 0
        for row in rows:
            combined.paste(row, (0, y))
            y += row.height + 8
        combined.save(staged_output / "all-crops-4x.png")

        aggregate_report = []
        for name, values in aggregate.items():
            aggregate_report.append(
                {
                    "method": name,
                    "meanPsnrDb": round(
                        sum(value["psnrDb"] for value in values) / len(values), 4
                    ),
                    "meanBoxWindowSsim": round(
                        sum(value["boxWindowSsim"] for value in values) / len(values),
                        7,
                    ),
                    "meanLumaGradientMae": round(
                        sum(value["lumaGradientMae"] for value in values) / len(values),
                        7,
                    ),
                }
            )
        report = {
            "schemaVersion": 1,
            "kind": "vfi-shootout-quality-report",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "methods": [name for name, _, _, _ in methods],
            "metricDefinitions": {
                "psnrDb": (
                    "RGB PSNR on decoded 8-bit PNG values; higher is better. "
                    "An exact match is represented by the finite 99 dB cap."
                ),
                "boxWindowSsim": (
                    "Local SSIM-like score on BT.601 luma using a uniform 11x11 "
                    "reflect-padded box window and population local moments; "
                    "higher is better. It is intentionally not labeled as the "
                    "standard Gaussian-window SSIM implementation."
                ),
                "lumaGradientMae": (
                    "Mean absolute error between luma gradient magnitudes; lower is better."
                ),
            },
            "visualization": {
                "cropSelection": (
                    "The fixed-size crop maximizes mean absolute error averaged "
                    "across all compared predictions."
                ),
                "scale": 4,
                "resampling": "nearest",
                "sceneHeaderPixels": 20,
            },
            "sourceRuns": {
                "framegen480": run_provenance(framegen_480_directory, framegen_480),
                "framegen720": run_provenance(framegen_720_directory, framegen_720),
                "reference": run_provenance(reference_directory, reference),
            },
            "performanceComparison": {
                "supported": False,
                "reason": (
                    "WebGPU and PyTorch runs may use different devices, frameworks, "
                    "precision, and preprocessing. Raw timing samples remain in each "
                    "source run and are not ranked here."
                ),
            },
            "qualityGate": {
                "enabled": False,
                "reason": (
                    "No baseline revision or acceptance thresholds were supplied; "
                    "this report describes quality and does not claim a pass."
                ),
            },
            "aggregate": aggregate_report,
            "scenes": scene_reports,
        }
        (staged_output / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"output directory already exists: {output_path}")
        shutil.move(str(staged_output), str(output_path))

    print(
        json.dumps(
            {
                "metrics": str(output_path / "metrics.json"),
                "comparison": str(output_path / "all-crops-4x.png"),
                "qualityGate": False,
                "performanceComparison": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
