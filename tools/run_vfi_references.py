#!/usr/bin/env python3
"""Run user-supplied Practical-RIFE and IFRNet references on the fixed shootout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = ROOT / "benchmarks/vfi-shootout/scenes.json"
SHOOTOUT_MANIFEST = ROOT / "benchmarks/vfi-shootout/manifest.json"
DEFAULT_RESULTS_ROOT = ROOT / ".bench/results/reference"
SCENE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")


def load_dependencies() -> None:
    global np, PIL, torch, F, Image
    np = importlib.import_module("numpy")
    PIL = importlib.import_module("PIL")
    torch = importlib.import_module("torch")
    F = importlib.import_module("torch.nn.functional")
    Image = importlib.import_module("PIL.Image")


def utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_label(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_dataset_file(relative_path: str, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} must be a relative path")
    resolved = (ROOT / relative_path).resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise ValueError(f"{label} escapes the repository")
    return resolved


def validate_scene_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SCENE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must match {SCENE_ID_PATTERN.pattern}")
    return value


def confined_output(root: Path, relative_path: str, label: str) -> Path:
    resolved_root = root.resolve()
    output = (resolved_root / relative_path).resolve()
    if output == resolved_root or resolved_root not in output.parents:
        raise ValueError(f"{label} escapes the staged output directory")
    return output


def validate_rife_layout(root: Path, checkpoint: Path, basename: str) -> Path:
    resolved_root = root.resolve()
    resolved_checkpoint = checkpoint.resolve()
    expected = (resolved_root / "train_log" / basename).resolve()
    if resolved_checkpoint != expected:
        raise ValueError(
            "Practical-RIFE checkpoint must be the exact canonical file "
            f"{expected}; received {resolved_checkpoint}"
        )
    return expected


def file_metadata(path: Path, base: Path = ROOT) -> dict[str, Any]:
    value = path.stat()
    return {
        "path": relative_label(path, base),
        "sizeBytes": value.st_size,
        "modifiedTimeNs": value.st_mtime_ns,
        "sha256": sha256(path),
    }


def imported_python_metadata(root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    imported_paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = Path(module_file).resolve()
        if path.suffix == ".py" and resolved_root in path.parents and path.is_file():
            imported_paths.add(path)
    if not imported_paths:
        raise RuntimeError(
            f"no imported Python sources were found under {resolved_root}"
        )
    return [file_metadata(path, resolved_root) for path in sorted(imported_paths)]


def stable_imported_python_metadata(
    start: list[dict[str, Any]], end: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    start_identity = [(row["path"], row["sha256"]) for row in start]
    end_identity = [(row["path"], row["sha256"]) for row in end]
    if start_identity != end_identity:
        raise RuntimeError(f"{label} imported Python sources changed during execution")
    return [
        {**row, "sha256End": end[index]["sha256"]} for index, row in enumerate(start)
    ]


def git_metadata(root: Path) -> dict[str, Any]:
    safe = ["git", "-c", f"safe.directory={root}", "-C", str(root)]

    def probe(arguments: list[str]) -> tuple[str | None, str | None]:
        result = subprocess.run(
            [*safe, *arguments],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip(), None
        return None, (
            result.stderr or result.stdout
        ).strip() or f"git exited {result.returncode}"

    commit, commit_error = probe(["rev-parse", "HEAD"])
    branch, branch_error = probe(["rev-parse", "--abbrev-ref", "HEAD"])
    status, status_error = probe(["status", "--porcelain=v1", "--untracked-files=all"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "statusPorcelain": status.splitlines() if status is not None else None,
        "errors": [
            error for error in (commit_error, branch_error, status_error) if error
        ],
    }


def require_clean_git_repository(
    name: str, root: Path, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = metadata if metadata is not None else git_metadata(root)
    if result.get("errors") or not result.get("commit"):
        details = "; ".join(result.get("errors") or []) or "missing commit"
        raise RuntimeError(f"{name} must be a readable Git repository: {details}")
    if result.get("dirty") or result.get("statusPorcelain"):
        raise RuntimeError(
            f"{name} repository must be clean; checkout or stash local changes first"
        )
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def device_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"type": device.type, "index": device.index}
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        result.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "cudaRuntime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    elif device.type == "mps":
        result["name"] = "Apple Metal Performance Shaders"
    else:
        result["name"] = platform.processor() or platform.machine()
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize_samples(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("timing samples cannot be empty")
    sorted_samples = sorted(samples)
    mean = statistics.fmean(samples)
    deviation = statistics.pstdev(samples)
    return {
        "count": len(samples),
        "minMs": sorted_samples[0],
        "medianMs": statistics.median(sorted_samples),
        "meanMs": mean,
        "p95Ms": sorted_samples[max(0, (len(sorted_samples) * 95 + 99) // 100 - 1)],
        "maxMs": sorted_samples[-1],
        "standardDeviationMs": deviation,
        "coefficientOfVariationPercent": deviation / mean * 100 if mean > 0 else 0.0,
    }


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB")).copy()
    return (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .float()
        .div_(255)
        .unsqueeze(0)
        .to(device)
    )


def swap_rgb_bgr(image: Any) -> Any:
    if image.shape[1] != 3:
        raise ValueError("RGB/BGR conversion requires a three-channel NCHW tensor")
    return image[:, [2, 1, 0], ...]


def save_image(tensor: torch.Tensor, path: Path) -> None:
    pixels = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()[0]
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(pixels, "RGB").save(path)


def pad_rife(image: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor
    return F.pad(image, (0, pad_width, 0, pad_height)), (height, width)


def pad_ifrnet(
    image: torch.Tensor, divisor: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    height, width = image.shape[-2:]
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor
    padding = (
        pad_width // 2,
        pad_width - pad_width // 2,
        pad_height // 2,
        pad_height - pad_height // 2,
    )
    return F.pad(image, padding, mode="replicate"), padding


def import_exact_module(name: str, expected_path: Path) -> tuple[Any, Path]:
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(name)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"{name} did not expose an import path")
    actual_path = Path(module_file).resolve()
    expected = expected_path.resolve()
    if actual_path != expected:
        raise RuntimeError(
            f"{name} resolved to {actual_path}, expected the pinned source {expected}"
        )
    return module, actual_path


def load_rife(root: Path, checkpoint: Path, device: torch.device) -> tuple[Any, Path]:
    sys.path.insert(0, str(root))
    module, module_path = import_exact_module(
        "train_log.RIFE_HDv3", root / "train_log" / "RIFE_HDv3.py"
    )

    model = module.Model()
    model.load_model(str(checkpoint.parent))
    model.flownet.to(device)
    model.eval()
    return model, module_path


def load_ifrnet(root: Path, checkpoint: Path, device: torch.device) -> tuple[Any, Path]:
    sys.path.insert(0, str(root))
    module, module_path = import_exact_module(
        "models.IFRNet", root / "models" / "IFRNet.py"
    )

    model = module.Model().to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    return model, module_path


def run_ifrnet_inference(
    model: Any, image0: Any, image1: Any, timestep: Any, scale_factor: float
) -> Any:
    return model.inference(image0, image1, timestep, scale_factor=scale_factor)


def measure(
    run: Callable[[], torch.Tensor],
    device: torch.device,
    warmups: int,
    repetitions: int,
) -> tuple[torch.Tensor, list[float]]:
    for _ in range(warmups):
        run()
    synchronize(device)
    samples: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = run()
        synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    if output is None:
        raise RuntimeError("reference measurement produced no output")
    return output, samples


def load_reference_configuration(path: Path = SHOOTOUT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    practical_rife = manifest.get("referenceMethods", {}).get("practicalRife")
    ifrnet = manifest.get("referenceMethods", {}).get("ifrnet")
    if practical_rife != {
        "checkpointBasename": "flownet.pkl",
        "inputColorOrder": "BGR",
        "outputColorOrder": "RGB",
        "paddingDivisor": 64,
        "paddingMode": "constant-zero",
    }:
        raise ValueError("shootout manifest has an unsupported Practical-RIFE contract")
    if ifrnet != {
        "inputColorOrder": "RGB",
        "outputColorOrder": "RGB",
        "paddingDivisor": 20,
        "paddingMode": "replicate",
        "scaleFactor": 0.8,
    }:
        raise ValueError("shootout manifest has an unsupported IFRNet contract")
    return {"practicalRife": practical_rife, "ifrnet": ifrnet}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trusted local reference outputs; timings are descriptive per method only."
    )
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rife-root", type=Path, required=True)
    parser.add_argument("--rife-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--rife-label", default="Practical-RIFE (user-supplied checkpoint)"
    )
    parser.add_argument(
        "--rife-device", choices=("auto", "cpu", "cuda", "mps"), default="cpu"
    )
    parser.add_argument("--ifrnet-root", type=Path, required=True)
    parser.add_argument("--ifrnet-checkpoint", type=Path, required=True)
    parser.add_argument("--ifrnet-label", default="IFRNet (user-supplied checkpoint)")
    parser.add_argument(
        "--ifrnet-device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    arguments = parser.parse_args()
    if not 0 <= arguments.warmups <= 100:
        parser.error("--warmups must be in [0, 100]")
    if not 1 <= arguments.repetitions <= 100:
        parser.error("--repetitions must be in [1, 100]")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    reference_configuration = load_reference_configuration()
    load_dependencies()
    scenes_path = arguments.scenes.resolve()
    output_path = (
        arguments.out.resolve()
        if arguments.out is not None
        else DEFAULT_RESULTS_ROOT / utc_label()
    )
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    roots = {
        "practicalRife": arguments.rife_root.resolve(),
        "ifrnet": arguments.ifrnet_root.resolve(),
    }
    checkpoints = {
        "practicalRife": arguments.rife_checkpoint.resolve(),
        "ifrnet": arguments.ifrnet_checkpoint.resolve(),
    }
    if not scenes_path.is_file():
        raise FileNotFoundError(f"scene manifest is missing: {scenes_path}")
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name} repository is missing: {root}")
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name} checkpoint is missing: {checkpoint}")
    validate_rife_layout(
        roots["practicalRife"],
        checkpoints["practicalRife"],
        reference_configuration["practicalRife"]["checkpointBasename"],
    )

    scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scene manifest must contain a non-empty array")
    seen_ids: set[str] = set()
    dataset_metadata: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes):
        scene_id = scene.get("id") if isinstance(scene, dict) else None
        scene_id = validate_scene_id(scene_id, f"scenes[{index}].id")
        if scene_id in seen_ids:
            raise ValueError(f"scenes[{index}].id is duplicated")
        seen_ids.add(scene_id)
        files: dict[str, Any] = {}
        for field in ("i0", "gt", "i1"):
            file_path = resolve_dataset_file(scene[field], f"{scene_id}.{field}")
            if not file_path.is_file():
                raise FileNotFoundError(f"{scene_id} {field} is missing: {file_path}")
            files[field] = file_metadata(file_path)
        dataset_metadata.append(
            {
                "id": scene_id,
                "dataset": scene.get("dataset"),
                "kind": scene.get("kind"),
                "files": files,
            }
        )

    runner_metadata_start = file_metadata(Path(__file__).resolve())
    scenes_metadata_start = file_metadata(scenes_path)
    shootout_manifest_metadata_start = file_metadata(SHOOTOUT_MANIFEST)
    checkpoint_metadata_start = {
        name: file_metadata(checkpoint, roots[name])
        for name, checkpoint in checkpoints.items()
    }
    repository_metadata_start = {
        name: require_clean_git_repository(name, root) for name, root in roots.items()
    }

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    rife_device = resolve_device(arguments.rife_device)
    ifrnet_device = resolve_device(arguments.ifrnet_device)
    print(
        "Reference timing devices are intentionally reported per method and are not "
        f"cross-method throughput evidence: Practical-RIFE={rife_device}, IFRNet={ifrnet_device}"
    )
    sys.dont_write_bytecode = True
    rife, rife_implementation_path = load_rife(
        roots["practicalRife"], checkpoints["practicalRife"], rife_device
    )
    ifrnet, ifrnet_implementation_path = load_ifrnet(
        roots["ifrnet"], checkpoints["ifrnet"], ifrnet_device
    )
    implementation_paths = {
        "practicalRife": rife_implementation_path,
        "ifrnet": ifrnet_implementation_path,
    }
    implementation_metadata_start = {
        name: file_metadata(path, roots[name])
        for name, path in implementation_paths.items()
    }
    imported_sources_metadata_start = {
        name: imported_python_metadata(root) for name, root in roots.items()
    }
    records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="framegen-vfi-reference-") as temporary:
        staged_output = Path(temporary) / "run"
        staged_output.mkdir()
        for scene in scenes:
            scene_id = scene["id"]
            i0_rgb = load_image(
                resolve_dataset_file(scene["i0"], f"{scene_id}.i0"), rife_device
            )
            i1_rgb = load_image(
                resolve_dataset_file(scene["i1"], f"{scene_id}.i1"), rife_device
            )
            rife0, (height, width) = pad_rife(
                swap_rgb_bgr(i0_rgb),
                reference_configuration["practicalRife"]["paddingDivisor"],
            )
            rife1, _ = pad_rife(
                swap_rgb_bgr(i1_rgb),
                reference_configuration["practicalRife"]["paddingDivisor"],
            )
            with torch.inference_mode():
                rife_output, rife_samples = measure(
                    lambda: rife.inference(rife0, rife1, timestep=0.5),
                    rife_device,
                    arguments.warmups,
                    arguments.repetitions,
                )
            rife_output = swap_rgb_bgr(rife_output[..., :height, :width])

            ifr0, padding = pad_ifrnet(
                i0_rgb.to(ifrnet_device),
                reference_configuration["ifrnet"]["paddingDivisor"],
            )
            ifr1, _ = pad_ifrnet(
                i1_rgb.to(ifrnet_device),
                reference_configuration["ifrnet"]["paddingDivisor"],
            )
            embt = torch.tensor(0.5, device=ifrnet_device).view(1, 1, 1, 1)
            with torch.inference_mode():
                ifrnet_output, ifrnet_samples = measure(
                    lambda: run_ifrnet_inference(
                        ifrnet,
                        ifr0,
                        ifr1,
                        embt,
                        reference_configuration["ifrnet"]["scaleFactor"],
                    ),
                    ifrnet_device,
                    arguments.warmups,
                    arguments.repetitions,
                )
            left, right, top, bottom = padding
            ifrnet_output = ifrnet_output[
                ...,
                top : ifrnet_output.shape[-2] - bottom,
                left : ifrnet_output.shape[-1] - right,
            ]
            if rife_output.shape != i0_rgb.shape:
                raise RuntimeError(
                    f"{scene_id} Practical-RIFE output shape {tuple(rife_output.shape)} "
                    f"does not match input shape {tuple(i0_rgb.shape)}"
                )
            if ifrnet_output.shape != i0_rgb.shape:
                raise RuntimeError(
                    f"{scene_id} IFRNet output shape {tuple(ifrnet_output.shape)} "
                    f"does not match input shape {tuple(i0_rgb.shape)}"
                )
            if not bool(torch.isfinite(rife_output).all().item()):
                raise RuntimeError(f"{scene_id} Practical-RIFE output is non-finite")
            if not bool(torch.isfinite(ifrnet_output).all().item()):
                raise RuntimeError(f"{scene_id} IFRNet output is non-finite")

            scene_output = confined_output(staged_output, scene_id, scene_id)
            scene_output.mkdir()
            rife_relative = f"{scene_id}/practical-rife.png"
            ifrnet_relative = f"{scene_id}/ifrnet.png"
            rife_output_path = confined_output(
                staged_output, rife_relative, f"{scene_id} Practical-RIFE output"
            )
            ifrnet_output_path = confined_output(
                staged_output, ifrnet_relative, f"{scene_id} IFRNet output"
            )
            save_image(rife_output, rife_output_path)
            save_image(ifrnet_output, ifrnet_output_path)
            records.append(
                {
                    "scene": scene_id,
                    "methods": {
                        "practicalRife": {
                            "wallMsSamples": rife_samples,
                            "wallMsSummary": summarize_samples(rife_samples),
                            "output": rife_relative,
                            "outputSha256": sha256(rife_output_path),
                        },
                        "ifrnet": {
                            "wallMsSamples": ifrnet_samples,
                            "wallMsSummary": summarize_samples(ifrnet_samples),
                            "output": ifrnet_relative,
                            "outputSha256": sha256(ifrnet_output_path),
                        },
                    },
                }
            )
            print(
                f"{scene_id}: Practical-RIFE {statistics.median(rife_samples):.2f} ms "
                f"on {rife_device}; IFRNet {statistics.median(ifrnet_samples):.2f} ms "
                f"on {ifrnet_device} (descriptive per-method timings)"
            )

        runner_metadata_end = file_metadata(Path(__file__).resolve())
        scenes_metadata_end = file_metadata(scenes_path)
        shootout_manifest_metadata_end = file_metadata(SHOOTOUT_MANIFEST)
        if runner_metadata_start["sha256"] != runner_metadata_end["sha256"]:
            raise RuntimeError("reference runner changed during execution")
        if scenes_metadata_start["sha256"] != scenes_metadata_end["sha256"]:
            raise RuntimeError("scene manifest changed during execution")
        if (
            shootout_manifest_metadata_start["sha256"]
            != shootout_manifest_metadata_end["sha256"]
        ):
            raise RuntimeError("shootout manifest changed during execution")
        for index, scene in enumerate(scenes):
            for field in ("i0", "gt", "i1"):
                current_hash = sha256(
                    resolve_dataset_file(scene[field], f"{scene['id']}.{field}")
                )
                if current_hash != dataset_metadata[index]["files"][field]["sha256"]:
                    raise RuntimeError(
                        f"{scene['id']} {field} changed during execution"
                    )
                dataset_metadata[index]["files"][field]["sha256End"] = current_hash
        checkpoint_metadata_end = {
            name: file_metadata(checkpoint, roots[name])
            for name, checkpoint in checkpoints.items()
        }
        repository_metadata_end = {
            name: require_clean_git_repository(name, root)
            for name, root in roots.items()
        }
        implementation_metadata_end = {
            name: file_metadata(path, roots[name])
            for name, path in implementation_paths.items()
        }
        imported_sources_metadata_end = {
            name: imported_python_metadata(root) for name, root in roots.items()
        }
        stable_imported_sources = {
            name: stable_imported_python_metadata(
                imported_sources_metadata_start[name],
                imported_sources_metadata_end[name],
                name,
            )
            for name in roots
        }
        for name in roots:
            if (
                checkpoint_metadata_start[name]["sha256"]
                != checkpoint_metadata_end[name]["sha256"]
            ):
                raise RuntimeError(f"{name} checkpoint changed during execution")
            if (
                repository_metadata_start[name]["commit"]
                != repository_metadata_end[name]["commit"]
                or repository_metadata_start[name]["statusPorcelain"]
                != repository_metadata_end[name]["statusPorcelain"]
            ):
                raise RuntimeError(f"{name} repository changed during execution")
            if (
                implementation_metadata_start[name]["sha256"]
                != implementation_metadata_end[name]["sha256"]
            ):
                raise RuntimeError(f"{name} implementation changed during execution")
        report = {
            "schemaVersion": 1,
            "kind": "pytorch-vfi-reference-shootout",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "source": {
                "runner": {
                    **runner_metadata_start,
                    "sha256End": runner_metadata_end["sha256"],
                },
                "scenes": {
                    **scenes_metadata_start,
                    "sha256End": scenes_metadata_end["sha256"],
                },
                "shootoutManifest": {
                    **shootout_manifest_metadata_start,
                    "sha256End": shootout_manifest_metadata_end["sha256"],
                },
                "dataset": dataset_metadata,
            },
            "host": {
                "platform": sys.platform,
                "arch": platform.machine(),
                "osRelease": platform.release(),
                "python": sys.version,
                "cpu": platform.processor() or platform.machine(),
            },
            "packages": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "models": {
                "practicalRife": {
                    "label": arguments.rife_label,
                    "repository": {
                        "name": roots["practicalRife"].name,
                        "git": repository_metadata_start["practicalRife"],
                        "gitEnd": repository_metadata_end["practicalRife"],
                    },
                    "checkpoint": {
                        **checkpoint_metadata_start["practicalRife"],
                        "sha256End": checkpoint_metadata_end["practicalRife"]["sha256"],
                    },
                    "implementation": {
                        **implementation_metadata_start["practicalRife"],
                        "sha256End": implementation_metadata_end["practicalRife"][
                            "sha256"
                        ],
                    },
                    "importedSources": stable_imported_sources["practicalRife"],
                    "preprocessing": reference_configuration["practicalRife"],
                    "device": device_metadata(rife_device),
                },
                "ifrnet": {
                    "label": arguments.ifrnet_label,
                    "repository": {
                        "name": roots["ifrnet"].name,
                        "git": repository_metadata_start["ifrnet"],
                        "gitEnd": repository_metadata_end["ifrnet"],
                    },
                    "checkpoint": {
                        **checkpoint_metadata_start["ifrnet"],
                        "sha256End": checkpoint_metadata_end["ifrnet"]["sha256"],
                    },
                    "implementation": {
                        **implementation_metadata_start["ifrnet"],
                        "sha256End": implementation_metadata_end["ifrnet"]["sha256"],
                    },
                    "importedSources": stable_imported_sources["ifrnet"],
                    "preprocessing": reference_configuration["ifrnet"],
                    "device": device_metadata(ifrnet_device),
                },
            },
            "measurement": {
                "timestep": 0.5,
                "warmupsPerSceneAndMethod": arguments.warmups,
                "repetitionsPerSceneAndMethod": arguments.repetitions,
                "sampleUnit": "synchronized model inference wall time",
                "crossMethodTimingComparable": False,
                "timingInterpretation": (
                    "Descriptive per-method latency only. Different devices, frameworks, "
                    "precision, or preprocessing invalidate cross-method throughput claims."
                ),
            },
            "records": records,
            "limitations": [
                "External repository imports execute their Python code; use only trusted, pinned sources.",
                "Checkpoint and repository identities are recorded but not certified as official releases.",
                "Reference timings are not compared with WebGPU timings by the report tool.",
                "Practical-RIFE uses its canonical BGR tensor order and outputs are converted back to RGB for scoring.",
                "IFRNet uses the published SNU-FILM divisor-20 replicate padding and scale_factor=0.8 contract for every shootout scene.",
            ],
        }
        (staged_output / "run.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"output directory already exists: {output_path}")
        shutil.move(str(staged_output), str(output_path))

    print(json.dumps({"output": str(output_path), "scenes": len(records)}, indent=2))


if __name__ == "__main__":
    main()
