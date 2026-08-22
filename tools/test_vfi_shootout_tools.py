from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


references = load_module("vfi_reference_runner", "tools/run_vfi_references.py")
reporter = load_module("vfi_reporter", "tools/report_vfi_shootout.py")
manifest = json.loads(
    (ROOT / "benchmarks/vfi-shootout/manifest.json").read_text(encoding="utf-8")
)
scene_ids = [
    scene["id"]
    for scene in json.loads(
        (ROOT / "benchmarks/vfi-shootout/scenes.json").read_text(encoding="utf-8")
    )
]


def stable_file(value: str = "a") -> dict[str, object]:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {
        "path": "fixture",
        "sizeBytes": 1,
        "sha256": digest,
        "sha256End": digest,
    }


def clean_git(commit: str = "1" * 40) -> dict[str, object]:
    return {
        "commit": commit,
        "branch": "main",
        "dirty": False,
        "statusPorcelain": [],
        "errors": [],
    }


def framegen_run(resolution: int) -> dict[str, object]:
    rung = manifest["inferenceRungs"][str(resolution)]
    adapter = {
        "vendor": "10de",
        "architecture": "ada",
        "device": "2803",
        "description": "NVIDIA RTX test adapter",
    }
    features = ["shader-f16", "subgroups"]
    limits = {
        "maxBufferSize": 2147483648,
        "maxTextureDimension2D": 32768,
    }
    return {
        "source": {
            "files": {
                key: stable_file(chr(ord("a") + index))
                for index, key in enumerate(reporter.FRAMEGEN_SOURCE_KEYS)
            }
        },
        "host": {
            "platform": "win32",
            "arch": "x64",
            "osRelease": "test",
            "node": "v24.0.0",
            "cpu": "test cpu",
        },
        "browser": {
            "channel": "chrome",
            "headed": True,
            "playwrightCliPackage": "@playwright/cli@0.1.17",
            "playwrightCliVersion": "0.1.17",
            "environment": {"userAgent": "test Chrome/1", "platform": "Win32"},
            "adapterIdentity": adapter,
            "deviceFeatures": features,
            "deviceLimits": limits,
        },
        "workload": {
            "model": "v7s",
            "inferenceResolution": resolution,
            "inferenceRung": rung,
            "timestep": 0.5,
            "scenes": scene_ids,
        },
        "records": [
            {
                "scene": scene_id,
                "rung": rung,
                "adapterIdentity": adapter,
                "deviceFeatures": features,
                "deviceLimits": limits,
            }
            for scene_id in scene_ids
        ],
    }


def reference_run() -> dict[str, object]:
    models = {}
    for key in ("practicalRife", "ifrnet"):
        git = clean_git()
        implementation_path = (
            "train_log/RIFE_HDv3.py" if key == "practicalRife" else "models/IFRNet.py"
        )
        checkpoint = stable_file("d")
        if key == "practicalRife":
            checkpoint["path"] = "train_log/flownet.pkl"
        models[key] = {
            "repository": {"git": git, "gitEnd": copy.deepcopy(git)},
            "checkpoint": checkpoint,
            "implementation": {**stable_file("e"), "path": implementation_path},
            "importedSources": [
                {**stable_file(f"{key}-source"), "path": implementation_path}
            ],
            "preprocessing": manifest["referenceMethods"][key],
        }
    return {
        "source": {
            "runner": stable_file("f"),
            "scenes": stable_file("1"),
            "shootoutManifest": stable_file("2"),
        },
        "models": models,
        "measurement": {"timestep": 0.5},
    }


class ReferenceRunnerTests(unittest.TestCase):
    def test_rife_channel_conversion_is_bgr_and_round_trips(self) -> None:
        rgb = np.array([[[[10]], [[20]], [[30]]]])
        bgr = references.swap_rgb_bgr(rgb)
        np.testing.assert_array_equal(bgr[:, :, 0, 0], [[30, 20, 10]])
        np.testing.assert_array_equal(references.swap_rgb_bgr(bgr), rgb)

    def test_rife_checkpoint_must_be_the_exact_canonical_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Practical-RIFE"
            train_log = root / "train_log"
            train_log.mkdir(parents=True)
            checkpoint = train_log / "flownet.pkl"
            checkpoint.write_bytes(b"weights")
            self.assertEqual(
                references.validate_rife_layout(root, checkpoint, "flownet.pkl"),
                checkpoint.resolve(),
            )
            decoy = train_log / "decoy.pkl"
            decoy.write_bytes(b"not loaded")
            with self.assertRaisesRegex(ValueError, "exact canonical file"):
                references.validate_rife_layout(root, decoy, "flownet.pkl")

    def test_dirty_or_missing_git_provenance_is_rejected(self) -> None:
        metadata = clean_git()
        self.assertEqual(
            references.require_clean_git_repository(
                "reference", Path("unused"), metadata
            ),
            metadata,
        )
        dirty = {**metadata, "dirty": True, "statusPorcelain": [" M model.py"]}
        with self.assertRaisesRegex(RuntimeError, "must be clean"):
            references.require_clean_git_repository("reference", Path("unused"), dirty)
        missing = {**metadata, "commit": None, "errors": ["not a repository"]}
        with self.assertRaisesRegex(RuntimeError, "readable Git repository"):
            references.require_clean_git_repository(
                "reference", Path("unused"), missing
            )

    def test_imported_reference_code_is_bound_to_its_exact_source(self) -> None:
        module_name = "_framegen_vfi_exact_import_fixture"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / f"{module_name}.py"
            source.write_text("VALUE = 7\n", encoding="utf-8")
            sys.path.insert(0, str(root))
            try:
                module, imported_path = references.import_exact_module(
                    module_name, source
                )
                self.assertEqual(module.VALUE, 7)
                self.assertEqual(imported_path, source.resolve())
                metadata = references.imported_python_metadata(root)
                self.assertEqual([row["path"] for row in metadata], [source.name])
                self.assertFalse((root / "__pycache__").exists())
                with self.assertRaisesRegex(RuntimeError, "expected the pinned source"):
                    references.import_exact_module(module_name, root / "other.py")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop(module_name, None)

    def test_scene_ids_and_staged_outputs_are_confined(self) -> None:
        self.assertEqual(
            references.validate_scene_id("middlebury-test", "id"), "middlebury-test"
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            references.validate_scene_id("../../escape", "id")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            self.assertEqual(
                references.confined_output(root, "scene/output.png", "output"),
                (root / "scene/output.png").resolve(),
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                references.confined_output(root, "../escape.png", "output")

    def test_manifest_pins_canonical_reference_preprocessing(self) -> None:
        configuration = references.load_reference_configuration()
        self.assertEqual(configuration["practicalRife"]["inputColorOrder"], "BGR")
        self.assertEqual(configuration["ifrnet"]["paddingDivisor"], 20)
        self.assertEqual(configuration["ifrnet"]["scaleFactor"], 0.8)

        class Model:
            def inference(self, image0, image1, timestep, *, scale_factor):
                return image0, image1, timestep, scale_factor

        self.assertEqual(
            references.run_ifrnet_inference(Model(), "a", "b", 0.5, 0.8),
            ("a", "b", 0.5, 0.8),
        )


class ReporterTests(unittest.TestCase):
    def test_identical_framegen_build_and_device_pair_is_accepted(self) -> None:
        reporter.validate_framegen_pair(framegen_run(480), framegen_run(720), manifest)

    def test_framegen_build_timestep_and_device_mismatches_are_rejected(self) -> None:
        run_480 = framegen_run(480)
        run_720 = framegen_run(720)
        run_720["source"]["files"]["weights"] = stable_file("9")
        with self.assertRaisesRegex(ValueError, "different source builds"):
            reporter.validate_framegen_pair(run_480, run_720, manifest)

        run_720 = framegen_run(720)
        run_720["workload"]["timestep"] = 0.25
        with self.assertRaisesRegex(ValueError, "workload identity"):
            reporter.validate_framegen_pair(run_480, run_720, manifest)

        run_720 = framegen_run(720)
        run_720["browser"]["adapterIdentity"]["device"] = "different"
        for record in run_720["records"]:
            record["adapterIdentity"] = run_720["browser"]["adapterIdentity"]
        with self.assertRaisesRegex(ValueError, "different host, browser, or GPU"):
            reporter.validate_framegen_pair(run_480, run_720, manifest)

    def test_reference_run_requires_clean_exact_provenance(self) -> None:
        run = reference_run()
        reporter.validate_reference_run(run, manifest)
        run["models"]["ifrnet"]["repository"]["git"]["dirty"] = True
        with self.assertRaisesRegex(ValueError, "dirty"):
            reporter.validate_reference_run(run, manifest)

    def test_metric_cap_and_scene_header_layout_are_explicit(self) -> None:
        pixels = np.zeros((16, 16, 3), dtype=np.float64)
        self.assertEqual(reporter.psnr(pixels, pixels), 99.0)
        part = Image.new("RGB", (8, 6), "#ff0000")
        sheet = reporter.compose_scene_sheet("scene", [part], header_height=20)
        self.assertEqual(sheet.size, (8, 26))
        self.assertEqual(sheet.getpixel((0, 20)), (255, 0, 0))
        self.assertNotEqual(sheet.getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
