"""Small CPU regressions for checkpoint safety, data labels, and COD metrics."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import torch
from torch import nn

from hfgf_dinov3.checkpoint import load_trained_weights
from hfgf_dinov3.evaluation.metrics import calc_metrics


class ReleaseTests(unittest.TestCase):
    def test_trained_checkpoint_roundtrip(self):
        source, target = nn.Linear(3, 2), nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save({"model": source.state_dict(), "iteration": 7}, checkpoint)
            load_trained_weights(target, checkpoint)
        for name, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

    def test_partial_checkpoint_rejected_without_mutation(self):
        model = nn.Linear(3, 2)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "partial.pth"
            torch.save({"model": {"weight": torch.ones_like(model.weight)}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Checkpoint/model mismatch"):
                load_trained_weights(model, checkpoint)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))

    def test_perfect_and_inverted_binary_masks(self):
        target = np.zeros((32, 40), dtype=np.float32)
        target[8:24, 10:30] = 1
        perfect = calc_metrics(target, target)
        inverted = calc_metrics(1 - target, target)
        self.assertTrue(np.isfinite(perfect).all())
        self.assertAlmostEqual(perfect[0], 1.0)
        self.assertAlmostEqual(perfect[2], 1.0)
        self.assertEqual(perfect[3], 0.0)
        self.assertEqual(inverted[3], 1.0)
        self.assertGreater(perfect[1], inverted[1])

    def test_empty_and_full_foreground_are_finite(self):
        for foreground in (0, 1):
            target = np.full((16, 20), foreground, dtype=np.float32)
            values = calc_metrics(target, target)
            self.assertTrue(np.isfinite(values).all())
            self.assertEqual(values[3], 0.0)

    def test_preparation_preserves_raw_masks_and_class_convention(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "prepare_datasets.py"
        spec = importlib.util.spec_from_file_location("prepare_datasets", script)
        preparation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparation)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "CAMO-V.1.0-CVIU2019"
            (base / "Images" / "Train").mkdir(parents=True)
            (base / "GT").mkdir()
            Image.new("RGB", (12, 10), (80, 100, 120)).save(base / "Images" / "Train" / "sample.jpg")
            foreground = np.zeros((10, 12), dtype=np.uint8)
            foreground[2:8, 3:9] = 255
            raw = base / "GT" / "sample.png"
            Image.fromarray(foreground).save(raw)
            original_bytes = raw.read_bytes()
            preparation.prepare_split(root, "camo", "Train")
            labels = np.asarray(Image.open(base / "annotations_detectron2" / "cod_semantic_train" / "sample.png"))
            self.assertTrue(np.array_equal(labels, (foreground == 0).astype(np.uint8)))
            annotation = json.loads((base / "COD_Train.json").read_text())["annotations"][0]
            self.assertTrue(np.array_equal(mask_util.decode(annotation["segmentation"]), foreground > 0))
            self.assertEqual(raw.read_bytes(), original_bytes)
            with self.assertRaises(FileExistsError):
                preparation.prepare_split(root, "camo", "Train")


if __name__ == "__main__":
    unittest.main()
