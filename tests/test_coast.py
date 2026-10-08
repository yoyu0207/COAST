"""Focused CPU tests for the public COAST interfaces."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataset import CDDataset
from losses import BCEHybridLoss, coast_loss
from models import COAST
from train import build_optimizer
from utils import MetricTracker, load_model
from tools.gwda import select_bandwidth_and_calibrator
from tools.make_spatial_split import Patch, verify_split


class COASTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_loss_optimizer_and_checkpoint(self):
        torch.manual_seed(42)
        model = COAST()
        image = torch.rand(2, 8, 32, 32)
        labels = (torch.rand(2, 1, 32, 32) > 0.8).float()
        posterior = torch.rand_like(labels)
        optimizer = build_optimizer(model)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        outputs = model(image, image, return_prior=True)
        self.assertTrue(all(output.shape == labels.shape for output in outputs))
        loss = coast_loss(outputs, labels, posterior, BCEHybridLoss())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.prior_encoder.eco_score[0].weight.grad)
        self.assertIsNotNone(model.boundary_head[-1].weight.grad)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(image, image)
            with tempfile.TemporaryDirectory() as folder:
                checkpoint = Path(folder) / "model.pth"
                torch.save(model.state_dict(), checkpoint)
                restored = load_model(checkpoint, torch.device("cpu"))
                torch.testing.assert_close(restored(image, image), expected, rtol=0, atol=0)

    def test_dataset_requires_targets_only_when_requested(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory in ("A", "B", "label"):
                (root / directory).mkdir()
            name = "region_0_0.npy"
            for directory in ("A", "B"):
                np.save(root / directory / name, np.ones((8, 32, 32), np.float32))
            np.save(root / "label" / name, np.zeros((32, 32), np.float32))
            with (root / "spatial_split_manifest.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["filename", "split"])
                writer.writerow([name, "train"])
            self.assertEqual(CDDataset(root)[0][0].shape, (8, 32, 32))
            with self.assertRaises(FileNotFoundError):
                CDDataset(root, prior_dir_name="spatial_prior_gwda_oof")
            (root / "spatial_prior_gwda_oof").mkdir()
            np.save(root / "spatial_prior_gwda_oof" / name, np.full((32, 32), 0.4, np.float32))
            sample = CDDataset(root, transform=True, prior_dir_name="spatial_prior_gwda_oof")[0]
            self.assertAlmostEqual(sample[3].mean().item(), 0.4, places=6)

    def test_pooled_metrics(self):
        tracker = MetricTracker()
        tracker.update(torch.tensor([4., 4., -4., -4.]), torch.tensor([1., 0., 1., 0.]))
        self.assertAlmostEqual(tracker.get_metrics()["F1"], 0.5, places=6)
        self.assertAlmostEqual(tracker.get_metrics()["IoU"], 1 / 3, places=6)

    def test_spatial_split_detects_overlap(self):
        left = Patch("region_0_0.npy", "region", 0, 0, 0.1)
        right = Patch("region_128_0.npy", "region", 128, 0, 0.1)
        with self.assertRaises(AssertionError):
            verify_split([left, right], {left.filename: "train", right.filename: "test"}, 256, 256)

    def test_gwda_spatial_selection(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(80, 8))
        labels = (features[:, 0] > 0).astype(np.uint8)
        coords = rng.uniform(size=(80, 2)) * 1000
        groups = np.repeat(np.arange(4), 20)
        neighbors, calibrator, _ = select_bandwidth_and_calibrator(
            features, labels, coords, groups, [16, 32], 4, 20, 1e-3, 42)
        self.assertIn(neighbors, [16, 32])
        self.assertTrue(np.isfinite(calibrator.predict([0., 0.5, 1.])).all())


if __name__ == "__main__":
    unittest.main()
