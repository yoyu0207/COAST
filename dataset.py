"""Aligned eight-channel image pairs, binary labels and optional GWDA targets."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_image(path):
    image = np.load(path, allow_pickle=False).astype(np.float32)
    if image.ndim != 3 or image.shape[0] != 8:
        raise ValueError(f"Expected [8, H, W] input: {path}, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Non-finite image values: {path}")
    return image


class CDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=False,
                 prior_dir_name=None, manifest_path=None):
        self.root = Path(root_dir).resolve()
        self.transform = transform
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        manifest = Path(manifest_path or "spatial_split_manifest.csv")
        if not manifest.is_absolute():
            manifest = self.root / manifest
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"filename", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("Manifest requires filename and split columns")
            rows = list(reader)
        seen = set()
        self.file_list = []
        for row in rows:
            name, assignment = row["filename"].strip(), row["split"].strip()
            if name in seen or Path(name).name != name or not name.endswith(".npy"):
                raise ValueError(f"Duplicate or invalid patch filename: {name}")
            if assignment not in {"train", "val", "test", "excluded"}:
                raise ValueError(f"Invalid split: {assignment}")
            seen.add(name)
            if assignment == split:
                self.file_list.append(name)
        self.file_list.sort()
        if not self.file_list:
            raise ValueError(f"No {split} patches in {manifest}")
        self.prior_dir = None
        if prior_dir_name is not None:
            self.prior_dir = self.root / prior_dir_name
        for name in self.file_list:
            paths = [self.root / folder / name for folder in ("A", "B", "label")]
            if self.prior_dir is not None:
                paths.append(self.prior_dir / name)
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        name = self.file_list[index]
        image_a = load_image(self.root / "A" / name)
        image_b = load_image(self.root / "B" / name)
        label = np.load(self.root / "label" / name, allow_pickle=False).astype(np.float32)
        if label.ndim == 3 and label.shape[0] == 1:
            label = label[0]
        shape = image_a.shape[-2:]
        if image_b.shape != image_a.shape or label.shape != shape:
            raise ValueError(f"Image/label shapes do not match: {name}")
        if not np.isfinite(label).all() or not np.isin(label, [0, 1]).all():
            raise ValueError(f"Expected binary 0/1 labels: {name}")
        prior = (np.load(self.prior_dir / name, allow_pickle=False).astype(np.float32)
                 if self.prior_dir is not None else np.zeros(shape, dtype=np.float32))
        if prior.ndim == 3 and prior.shape[0] == 1:
            prior = prior[0]
        if prior.shape != shape or not np.isfinite(prior).all():
            raise ValueError(f"Invalid GWDA target shape/values: {name}")
        if prior.min() < 0 or prior.max() > 1:
            raise ValueError(f"GWDA target must lie in [0, 1]: {name}")
        if self.transform:
            if np.random.rand() > 0.5:
                image_a, image_b = np.flip(image_a, 2), np.flip(image_b, 2)
                label, prior = np.flip(label, 1), np.flip(prior, 1)
            if np.random.rand() > 0.5:
                image_a, image_b = np.flip(image_a, 1), np.flip(image_b, 1)
                label, prior = np.flip(label, 0), np.flip(prior, 0)
            k = np.random.randint(0, 4)
            image_a, image_b = np.rot90(image_a, k, (1, 2)), np.rot90(image_b, k, (1, 2))
            label, prior = np.rot90(label, k), np.rot90(prior, k)
        return (torch.from_numpy(image_a.copy()), torch.from_numpy(image_b.copy()),
                torch.from_numpy(label[None].copy()), torch.from_numpy(prior[None].copy()))
