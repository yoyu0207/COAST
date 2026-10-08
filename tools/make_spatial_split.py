"""Create a leakage-safe spatial train/validation/test manifest.

Patch filenames must end in ``_<x>_<y>`` where ``x`` and ``y`` are the
upper-left pixel coordinates in the source raster. Patches are assigned by
large spatial blocks. Samples that violate the requested inter-split buffer
are marked as excluded instead of being silently reassigned.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Patch:
    filename: str
    region: str
    x: int
    y: int
    positive_fraction: float

    @property
    def has_change(self) -> bool:
        return self.positive_fraction > 0.0


def parse_patch_name(filename: str) -> tuple[str, int, int]:
    stem = Path(filename).stem
    try:
        region, x, y = stem.rsplit("_", 2)
        return region, int(x), int(y)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot parse spatial coordinates from patch name: {filename}. "
            "Expected a name ending in _<x>_<y>."
        ) from exc


def load_label(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        label = np.load(path)
    else:
        label = np.asarray(Image.open(path))
    if label.ndim > 2:
        label = np.squeeze(label)
    return label > 0.5


def discover_patches(label_dir: Path) -> list[Patch]:
    files = sorted(label_dir.glob("*.npy"))
    if not files:
        files = sorted(label_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .npy or .png labels found in {label_dir}")

    patches = []
    for path in files:
        region, x, y = parse_patch_name(path.name)
        patches.append(
            Patch(
                filename=path.name,
                region=region,
                x=x,
                y=y,
                positive_fraction=float(load_label(path).mean()),
            )
        )
    return patches


def buffered_conflict(a: Patch, b: Patch, patch_size: int, buffer: int) -> bool:
    if a.region != b.region:
        return False
    return (
        max(a.x - buffer, b.x) < min(a.x + patch_size + buffer, b.x + patch_size)
        and max(a.y - buffer, b.y)
        < min(a.y + patch_size + buffer, b.y + patch_size)
    )


def build_conflicts(
    patches: list[Patch], patch_size: int, buffer: int
) -> dict[str, set[str]]:
    conflicts = {patch.filename: set() for patch in patches}
    for index, left in enumerate(patches):
        for right in patches[index + 1 :]:
            if buffered_conflict(left, right, patch_size, buffer):
                conflicts[left.filename].add(right.filename)
                conflicts[right.filename].add(left.filename)
    return conflicts


def positive_mean(patches: list[Patch]) -> float:
    return float(np.mean([patch.positive_fraction for patch in patches]))


def choose_assignment(
    patches: list[Patch],
    block_size: int,
    patch_size: int,
    buffer: int,
    seed: int,
    trials: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, str], dict]:
    blocks: dict[tuple[str, int, int], list[Patch]] = defaultdict(list)
    for patch in patches:
        block = (
            patch.region,
            (patch.x + patch_size // 2) // block_size,
            (patch.y + patch_size // 2) // block_size,
        )
        blocks[block].append(patch)

    block_keys = sorted(blocks)
    conflicts = build_conflicts(patches, patch_size, buffer)
    overall_positive = positive_mean(patches)
    rng = np.random.default_rng(seed)
    target = {
        "train": len(patches) * train_ratio,
        "val": len(patches) * val_ratio,
        "test": len(patches) * test_ratio,
    }

    best = None
    probabilities = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    probabilities /= probabilities.sum()

    for _ in range(trials):
        categories = rng.choice(3, len(block_keys), p=probabilities)
        if np.sum(categories == 1) < 3 or np.sum(categories == 2) < 3:
            continue

        raw = {split: [] for split in VALID_SPLITS}
        for block, category in zip(block_keys, categories):
            raw[VALID_SPLITS[int(category)]].extend(blocks[block])

        minimum_eval = max(20, int(len(patches) * 0.08))
        if len(raw["val"]) < minimum_eval or len(raw["test"]) < minimum_eval:
            continue

        test = raw["test"]
        test_names = {patch.filename for patch in test}
        val = [
            patch
            for patch in raw["val"]
            if not (conflicts[patch.filename] & test_names)
        ]
        evaluation_names = test_names | {patch.filename for patch in val}
        train = [
            patch
            for patch in raw["train"]
            if not (conflicts[patch.filename] & evaluation_names)
        ]

        if len(val) < minimum_eval or len(test) < minimum_eval:
            continue
        if len(train) < int(len(patches) * 0.50):
            continue
        if not all(any(patch.has_change for patch in group) for group in (train, val, test)):
            continue

        retained = {"train": train, "val": val, "test": test}
        excluded_count = len(patches) - sum(len(group) for group in retained.values())
        count_error = sum(
            abs(len(retained[split]) - target[split]) / max(target[split], 1.0)
            for split in VALID_SPLITS
        )
        prevalence_error = sum(
            abs(positive_mean(retained[split]) - overall_positive)
            / max(overall_positive, 1e-8)
            for split in VALID_SPLITS
        )
        exclusion_penalty = 2.0 * excluded_count / len(patches)
        score = count_error + prevalence_error + exclusion_penalty

        if best is None or score < best[0]:
            best = (score, categories.copy(), retained)

    if best is None:
        raise RuntimeError(
            "No valid spatial split was found. Increase --trials or adjust the "
            "block size, buffer, or target ratios."
        )

    score, categories, retained = best
    assignments = {patch.filename: "excluded" for patch in patches}
    for split, group in retained.items():
        for patch in group:
            assignments[patch.filename] = split

    block_assignments = {
        "|".join(map(str, block)): VALID_SPLITS[int(category)]
        for block, category in zip(block_keys, categories)
    }
    summary = {
        "strategy": "large_spatial_blocks_with_cross_split_buffer",
        "seed": seed,
        "trials": trials,
        "target_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "patch_size_pixels": patch_size,
        "block_size_pixels": block_size,
        "buffer_pixels": buffer,
        "source_patch_count": len(patches),
        "retained_patch_count": sum(len(group) for group in retained.values()),
        "excluded_patch_count": sum(split == "excluded" for split in assignments.values()),
        "counts": {split: len(retained[split]) for split in VALID_SPLITS},
        "positive_pixel_fraction": {
            split: positive_mean(retained[split]) for split in VALID_SPLITS
        },
        "positive_patch_count": {
            split: sum(patch.has_change for patch in retained[split])
            for split in VALID_SPLITS
        },
        "objective_score": score,
        "block_assignments": block_assignments,
    }
    return assignments, summary


def verify_split(
    patches: list[Patch],
    assignments: dict[str, str],
    patch_size: int,
    buffer: int,
) -> None:
    retained = [patch for patch in patches if assignments[patch.filename] in VALID_SPLITS]
    for index, left in enumerate(retained):
        left_split = assignments[left.filename]
        for right in retained[index + 1 :]:
            right_split = assignments[right.filename]
            if left_split == right_split:
                continue
            if buffered_conflict(left, right, patch_size, buffer):
                raise AssertionError(
                    f"Spatial leakage between {left.filename} ({left_split}) and "
                    f"{right.filename} ({right_split})"
                )


def write_outputs(
    patches: list[Patch],
    assignments: dict[str, str],
    summary: dict,
    output_path: Path,
    block_size: int,
    patch_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "region",
                "x",
                "y",
                "block_x",
                "block_y",
                "split",
                "positive_fraction",
            ],
        )
        writer.writeheader()
        for patch in sorted(patches, key=lambda item: item.filename):
            writer.writerow(
                {
                    "filename": patch.filename,
                    "region": patch.region,
                    "x": patch.x,
                    "y": patch.y,
                    "block_x": (patch.x + patch_size // 2) // block_size,
                    "block_y": (patch.y + patch_size // 2) // block_size,
                    "split": assignments[patch.filename],
                    "positive_fraction": f"{patch.positive_fraction:.8f}",
                }
            )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--buffer", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    label_dir = args.data_root / "label"
    output = args.output or args.data_root / "spatial_split_manifest.csv"
    patches = discover_patches(label_dir)
    assignments, summary = choose_assignment(
        patches=patches,
        block_size=args.block_size,
        patch_size=args.patch_size,
        buffer=args.buffer,
        seed=args.seed,
        trials=args.trials,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    verify_split(patches, assignments, args.patch_size, args.buffer)
    write_outputs(
        patches,
        assignments,
        summary,
        output,
        args.block_size,
        args.patch_size,
    )

    print(json.dumps(summary, indent=2))
    print(f"Manifest: {output}")
    print(f"Summary : {output.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
