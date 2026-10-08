"""Evaluate a COAST checkpoint on a fixed validation or test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CDDataset
from train import validate
from utils import MetricTracker, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tracker = MetricTracker(args.threshold)
    manifest = None if args.split_manifest is None else args.split_manifest.resolve()
    dataset = CDDataset(args.data_root, args.split, manifest_path=manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = validate(load_model(args.checkpoint, device), loader, device, tracker)
    result = {"model": "COAST", "checkpoint": str(args.checkpoint), "split": args.split,
              "patch_count": len(dataset), "threshold": args.threshold, "metrics": metrics}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
