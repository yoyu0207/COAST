"""Predict removal-related change from paired eight-channel NumPy patches."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dataset import load_image
from utils import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--t2", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    image_a, image_b = load_image(args.t1), load_image(args.t2)
    if image_a.shape != image_b.shape:
        raise ValueError("T1 and T2 must have identical shapes")
    if image_a.shape[-2:] != (256, 256):
        raise ValueError("This patch entry point expects 256 x 256 inputs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tensor_a = torch.from_numpy(image_a[None]).to(device)
    tensor_b = torch.from_numpy(image_b[None]).to(device)
    logits, prior, _ = model(tensor_a, tensor_b, return_prior=True)
    probability = logits.sigmoid()[0, 0].cpu().numpy()
    prior = prior[0, 0].cpu().numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "change_probability.npy", probability)
    np.save(args.output_dir / "online_prior.npy", prior)
    Image.fromarray(((probability > args.threshold) * 255).astype(np.uint8)).save(
        args.output_dir / "change_mask.png")
    print(f"Predictions saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
