"""Train COAST with spatial-block OOF GWDA posterior soft supervision."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CDDataset
from losses import BCEHybridLoss, coast_loss
from models import COAST
from utils import MetricTracker, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--prior_dir", default="spatial_prior_gwda_oof")
    parser.add_argument("--output_root", type=Path, default=Path("experiments"))
    parser.add_argument("--run_name")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=60, help="0 disables early stopping")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--spg_lr", type=float, default=1e-4)
    parser.add_argument("--spg_gamma_lr", type=float)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--boundary_weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--deterministic_warn_only", action="store_true")
    parser.add_argument("--skip_test", action="store_true", help="Validation-only configuration selection")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        parser.error("epochs/batch_size must be positive; patience must be non-negative")
    return args


def build_optimizer(model, lr=5e-5, spg_lr=1e-4, spg_gamma_lr=None):
    excluded = {id(p) for module in (model.prior_encoder, model.spg1, model.spg2)
                for p in module.parameters()}
    gamma = [model.spg1.gamma, model.spg2.gamma]
    gamma_ids = {id(p) for p in gamma}
    projection = [p for module in (model.spg1, model.spg2) for p in module.parameters()
                  if id(p) not in gamma_ids]
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in excluded], "lr": lr},
        {"params": list(model.prior_encoder.parameters()), "lr": 5e-5},
        {"params": projection, "lr": spg_lr},
        {"params": gamma, "lr": spg_lr if spg_gamma_lr is None else spg_gamma_lr,
         "weight_decay": 0.0},
    ], weight_decay=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha=0.1,
                    boundary_weight=0.2, scaler=None, amp=False, amp_dtype=torch.float16):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        image_a, image_b, labels, posterior = [item.to(device) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
            outputs = model(image_a, image_b, return_prior=True)
            loss = coast_loss(outputs, labels, posterior, criterion, alpha, boundary_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite COAST training loss")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device, tracker=None):
    model.eval()
    tracker = tracker or MetricTracker()
    tracker.reset()
    for image_a, image_b, labels, _ in tqdm(loader, desc="Evaluate", leave=False):
        tracker.update(model(image_a.to(device), image_b.to(device)), labels.to(device))
    return tracker.get_metrics()


def main():
    args = parse_args()
    set_global_seed(args.seed, args.deterministic_warn_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root.resolve()
    manifest = (args.split_manifest.resolve() if args.split_manifest is not None
                else root / "spatial_split_manifest.csv")
    train_ds = CDDataset(root, "train", transform=True, prior_dir_name=args.prior_dir,
                         manifest_path=manifest)
    val_ds = CDDataset(root, "val", manifest_path=manifest)
    test_ds = None if args.skip_test else CDDataset(root, "test", manifest_path=manifest)
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = None if test_ds is None else DataLoader(test_ds, shuffle=False, **kwargs)
    run_name = args.run_name or f"COAST_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, run_dir / "split_manifest.csv")
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    model = COAST().to(device)
    optimizer = build_optimizer(model, args.lr, args.spg_lr, args.spg_gamma_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = BCEHybridLoss()
    amp = args.amp and device.type == "cuda"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    best_f1, stale, best_epoch = -1.0, 0, 0
    started = time.perf_counter()
    print(f"COAST | {device} | seed {args.seed} | {run_dir}", flush=True)
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_f1", "val_iou", "val_precision", "val_recall", "lr"])
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                   args.alpha, args.boundary_weight, scaler, amp, amp_dtype)
            metrics = validate(model, val_loader, device)
            scheduler.step()
            writer.writerow([epoch, loss, metrics["F1"], metrics["IoU"], metrics["Precision"],
                             metrics["Recall"], scheduler.get_last_lr()[0]])
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs} | loss {loss:.4f} | val F1 {metrics['F1']:.4f}", flush=True)
            if metrics["F1"] > best_f1:
                best_f1, stale, best_epoch = metrics["F1"], 0, epoch
                torch.save(model.state_dict(), run_dir / "best_model.pth")
            else:
                stale += 1
            if args.patience and stale >= args.patience:
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    test_metrics = None if test_loader is None else validate(model, test_loader, device)
    summary = {"model": "COAST", "seed": args.seed, "best_val_f1": best_f1,
               "best_epoch": best_epoch, "epochs_completed": epoch,
               "training_seconds": training_seconds, "test_metrics": test_metrics,
               "split_counts": {"train": len(train_ds), "val": len(val_ds),
                                "test": None if test_ds is None else len(test_ds)},
               "configuration": configuration}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
