"""Build spatial-block cross-fitted GWDA priors without overwriting legacy priors.

For every training spatial block, fitting, neighbour selection, feature
standardization, and probability calibration exclude that target block.  The
validation and test priors are generated from a final GWDA model fitted only
to all training blocks.  The script is resumable at the outer-block level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from tools.gwda import (
    bilinear_resize,
    build_query_grid,
    collect_unique_training_samples,
    evaluate_saved_prior,
    load_manifest,
    local_gwda_predict,
    select_bandwidth_and_calibrator,
    standardize_fit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sample_stride", type=int, default=16)
    parser.add_argument("--prediction_stride", type=int, default=16)
    parser.add_argument(
        "--candidate_neighbors", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument("--inner_cv_folds", type=int, default=4)
    parser.add_argument("--cv_points_per_fold", type=int, default=2500)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--pixel_size_m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def row_group(row: dict[str, str]) -> str:
    return f'{row["region"]}|{row["block_x"]}|{row["block_y"]}'


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def predict_rows(
    root: Path,
    rows: list[dict[str, str]],
    output_dir: Path,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_coords: np.ndarray,
    neighbors: int,
    calibrator,
    ridge: float,
    prediction_stride: int,
) -> dict:
    mean, scale = standardize_fit(fit_x)
    query_x, query_coords, patch_keys = build_query_grid(root, rows, prediction_stride)
    raw, bandwidth = local_gwda_predict(
        (fit_x - mean) / scale,
        fit_y,
        fit_coords,
        (query_x - mean) / scale,
        query_coords,
        neighbors,
        ridge,
    )
    probability = calibrator.predict(raw)
    unique_keys = sorted({key for keys in patch_keys.values() for key in keys})
    if len(unique_keys) != len(probability):
        raise AssertionError("Query-key and probability counts differ")
    lookup = dict(zip(unique_keys, probability, strict=True))
    coarse_size = 256 // prediction_stride
    for name, keys in patch_keys.items():
        coarse = np.asarray([lookup[key] for key in keys], dtype=np.float32)
        coarse = coarse.reshape(coarse_size, coarse_size)
        prior = np.clip(bilinear_resize(coarse), 0.0, 1.0).astype(np.float32)
        atomic_save(output_dir / name, prior)
    return {
        "query_point_count": int(len(probability)),
        "patch_count": int(len(patch_keys)),
        "median_adaptive_bandwidth_pixels": float(np.median(bandwidth)),
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
        "probability_mean": float(np.mean(probability)),
    }


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.prediction_stride < 1 or 256 % args.prediction_stride:
        raise ValueError("Strides must be positive and prediction_stride must divide 256")
    if args.inner_cv_folds < 2 or args.ridge <= 0 or min(args.candidate_neighbors) < 16:
        raise ValueError("Invalid GWDA cross-validation, ridge or neighbour settings")
    started = time.time()
    root = args.data_root.resolve()
    manifest = (args.manifest or root / "spatial_split_manifest.csv").resolve()
    output_dir = (args.output_dir or root / "spatial_prior_gwda_oof").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    configuration["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    configuration["data_root"] = str(root)
    config_path = output_dir / "configuration.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError("GWDA configuration changed; use a new output directory")
    elif any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory for a new GWDA run")
    atomic_write_json(config_path, configuration)
    fold_dir = output_dir / "outer_folds"
    fold_dir.mkdir(exist_ok=True)
    status_path = output_dir / "pipeline_status.json"
    rows = load_manifest(manifest)

    x, y, coords, groups, source_files = collect_unique_training_samples(
        root, rows, args.sample_stride
    )
    train_groups = sorted(np.unique(groups).tolist())
    train_rows = [row for row in rows if row["split"] == "train"]
    nontrain_rows = [row for row in rows if row["split"] in {"val", "test"}]
    completed: list[str] = []
    outer_results: dict[str, dict] = {}

    atomic_write_json(
        status_path,
        {
            "status": "running",
            "stage": "outer_cross_fitting",
            "outer_group_count": len(train_groups),
            "completed_groups": completed,
            "started_unix": started,
        },
    )

    for index, target_group in enumerate(train_groups, start=1):
        target_rows = [row for row in train_rows if row_group(row) == target_group]
        expected = [output_dir / row["filename"] for row in target_rows]
        fold_path = fold_dir / f"{safe_name(target_group)}.json"
        if fold_path.exists() and expected and all(path.exists() for path in expected):
            outer_results[target_group] = json.loads(fold_path.read_text(encoding="utf-8"))
            completed.append(target_group)
            print(f"[skip {index}/{len(train_groups)}] {target_group}", flush=True)
            continue

        print(f"[outer {index}/{len(train_groups)}] {target_group}", flush=True)
        fit_mask = groups != target_group
        if not np.any(~fit_mask):
            raise AssertionError(f"No sampled points found for target group {target_group}")
        selected, calibrator, inner_cv = select_bandwidth_and_calibrator(
            x[fit_mask],
            y[fit_mask],
            coords[fit_mask],
            groups[fit_mask],
            args.candidate_neighbors,
            args.inner_cv_folds,
            args.cv_points_per_fold,
            args.ridge,
            args.seed + index,
        )
        prediction = predict_rows(
            root,
            target_rows,
            output_dir,
            x[fit_mask],
            y[fit_mask],
            coords[fit_mask],
            selected,
            calibrator,
            args.ridge,
            args.prediction_stride,
        )
        result = {
            "target_group": target_group,
            "target_patch_count": len(target_rows),
            "target_sample_count_excluded": int((~fit_mask).sum()),
            "fit_sample_count": int(fit_mask.sum()),
            "fit_group_count": int(len(np.unique(groups[fit_mask]))),
            "selected_neighbors": int(selected),
            "inner_cv": inner_cv,
            "prediction": prediction,
            "calibration_x": calibrator.X_thresholds_.tolist(),
            "calibration_y": calibrator.y_thresholds_.tolist(),
        }
        atomic_write_json(fold_path, result)
        outer_results[target_group] = result
        completed.append(target_group)
        atomic_write_json(
            status_path,
            {
                "status": "running",
                "stage": "outer_cross_fitting",
                "outer_group_count": len(train_groups),
                "completed_groups": completed,
                "current_group": target_group,
                "started_unix": started,
                "updated_unix": time.time(),
            },
        )

    print("[final] selecting GWDA settings on all training blocks", flush=True)
    selected, calibrator, full_cv = select_bandwidth_and_calibrator(
        x,
        y,
        coords,
        groups,
        args.candidate_neighbors,
        args.inner_cv_folds,
        args.cv_points_per_fold,
        args.ridge,
        args.seed,
    )
    final_prediction = predict_rows(
        root,
        nontrain_rows,
        output_dir,
        x,
        y,
        coords,
        selected,
        calibrator,
        args.ridge,
        args.prediction_stride,
    )

    expected_names = sorted(row["filename"] for row in rows if row["split"] in {"train", "val", "test"})
    missing = [name for name in expected_names if not (output_dir / name).exists()]
    if missing:
        raise AssertionError(f"Missing {len(missing)} prior patches; first: {missing[:5]}")
    selected_by_outer = [value["selected_neighbors"] for value in outer_results.values()]
    bandwidth_by_outer = [
        value["prediction"]["median_adaptive_bandwidth_pixels"]
        for value in outer_results.values()
    ]
    source_digest = hashlib.sha256("\n".join(source_files).encode("utf-8")).hexdigest()
    metadata = {
        "method": "geographically_weighted_linear_discriminant_analysis",
        "fit_split": "train_only_spatial_block_cross_fitted",
        "training_prior_protocol": (
            "leave-one-training-spatial-block-out outer cross-fitting; target block excluded "
            "from fitting, neighbour selection, standardization, and calibration"
        ),
        "validation_test_prior_protocol": (
            "fit on all training blocks after training-only spatial-block hyperparameter selection"
        ),
        "label_source": "change reference labels from training spatial blocks only",
        "feature_source": "eight pre-eradication image channels",
        "coordinate_role": "adaptive Gaussian geographic weights only",
        "manifest": str(manifest),
        "training_source_file_count": len(source_files),
        "training_source_files_sha256": source_digest,
        "unique_training_sample_count": int(len(y)),
        "positive_training_sample_count": int(y.sum()),
        "negative_training_sample_count": int((1 - y).sum()),
        "sample_stride_pixels": args.sample_stride,
        "prediction_stride_pixels": args.prediction_stride,
        "kernel": "adaptive Gaussian truncated to k nearest samples",
        "candidate_neighbors": args.candidate_neighbors,
        "ridge": args.ridge,
        "pixel_size_m": args.pixel_size_m,
        "outer_fold_count": len(train_groups),
        "outer_selected_neighbors": selected_by_outer,
        "outer_selected_neighbors_distribution": {
            str(value): selected_by_outer.count(value) for value in sorted(set(selected_by_outer))
        },
        "median_outer_bandwidth_pixels": float(np.median(bandwidth_by_outer)),
        "median_outer_bandwidth_m": float(np.median(bandwidth_by_outer) * args.pixel_size_m),
        "outer_folds": outer_results,
        "final_selected_neighbors": int(selected),
        "final_bandwidth_selection": full_cv,
        "final_prediction": final_prediction,
        "final_calibration_x": calibrator.X_thresholds_.tolist(),
        "final_calibration_y": calibrator.y_thresholds_.tolist(),
        "posterior_metrics": evaluate_saved_prior(root, rows, output_dir),
        "seed": args.seed,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_write_json(output_dir / "gwda_metadata.json", metadata)
    atomic_write_json(
        status_path,
        {
            "status": "completed",
            "stage": "completed",
            "outer_group_count": len(train_groups),
            "completed_groups": train_groups,
            "selected_neighbors": int(selected),
            "elapsed_seconds": metadata["elapsed_seconds"],
            "updated_unix": time.time(),
        },
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "outer_fold_count": len(train_groups),
        "selected_neighbors": selected,
        "posterior_metrics": metadata["posterior_metrics"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
