"""Run the Issue 89 synthetic and real-data backend entry gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from recad.evaluate.benchmark_backends import (
    PATCH_FAMILIES,
    TABULAR_FAMILIES,
    fit_patch_estimator,
    fit_tabular_estimator,
    predict_patch_estimator,
    predict_tabular_estimator,
)
from recad.evaluate.coastal_patches import CoastalPatchReader
from recad.evaluate.model_zoo import TabularTransform
from run_p2_validation_repair_v2_4 import SSS_RESIDUAL_FEATURES, load_frame

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_model_family_benchmark_v2.5")
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
SEED = 89001


SMOKE_PARAMETERS: dict[str, dict[str, object]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 32,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.8,
    },
    "extra_trees": {
        "n_estimators": 32,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": 0.8,
    },
    "hist_gradient_boosting": {
        "max_iter": 50,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.1,
    },
    "xgboost": {
        "n_estimators": 50,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "lightgbm": {
        "n_estimators": 50,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    },
    "catboost": {
        "iterations": 50,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.1,
    },
    "residual_mlp": {
        "width": 64,
        "depth": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 2,
    },
    "realmlp": {
        "hidden_sizes": "256-256-256",
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 2,
    },
    "tabm": {
        "width": 64,
        "depth": 2,
        "ensemble_size": 8,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 3e-4,
        "epochs": 2,
    },
    "ft_transformer": {
        "width": 64,
        "depth": 2,
        "heads": 4,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 2,
    },
    "tabpfn_3": {
        "n_estimators": 4,
        "inference_precision": "auto",
        "internal_subsample": None,
        "fit_mode": "fit_preprocessors",
    },
    "unet_convlstm": {
        "base_channels": 8,
        "depth": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "epochs": 2,
        "batch_size": 32,
    },
    "vision_transformer": {
        "width": 64,
        "depth": 2,
        "heads": 4,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "epochs": 2,
        "batch_size": 32,
    },
}


def stable_rows() -> tuple[object, object]:
    frame = load_frame("SSS")
    pool = frame.loc[frame.yr.eq(2018) & frame.mon.isin([6, 7, 8])].copy()
    key = pool.record_id.map(lambda value: hashlib.sha256(f"issue89:{value}".encode()).hexdigest())
    pool = pool.assign(_key=key).sort_values("_key").drop(columns="_key")
    if len(pool) < 2560:
        raise RuntimeError("real-data smoke pool is unexpectedly small")
    return pool.iloc[:2048].reset_index(drop=True), pool.iloc[2048:2560].reset_index(drop=True)


def run(family: str) -> dict[str, object]:
    train, validation = stable_rows()
    parameters = SMOKE_PARAMETERS[family]
    started = time.perf_counter()
    if family in TABULAR_FAMILIES:
        transform = TabularTransform.fit(train, SSS_RESIDUAL_FEATURES)
        train_x = transform.apply(train)
        validation_x = transform.apply(validation)
        model = fit_tabular_estimator(
            family,
            train_x,
            train.truth.to_numpy("float32"),
            parameters,
            seed=SEED,
        )
        prediction = predict_tabular_estimator(model, validation_x)
        train_rows = len(train)
        validation_rows = len(validation)
    else:
        patch_train = train.iloc[:256].reset_index(drop=True)
        patch_validation = validation.iloc[:64].reset_index(drop=True)
        with CoastalPatchReader(CUBE, monthly_cache_size=8) as reader:
            train_x = reader.read(patch_train, sequence_months=3, patch_size=8)
            validation_x = reader.read(patch_validation, sequence_months=3, patch_size=8)
        mean = train_x.mean(axis=(0, 1, 3, 4), keepdims=True)
        scale = train_x.std(axis=(0, 1, 3, 4), keepdims=True)
        scale = np.where(scale > 1e-6, scale, 1.0)
        train_x = (train_x - mean) / scale
        validation_x = (validation_x - mean) / scale
        model = fit_patch_estimator(
            family,
            train_x,
            patch_train.truth.to_numpy("float32"),
            parameters,
            seed=SEED,
        )
        prediction = predict_patch_estimator(model, validation_x)
        validation = patch_validation
        train_rows = len(patch_train)
        validation_rows = len(patch_validation)
    error = prediction - validation.truth.to_numpy(float)
    return {
        "family": family,
        "status": "passed",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "parameters": parameters,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "device": model.device,
        "parameter_count": model.parameter_count,
        "updates": model.updates,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=(*TABULAR_FAMILIES, *PATCH_FAMILIES), required=True)
    args = parser.parse_args()
    destination = OUTPUT / "entry_gate" / f"{args.family}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.family)
    except Exception as error:  # preserve every failed backend as scientific evidence
        failure_traceback = traceback.format_exc()
        status = (
            "license_unavailable"
            if args.family == "tabpfn_3" and "license" in failure_traceback.lower()
            else "failed"
        )
        result = {
            "family": args.family,
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": failure_traceback,
        }
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] not in {"passed", "license_unavailable"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
