"""Evaluate Issue 89 families under strict reverse-time early-era folds."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recad.evaluate.benchmark_backends import PATCH_FAMILIES, TABULAR_FAMILIES
from recad.evaluate.benchmark_contract import load_benchmark_contract
from recad.evaluate.benchmark_patch_targets import (
    fit_patch_target_estimator,
    predict_patch_target_estimator,
)
from recad.evaluate.benchmark_search import stable_seed
from recad.evaluate.benchmark_targets import fit_target_estimator, predict_target_estimator
from recad.evaluate.coastal_patches import CoastalPatchMemmap
from recad.evaluate.early_era import complete_group_indices, load_early_era_contract
from run_early_era_validation_v2_5 import nested_later_sss, split_later_training
from run_global_tabular_benchmark_v2_5 import features_for, write_json
from run_p2_validation_repair_v2_4 import load_frame

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/frozen/model_family_benchmark_v2.5.in"
EARLY_CONTRACT = ROOT / "configs/frozen/early_era_validation_v2.5.in"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_model_family_benchmark_v2.5")
PATCH_CACHE = OUTPUT / "patch_cache"
PARTITIONS = OUTPUT / "partitions"
RESOURCES = OUTPUT / "resources"


def forward_decision(target: str, fold: int, family: str) -> dict[str, object] | None:
    path = OUTPUT / "search" / f"{target}_forward_fold{fold}_{family}" / "decision.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload["converged"] else None


def fit_predict(
    target: str,
    family: str,
    parameters: dict[str, object],
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    if family in PATCH_FAMILIES:
        cache = CoastalPatchMemmap(PATCH_CACHE)
        model = fit_patch_target_estimator(family, parameters, target, train, cache, seed=seed)
        return predict_patch_target_estimator(model, query, cache)
    model = fit_target_estimator(family, parameters, target, train, features_for(target), seed=seed)
    return predict_target_estimator(model, query)


def run_cell(target: str, family: str, fold_spec: object) -> None:
    decision = forward_decision(target, fold_spec.fold, family)
    status_path = OUTPUT / "early_era_status" / f"{target}_{family}_fold{fold_spec.fold}.json"
    if decision is None:
        write_json(
            status_path,
            {
                "target": target,
                "family": family,
                "fold": fold_spec.fold,
                "status": "ineligible_missing_converged_forward_configuration",
            },
        )
        return
    contract = load_early_era_contract(EARLY_CONTRACT)
    frame = load_frame(target)
    frame = frame.loc[frame.yr.le(contract.maximum_label_year)].reset_index(drop=True)
    train_index, held_index, _ = complete_group_indices(frame, fold_spec)
    later = frame.iloc[train_index].reset_index(drop=True)
    held = frame.iloc[held_index].reset_index(drop=True)
    model_fit, calibration = split_later_training(
        later,
        folds=contract.calibration_folds,
        calibration_fold=contract.calibration_fold,
        salt=f"issue89-early-era-calibration-{target}-{fold_spec.fold}",
    )
    if target == "fCO2":
        sss_frame = load_frame("SSS")
        sss_frame = sss_frame.loc[sss_frame.yr.le(contract.maximum_label_year)].reset_index(
            drop=True
        )
        fit_sss, calibration_sss, held_sss = nested_later_sss(
            sss_frame,
            model_fit,
            calibration,
            held,
            fold=fold_spec.fold,
            inner_folds=contract.inner_folds,
        )
        model_fit["sss_crossfit"] = fit_sss
        calibration["sss_crossfit"] = calibration_sss
        held["sss_crossfit"] = held_sss
    query = pd.concat([calibration, held], ignore_index=True)
    family_contract = load_benchmark_contract(CONTRACT_PATH).family_contracts[family]
    seeds = [89001, 89002, 89003]
    if family_contract.neural:
        seeds.extend([89004, 89005])
    parameters = dict(decision["selected_parameters"])
    for seed in seeds:
        destination = (
            PARTITIONS / f"{target}_early_era_fold{fold_spec.fold}_{family}_seed{seed}.parquet"
        )
        if destination.is_file():
            continue
        started = time.perf_counter()
        prediction = fit_predict(
            target,
            family,
            parameters,
            model_fit,
            query,
            seed=stable_seed(target, "early_era", fold_spec.fold, family, seed),
        )[len(calibration) :]
        output = held[
            [
                "record_id",
                "group_key",
                "yr",
                "mon",
                "grid_flat",
                "latitude",
                "longitude",
                "lme_id",
                "truth",
            ]
        ].copy()
        output["prediction"] = prediction
        output["target"] = target
        output["scheme"] = "early_era"
        output["protocol"] = "early_era"
        output["fold"] = fold_spec.fold
        output["model"] = family
        output["seed"] = seed
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(destination, index=False, compression="zstd")
        write_json(
            RESOURCES / f"{target}_early_era_fold{fold_spec.fold}_{family}_seed{seed}.json",
            {
                "target": target,
                "scheme": "early_era",
                "fold": fold_spec.fold,
                "family": family,
                "seed": seed,
                "model_fit_rows": len(model_fit),
                "calibration_rows": len(calibration),
                "held_rows": len(held),
                "parameters_from": f"forward_fold{fold_spec.fold}",
                "parameters": parameters,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        status_path,
        {
            "target": target,
            "family": family,
            "fold": fold_spec.fold,
            "status": "complete",
            "parameters_from": f"forward_fold{fold_spec.fold}",
        },
    )


def main() -> None:
    early = load_early_era_contract(EARLY_CONTRACT)
    for target in ("SSS", "fCO2"):
        for family in (*TABULAR_FAMILIES, *PATCH_FAMILIES):
            if (
                family == "tabpfn_3"
                and (OUTPUT / "family_status" / f"{target}_{family}.json").is_file()
            ):
                continue
            for fold_spec in early.folds:
                run_cell(target, family, fold_spec)
    (OUTPUT / "early_era.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
