"""Run one resumable tabular cell of the frozen Issue 89 benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psutil
import torch

from recad.evaluate.benchmark_backends import TABULAR_FAMILIES
from recad.evaluate.benchmark_contract import (
    assess_empirical_convergence,
    load_benchmark_contract,
)
from recad.evaluate.benchmark_search import (
    canonical_parameters,
    capacity_tail_improvement,
    common_fidelity_rows,
    stable_seed,
    suggest_parameters,
)
from recad.evaluate.benchmark_targets import fit_target_estimator, predict_target_estimator
from recad.evaluate.model_zoo import group_equal_rmse
from run_p2_model_zoo_v2_4 import prepare_split
from run_p2_validation_repair_v2_4 import (
    MODERN_RF_FEATURES,
    SSS_RESIDUAL_FEATURES,
    cached_splits,
    load_frame,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs/frozen/model_family_benchmark_v2.5.in"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_model_family_benchmark_v2.5")
SEARCH_ROOT = OUTPUT / "search"
PARTITIONS = OUTPUT / "partitions"
RESOURCES = OUTPUT / "resources"
PROTOCOL_MAP = {
    "cruise": "whole_cruise",
    "geographic": "buffered_geographic",
    "whole_region": "whole_region",
    "forward": "forward_time",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def features_for(target: str) -> tuple[str, ...]:
    return SSS_RESIDUAL_FEATURES if target == "SSS" else MODERN_RF_FEATURES


def score(
    target: str,
    family: str,
    parameters: dict[str, object],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int,
) -> tuple[float, dict[str, object], np.ndarray]:
    process = psutil.Process()
    memory_before = process.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = fit_target_estimator(
        family,
        parameters,
        target,
        train,
        features_for(target),
        seed=seed,
    )
    prediction = predict_target_estimator(model, validation)
    elapsed = time.perf_counter() - started
    objective = group_equal_rmse(validation, prediction)
    resources = {
        "wall_seconds": elapsed,
        "rss_delta_bytes": max(process.memory_info().rss - memory_before, 0),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "device": model.direct.model.device,
        "parameter_count": model.direct.model.parameter_count
        + (model.residual.model.parameter_count if model.residual else 0),
        "updates": model.direct.model.updates
        + (model.residual.model.updates if model.residual else 0),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return objective, resources, prediction


def validation_cap(fidelity: int | None) -> int | None:
    if fidelity == 120000:
        return 60000
    if fidelity == 300000:
        return 100000
    return None


def fidelity_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fidelity: int | None,
    *,
    target: str,
    scheme: str,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        common_fidelity_rows(
            train,
            fidelity,
            target=target,
            scheme=scheme,
            fold=fold,
            role=f"train-{fidelity or 'FULL'}",
        ),
        common_fidelity_rows(
            validation,
            validation_cap(fidelity),
            target=target,
            scheme=scheme,
            fold=fold,
            role=f"validation-{fidelity or 'FULL'}",
        ),
    )


def run_search(
    target: str,
    scheme: str,
    fold: int,
    family: str,
    tune_train: pd.DataFrame,
    tune_validation: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    contract = load_benchmark_contract(CONTRACT_PATH)
    family_contract = contract.family_contracts[family]
    cell = SEARCH_ROOT / f"{target}_{scheme}_fold{fold}_{family}"
    cell.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(cell / 'study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name="fidelity120k",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=stable_seed(target, scheme, fold, family)),
        storage=storage,
        load_if_exists=True,
    )
    train120, validation120 = fidelity_frames(
        tune_train,
        tune_validation,
        120000,
        target=target,
        scheme=scheme,
        fold=fold,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial, family_contract.search_space)
        value, resources, prediction = score(
            target,
            family,
            parameters,
            train120,
            validation120,
            seed=stable_seed(target, scheme, fold, family, "trial", trial.number),
        )
        trial.set_user_attr("resources", resources)
        trial.set_user_attr("parameter_identity", canonical_parameters(parameters))
        inner = pd.DataFrame(
            {
                "record_id": validation120.record_id,
                "group_key": validation120.group_key,
                "truth": validation120.truth,
                "prediction": prediction,
                "trial": trial.number,
            }
        )
        inner.to_parquet(cell / f"trial{trial.number:04d}_inner.parquet", index=False)
        return value

    completed = len(
        [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    )
    while completed < family_contract.minimum_trials:
        if len(study.trials) >= family_contract.maximum_trials:
            raise RuntimeError("minimum successful trial count was not reached before the ceiling")
        count = min(
            8,
            family_contract.minimum_trials - completed,
            family_contract.maximum_trials - len(study.trials),
        )
        study.optimize(objective, n_trials=count, catch=(Exception,), gc_after_trial=True)
        completed = len(
            [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        )
    objective_history = [
        float(trial.value)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    while len(study.trials) < family_contract.maximum_trials:
        preliminary = assess_empirical_convergence(
            contract,
            family,
            objective_history,
            full_data_completed=True,
            seeds_completed=family_contract.minimum_seeds,
            capacity_tail_relative_improvement=0.0,
        )
        if "search_tail_not_flat" not in preliminary.reasons:
            break
        study.optimize(
            objective,
            n_trials=min(8, family_contract.maximum_trials - len(study.trials)),
            catch=(Exception,),
            gc_after_trial=True,
        )
        objective_history = [
            float(trial.value)
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]
    complete_trials = sorted(
        (
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        ),
        key=lambda trial: (float(trial.value), trial.number),
    )
    if not complete_trials:
        raise RuntimeError("all search trials failed")
    promoted300 = complete_trials[: max(4, math.ceil(len(complete_trials) * 0.25))]
    train300, validation300 = fidelity_frames(
        tune_train,
        tune_validation,
        300000,
        target=target,
        scheme=scheme,
        fold=fold,
    )
    fidelity_rows: list[dict[str, object]] = []
    for trial in promoted300:
        parameters = dict(trial.params)
        value, resources, _ = score(
            target,
            family,
            parameters,
            train300,
            validation300,
            seed=stable_seed(target, scheme, fold, family, "300k", trial.number),
        )
        fidelity_rows.append(
            {
                "trial": trial.number,
                "fidelity": "300000",
                "objective": value,
                "parameters": canonical_parameters(parameters),
                **resources,
            }
        )
    table300 = pd.DataFrame(fidelity_rows).sort_values(["objective", "trial"])
    promoted_full = table300.head(4)
    train_full, validation_full = fidelity_frames(
        tune_train,
        tune_validation,
        None,
        target=target,
        scheme=scheme,
        fold=fold,
    )
    for row in promoted_full.itertuples(index=False):
        parameters = json.loads(row.parameters)
        value, resources, prediction = score(
            target,
            family,
            parameters,
            train_full,
            validation_full,
            seed=stable_seed(target, scheme, fold, family, "FULL", row.trial),
        )
        fidelity_rows.append(
            {
                "trial": row.trial,
                "fidelity": "FULL",
                "objective": value,
                "parameters": canonical_parameters(parameters),
                **resources,
            }
        )
        pd.DataFrame(
            {
                "record_id": validation_full.record_id,
                "group_key": validation_full.group_key,
                "truth": validation_full.truth,
                "prediction": prediction,
                "trial": row.trial,
            }
        ).to_parquet(cell / f"trial{row.trial:04d}_full_inner.parquet", index=False)
    fidelity_table = pd.DataFrame(fidelity_rows)
    fidelity_table.to_csv(cell / "fidelity_scores.csv", index=False)
    full = fidelity_table.loc[fidelity_table.fidelity.eq("FULL")].sort_values(
        ["objective", "trial"]
    )
    winner = full.iloc[0]
    capacity_improvement = capacity_tail_improvement(
        full.objective.astype(float).tolist(),
        full.parameter_count.astype(int).tolist(),
        tail_fraction=contract.tail_fraction,
    )
    convergence = assess_empirical_convergence(
        contract,
        family,
        objective_history,
        full_data_completed=True,
        seeds_completed=family_contract.minimum_seeds,
        capacity_tail_relative_improvement=float(capacity_improvement),
    )
    decision: dict[str, object] = {
        "target": target,
        "scheme": scheme,
        "protocol": PROTOCOL_MAP[scheme],
        "fold": fold,
        "family": family,
        "search_trials": len(study.trials),
        "completed_trials": len(complete_trials),
        "failed_trials": len(study.trials) - len(complete_trials),
        "selected_trial": int(winner.trial),
        "selected_parameters": json.loads(winner.parameters),
        "selected_full_objective": float(winner.objective),
        "capacity_tail_relative_improvement": float(capacity_improvement),
        "convergence_status": convergence.status,
        "converged": convergence.converged,
        "search_tail_relative_improvement": convergence.tail_relative_improvement,
        "convergence_reasons": list(convergence.reasons),
        "train_rows_120k": len(train120),
        "validation_rows_120k": len(validation120),
        "train_rows_300k": len(train300),
        "validation_rows_300k": len(validation300),
        "train_rows_full": len(train_full),
        "validation_rows_full": len(validation_full),
    }
    write_json(cell / "decision.json", decision)
    return decision, fidelity_table


def run_outer_seeds(
    decision: dict[str, object],
    target: str,
    scheme: str,
    fold: int,
    family: str,
    model_fit: pd.DataFrame,
    calibration: pd.DataFrame,
    held: pd.DataFrame,
) -> None:
    contract = load_benchmark_contract(CONTRACT_PATH)
    seeds = [89001, 89002, 89003]
    if contract.family_contracts[family].neural:
        seeds.extend([89004, 89005])
    query = pd.concat([calibration, held], ignore_index=True)
    parameters = dict(decision["selected_parameters"])
    for seed in seeds:
        destination = PARTITIONS / f"{target}_{scheme}_fold{fold}_{family}_seed{seed}.parquet"
        if destination.is_file():
            continue
        started = time.perf_counter()
        model = fit_target_estimator(
            family,
            parameters,
            target,
            model_fit,
            features_for(target),
            seed=stable_seed(target, scheme, fold, family, seed),
        )
        prediction = predict_target_estimator(model, query)
        held_prediction = prediction[len(calibration) :]
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
        output["prediction"] = held_prediction
        output["target"] = target
        output["scheme"] = scheme
        output["protocol"] = PROTOCOL_MAP[scheme]
        output["fold"] = fold
        output["model"] = family
        output["seed"] = seed
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(destination, index=False, compression="zstd")
        write_json(
            RESOURCES / f"{target}_{scheme}_fold{fold}_{family}_seed{seed}.json",
            {
                "target": target,
                "scheme": scheme,
                "fold": fold,
                "family": family,
                "seed": seed,
                "model_fit_rows": len(model_fit),
                "calibration_rows": len(calibration),
                "held_rows": len(held),
                "wall_seconds": time.perf_counter() - started,
                "parameters": parameters,
                "partition": str(destination),
            },
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("SSS", "fCO2"), required=True)
    parser.add_argument("--scheme", choices=tuple(PROTOCOL_MAP), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--family", choices=TABULAR_FAMILIES, required=True)
    args = parser.parse_args()
    frame = load_frame(args.target)
    split = next(
        item
        for item in cached_splits(args.target, frame)
        if item.scheme == args.scheme and item.fold == args.fold
    )
    tune_train, tune_validation, model_fit, calibration, held = prepare_split(
        args.target, frame, split
    )
    failure = (
        OUTPUT / "failures" / (f"{args.target}_{args.scheme}_fold{args.fold}_{args.family}.json")
    )
    try:
        decision, _ = run_search(
            args.target,
            args.scheme,
            args.fold,
            args.family,
            tune_train,
            tune_validation,
        )
        if decision["converged"]:
            run_outer_seeds(
                decision,
                args.target,
                args.scheme,
                args.fold,
                args.family,
                model_fit,
                calibration,
                held,
            )
    except Exception as error:
        write_json(
            failure,
            {
                "target": args.target,
                "scheme": args.scheme,
                "fold": args.fold,
                "family": args.family,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
