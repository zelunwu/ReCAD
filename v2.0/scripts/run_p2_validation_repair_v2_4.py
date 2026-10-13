"""Run the Issue #61 exposure, split, and corrected-baseline repair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from recad.evaluate.fco2_bakeoff import interval_scores
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.sss_bakeoff import NumericTransform
from recad.evaluate.validation_repair import (
    RepairedSplit,
    assert_split_integrity,
    conformal_widths,
    cruise_splits,
    geographic_block_id,
    geographic_splits,
    grouped_fit_calibration,
    inner_group_folds,
    normalize_group_key,
    rolling_forward_splits,
    whole_region_splits,
)
from recad.utils.chem import xco2air_to_pco2air

ROOT = Path(__file__).resolve().parents[1]
MATCHUP = Path(
    r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3"
    r"\socat_predictor_matchup_v2.3.parquet"
)
LEGACY_SPLIT = Path(r"C:\backup\phd\data\processed\recad_v2_2\split_manifest_v2.2.parquet")
GLOBAL_GROUP = Path(
    r"C:\backup\phd\data\processed\recad_v2_3"
    r"\global_group_split_manifest_v2.3.parquet"
)
CONFIG = ROOT / "configs/p2_validation_repair_v2.4.json"
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_validation_repair_v2.4")
PARTITIONS = OUTPUT / "partitions"
SPLITS = OUTPUT / "splits"
CHECKPOINTS = OUTPUT / "checkpoints"
UPSTREAM = OUTPUT / "upstream_sss"
TABLES = OUTPUT / "tables"

BASE_FEATURES = (
    "year_centered",
    "month_sin",
    "month_cos",
    "latitude_sin",
    "latitude_cos",
    "longitude_sin",
    "longitude_cos",
    "distance_to_significant_land_km",
    "bathymetry_m",
    "shelf_core_mask",
    "slope_deep_mask",
    "sst",
    "wspd",
    "xco2air",
)
SSS_RESIDUAL_FEATURES = (*BASE_FEATURES, "sss_bg", "sss_bg_available")
MODERN_RF_FEATURES = (*BASE_FEATURES, "sss_crossfit")
STRICT_V1_FEATURES = (
    "longitude",
    "latitude",
    "strict_month",
    "strict_sss",
    "sst",
    "adt",
    "strict_pco2air",
)
FORWARD_WINDOWS = ((2013, 2016), (2017, 2020), (2021, 2025))


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash one artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def audit_exposure() -> None:
    """Build exact and normalized cross-release label-exposure ledgers."""

    TABLES.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_parquet(LEGACY_SPLIT).copy()
    groups = pd.read_parquet(GLOBAL_GROUP).copy()
    matchup = pd.read_parquet(
        MATCHUP,
        columns=["group_key", "fco2_n", "sss_n", "record_id"],
    )
    for frame in (legacy, groups, matchup):
        frame["normalized_group_key"] = frame.group_key.map(normalize_group_key)
    matchup_summary = (
        matchup.groupby(["group_key", "normalized_group_key"], as_index=False)
        .agg(
            matchup_rows=("record_id", "size"),
            fco2_matchup_rows=("fco2_n", lambda values: int((values > 0).sum())),
            sss_matchup_rows=("sss_n", lambda values: int((values > 0).sum())),
        )
        .sort_values("group_key")
    )
    global_columns = [
        "group_key",
        "normalized_group_key",
        "development_role",
        "label_exposure",
        "fco2_rows",
        "sss_rows",
    ]
    exact = legacy.merge(
        groups[global_columns],
        on=["group_key", "normalized_group_key"],
        how="left",
        indicator="global_membership",
    ).merge(
        matchup_summary,
        on=["group_key", "normalized_group_key"],
        how="left",
        indicator="matchup_membership",
    )
    exact["in_global_manifest"] = exact.global_membership.eq("both")
    exact["in_matchup"] = exact.matchup_membership.eq("both")
    exact["legacy_sensitive"] = exact.split.isin(["locked_test", "external_independent"])
    exact["repaired_status"] = np.select(
        [
            exact.legacy_sensitive & exact.in_matchup,
            exact.legacy_sensitive & exact.in_global_manifest,
            exact.legacy_sensitive,
        ],
        [
            "exposed_in_p2_matchup_development_only",
            "present_in_p2_manifest_metadata_only_review",
            "candidate_unexposed_metadata_only",
        ],
        default="development_or_train",
    )
    exact.drop(columns=["global_membership", "matchup_membership"]).to_csv(
        TABLES / "exposure_ledger_exact.csv", index=False
    )
    normalized = (
        exact.groupby(["normalized_group_key", "dataset"], as_index=False)
        .agg(
            legacy_splits=("split", lambda values: ";".join(sorted(set(values)))),
            exact_keys=("group_key", lambda values: ";".join(sorted(set(values)))),
            in_global_manifest=("in_global_manifest", "max"),
            in_matchup=("in_matchup", "max"),
            matchup_rows=("matchup_rows", "max"),
        )
        .sort_values(["normalized_group_key", "dataset"])
    )
    normalized.to_csv(TABLES / "exposure_ledger_normalized.csv", index=False)
    cross_target = (
        exact.groupby("normalized_group_key")
        .filter(lambda part: part.dataset.nunique() > 1)
        .sort_values(["normalized_group_key", "dataset"])
    )
    cross_target.to_csv(TABLES / "cross_target_lineage.csv", index=False)
    collisions = (
        legacy.groupby("normalized_group_key")
        .agg(exact_keys=("group_key", "nunique"), datasets=("dataset", "nunique"))
        .query("exact_keys > 1")
        .reset_index()
    )
    collisions.to_csv(TABLES / "normalized_alias_collisions.csv", index=False)
    sensitive = exact.loc[exact.legacy_sensitive]
    rows = []
    for (dataset, split), part in sensitive.groupby(["dataset", "split"]):
        rows.append(
            {
                "dataset": dataset,
                "legacy_split": split,
                "legacy_groups": part.group_key.nunique(),
                "groups_in_global_manifest": part.loc[
                    part.in_global_manifest, "group_key"
                ].nunique(),
                "groups_in_matchup": part.loc[part.in_matchup, "group_key"].nunique(),
                "matchup_rows": int(part.matchup_rows.fillna(0).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "exposure_summary.csv", index=False)
    write_json(
        OUTPUT / "exposure_decision.json",
        {
            "rule": "every legacy sensitive group found in P2 matchup is permanently development-only",
            "repartitioning_restores_blind_status": False,
            "summary": summary.to_dict("records"),
            "cross_target_normalized_groups": int(cross_target.normalized_group_key.nunique()),
            "normalized_alias_collisions": len(collisions),
            "new_target_columns_opened": False,
        },
    )


def load_frame(target: str) -> pd.DataFrame:
    """Load the frozen Issue #54 matchup for one target."""

    frame = pd.read_parquet(MATCHUP)
    frame = frame.loc[frame.yr.le(2025)].copy()
    frame["group_key"] = frame.group_key.map(normalize_group_key)
    if target == "SSS":
        frame = frame.loc[frame.salinity.notna()].copy()
        frame["truth"] = frame.salinity.astype("float32")
    elif target == "fCO2":
        frame = frame.loc[frame.fco2.notna()].copy()
        frame["truth"] = frame.fco2.astype("float32")
    else:
        raise ValueError(f"unknown target: {target}")
    frame["strict_month"] = frame.mon.astype("float32")
    frame["strict_sss"] = frame.sss_bg.astype("float32")
    return frame.reset_index(drop=True)


def build_splits(frame: pd.DataFrame) -> list[RepairedSplit]:
    """Construct every preregistered repaired outer split."""

    return [
        *cruise_splits(frame),
        *geographic_splits(frame, block_degrees=5.0, buffer_km=100.0),
        *whole_region_splits(frame),
        *rolling_forward_splits(frame, FORWARD_WINDOWS),
    ]


def hash_values(values: np.ndarray) -> str:
    """Hash a sorted vector for a compact split manifest."""

    ordered = np.sort(np.asarray(values))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def audit_splits(target: str, frame: pd.DataFrame, splits: list[RepairedSplit]) -> None:
    """Persist zero-leakage split evidence and held-row assignments."""

    SPLITS.mkdir(parents=True, exist_ok=True)
    rows = []
    assignments = []
    index_payload: dict[str, np.ndarray] = {}
    for split in splits:
        checks = assert_split_integrity(frame, split)
        fit = frame.iloc[split.fit_index]
        held = frame.iloc[split.held_index]
        model_fit_index, calibration_index = grouped_fit_calibration(
            fit,
            salt=f"v2.4-calibration-{target}-{split.scheme}-{split.fold}",
        )
        model_fit = fit.iloc[model_fit_index]
        calibration = fit.iloc[calibration_index]
        fit_blocks = set(geographic_block_id(fit.latitude, fit.longitude))
        held_blocks = set(geographic_block_id(held.latitude, held.longitude))
        rows.append(
            {
                "target": target,
                "scheme": split.scheme,
                "fold": split.fold,
                "fit_rows": len(fit),
                "held_rows": len(held),
                "fit_groups": fit.group_key.nunique(),
                "held_groups": held.group_key.nunique(),
                "model_fit_rows": len(model_fit),
                "calibration_rows": len(calibration),
                "model_fit_groups": model_fit.group_key.nunique(),
                "calibration_groups": calibration.group_key.nunique(),
                "fit_calibration_group_overlap": len(
                    set(model_fit.group_key) & set(calibration.group_key)
                ),
                "fit_blocks": len(fit_blocks),
                "held_blocks": len(held_blocks),
                "exact_block_overlap": len(fit_blocks & held_blocks),
                "exact_grid_overlap": len(set(fit.grid_flat) & set(held.grid_flat)),
                "group_overlap": checks["group_overlap"],
                "record_overlap": checks["record_overlap"],
                "forbidden_time_rows": checks["forbidden_time_rows"],
                "fit_record_hash": hash_values(fit.record_id.to_numpy(np.int64)),
                "held_record_hash": hash_values(held.record_id.to_numpy(np.int64)),
                **split.metadata,
            }
        )
        assignments.append(
            held[["record_id", "group_key", "yr", "grid_flat", "latitude", "longitude"]]
            .assign(target=target, scheme=split.scheme, fold=split.fold, role="held")
            .copy()
        )
        key = f"{split.scheme}__{split.fold}"
        index_payload[f"{key}__fit"] = split.fit_index.astype(np.int64)
        index_payload[f"{key}__held"] = split.held_index.astype(np.int64)
    pd.DataFrame(rows).to_csv(SPLITS / f"{target}_split_audit.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_parquet(
        SPLITS / f"{target}_held_assignments.parquet", index=False, compression="zstd"
    )
    np.savez_compressed(SPLITS / f"{target}_split_indices.npz", **index_payload)


def cached_splits(target: str, frame: pd.DataFrame) -> list[RepairedSplit]:
    """Load audited split indices and verify they still match the current frame."""

    path = SPLITS / f"{target}_split_indices.npz"
    audit_path = SPLITS / f"{target}_split_audit.csv"
    if not path.is_file() or not audit_path.is_file():
        splits = build_splits(frame)
        audit_splits(target, frame, splits)
        return splits
    audit = pd.read_csv(audit_path).set_index(["scheme", "fold"])
    stored = np.load(path)
    splits = []
    for scheme, fold in audit.index:
        key = f"{scheme}__{int(fold)}"
        fit_index = stored[f"{key}__fit"]
        held_index = stored[f"{key}__held"]
        row = audit.loc[(scheme, fold)]
        fit_hash = hash_values(frame.iloc[fit_index].record_id.to_numpy(np.int64))
        held_hash = hash_values(frame.iloc[held_index].record_id.to_numpy(np.int64))
        if fit_hash != row.fit_record_hash or held_hash != row.held_record_hash:
            raise RuntimeError(f"cached split drift: {target} {scheme}/{fold}")
        splits.append(
            RepairedSplit(
                scheme=str(scheme),
                fold=int(fold),
                fit_index=fit_index,
                held_index=held_index,
                held_groups=tuple(sorted(set(frame.iloc[held_index].group_key.astype(str)))),
                metadata={},
            )
        )
    return splits


def catboost_model(seed: int, iterations: int = 300) -> CatBoostRegressor:
    """Return the frozen baseline CatBoost."""

    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        task_type="GPU",
        devices="0",
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
    )


def fit_predict_sss(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Fit the frozen direct plus background-residual SSS baseline."""

    labelled = train.loc[train.salinity.notna()].reset_index(drop=True)
    if labelled.group_key.nunique() < 20:
        raise RuntimeError("insufficient SSS groups")
    weight = balanced_sample_weight(labelled)
    direct = catboost_model(seed)
    direct.fit(labelled.loc[:, BASE_FEATURES], labelled.salinity, sample_weight=weight)
    prediction = direct.predict(predict.loc[:, BASE_FEATURES]).astype("float32")
    available_train = labelled.sss_bg.notna().to_numpy()
    residual = catboost_model(seed + 1)
    residual.fit(
        labelled.loc[available_train, SSS_RESIDUAL_FEATURES],
        labelled.loc[available_train, "salinity"].to_numpy(float)
        - labelled.loc[available_train, "sss_bg"].to_numpy(float),
        sample_weight=weight[available_train],
    )
    available_predict = predict.sss_bg.notna().to_numpy()
    prediction[available_predict] = predict.loc[available_predict, "sss_bg"].to_numpy(
        float
    ) + residual.predict(predict.loc[available_predict, SSS_RESIDUAL_FEATURES])
    del direct, residual
    gc.collect()
    return prediction


def crossfit_sss(
    target: str,
    split: RepairedSplit,
    fit: pd.DataFrame,
    held: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outer-training OOF SSS and all-outer-training held SSS."""

    path = UPSTREAM / f"{target}_{split.scheme}_fold{split.fold}.parquet"
    if path.is_file():
        cached = pd.read_parquet(path)
        fit_index = cached.loc[cached.role.eq("fit")].set_index("record_id").sss_crossfit
        held_index = cached.loc[cached.role.eq("held")].set_index("record_id").sss_crossfit
        return (
            fit.record_id.map(fit_index).to_numpy("float32"),
            held.record_id.map(held_index).to_numpy("float32"),
        )
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    assignment = inner_group_folds(
        fit, n_folds=2, salt=f"v2.4-sss-{target}-{split.scheme}-{split.fold}"
    )
    fit_prediction = np.full(len(fit), np.nan, dtype="float32")
    for inner in range(2):
        train = fit.loc[assignment != inner]
        calibration = assignment == inner
        fit_prediction[calibration] = fit_predict_sss(
            train,
            fit.loc[calibration],
            seed=61024 + split.fold * 100 + inner * 10,
        )
    held_prediction = fit_predict_sss(
        fit,
        held,
        seed=61024 + split.fold * 100 + 90,
    )
    if not np.isfinite(fit_prediction).all() or not np.isfinite(held_prediction).all():
        raise RuntimeError("non-finite cross-fitted SSS")
    pd.concat(
        [
            pd.DataFrame(
                {
                    "role": "fit",
                    "record_id": fit.record_id,
                    "sss_crossfit": fit_prediction,
                }
            ),
            pd.DataFrame(
                {
                    "role": "held",
                    "record_id": held.record_id,
                    "sss_crossfit": held_prediction,
                }
            ),
        ],
        ignore_index=True,
    ).to_parquet(path, index=False, compression="zstd")
    return fit_prediction, held_prediction


def add_pco2air(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the seventh strict-v1 input in memory-bounded chunks."""

    if "strict_pco2air" in frame:
        return frame
    output = np.full(len(frame), np.nan, dtype="float32")
    for start in range(0, len(frame), 50_000):
        stop = min(start + 50_000, len(frame))
        output[start:stop] = xco2air_to_pco2air(
            frame.xco2air.iloc[start:stop].to_numpy(float),
            frame.sst.iloc[start:stop].to_numpy(float),
            frame.strict_sss.iloc[start:stop].to_numpy(float),
        ).astype("float32")
    frame = frame.copy()
    frame["strict_pco2air"] = output
    return frame


def fit_rf(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    features: tuple[str, ...],
    strict: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit either exact strict-v1 or modernized RF and return prediction plus eligibility."""

    eligible_train = np.isfinite(train.loc[:, features]).all(axis=1).to_numpy()
    eligible_predict = np.isfinite(predict.loc[:, features]).all(axis=1).to_numpy()
    transform = NumericTransform.fit(train.loc[eligible_train], features)
    if strict:
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            max_features=1.0,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
        sample_weight = None
    else:
        model = RandomForestRegressor(
            n_estimators=96,
            max_depth=18,
            min_samples_leaf=20,
            max_features=0.7,
            max_samples=0.5,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
        sample_weight = balanced_sample_weight(train.loc[eligible_train])
    model.fit(
        transform.apply(train.loc[eligible_train]),
        train.loc[eligible_train, "truth"].to_numpy(float),
        sample_weight=sample_weight,
    )
    prediction = np.full(len(predict), np.nan, dtype="float32")
    prediction[eligible_predict] = model.predict(
        transform.apply(predict.loc[eligible_predict])
    ).astype("float32")
    del model
    gc.collect()
    return prediction, eligible_predict


def fit_direct_catboost(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen direct CatBoost baseline."""

    model = catboost_model(seed)
    model.fit(
        train.loc[:, BASE_FEATURES],
        train.truth,
        sample_weight=balanced_sample_weight(train),
    )
    prediction = model.predict(predict.loc[:, BASE_FEATURES]).astype("float32")
    del model
    gc.collect()
    return prediction, np.ones(len(predict), dtype=bool)


def fit_fco2_model(
    model_name: str,
    split: RepairedSplit,
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one fCO2 baseline without calibration labels and predict requested rows."""

    def fit_one(train: pd.DataFrame, predict: pd.DataFrame, seed: int):
        if model_name == "strict_v1_rf":
            return fit_rf(
                train,
                predict,
                features=STRICT_V1_FEATURES,
                strict=True,
                seed=seed,
            )
        if model_name == "modernized_rf":
            return fit_rf(
                train,
                predict,
                features=MODERN_RF_FEATURES,
                strict=False,
                seed=seed,
            )
        if model_name == "direct_catboost":
            return fit_direct_catboost(train, predict, seed=seed)
        raise ValueError(model_name)

    return fit_one(
        train,
        predict,
        61024 + split.fold * 100 + 90,
    )


def output_partition(
    target: str,
    split: RepairedSplit,
    model: str,
    fit: pd.DataFrame,
    held: pd.DataFrame,
    calibration_prediction: np.ndarray,
    held_prediction: np.ndarray,
    calibration_eligible: np.ndarray,
    held_eligible: np.ndarray,
) -> None:
    """Calibrate and save one held partition."""

    destination = PARTITIONS / f"{target}_{split.scheme}_fold{split.fold}_{model}.parquet"
    if destination.is_file():
        return
    valid_calibration = calibration_eligible & np.isfinite(calibration_prediction)
    valid_held = held_eligible & np.isfinite(held_prediction)
    width50 = np.full(len(held), np.nan, dtype="float32")
    width90 = np.full(len(held), np.nan, dtype="float32")
    bands = np.full(len(held), "unavailable", dtype=object)
    calibrated = valid_held.copy()
    if valid_calibration.any() and valid_held.any():
        w50, w90, prediction_band = conformal_widths(
            fit.loc[valid_calibration, "truth"].to_numpy(float),
            calibration_prediction[valid_calibration],
            held_prediction[valid_held],
            target=target,
        )
        width50[valid_held] = w50
        width90[valid_held] = w90
        bands[valid_held] = prediction_band
    result = held[
        [
            "record_id",
            "group_key",
            "yr",
            "mon",
            "grid_flat",
            "latitude",
            "longitude",
            "lme_id",
            "distance_to_significant_land_km",
            "shelf_core_mask",
            "truth",
        ]
    ].copy()
    result["prediction"] = held_prediction
    result["eligible"] = valid_held
    result["calibrated"] = calibrated
    result["width50"] = width50
    result["width90"] = width90
    result["prediction_band"] = bands
    result["target"] = target
    result["scheme"] = split.scheme
    result["fold"] = split.fold
    result["model"] = model
    result.to_parquet(destination, index=False, compression="zstd")


def run_sss_split(frame: pd.DataFrame, split: RepairedSplit) -> None:
    """Run raw GLORYS and corrected SSS CatBoost for one outer split."""

    outer_fit = frame.iloc[split.fit_index].reset_index(drop=True)
    held = frame.iloc[split.held_index].reset_index(drop=True)
    model_fit_index, calibration_index = grouped_fit_calibration(
        outer_fit,
        salt=f"v2.4-calibration-SSS-{split.scheme}-{split.fold}",
    )
    model_fit = outer_fit.iloc[model_fit_index].reset_index(drop=True)
    calibration = outer_fit.iloc[calibration_index].reset_index(drop=True)
    prediction_frame = pd.concat([calibration, held], ignore_index=True)
    combined_prediction = fit_predict_sss(
        model_fit,
        prediction_frame,
        seed=61024 + split.fold * 100 + 90,
    )
    calibration_prediction = combined_prediction[: len(calibration)]
    prediction = combined_prediction[len(calibration) :]
    output_partition(
        "SSS",
        split,
        "catboost_background_residual",
        calibration,
        held,
        calibration_prediction,
        prediction,
        np.isfinite(calibration_prediction),
        np.isfinite(prediction),
    )
    output_partition(
        "SSS",
        split,
        "raw_glorys",
        calibration,
        held,
        calibration.sss_bg.to_numpy(float),
        held.sss_bg.to_numpy(float),
        calibration.sss_bg.notna().to_numpy(),
        held.sss_bg.notna().to_numpy(),
    )


def run_fco2_split(frame: pd.DataFrame, split: RepairedSplit) -> None:
    """Run strict-v1, modernized RF, and direct CatBoost for one outer split."""

    outer_fit = frame.iloc[split.fit_index].reset_index(drop=True)
    held = frame.iloc[split.held_index].reset_index(drop=True)
    model_fit_index, calibration_index = grouped_fit_calibration(
        outer_fit,
        salt=f"v2.4-calibration-fCO2-{split.scheme}-{split.fold}",
    )
    model_fit = outer_fit.iloc[model_fit_index].reset_index(drop=True)
    calibration = outer_fit.iloc[calibration_index].reset_index(drop=True)
    predict_frame = pd.concat([calibration, held], ignore_index=True)
    model_fit_sss, predict_sss = crossfit_sss("fCO2", split, model_fit, predict_frame)
    model_fit["sss_crossfit"] = model_fit_sss
    predict_frame["sss_crossfit"] = predict_sss
    model_fit = add_pco2air(model_fit)
    predict_frame = add_pco2air(predict_frame)
    calibration = predict_frame.iloc[: len(calibration)].reset_index(drop=True)
    held = predict_frame.iloc[len(calibration) :].reset_index(drop=True)
    model_fit = add_pco2air(model_fit)
    held = add_pco2air(held)
    for model in ("strict_v1_rf", "modernized_rf", "direct_catboost"):
        destination = PARTITIONS / f"fCO2_{split.scheme}_fold{split.fold}_{model}.parquet"
        if destination.is_file():
            continue
        combined_prediction, combined_eligible = fit_fco2_model(
            model, split, model_fit, predict_frame
        )
        calibration_prediction = combined_prediction[: len(calibration)]
        prediction = combined_prediction[len(calibration) :]
        calibration_eligible = combined_eligible[: len(calibration)]
        held_eligible = combined_eligible[len(calibration) :]
        output_partition(
            "fCO2",
            split,
            model,
            calibration,
            held,
            calibration_prediction,
            prediction,
            calibration_eligible,
            held_eligible,
        )


def run_models(
    target: str | None = None,
    only_scheme: str | None = None,
    only_fold: int | None = None,
) -> None:
    """Run the preregistered corrected baselines with resumable partitions."""

    PARTITIONS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "training_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.is_file() else []
    targets = [target] if target else ["SSS", "fCO2"]
    for target_name in targets:
        frame = load_frame(target_name)
        if target_name == "fCO2":
            frame = add_pco2air(frame)
        splits = cached_splits(target_name, frame)
        for split in splits:
            if only_scheme and split.scheme != only_scheme:
                continue
            if only_fold is not None and split.fold != only_fold:
                continue
            started = time.perf_counter()
            print(
                f"{target_name} {split.scheme}/{split.fold}: "
                f"fit={len(split.fit_index):,} held={len(split.held_index):,}",
                flush=True,
            )
            if target_name == "SSS":
                run_sss_split(frame, split)
            else:
                run_fco2_split(frame, split)
            log.append(
                {
                    "target": target_name,
                    "scheme": split.scheme,
                    "fold": split.fold,
                    "fit_rows": len(split.fit_index),
                    "held_rows": len(split.held_index),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            pd.DataFrame(log).to_csv(log_path, index=False)


def metric_row(frame: pd.DataFrame, **keys: object) -> dict[str, object]:
    """Return point and interval metrics for eligible rows."""

    frame = frame.loc[frame.eligible & frame.prediction.notna()].copy()
    error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
    denominator = np.sum((frame.truth - frame.truth.mean()) ** 2)
    per_group = (
        frame.assign(squared_error=error**2)
        .groupby("group_key", observed=True)
        .squared_error.mean()
    )
    output: dict[str, object] = {
        **keys,
        "n": len(frame),
        "groups": frame.group_key.nunique(),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator else np.nan,
        "cruise_equal_rmse": float(np.sqrt(per_group.mean())),
    }
    calibrated = frame.loc[frame.calibrated & frame.width90.notna()]
    if not calibrated.empty:
        output.update(
            interval_scores(
                calibrated.truth.to_numpy(float),
                calibrated.prediction.to_numpy(float),
                calibrated.width50.to_numpy(float),
                calibrated.width90.to_numpy(float),
            )
        )
    return output


def analyze() -> None:
    """Create corrected point, common-grid, calibration, risk, and gate tables."""

    paths = sorted(PARTITIONS.glob("*.parquet"))
    if not paths:
        raise RuntimeError("no repaired prediction partitions")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rows = [
        metric_row(part, target=target, scheme=scheme, model=model)
        for (target, scheme, model), part in frame.groupby(["target", "scheme", "model"])
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / "corrected_native_metrics.csv", index=False)
    coverage = (
        frame.groupby(["target", "scheme", "model"], as_index=False)
        .agg(total_rows=("record_id", "size"), eligible_rows=("eligible", "sum"))
        .assign(eligible_fraction=lambda values: values.eligible_rows / values.total_rows)
    )
    coverage.to_csv(TABLES / "model_coverage.csv", index=False)

    common_rows = []
    for (target, scheme, fold), part in frame.groupby(["target", "scheme", "fold"]):
        models = sorted(part.model.unique())
        wide = part.pivot_table(
            index=[
                "record_id",
                "group_key",
                "yr",
                "mon",
                "latitude",
                "longitude",
                "truth",
            ],
            columns="model",
            values="prediction",
            aggfunc="first",
        ).dropna(subset=models)
        if wide.empty:
            continue
        work = wide.reset_index()
        work["lat025"] = np.floor((work.latitude + 90.0) / 0.25).astype(int)
        work["lon025"] = np.floor(np.mod(work.longitude, 360.0) / 0.25).astype(int)
        aggregation = {
            "truth": "mean",
            "group_key": lambda values: ";".join(sorted(set(values))),
            **dict.fromkeys(models, "mean"),
        }
        gridded = work.groupby(["yr", "mon", "lat025", "lon025"], as_index=False).agg(aggregation)
        for model in models:
            error = gridded[model] - gridded.truth
            common_rows.append(
                {
                    "target": target,
                    "scheme": scheme,
                    "fold": fold,
                    "model": model,
                    "grid_months": len(gridded),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                }
            )
    pd.DataFrame(common_rows).to_csv(TABLES / "corrected_common_grid_metrics.csv", index=False)

    risk_rows = []
    for (target, scheme, model), part in frame.groupby(["target", "scheme", "model"]):
        part = part.loc[part.eligible & part.width90.notna()].copy()
        part["squared_error"] = (part.prediction - part.truth) ** 2
        part = part.sort_values("width90")
        for retained in (0.25, 0.5, 0.75, 1.0):
            selected = part.iloc[: max(1, int(len(part) * retained))]
            risk_rows.append(
                {
                    "target": target,
                    "scheme": scheme,
                    "model": model,
                    "retained_fraction": retained,
                    "n": len(selected),
                    "rmse": float(np.sqrt(selected.squared_error.mean())),
                }
            )
    pd.DataFrame(risk_rows).to_csv(TABLES / "corrected_risk_coverage.csv", index=False)

    correction = pd.DataFrame(
        [
            {
                "archive": "p2_sss_global_bakeoff_v2.3",
                "retained": "point predictions as development diagnostics",
                "withdrawn": "sealed-label, strict-spatial, leakage-safe interval and risk claims",
                "replacement": "Issue 61 corrected SSS baselines",
            },
            {
                "archive": "p2_fco2_global_bakeoff_v2.3",
                "retained": "point rankings as development diagnostics",
                "withdrawn": "sealed-label, strict-spatial, nested-calibration and exact-v1 identity claims",
                "replacement": "Issue 61 strict-v1 and modernized RF baselines",
            },
            {
                "archive": "p2_context_connectivity_v2.3",
                "retained": "negative result for exact GRU/ST96/four-neighbour implementation",
                "withdrawn": "family-wide rejection and deployable uncertainty claims",
                "replacement": "Issue 62 fair model-family screen",
            },
        ]
    )
    correction.to_csv(TABLES / "archive_correction_matrix.csv", index=False)

    gate_rows = []
    for row in metrics.itertuples():
        target_ceiling = 1.0 if row.target == "SSS" and row.scheme == "cruise" else None
        if row.target == "SSS" and row.scheme in {"geographic", "forward"}:
            target_ceiling = 1.2
        if row.target == "fCO2" and row.scheme == "cruise":
            target_ceiling = 30.0
        if row.target == "fCO2" and row.scheme in {"geographic", "forward"}:
            target_ceiling = 40.0
        gate_rows.append(
            {
                "target": row.target,
                "scheme": row.scheme,
                "model": row.model,
                "rmse_ceiling": target_ceiling,
                "rmse_pass": None if target_ceiling is None else bool(row.rmse <= target_ceiling),
                "coverage50_pass": bool(
                    pd.notna(getattr(row, "coverage50", np.nan)) and 0.45 <= row.coverage50 <= 0.55
                ),
                "coverage90_pass": bool(
                    pd.notna(getattr(row, "coverage90", np.nan)) and 0.85 <= row.coverage90 <= 0.95
                ),
            }
        )
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(TABLES / "corrected_gate_matrix.csv", index=False)
    gate_records = gates.astype(object).where(pd.notna(gates), None).to_dict("records")
    write_json(
        OUTPUT / "decision.json",
        {
            "status": "corrected_development_baselines_only",
            "issue_62_unblocked": bool(
                pd.concat(
                    [
                        pd.read_csv(SPLITS / "SSS_split_audit.csv"),
                        pd.read_csv(SPLITS / "fCO2_split_audit.csv"),
                    ],
                    ignore_index=True,
                )[["group_overlap", "record_overlap", "forbidden_time_rows"]]
                .to_numpy()
                .sum()
                == 0
            ),
            "complex_model_intervals_recomputed": False,
            "complex_model_policy": "withdraw affected claims and rerun families fairly in Issue 62",
            "new_independent_labels_opened": False,
            "gate_rows": gate_records,
        },
    )


def write_manifests() -> None:
    """Write source and environment manifests after outputs exist."""

    files = [MATCHUP, LEGACY_SPLIT, GLOBAL_GROUP, CONFIG]
    write_json(
        OUTPUT / "data_manifest.json",
        {
            "sources": {
                str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files
            },
            "prediction_partitions": len(list(PARTITIONS.glob("*.parquet"))),
            "upstream_sss_partitions": len(list(UPSTREAM.glob("*.parquet"))),
            "new_independent_labels_opened": False,
        },
    )
    write_json(
        OUTPUT / "protocol.json",
        json.loads(CONFIG.read_text(encoding="utf-8")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "models", "analyze", "all"), default="all")
    parser.add_argument("--target", choices=("SSS", "fCO2"))
    parser.add_argument("--scheme", choices=("cruise", "geographic", "whole_region", "forward"))
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    if args.stage in {"audit", "all"}:
        audit_exposure()
        for target in ("SSS", "fCO2"):
            frame = load_frame(target)
            if not (SPLITS / f"{target}_split_indices.npz").is_file():
                audit_splits(target, frame, build_splits(frame))
        write_manifests()
    if args.stage in {"models", "all"}:
        run_models(args.target, args.scheme, args.fold)
        write_manifests()
    if args.stage in {"analyze", "all"}:
        analyze()
        write_manifests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
