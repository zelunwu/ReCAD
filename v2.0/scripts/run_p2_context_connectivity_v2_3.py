"""Run the full Issue #42 temporal-context and coastal-connectivity experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from netCDF4 import Dataset
from sklearn.ensemble import RandomForestRegressor

from recad.evaluate.context_connectivity import (
    ArrayStandardizer,
    ContextResidualNet,
    adjacent_residual_signal,
    count_parameters,
    water_neighbors,
)
from recad.evaluate.fco2_bakeoff import SeasonalTrendBackground, fco2_band
from recad.evaluate.predictor_ablation import balanced_sample_weight, frozen_splits
from recad.evaluate.sss_bakeoff import NumericTransform

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3")
MATCHUP = SOURCE / "socat_predictor_matchup_v2.3.parquet"
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
ISSUE41 = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_fco2_global_bakeoff_v2.3")
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_3\p2_context_connectivity_v2.3")
PARTITIONS = OUTPUT / "partitions"
CHECKPOINTS = OUTPUT / "checkpoints"
CONTEXT = OUTPUT / "context_cache"
CONTRACT = ROOT / "configs/frozen/context_connectivity_contract_v2.3.json"

STATIC_FEATURES = (
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
    "sss_crossfit",
)
HISTORY_FIELDS = ("sst", "wspd", "xco2air", "sss_bg")
MODELS = ("v11_random_forest", "lag_catboost", "gru_context", "st96_local", "st96_graph")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(MATCHUP)
    frame = frame.loc[frame.fco2.notna()].reset_index(drop=True)
    frame["truth"] = frame.fco2.astype("float32")
    frame["fco2_band"] = fco2_band(frame.truth).astype(str)
    frame["coast_band"] = pd.cut(
        frame.distance_to_significant_land_km,
        [-np.inf, 25, 100, 200, np.inf],
        labels=["0_25", "25_100", "100_200", "200_400"],
        right=False,
    ).astype(str)
    return frame


def build_context_cache(frame: pd.DataFrame, history: int) -> None:
    """Materialize label-free six-month local and water-neighbour histories."""

    manifest_path = CONTEXT / "manifest.json"
    expected = [CONTEXT / name for name in ("own.npy", "neighbour.npy", "neighbours.npy")]
    if manifest_path.exists() and all(path.exists() for path in expected):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["rows"] == len(frame) and manifest["history_months"] == history:
            return
    CONTEXT.mkdir(parents=True, exist_ok=True)
    own = np.lib.format.open_memmap(
        CONTEXT / "own.npy",
        mode="w+",
        dtype="float32",
        shape=(len(frame), history, len(HISTORY_FIELDS)),
    )
    neighbour = np.lib.format.open_memmap(
        CONTEXT / "neighbour.npy",
        mode="w+",
        dtype="float32",
        shape=(len(frame), history, len(HISTORY_FIELDS)),
    )
    own[:] = np.nan
    neighbour[:] = np.nan
    with Dataset(CUBE) as cube:
        grid_flat = np.asarray(cube.variables["grid_flat"][:], dtype=np.int64)
        node_neighbours = water_neighbors(grid_flat)
        np.save(CONTEXT / "neighbours.npy", node_neighbours)
        by_time = {
            int(time_index): indices
            for time_index, indices in frame.groupby("time_index", sort=False).indices.items()
        }
        for source_time in range(len(cube.dimensions["time"])):
            targets = [
                (lag, by_time[source_time + lag])
                for lag in range(history)
                if source_time + lag in by_time
            ]
            if not targets:
                continue
            field_values = {
                field: np.asarray(cube.variables[field][source_time, :], dtype="float32")
                for field in HISTORY_FIELDS
            }
            for lag, positions in targets:
                nodes = frame.coastal_node.to_numpy(np.int64)[positions]
                connected = node_neighbours[nodes]
                valid_connected = connected >= 0
                for field_index, field in enumerate(HISTORY_FIELDS):
                    values = field_values[field]
                    own[positions, lag, field_index] = values[nodes]
                    connected_values = np.full(connected.shape, np.nan, dtype="float32")
                    connected_values[valid_connected] = values[connected[valid_connected]]
                    count = np.isfinite(connected_values).sum(axis=1)
                    total = np.nansum(connected_values, axis=1)
                    neighbour[positions, lag, field_index] = np.divide(
                        total,
                        count,
                        out=np.full(len(positions), np.nan, dtype="float32"),
                        where=count > 0,
                    )
            if source_time % 24 == 0:
                print(f"context source month {source_time + 1}/396", flush=True)
    own.flush()
    neighbour.flush()
    del own, neighbour
    manifest_path.write_text(
        json.dumps(
            {
                "rows": len(frame),
                "history_months": history,
                "history_fields": HISTORY_FIELDS,
                "matchup_sha256": sha256(MATCHUP),
                "cube_sha256": sha256(CUBE),
                "target_labels_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def context_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(CONTEXT / "own.npy", mmap_mode="r"),
        np.load(CONTEXT / "neighbour.npy", mmap_mode="r"),
        np.load(CONTEXT / "neighbours.npy", mmap_mode="r"),
    )


def graph_trigger(
    frame: pd.DataFrame, node_neighbours: np.ndarray
) -> dict[str, float | int | bool]:
    path = OUTPUT / "graph_trigger.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    parts = [
        pd.read_parquet(partition, columns=["record_id", "truth", "prediction"])
        for partition in sorted(
            (ISSUE41 / "partitions").glob("fCO2_cruise_fold*_v11_random_forest.parquet")
        )
    ]
    residual = pd.concat(parts, ignore_index=True)
    residual["residual"] = residual.truth - residual.prediction
    residual = residual.merge(
        frame[["record_id", "time_index", "coastal_node"]],
        on="record_id",
        validate="one_to_one",
    )
    nodes = (
        residual.groupby(["time_index", "coastal_node"], as_index=False)
        .residual.mean()
        .reset_index(drop=True)
    )
    pairs = []
    for slot in range(node_neighbours.shape[1]):
        left = nodes.copy()
        left["neighbour_node"] = node_neighbours[left.coastal_node.to_numpy(int), slot]
        left = left.loc[left.neighbour_node.ge(0)]
        right = nodes.rename(
            columns={"coastal_node": "neighbour_node", "residual": "neighbour_residual"}
        )[["time_index", "neighbour_node", "neighbour_residual"]]
        pairs.append(left.merge(right, on=["time_index", "neighbour_node"], how="inner"))
    pair = pd.concat(pairs, ignore_index=True).drop_duplicates(
        ["time_index", "coastal_node", "neighbour_node"]
    )
    signal = adjacent_residual_signal(pair.residual, pair.neighbour_residual)
    signal["trigger_threshold"] = 0.15
    signal["triggered"] = bool(signal["correlation"] >= signal["trigger_threshold"])
    path.write_text(json.dumps(signal, indent=2) + "\n", encoding="utf-8")
    return signal


def outer_sss(
    scheme: str, fold: int, fit: pd.DataFrame, held: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    values = pd.read_parquet(ISSUE41 / "crossfit_sss" / f"{scheme}_fold{fold}.parquet")
    lookup = values.set_index(["role", "record_id"]).sss_crossfit
    fit_index = pd.MultiIndex.from_arrays(
        [np.repeat("fit", len(fit)), fit.record_id], names=["role", "record_id"]
    )
    held_index = pd.MultiIndex.from_arrays(
        [np.repeat("held", len(held)), held.record_id], names=["role", "record_id"]
    )
    fit_sss = lookup.reindex(fit_index).to_numpy("float32")
    held_sss = lookup.reindex(held_index).to_numpy("float32")
    if not np.isfinite(fit_sss).all() or not np.isfinite(held_sss).all():
        raise RuntimeError("Issue #41 cross-fitted SSS join failed")
    return fit_sss, held_sss


def catboost_model(seed: int, contract: dict[str, object]) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(contract["iterations"]),
        depth=int(contract["depth"]),
        learning_rate=float(contract["learning_rate"]),
        loss_function="RMSE",
        random_seed=seed,
        task_type="GPU",
        devices="0",
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
    )


def lag_matrix(static: pd.DataFrame, own: np.ndarray) -> np.ndarray:
    static_values = static.loc[:, STATIC_FEATURES].to_numpy("float32")
    history = np.asarray(own, dtype="float32").reshape(len(static), -1)
    available = np.isfinite(history).astype("float32")
    return np.concatenate([static_values, history, available], axis=1)


def fit_lag_catboost(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    own_fit: np.ndarray,
    own_held: np.ndarray,
    background_fit: np.ndarray,
    background_held: np.ndarray,
    *,
    seed: int,
    budget: dict[str, object],
    checkpoint: Path,
) -> tuple[np.ndarray, dict[str, float | int]]:
    model = catboost_model(seed, budget)
    train = lag_matrix(fit, own_fit)
    query = lag_matrix(held, own_held)
    start = time.perf_counter()
    model.fit(
        train,
        fit.truth.to_numpy(float) - background_fit,
        sample_weight=balanced_sample_weight(fit),
    )
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = background_held + model.predict(query)
    inference_seconds = time.perf_counter() - start
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(checkpoint)
    return prediction, {
        "epochs": int(budget["iterations"]),
        "parameters": int(model.tree_count_),
        "training_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_mb": np.nan,
    }


def _stable_inner_validation(frame: pd.DataFrame) -> np.ndarray:
    hashed = pd.util.hash_pandas_object(frame.group_key.astype(str), index=False).to_numpy(
        np.uint64
    )
    return hashed % 10 == 0


def _tensor_batch(
    index: np.ndarray,
    own: np.ndarray,
    neighbour: np.ndarray,
    static: np.ndarray,
    own_transform: ArrayStandardizer,
    neighbour_transform: ArrayStandardizer,
    static_transform: ArrayStandardizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    own_values, own_available = own_transform.apply(own[index])
    neighbour_values, neighbour_available = neighbour_transform.apply(neighbour[index])
    static_values, _ = static_transform.apply(static[index])
    return tuple(
        torch.as_tensor(value, device=device)
        for value in (
            own_values,
            own_available,
            static_values,
            neighbour_values,
            neighbour_available,
        )
    )


def fit_neural_context(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    own_fit: np.ndarray,
    own_held: np.ndarray,
    neighbour_fit: np.ndarray,
    neighbour_held: np.ndarray,
    background_fit: np.ndarray,
    background_held: np.ndarray,
    *,
    mode: str,
    seed: int,
    budget: dict[str, object],
    checkpoint: Path,
) -> tuple[np.ndarray, dict[str, float | int]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner_validation = _stable_inner_validation(fit)
    train_index = np.flatnonzero(~inner_validation)
    validation_index = np.flatnonzero(inner_validation)
    static = fit.loc[:, STATIC_FEATURES].to_numpy("float32")
    held_static = held.loc[:, STATIC_FEATURES].to_numpy("float32")
    own_transform = ArrayStandardizer.fit(own_fit[train_index])
    neighbour_transform = ArrayStandardizer.fit(neighbour_fit[train_index])
    static_transform = ArrayStandardizer.fit(static[train_index])
    target = fit.truth.to_numpy("float32") - np.asarray(background_fit, dtype="float32")
    target_scale = float(np.std(target[train_index]))
    target_scale = target_scale if target_scale > 1e-6 else 1.0
    target /= target_scale
    weights = balanced_sample_weight(fit).astype("float32")
    model = ContextResidualNet(
        sequence_features=len(HISTORY_FIELDS),
        static_features=len(STATIC_FEATURES),
        mode=mode,
        sequence_length=own_fit.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    batch_size = int(budget["batch_size"])
    rng = np.random.default_rng(seed)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    epochs_completed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_training = time.perf_counter()
    for epoch in range(int(budget["maximum_epochs"])):
        model.train()
        shuffled = rng.permutation(train_index)
        for start in range(0, len(shuffled), batch_size):
            index = shuffled[start : start + batch_size]
            batch = _tensor_batch(
                index,
                own_fit,
                neighbour_fit,
                static,
                own_transform,
                neighbour_transform,
                static_transform,
                device,
            )
            batch_target = torch.as_tensor(target[index], device=device)
            batch_weight = torch.as_tensor(weights[index], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                prediction = model(*batch)
                loss = torch.sum(batch_weight * (prediction - batch_target) ** 2) / torch.sum(
                    batch_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        total = 0.0
        total_weight = 0.0
        with torch.inference_mode():
            for start in range(0, len(validation_index), batch_size):
                index = validation_index[start : start + batch_size]
                batch = _tensor_batch(
                    index,
                    own_fit,
                    neighbour_fit,
                    static,
                    own_transform,
                    neighbour_transform,
                    static_transform,
                    device,
                )
                prediction = model(*batch).float().cpu().numpy()
                total += float(np.sum(weights[index] * (prediction - target[index]) ** 2))
                total_weight += float(np.sum(weights[index]))
        validation_loss = total / total_weight
        epochs_completed = epoch + 1
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= int(budget["early_stopping_patience"]):
            break
    training_seconds = time.perf_counter() - start_training
    if best_state is None:
        raise RuntimeError("neural context training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    prediction = np.empty(len(held), dtype="float32")
    held_index = np.arange(len(held))
    start_inference = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(held), batch_size):
            index = held_index[start : start + batch_size]
            batch = _tensor_batch(
                index,
                own_held,
                neighbour_held,
                held_static,
                own_transform,
                neighbour_transform,
                static_transform,
                device,
            )
            prediction[index] = model(*batch).float().cpu().numpy() * target_scale
    inference_seconds = time.perf_counter() - start_inference
    peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "mode": mode,
            "own_transform": own_transform,
            "neighbour_transform": neighbour_transform,
            "static_transform": static_transform,
            "target_scale": target_scale,
            "best_inner_validation_mse": best_loss,
            "epochs": epochs_completed,
        },
        checkpoint,
    )
    return background_held + prediction, {
        "epochs": epochs_completed,
        "parameters": count_parameters(model),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_mb": peak_gpu_mb,
        "best_inner_validation_rmse": math.sqrt(best_loss) * target_scale,
    }


def source_baseline(scheme: str, fold: int, held: pd.DataFrame) -> np.ndarray:
    source = pd.read_parquet(
        ISSUE41 / "partitions" / f"fCO2_{scheme}_fold{fold}_v11_random_forest.parquet",
        columns=["record_id", "prediction"],
    ).set_index("record_id")
    prediction = held.record_id.map(source.prediction)
    if prediction.isna().any():
        raise RuntimeError("Issue #41 random-forest prediction join failed")
    return prediction.to_numpy("float32")


def support_band(fit: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = fit.groupby("lme_id", observed=True).group_key.nunique()
    support = held.lme_id.map(counts).fillna(0).to_numpy(int)
    band = pd.cut(
        support,
        [-1, 4, 14, 49, np.inf],
        labels=["lt5", "5_14", "15_49", "ge50"],
    ).astype(str)
    return support, np.asarray(band)


def write_prediction(
    scheme: str,
    fold: int,
    model: str,
    held: pd.DataFrame,
    prediction: np.ndarray,
    background: np.ndarray,
    neighbour_count: np.ndarray,
    lme_support: np.ndarray,
    lme_support_band: np.ndarray,
) -> None:
    columns = [
        "record_id",
        "group_key",
        "yr",
        "mon",
        "time_index",
        "coastal_node",
        "latitude",
        "longitude",
        "lme_id",
        "truth",
        "distance_to_significant_land_km",
        "shelf_core_mask",
        "fco2_band",
        "coast_band",
        "sss_crossfit",
    ]
    result = held.loc[:, columns].copy()
    result["background"] = np.asarray(background, dtype="float32")
    result["prediction"] = np.asarray(prediction, dtype="float32")
    result["water_neighbour_count"] = neighbour_count.astype("int8")
    result["training_lme_cruises"] = lme_support.astype("int32")
    result["support_band"] = lme_support_band
    result["scheme"] = scheme
    result["fold"] = fold
    result["model"] = model
    result.to_parquet(
        PARTITIONS / f"fCO2_{scheme}_fold{fold}_{model}.parquet",
        index=False,
        compression="zstd",
    )


def run(*, only_scheme: str | None = None, only_model: str | None = None) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path in (OUTPUT, PARTITIONS, CHECKPOINTS, CONTEXT):
        path.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    build_context_cache(frame, int(contract["history_months"]))
    own, neighbour, node_neighbours = context_arrays()
    trigger = graph_trigger(frame, node_neighbours)
    active_models = [model for model in MODELS if model != "st96_graph" or trigger["triggered"]]
    if only_model:
        active_models = [model for model in active_models if model == only_model]
    log_path = OUTPUT / "training_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    for scheme_index, scheme in enumerate(contract["outer_schemes"]):
        if only_scheme and scheme != only_scheme:
            continue
        for split in frozen_splits(frame, scheme):
            expected = [
                PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_{model}.parquet"
                for model in active_models
            ]
            if expected and all(path.exists() for path in expected):
                print(f"reuse complete {scheme}/{split.fold}", flush=True)
                continue
            fit = frame.iloc[split.fit_index].reset_index(drop=True)
            held = frame.iloc[split.held_index].reset_index(drop=True)
            fit_sss, held_sss = outer_sss(scheme, split.fold, fit, held)
            fit["sss_crossfit"] = fit_sss
            held["sss_crossfit"] = held_sss
            background_model = SeasonalTrendBackground().fit(fit)
            background_fit = background_model.predict(fit)
            background_held = background_model.predict(held)
            own_fit = own[split.fit_index]
            own_held = own[split.held_index]
            neighbour_fit = neighbour[split.fit_index]
            neighbour_held = neighbour[split.held_index]
            support, bands = support_band(fit, held)
            neighbour_count = np.sum(node_neighbours[held.coastal_node.to_numpy(int)] >= 0, axis=1)
            seed = 76023 + scheme_index * 100 + split.fold * 10
            print(f"{scheme}/{split.fold}: fit={len(fit):,} held={len(held):,}", flush=True)
            for model_name in active_models:
                destination = PARTITIONS / f"fCO2_{scheme}_fold{split.fold}_{model_name}.parquet"
                if destination.exists():
                    continue
                if model_name == "v11_random_forest":
                    start = time.perf_counter()
                    prediction = source_baseline(scheme, split.fold, held)
                    resources = {
                        "epochs": 96,
                        "parameters": 96,
                        "training_seconds": 0.0,
                        "inference_seconds": time.perf_counter() - start,
                        "peak_gpu_mb": 0.0,
                    }
                elif model_name == "lag_catboost":
                    prediction, resources = fit_lag_catboost(
                        fit,
                        held,
                        own_fit,
                        own_held,
                        background_fit,
                        background_held,
                        seed=seed + 1,
                        budget=contract["catboost_budget"],
                        checkpoint=CHECKPOINTS / f"{scheme}_fold{split.fold}_lag_catboost.cbm",
                    )
                else:
                    mode = {
                        "gru_context": "gru",
                        "st96_local": "st96_local",
                        "st96_graph": "st96_graph",
                    }[model_name]
                    prediction, resources = fit_neural_context(
                        fit,
                        held,
                        own_fit,
                        own_held,
                        neighbour_fit,
                        neighbour_held,
                        background_fit,
                        background_held,
                        mode=mode,
                        seed=seed
                        + {"gru_context": 2, "st96_local": 3, "st96_graph": 4}[model_name],
                        budget=contract["neural_budget"],
                        checkpoint=CHECKPOINTS / f"{scheme}_fold{split.fold}_{model_name}.pt",
                    )
                write_prediction(
                    scheme,
                    split.fold,
                    model_name,
                    held,
                    prediction,
                    background_held,
                    neighbour_count,
                    support,
                    bands,
                )
                log.append(
                    {
                        "scheme": scheme,
                        "fold": split.fold,
                        "model": model_name,
                        "fit_rows": len(fit),
                        "held_rows": len(held),
                        **resources,
                    }
                )
                pd.DataFrame(log).drop_duplicates(["scheme", "fold", "model"], keep="last").to_csv(
                    log_path, index=False
                )
                print(
                    f"  {model_name}: train={resources['training_seconds'] / 60:.1f} min",
                    flush=True,
                )


def _nested_cruise_subset(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    cruises = frame[["lme_id", "group_key"]].drop_duplicates().copy()
    cruises["rank_key"] = pd.util.hash_pandas_object(
        cruises.group_key.astype(str), index=False
    ).to_numpy(np.uint64)
    cruises["rank"] = cruises.groupby("lme_id", observed=True).rank_key.rank(
        method="first", pct=True
    )
    selected = set(cruises.loc[cruises["rank"].le(fraction), "group_key"].astype(str))
    return frame.group_key.astype(str).isin(selected).to_numpy()


def _curve_metrics(
    held: pd.DataFrame,
    prediction: np.ndarray,
    *,
    model: str,
    fraction: float,
    fit: pd.DataFrame,
    resources: dict[str, float | int],
) -> dict[str, float | int | str]:
    error = np.asarray(prediction) - held.truth.to_numpy(float)
    per_cruise = (
        pd.DataFrame({"group": held.group_key, "sq": error**2})
        .groupby("group", observed=True)
        .sq.mean()
    )
    return {
        "model": model,
        "cruise_fraction": fraction,
        "fit_rows": len(fit),
        "fit_cruises": fit.group_key.nunique(),
        "held_rows": len(held),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "cruise_equal_rmse": float(np.sqrt(per_cruise.mean())),
        "bias": float(np.mean(error)),
        **resources,
    }


def run_learning_curve() -> None:
    """Measure model-capacity scaling under nested fCO2 cruise-label budgets."""

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    frame = load_frame()
    build_context_cache(frame, int(contract["history_months"]))
    own, neighbour, _ = context_arrays()
    split = frozen_splits(frame, "cruise")[0]
    full_fit = frame.iloc[split.fit_index].reset_index(drop=True)
    held = frame.iloc[split.held_index].reset_index(drop=True)
    full_fit_sss, held_sss = outer_sss("cruise", split.fold, full_fit, held)
    full_fit["sss_crossfit"] = full_fit_sss
    held["sss_crossfit"] = held_sss
    own_full = own[split.fit_index]
    own_held = own[split.held_index]
    neighbour_full = neighbour[split.fit_index]
    neighbour_held = neighbour[split.held_index]
    destination = OUTPUT / "learning_curve_partitions"
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for fraction in contract["learning_curve_cruise_fractions"]:
        fraction = float(fraction)
        selected = _nested_cruise_subset(full_fit, fraction)
        fit = full_fit.loc[selected].reset_index(drop=True)
        own_fit = own_full[selected]
        neighbour_fit = neighbour_full[selected]
        background_model = SeasonalTrendBackground().fit(fit)
        background_fit = background_model.predict(fit)
        background_held = background_model.predict(held)
        fraction_label = f"{round(fraction * 100):03d}"
        for model_name in ("v11_random_forest", "lag_catboost", "st96_local"):
            partition = destination / f"fraction{fraction_label}_{model_name}.parquet"
            if partition.exists():
                saved = pd.read_parquet(partition)
                resources = json.loads(saved.attrs.get("resources", "{}"))
                prediction = saved.prediction.to_numpy(float)
            elif model_name == "v11_random_forest":
                transform = NumericTransform.fit(fit, STATIC_FEATURES)
                model = RandomForestRegressor(
                    n_estimators=96,
                    max_depth=18,
                    min_samples_leaf=20,
                    max_features=0.7,
                    max_samples=0.5,
                    n_jobs=-1,
                    random_state=86023 + int(fraction * 100),
                )
                start = time.perf_counter()
                model.fit(
                    transform.apply(fit),
                    fit.truth,
                    sample_weight=balanced_sample_weight(fit),
                )
                training_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = model.predict(transform.apply(held))
                resources = {
                    "epochs": 96,
                    "parameters": 96,
                    "training_seconds": training_seconds,
                    "inference_seconds": time.perf_counter() - start,
                    "peak_gpu_mb": 0.0,
                }
                joblib.dump(
                    {"model": model, "transform": transform},
                    CHECKPOINTS / f"curve_fraction{fraction_label}_rf.joblib",
                    compress=3,
                )
            elif model_name == "lag_catboost":
                prediction, resources = fit_lag_catboost(
                    fit,
                    held,
                    own_fit,
                    own_held,
                    background_fit,
                    background_held,
                    seed=87023 + int(fraction * 100),
                    budget=contract["catboost_budget"],
                    checkpoint=CHECKPOINTS / f"curve_fraction{fraction_label}_lag.cbm",
                )
            else:
                prediction, resources = fit_neural_context(
                    fit,
                    held,
                    own_fit,
                    own_held,
                    neighbour_fit,
                    neighbour_held,
                    background_fit,
                    background_held,
                    mode="st96_local",
                    seed=88023 + int(fraction * 100),
                    budget=contract["neural_budget"],
                    checkpoint=CHECKPOINTS / f"curve_fraction{fraction_label}_st96.pt",
                )
            pd.DataFrame(
                {
                    "record_id": held.record_id,
                    "truth": held.truth,
                    "prediction": prediction,
                    "group_key": held.group_key,
                }
            ).to_parquet(partition, index=False, compression="zstd")
            rows.append(
                _curve_metrics(
                    held,
                    prediction,
                    model=model_name,
                    fraction=fraction,
                    fit=fit,
                    resources=resources,
                )
            )
            print(
                f"curve {fraction_label}% {model_name}: RMSE={rows[-1]['rmse']:.2f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "learning_curve.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("cruise", "spatial", "whole_lme", "forward"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--build-context-only", action="store_true")
    parser.add_argument("--learning-curve", action="store_true")
    args = parser.parse_args()
    if args.learning_curve:
        run_learning_curve()
    elif args.build_context_only:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        build_context_cache(load_frame(), int(contract["history_months"]))
    else:
        run(only_scheme=args.scheme, only_model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
