"""Fit the frozen Issue 44 candidates and project the complete 1993-2025 domain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import netCDF4
import numpy as np
import pandas as pd

from recad.evaluate.model_zoo import fit_model, predict_model, save_model
from recad.evaluate.predictor_ablation import balanced_sample_weight
from recad.evaluate.process_predictors import ContinuousBackground
from recad.evaluate.reliability_contract import DeployableSupportIndex, conformal_quantile
from recad.evaluate.validation_repair import conformal_widths
from run_p2_process_predictors_v2_4 import load_augmented_frame
from run_p2_validation_repair_v2_4 import MODERN_RF_FEATURES, SSS_RESIDUAL_FEATURES

OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_global_reliability_contract_v2.4")
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
CHECKPOINTS = OUTPUT / "checkpoints"
PRODUCTS = OUTPUT / "products"
TABLES = OUTPUT / "tables"
UPSTREAM_SSS = Path(
    r"C:\backup\phd\data\processed\recad_v2_4\p2_validation_repair_v2.4\upstream_sss"
)
ENVIRONMENT = (
    "sst",
    "wspd",
    "xco2air",
    "sss_bg",
    "distance_to_significant_land_km",
    "bathymetry_m",
)
RISK_FEATURES = (
    "log1p_geographic_km",
    "log1p_environment_k16",
    "effective_groups",
    "unique_cruises",
    "support_entropy",
    "time_extrapolation_years",
    "missing_mandatory_input",
    "model_disagreement",
    "background_disagreement",
)
SSS_CONFIG = [700, 10, 0.02, 12.0]
F4_CONFIG = [256, 24, 5, 0.7]
CAT_CONFIG = [700, 10, 0.02, 12.0]
FCO2_CAT_WEIGHT = 0.30
SEEDS = (62024, 62025, 62026)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_cruise(target: str) -> pd.DataFrame:
    paths = sorted((OUTPUT / "partitions").glob(f"{target}_cruise_fold*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def attach_final_sss_crossfit(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the existing leakage-safe SSS predictions for final fCO2 fitting."""

    paths = sorted(UPSTREAM_SSS.glob("fCO2_cruise_fold*.parquet"))
    if not paths:
        raise FileNotFoundError("Issue 61 fCO2 upstream SSS partitions are missing")
    predictions = pd.concat(
        [pd.read_parquet(path, columns=["record_id", "sss_crossfit"]) for path in paths],
        ignore_index=True,
    )
    mean = predictions.groupby("record_id", observed=True).sss_crossfit.mean()
    result = frame.copy()
    result["sss_crossfit"] = result.record_id.map(mean).astype("float32")
    if result.sss_crossfit.isna().any():
        raise RuntimeError("final fCO2 frame has missing cross-fitted SSS")
    return result


def fit_sss(frame: pd.DataFrame):
    weight = balanced_sample_weight(frame)
    result = []
    available = frame.sss_bg.notna().to_numpy()
    residual_frame = frame.loc[available].reset_index(drop=True)
    for seed in SEEDS:
        direct = fit_model(
            "catboost",
            SSS_CONFIG,
            frame,
            SSS_RESIDUAL_FEATURES,
            frame.truth.to_numpy(float),
            weight,
            seed,
        )
        residual = fit_model(
            "catboost",
            SSS_CONFIG,
            residual_frame,
            SSS_RESIDUAL_FEATURES,
            residual_frame.truth.to_numpy(float) - residual_frame.sss_bg.to_numpy(float),
            balanced_sample_weight(residual_frame),
            seed + 1,
        )
        save_model(direct, CHECKPOINTS / f"SSS_seed{seed}_direct")
        save_model(residual, CHECKPOINTS / f"SSS_seed{seed}_residual")
        result.append((direct, residual))
    return result


def predict_sss(models, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    members = []
    available = frame.sss_bg.notna().to_numpy()
    for direct, residual in models:
        prediction, _ = predict_model(direct, frame)
        correction, _ = predict_model(residual, frame.loc[available])
        prediction[available] = frame.loc[available, "sss_bg"].to_numpy(float) + correction
        members.append(prediction)
    array = np.column_stack(members)
    return array.mean(axis=1).astype("float32"), array.std(axis=1).astype("float32")


def fit_fco2(frame: pd.DataFrame):
    weight = balanced_sample_weight(frame)
    background = ContinuousBackground.fit(frame, frame.truth.to_numpy(float), weight, alpha=10.0)
    residual = fit_model(
        "extra_trees",
        F4_CONFIG,
        frame,
        MODERN_RF_FEATURES,
        frame.truth.to_numpy(float) - background.predict(frame),
        weight,
        64024,
    )
    direct = fit_model(
        "catboost",
        CAT_CONFIG,
        frame,
        MODERN_RF_FEATURES,
        frame.truth.to_numpy(float),
        weight,
        64025,
    )
    save_model(residual, CHECKPOINTS / "fCO2_F4_residual")
    save_model(direct, CHECKPOINTS / "fCO2_catboost")
    joblib.dump(background.estimator, CHECKPOINTS / "fCO2_F4_background.joblib")
    return background, residual, direct


def predict_fco2(models, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    background, residual, direct = models
    residual_prediction, _ = predict_model(residual, frame)
    f4 = background.predict(frame) + residual_prediction
    cat, _ = predict_model(direct, frame)
    return (
        (f4 + FCO2_CAT_WEIGHT * (cat - f4)).astype("float32"),
        np.abs(cat - f4).astype("float32"),
    )


def fit_risk(target: str, observations: pd.DataFrame):
    oof = pooled_cruise(target)
    oof["log1p_geographic_km"] = np.log1p(oof.geographic_km)
    oof["log1p_environment_k16"] = np.log1p(oof.environment_k16)
    metadata = observations[["record_id", "lme_id"]].drop_duplicates("record_id")
    oof = oof.merge(metadata, on="record_id", how="left", validate="one_to_one")
    response = np.log1p(np.abs(oof.truth.to_numpy(float) - oof.prediction.to_numpy(float)))
    model = fit_model(
        "extra_trees",
        [192, 24, 20, 0.7],
        oof,
        RISK_FEATURES,
        response,
        balanced_sample_weight(oof),
        44024,
    )
    save_model(model, CHECKPOINTS / f"{target}_risk")
    thresholds = np.quantile(oof.predicted_error_scale, [0.5, 0.75, 0.9])
    contract = {
        "target": target,
        "risk_thresholds": thresholds.tolist(),
        "geographic_max_km": float(np.quantile(oof.geographic_km, 0.99)),
        "environment_max": float(np.quantile(oof.environment_k16, 0.99)),
        "disagreement_max": float(np.quantile(oof.model_disagreement, 0.99)),
        "interval_method": "support_risk_scaled" if target == "SSS" else "prediction_stratified",
    }
    if target == "SSS":
        scale = np.maximum(oof.predicted_error_scale.to_numpy(float), 1e-3)
        score = np.abs(oof.truth - oof.prediction).to_numpy(float) / scale
        contract["interval_q50"] = conformal_quantile(score, 0.5)
        contract["interval_q90"] = conformal_quantile(score, 0.9)
    else:
        # Store cruise OOF arrays only in memory for the frozen prediction-stratified control.
        contract["interval_calibration_rows"] = len(oof)
    return model, oof, contract


def static_frame(source: netCDF4.Dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: np.asarray(source.variables[name][:])
            for name in (
                "latitude",
                "longitude",
                "latitude_sin",
                "latitude_cos",
                "longitude_sin",
                "longitude_cos",
                "distance_to_significant_land_km",
                "bathymetry_m",
                "shelf_core_mask",
                "slope_deep_mask",
            )
        }
    )


def base_frame(source: netCDF4.Dataset, time_index: int, static: pd.DataFrame) -> pd.DataFrame:
    year = int(source.variables["year"][time_index])
    month = int(source.variables["month"][time_index])
    frame = static.copy()
    frame["yr"] = np.full(len(frame), year, dtype=np.int16)
    frame["mon"] = np.full(len(frame), month, dtype=np.uint8)
    frame["year_centered"] = np.full(len(frame), year - 2009, dtype=np.float32)
    frame["month_sin"] = np.full(
        len(frame), source.variables["month_sin"][time_index], dtype=np.float32
    )
    frame["month_cos"] = np.full(
        len(frame), source.variables["month_cos"][time_index], dtype=np.float32
    )
    for name in ("sst", "wspd", "xco2air", "sss_bg"):
        frame[name] = np.asarray(source.variables[name][time_index, :].filled(np.nan))
    frame["sss_bg_available"] = np.asarray(source.variables["sss_bg_available"][time_index, :])
    return frame


def create_product(path: Path, source: netCDF4.Dataset, target: str) -> netCDF4.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = netCDF4.Dataset(path, "w", format="NETCDF4")
    dataset.createDimension("time", len(source.dimensions["time"]))
    dataset.createDimension("node", len(source.dimensions["node"]))
    for name, dtype, dims in (
        ("time", "i8", ("time",)),
        ("latitude", "f4", ("node",)),
        ("longitude", "f4", ("node",)),
        ("cell_area_km2", "f4", ("node",)),
    ):
        variable = dataset.createVariable(name, dtype, dims)
        if name == "time":
            variable[:] = source.variables["time"][:]
            variable.units = source.variables["time"].units
            variable.calendar = getattr(source.variables["time"], "calendar", "proleptic_gregorian")
        else:
            variable[:] = source.variables[name][:]
    chunks = (1, 65536)
    for name in (
        "prediction",
        "width50",
        "width90",
        "risk_score",
        "training_support_km",
        "environmental_dissimilarity",
        "effective_groups",
        "model_disagreement",
    ):
        dataset.createVariable(
            name,
            "f4",
            ("time", "node"),
            zlib=True,
            complevel=2,
            chunksizes=chunks,
            fill_value=np.nan,
        )
    for name, dtype in (("applicability_grade", "u1"), ("status", "u1"), ("reason_bits", "u2")):
        dataset.createVariable(
            name, dtype, ("time", "node"), zlib=True, complevel=2, chunksizes=chunks
        )
    dataset.target = target
    dataset.grade_codes = "1=A, 2=B, 3=C, 4=D"
    dataset.status_codes = "1=trusted, 2=diagnostic, 3=suppressed"
    dataset.model_version = "ReCAD-v2.4-Issue44"
    dataset.calibration_version = "p2_global_reliability_contract_v2.4"
    return dataset


def assign_contract(
    frame: pd.DataFrame, risk: np.ndarray, contract: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q50, q75, q90 = contract["risk_thresholds"]
    grade = np.select([risk <= q50, risk <= q75, risk <= q90], [1, 2, 3], default=4).astype("uint8")
    reason = np.zeros(len(frame), dtype="uint16")
    reason[frame.geographic_km.to_numpy(float) > 200] |= 1
    reason[frame.environment_k16.to_numpy(float) > contract["environment_max"]] |= 16
    low_group = frame.effective_groups.to_numpy(float) < 2
    reason[low_group] |= 32
    missing = frame.missing_mandatory_input.to_numpy(bool)
    reason[missing] |= 64
    temporal = frame.time_extrapolation_years.to_numpy(float) > 0
    reason[temporal] |= 512
    disagreement = frame.model_disagreement.to_numpy(float) > contract["disagreement_max"]
    reason[disagreement] |= 1024
    hard = (
        (frame.geographic_km.to_numpy(float) > contract["geographic_max_km"])
        | (frame.environment_k16.to_numpy(float) > contract["environment_max"])
        | missing
    )
    grade[hard] = 4
    grade[(grade < 3) & (low_group | temporal | disagreement)] = 3
    status = np.select([grade <= 2, grade == 3], [1, 2], default=3).astype("uint8")
    return grade, status, reason


def main() -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    sss_train = load_augmented_frame("SSS")
    fco2_train = attach_final_sss_crossfit(load_augmented_frame("fCO2"))
    sss_models = fit_sss(sss_train)
    fco2_models = fit_fco2(fco2_train)
    sss_risk, _sss_oof, sss_contract = fit_risk("SSS", sss_train)
    fco2_risk, fco2_oof, fco2_contract = fit_risk("fCO2", fco2_train)
    contracts = {"SSS": sss_contract, "fCO2": fco2_contract}
    frozen = {
        "contracts": contracts,
        "sss_config": SSS_CONFIG,
        "fco2_f4_config": F4_CONFIG,
        "fco2_catboost_config": CAT_CONFIG,
        "fco2_catboost_weight": FCO2_CAT_WEIGHT,
        "independent_labels_opened": False,
        "labels_2026_opened": False,
    }
    (OUTPUT / "calibration_contract_frozen.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    support = {
        "SSS": DeployableSupportIndex(ENVIRONMENT).fit(sss_train),
        "fCO2": DeployableSupportIndex(ENVIRONMENT).fit(fco2_train),
    }
    write = {}
    rows = []
    with netCDF4.Dataset(CUBE) as source:
        static = static_frame(source)
        geographic = {
            target: index.query_geographic(static[["latitude", "longitude"]])
            for target, index in support.items()
        }
        for target in ("SSS", "fCO2"):
            write[target] = create_product(
                PRODUCTS / f"{target}_reliability_v2.4.nc", source, target
            )
        try:
            for time_index in range(len(source.dimensions["time"])):
                query = base_frame(source, time_index, static)
                sss_prediction, sss_disagreement = predict_sss(sss_models, query)
                query["sss_crossfit"] = sss_prediction
                predictions = {"SSS": (sss_prediction, sss_disagreement)}
                predictions["fCO2"] = predict_fco2(fco2_models, query)
                for target in ("SSS", "fCO2"):
                    prediction, disagreement = predictions[target]
                    features = pd.concat(
                        [geographic[target], support[target].query_environmental(query)], axis=1
                    )
                    features["log1p_geographic_km"] = np.log1p(features.geographic_km)
                    features["log1p_environment_k16"] = np.log1p(features.environment_k16)
                    features["model_disagreement"] = disagreement
                    features["background_disagreement"] = (
                        np.abs(prediction - query.sss_bg.to_numpy(float))
                        if target == "SSS"
                        else disagreement
                    )
                    log_risk, _ = predict_model(
                        sss_risk if target == "SSS" else fco2_risk, features
                    )
                    risk = np.maximum(np.expm1(log_risk), 1e-3)
                    contract = contracts[target]
                    if target == "SSS":
                        width50 = contract["interval_q50"] * risk
                        width90 = contract["interval_q90"] * risk
                    else:
                        width50, width90, _ = conformal_widths(
                            fco2_oof.truth.to_numpy(float),
                            fco2_oof.prediction.to_numpy(float),
                            prediction,
                            target="fCO2",
                        )
                    grade, status, reason = assign_contract(features, risk, contract)
                    destination = write[target]
                    values = {
                        "prediction": prediction,
                        "width50": width50,
                        "width90": width90,
                        "risk_score": risk,
                        "training_support_km": features.geographic_km,
                        "environmental_dissimilarity": features.environment_k16,
                        "effective_groups": features.effective_groups,
                        "model_disagreement": disagreement,
                        "applicability_grade": grade,
                        "status": status,
                        "reason_bits": reason,
                    }
                    for name, value in values.items():
                        destination.variables[name][time_index, :] = np.asarray(value)
                    area = np.asarray(source.variables["cell_area_km2"][:], float)
                    for code, label in ((1, "A"), (2, "B"), (3, "C"), (4, "D")):
                        selected = grade == code
                        rows.append(
                            {
                                "target": target,
                                "time_index": time_index,
                                "year": int(query.yr.iloc[0]),
                                "month": int(query.mon.iloc[0]),
                                "grade": label,
                                "nodes": int(selected.sum()),
                                "area_fraction": float(area[selected].sum() / area.sum()),
                            }
                        )
                if time_index % 12 == 11:
                    pd.DataFrame(rows).to_csv(
                        TABLES / "product_monthly_grade_coverage.csv", index=False
                    )
                    print(f"completed month {time_index + 1}/396", flush=True)
        finally:
            for dataset in write.values():
                dataset.close()
    pd.DataFrame(rows).to_csv(TABLES / "product_monthly_grade_coverage.csv", index=False)
    payload = {
        "contracts": contracts,
        "fco2_catboost_weight": FCO2_CAT_WEIGHT,
        "products": {
            target: {
                "path": str(PRODUCTS / f"{target}_reliability_v2.4.nc"),
                "sha256": sha256(PRODUCTS / f"{target}_reliability_v2.4.nc"),
            }
            for target in ("SSS", "fCO2")
        },
        "independent_labels_opened": False,
        "labels_2026_opened": False,
    }
    (OUTPUT / "calibration_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUTPUT / "product.complete").write_text("complete\n")


if __name__ == "__main__":
    main()
