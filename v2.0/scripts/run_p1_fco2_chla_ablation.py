"""Run the Issue #26 Chl-a single-factor ablation on identical support."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from catboost import CatBoostRegressor

from recad.evaluate.applicability import outer_splits
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose
from run_p1_fco2_reliability import (
    CATEGORICAL,
    MODEL_COLUMNS,
    NUMERIC,
    ROOT,
    SeasonalTrendClimatology,
    add_crossfit_sss,
    balanced_weights,
    metric_values,
    prepare,
)

EXPERIMENT_ID = "p1_fco2_reliability_v2.2"


def add_chla(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["log_chla"] = np.nan
    with xr.open_dataset(path) as dataset:
        lat0, lon0 = float(dataset.lat[0]), float(dataset.lon[0])
        dlat = float(dataset.lat[1] - dataset.lat[0])
        dlon = float(dataset.lon[1] - dataset.lon[0])
        for (year, month), index in result.groupby(["year", "month"]).groups.items():
            lat = result.loc[index, "latitude"].to_numpy(float)
            lon = result.loc[index, "longitude"].to_numpy(float)
            lat_index = np.rint((lat - lat0) / dlat).astype(int)
            lon_index = np.rint((lon - lon0) / dlon).astype(int)
            inside = (
                (lat_index >= 0)
                & (lat_index < dataset.sizes["lat"])
                & (lon_index >= 0)
                & (lon_index < dataset.sizes["lon"])
            )
            if not inside.any():
                continue
            field = dataset.chla.sel(time=f"{int(year)}-{int(month):02d}-15").to_numpy()
            values = np.full(len(index), np.nan)
            values[inside] = field[lat_index[inside], lon_index[inside]]
            values = np.where(values > 0, np.log10(values), np.nan)
            result.loc[index, "log_chla"] = values
    return result


def predict(
    fit: pd.DataFrame,
    held: pd.DataFrame,
    *,
    include_chla: bool,
    iterations: int,
    seed: int,
) -> np.ndarray:
    numeric = [*NUMERIC, "log_chla"] if include_chla else NUMERIC
    columns = numeric + CATEGORICAL
    x_fit, x_held = fit[columns].copy(), held[columns].copy()
    for column in numeric:
        median = float(x_fit[column].median())
        x_fit[column] = x_fit[column].fillna(median)
        x_held[column] = x_held[column].fillna(median)
    for column in CATEGORICAL:
        x_fit[column] = x_fit[column].fillna(-999).astype(int).astype(str)
        x_held[column] = x_held[column].fillna(-999).astype(int).astype(str)
    model = CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        random_strength=0.5,
        l2_leaf_reg=5.0,
        allow_writing_files=False,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    model.fit(
        x_fit,
        fit.residual,
        cat_features=[columns.index(column) for column in CATEGORICAL],
        sample_weight=balanced_weights(fit) * len(fit),
        verbose=False,
    )
    return held.background.to_numpy(float) + model.predict(x_held)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_fco2_reliability_v2.2.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    gateway = P1DataGateway(manifest)
    train = prepare(gateway.load_labels("fco2", Purpose.TRAIN, columns=MODEL_COLUMNS))
    development = prepare(gateway.load_labels("fco2", Purpose.SELECTION, columns=MODEL_COLUMNS))
    train = add_chla(train, Path(config["chla_path"]))
    development = add_chla(development, Path(config["chla_path"]))
    rows = []
    predictions = []
    sss_path = Path(config["sss_reliability_output"]) / "development_oof_predictions.parquet"
    for scheme in config["outer_schemes"]:
        source = (
            pd.concat([train, development], ignore_index=True) if scheme == "forward" else train
        )
        source = source.loc[source.log_chla.notna()].reset_index(drop=True)
        source = add_crossfit_sss(source, sss_path, str(scheme))
        for split in outer_splits(source, str(scheme), n_folds=int(config["outer_folds"])):
            fit = source.iloc[split.fit_index].copy().reset_index(drop=True)
            held = source.iloc[split.held_index].copy().reset_index(drop=True)
            background = SeasonalTrendClimatology().fit(fit)
            for frame in (fit, held):
                frame["background"] = background.predict(frame)
                frame["residual"] = frame.truth - frame.background
            print(
                f"CHLA {scheme}/{split.fold} fit={len(fit):,} held={len(held):,}",
                flush=True,
            )
            for candidate, include_chla in (("core", False), ("core_plus_chla", True)):
                estimate = predict(
                    fit,
                    held,
                    include_chla=include_chla,
                    iterations=int(config["models"]["catboost_iterations"]),
                    seed=int(config["models"]["oof_seed"]),
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "outer_scheme": scheme,
                        "outer_fold": split.fold,
                        **metric_values(held.assign(prediction=estimate), "prediction"),
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "candidate": candidate,
                            "outer_scheme": scheme,
                            "outer_fold": split.fold,
                            "record_id": held.record_id,
                            "truth": held.truth,
                            "background": held.background,
                            "prediction": estimate,
                        }
                    )
                )
    pd.DataFrame(rows).to_csv(args.output / "chla_ablation_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output / "chla_ablation_predictions.parquet", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
