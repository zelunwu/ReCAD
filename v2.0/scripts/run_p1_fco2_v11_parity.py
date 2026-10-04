"""Run the historical ReCAD v1.1 comparability stage for Issue #32."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from catboost import CatBoostRegressor

from recad.evaluate.v11_parity import (
    KEY_TO_DISPLAY,
    parity_gate,
    published_reference,
    regression_metrics,
)
from recad.viz.regions import region_masks, v11_region_keys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_fco2_v11_parity_v2.2"
NUMERIC = [
    "sss",
    "sst",
    "adt",
    "wspd",
    "pco2air",
    "xco2air",
    "year",
    "latitude",
    "lon_sin",
    "lon_cos",
    "month_sin",
    "month_cos",
]
CATEGORICAL = ["legacy_region"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_frame(prepared_path: Path, masks_path: Path, period_end: int) -> pd.DataFrame:
    """Extract protocol-aligned historical rows from the repeatedly viewed cache."""

    with xr.open_dataset(prepared_path) as prepared, xr.open_dataset(masks_path) as masks:
        years = prepared.year.to_numpy().astype(int)
        months = prepared.month.to_numpy().astype(int)
        latitude = prepared.lat.to_numpy(float)
        longitude = prepared.lon.to_numpy(float)
        shapes = (len(years), len(months), len(latitude), len(longitude))
        active = np.zeros(shapes, dtype=bool)
        split_code = np.full(shapes, -1, dtype=np.int8)
        for code, name in enumerate(("train", "val", "test")):
            values = masks[name].to_numpy().astype(bool)
            active |= values
            split_code[values] = code
        active &= years[:, None, None, None] <= period_end
        spatial_masks = region_masks(longitude, latitude)
        active &= spatial_masks["Atlantic"][None, None, :, :]
        flat = np.flatnonzero(active)
        year_index, month_index, lat_index, lon_index = np.unravel_index(flat, shapes)
        result = pd.DataFrame(
            {
                "year": years[year_index],
                "month": months[month_index],
                "latitude": latitude[lat_index],
                "longitude": longitude[lon_index],
                "split_code": split_code.ravel()[flat],
            }
        )
        split_names = np.array(["Train", "Test", "Validation"], dtype=object)
        result["Type"] = split_names[result.split_code]
        keys = v11_region_keys()[:-1]
        region_code = np.full(len(result), "unassigned", dtype=object)
        for key in keys:
            mask = spatial_masks[key][lat_index, lon_index]
            region_code[mask] = key
        if np.any(region_code == "unassigned"):
            raise RuntimeError("v1.1 Atlantic rows do not map to exactly one subregion")
        result["region_key"] = region_code
        radians = np.deg2rad(result.longitude.to_numpy(float))
        phase = 2.0 * np.pi * (result.month.to_numpy(float) - 1.0) / 12.0
        result["lon_sin"], result["lon_cos"] = np.sin(radians), np.cos(radians)
        result["month_sin"], result["month_cos"] = np.sin(phase), np.cos(phase)
        result["legacy_region"] = result.region_key
        for variable in ("fco2", "sss", "sst", "adt", "wspd", "pco2air", "xco2air"):
            values = prepared[variable].to_numpy().ravel()[flat]
            result["truth" if variable == "fco2" else variable] = values
    required = ["truth", *NUMERIC[:-6], "year", "latitude"]
    return result.loc[np.isfinite(result[required]).all(axis=1)].reset_index(drop=True)


class RegionalSeasonalTrend:
    """Training-only legacy-region/month background with a shared trend."""

    def fit(self, frame: pd.DataFrame) -> RegionalSeasonalTrend:
        self.center_year = float(frame.year.median())
        time = frame.year.to_numpy(float) - self.center_year
        truth = frame.truth.to_numpy(float)
        self.slope = float(np.cov(time, truth, ddof=0)[0, 1] / max(np.var(time), 1e-12))
        work = frame.assign(detrended=truth - self.slope * time)
        self.global_mean = float(work.detrended.mean())
        self.month = work.groupby("month").detrended.mean().to_dict()
        self.region_month = work.groupby(["region_key", "month"]).detrended.mean().to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        base = np.asarray(
            [
                self.region_month.get(
                    (row.region_key, row.month),
                    self.month.get(row.month, self.global_mean),
                )
                for row in frame.itertuples()
            ]
        )
        return base + self.slope * (frame.year.to_numpy(float) - self.center_year)


def balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    regions = frame.region_key.nunique()
    months = frame.groupby("region_key").month.transform("nunique")
    records = frame.groupby(["region_key", "month"])["month"].transform("size")
    values = (1.0 / (regions * months * records)).to_numpy(float)
    return values / values.sum()


def fit_catboost(frame: pd.DataFrame, config: dict[str, object], seed: int) -> CatBoostRegressor:
    columns = NUMERIC + CATEGORICAL
    values = frame[columns].copy()
    medians = {}
    for column in NUMERIC:
        medians[column] = float(values[column].median())
        values[column] = values[column].fillna(medians[column])
    model = CatBoostRegressor(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
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
        values,
        frame.residual,
        cat_features=[columns.index(column) for column in CATEGORICAL],
        sample_weight=balanced_weights(frame) * len(frame),
        verbose=False,
    )
    model._recad_medians = medians
    return model


def predict_catboost(model: CatBoostRegressor, frame: pd.DataFrame) -> np.ndarray:
    columns = NUMERIC + CATEGORICAL
    values = frame[columns].copy()
    for column in NUMERIC:
        values[column] = values[column].fillna(model._recad_medians[column])
    return model.predict(values)


def candidate_predictions(
    frame: pd.DataFrame,
    config: dict[str, object],
    output: Path,
) -> pd.DataFrame:
    train = frame.loc[frame.Type.eq("Train")].copy().reset_index(drop=True)
    background = RegionalSeasonalTrend().fit(train)
    for part in (train, frame):
        part["background"] = background.predict(part)
        part["residual"] = part.truth - part.background
    rows = []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("global_balanced", "regional_experts"):
        seed_predictions = []
        for seed in config["seeds"]:
            print(f"HISTORICAL {candidate} seed={seed}", flush=True)
            if candidate == "global_balanced":
                model = fit_catboost(train, config, int(seed))
                residual = predict_catboost(model, frame)
                model.save_model(checkpoint_dir / f"{candidate}_seed{seed}.cbm")
            else:
                residual = np.full(len(frame), np.nan)
                for region in sorted(train.region_key.unique()):
                    fit = train.loc[train.region_key.eq(region)].reset_index(drop=True)
                    held = frame.loc[frame.region_key.eq(region)]
                    model = fit_catboost(fit, config, int(seed))
                    residual[held.index] = predict_catboost(model, held)
                    model.save_model(checkpoint_dir / f"{candidate}_{region}_seed{seed}.cbm")
            seed_predictions.append(frame.background.to_numpy(float) + residual)
        ensemble = np.vstack(seed_predictions)
        part = frame[
            ["year", "month", "latitude", "longitude", "region_key", "Type", "truth", "background"]
        ].copy()
        part.insert(0, "candidate", candidate)
        part["prediction"] = ensemble.mean(axis=0)
        part["epistemic_sd"] = ensemble.std(axis=0, ddof=1)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, candidate_part in predictions.groupby("candidate"):
        for split, split_part in candidate_part.groupby("Type"):
            for key in [*v11_region_keys()[:-1], "Atlantic"]:
                part = (
                    split_part
                    if key == "Atlantic"
                    else split_part.loc[split_part.region_key.eq(key)]
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "Region": KEY_TO_DISPLAY[key],
                        "Type": split,
                        **regression_metrics(part.truth, part.prediction),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/p1_fco2_v11_parity_v2.2.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / f"outputs/experiments/{EXPERIMENT_ID}"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["locked_fco2_access"] != "forbidden":
        raise ValueError("Issue #32 must keep the current locked fCO2 labels sealed")
    args.output.mkdir(parents=True, exist_ok=True)
    reference_path = (ROOT / config["published_reference"]).resolve()
    reference = published_reference(str(reference_path))
    reference.to_csv(args.output / "published_v11_reference.csv", index=False)
    frame = legacy_frame(
        Path(config["historical_prepared"]),
        Path(config["historical_masks"]),
        int(config["historical_period_end"]),
    )
    predictions = candidate_predictions(frame, config["model"], args.output)
    metrics = metric_table(predictions)
    predictions.to_parquet(args.output / "historical_predictions.parquet", index=False)
    metrics.to_csv(args.output / "historical_metrics.csv", index=False)
    decisions = {}
    for candidate, part in metrics.groupby("candidate"):
        decisions[candidate] = parity_gate(part, config["historical_gate"])
    decision = {
        "candidates": decisions,
        "historical_evidence_only": True,
        "locked_fco2_opened": False,
        "external_independent_opened": False,
    }
    (args.output / "historical_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    protocol = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": sha256(args.config),
        "published_reference_sha256": sha256(reference_path),
        "prepared_sha256": sha256(Path(config["historical_prepared"])),
        "masks_sha256": sha256(Path(config["historical_masks"])),
        "rows": len(frame),
        "split_counts": frame.Type.value_counts().to_dict(),
        "locked_fco2_opened": False,
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
