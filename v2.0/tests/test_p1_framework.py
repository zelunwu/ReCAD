from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from recad.evaluate.p1_framework import (
    AccessViolation,
    BalancedBatchSampler,
    FrozenManifest,
    P1DataGateway,
    Purpose,
    aggregate_seeds,
    evaluate_predictions,
    leave_group_out,
    select_checkpoint,
    standardize_predictions,
)


def _gateway(tmp_path: Path) -> P1DataGateway:
    data = tmp_path / "data"
    data.mkdir()
    split_rows = []
    socat_rows = []
    carbon_rows = []
    specs = [
        ("train-a", "train", 0, 2017),
        ("train-b", "train", 1, 2025),
        ("dev-a", "development", 2, 2020),
        ("lock-a", "locked_test", 3, 2024),
        ("external-a", "external_independent", 4, 2024),
    ]
    for i, (group, split, fold, year) in enumerate(specs):
        split_rows.append({"group_key": group, "split": split, "cv_fold": fold,
                           "dataset": "socat", "max_year": year,
                           "forward_split": "train" if year <= 2018 else "development" if year <= 2021 else "locked_test"})
        base = {"group_key": group, "yr": year, "mon": i % 12 + 1, "fco2": 350.0 + i,
                "f_n": 1, "salinity": 33.0 + i, "sal_n": 1, "latitude": i,
                "longitude": i, "lme_id": i % 2, "basin_id": 1, "regime_id": i % 3,
                "nearest_carbon_km": i * 10.0, "split": split, "cv_fold": fold,
                "record_status": "test"}
        socat_rows.append(base)
        carbon_rows.append({**{k: v for k, v in base.items() if k not in {"yr", "mon", "f_n", "sal_n"}},
                            "year": min(year, 2024), "month": i % 12 + 1, "ta": 2200.0 + i,
                            "dic": 2000.0 + i, "label_ta_ok": True, "label_dic_ok": True})
    pd.DataFrame(split_rows).to_parquet(data / "split_manifest_v2.2.parquet", index=False)
    pd.DataFrame(socat_rows).to_parquet(data / "na_socat_cache_v2.2.parquet", index=False)
    pd.DataFrame(carbon_rows).to_parquet(data / "na_carbon_cache_v2.2.parquet", index=False)
    ds = xr.Dataset({"strict_all_inputs_ready": (("year", "month", "node"), np.ones((2, 1, 2), dtype=bool))},
                    coords={"year": [2025, 2026], "month": [1], "node": [0, 1]})
    ds.to_netcdf(data / "inference_availability_v2.2.nc")
    external = tmp_path / "configs" / "frozen" / "external.json"
    external.parent.mkdir(parents=True)
    external.write_text("{}", encoding="utf-8")
    import hashlib
    ext_hash = hashlib.sha256(external.read_bytes()).hexdigest()
    manifest = {"split_manifest": str(data / "split_manifest_v2.2.parquet"),
                "inputs_sha256": {}, "artifacts_sha256": {},
                "independent_validation": {"path": "configs/frozen/external.json", "sha256": ext_hash}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # The production resolver expects a manifest under configs/frozen. Keep the
    # fixture's independent file addressable relative to its synthetic root.
    loaded = FrozenManifest.load(manifest_path)
    object.__setattr__(loaded, "content", {**manifest, "independent_validation": {"path": "external.json", "sha256": ext_hash}})
    return P1DataGateway(loaded)


def test_training_selection_and_sealed_access(tmp_path):
    gateway = _gateway(tmp_path)
    train = gateway.load_labels("fco2", Purpose.TRAIN)
    dev = gateway.load_labels("sss", Purpose.SELECTION)
    assert set(train.group_key) == {"train-a", "train-b"}
    assert train.year.max() == 2025
    assert set(dev.group_key) == {"dev-a"}
    with pytest.raises(AccessViolation):
        gateway.load_labels("ta", Purpose.LOCKED_TEST)
    with pytest.raises(AccessViolation):
        gateway.load_external_labels("ta")
    with pytest.raises(AccessViolation):
        gateway.load_labels("sss", Purpose.PREDICTION)


def test_cv_is_cruise_grouped_and_leave_group_is_complete(tmp_path):
    gateway = _gateway(tmp_path)
    fit, held = gateway.cv_fold("fco2", 0)
    assert set(fit.group_key).isdisjoint(set(held.group_key))
    frame = gateway.load_labels("fco2", Purpose.TRAIN)
    for value, remain, excluded in leave_group_out(frame, "lme_id"):
        assert excluded.lme_id.eq(value).all()
        assert not remain.lme_id.eq(value).any()


def test_forward_chain_is_grouped_and_predicate_filtered(tmp_path):
    gateway = _gateway(tmp_path)
    train = gateway.load_labels("fco2", Purpose.TRAIN, split_scheme="forward")
    development = gateway.load_labels("fco2", Purpose.SELECTION, split_scheme="forward")
    assert set(train.group_key) == {"train-a"}
    assert set(development.group_key) == {"dev-a"}
    assert train.evaluation_split.eq("train").all()
    assert development.evaluation_split.eq("development").all()


def test_2026_is_prediction_only_and_provisional(tmp_path):
    gateway = _gateway(tmp_path)
    available = gateway.prediction_availability()
    assert available.loc[available.year.eq(2026), "prediction_status"].eq("provisional").all()
    row = dict.fromkeys((
        "record_id", "target", "prediction", "uncertainty", "lower", "upper",
        "target_provenance", "model_provenance", "group_key", "year", "month",
        "latitude", "longitude", "split", "cv_fold", "lme_id", "regime_id",
        "nearest_carbon_km", "ood", "prediction_status", "seed"), np.nan)
    row.update(record_id="x", target="sss", prediction=34.0, year=2026, month=1,
               prediction_status="core", truth=np.nan)
    with pytest.raises(AccessViolation):
        standardize_predictions(pd.DataFrame([row]))
    row["prediction_status"] = "provisional"
    row["truth"] = 34.0
    with pytest.raises(AccessViolation):
        standardize_predictions(pd.DataFrame([row]))


def test_metrics_use_true_r2_and_all_required_aggregations():
    frame = pd.DataFrame({"truth": [1.0, 2.0, 3.0, 4.0], "prediction": [1.0, 2.0, 2.0, 5.0],
                          "group_key": ["a", "a", "b", "b"], "lme_id": [1, 1, 2, 2],
                          "regime_id": [0, 1, 0, 1], "nearest_carbon_km": [0, 20, 60, 600]})
    result = evaluate_predictions(frame)
    kinds = set(result.aggregation)
    assert {"pooled", "cruise_equal", "lme_macro", "regime_macro", "worst_lme", "support_distance"} <= kinds
    pooled = result.loc[result.aggregation.eq("pooled")].iloc[0]
    assert pooled.rmse == pytest.approx(np.sqrt(0.5))
    assert pooled.r2 == pytest.approx(0.6)


def test_sampler_checkpoint_and_three_seed_contract():
    frame = pd.DataFrame({"lme_id": [1, 1, 2, 2], "group_key": ["a", "b", "c", "d"],
                          "month": [1, 2, 1, 2]})
    sampler = BalancedBatchSampler(frame, seed=100)
    assert np.array_equal(sampler.sample(20, step=3), sampler.sample(20, step=3))
    checkpoints = pd.DataFrame({"checkpoint": ["a", "b"], "lme_macro_rmse": [2.0, 1.0],
                                "eligible": [True, True]})
    assert select_checkpoint(checkpoints).checkpoint == "b"
    records = pd.DataFrame({"model": ["x"] * 3, "seed": [100, 101, 102], "rmse": [1.0, 2.0, 3.0]})
    result = aggregate_seeds(records, group_columns=["model"], metrics=["rmse"])
    assert result.rmse_mean.iloc[0] == 2.0
