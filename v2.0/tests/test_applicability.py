from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recad.evaluate.applicability import (
    CoastalGraphIndex,
    ReasonBit,
    SupportIndex,
    add_label_free_risk_scores,
    assign_spatial_blocks,
    nested_grouped_ridge,
    outer_splits,
    reliability_schema,
    risk_coverage_curve,
    risk_curve_auc,
    validate_reliability_frame,
)


def synthetic_frame(rows: int = 60) -> pd.DataFrame:
    index = np.arange(rows)
    group = np.repeat([f"cruise-{i}" for i in range(10)], rows // 10)
    longitude = -75.0 + 0.1 * index
    month = index % 12 + 1
    sst = 10.0 + 0.2 * index
    return pd.DataFrame(
        {
            "group_key": group,
            "latitude": 30.0 + 0.05 * index,
            "longitude": longitude,
            "year": 2000 + index % 10,
            "month": month,
            "lme_id": index % 3,
            "basin_id": 2,
            "regime_id": index % 4,
            "cv_fold": np.array([i % 5 for i in range(10) for _ in range(rows // 10)]),
            "forward_split": np.where(index < rows * 0.7, "train", "development"),
            "sst": sst,
            "sss": 32.0 + 0.01 * index,
            "adt": np.sin(index),
            "wspd": 5.0 + index % 5,
            "pco2air": 380.0 + 0.5 * index,
            "latitude_sin": np.sin(np.deg2rad(30.0 + 0.05 * index)),
            "longitude_sin": np.sin(np.deg2rad(longitude)),
            "longitude_cos": np.cos(np.deg2rad(longitude)),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "truth": 2.0 * sst + 0.1 * index,
        }
    )


@pytest.mark.l1
def test_outer_splits_keep_cruises_regions_and_blocks_whole() -> None:
    frame = synthetic_frame()
    for scheme in ("cruise", "spatial_block", "whole_lme"):
        for split in outer_splits(frame, scheme):
            fit = frame.iloc[split.fit_index]
            held = frame.iloc[split.held_index]
            if scheme == "cruise":
                assert set(fit.group_key).isdisjoint(held.group_key)
            elif scheme == "whole_lme":
                assert set(fit.lme_id).isdisjoint(held.lme_id)
            else:
                fit_block = assign_spatial_blocks(fit)
                held_block = assign_spatial_blocks(held)
                assert set(fit_block[fit_block.eq(split.fold)].index).isdisjoint(held_block.index)


@pytest.mark.l1
def test_nested_prediction_does_not_read_outer_truth() -> None:
    frame = synthetic_frame()
    fit, held = frame.iloc[:42].copy(), frame.iloc[42:].copy()
    columns = [
        "sst",
        "sss",
        "latitude_sin",
        "longitude_sin",
        "longitude_cos",
        "month_sin",
        "month_cos",
    ]
    first = nested_grouped_ridge(
        fit,
        held.drop(columns="truth"),
        numeric=columns,
        categorical=["lme_id", "regime_id"],
        alphas=[0.1, 10.0],
    )
    changed = held.copy()
    changed["truth"] = changed.truth + 1e6
    second = nested_grouped_ridge(
        fit,
        changed.drop(columns="truth"),
        numeric=columns,
        categorical=["lme_id", "regime_id"],
        alphas=[0.1, 10.0],
    )
    assert first.selected_alpha == second.selected_alpha
    np.testing.assert_allclose(first.prediction, second.prediction)


@pytest.mark.l1
def test_residual_probe_requires_finite_background() -> None:
    frame = synthetic_frame()
    frame.loc[0, "sss"] = np.nan
    eligible = frame.loc[frame.sss.notna()]
    assert len(eligible) == len(frame) - 1
    assert np.isfinite(eligible.truth.to_numpy() - eligible.sss.to_numpy()).all()


@pytest.mark.l1
def test_coastal_graph_reports_disconnected_component() -> None:
    graph = sparse.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        )
    )
    index = CoastalGraphIndex(graph, np.array([0, 0, 10, 10]), np.array([0, 1, 0, 1]))
    distance = index.distance_to_sources([0])
    assert distance[0] == 0
    assert 100 < distance[1] < 120
    assert np.isinf(distance[2:]).all()


@pytest.mark.l2
def test_support_index_is_training_fitted_and_complete() -> None:
    train = synthetic_frame(60)
    held = synthetic_frame(60).iloc[[0, 20, 40]].copy()
    held["sst"] = [10.0, 100.0, np.nan]
    index = SupportIndex(
        environmental_columns=["sst", "sss", "latitude_sin", "longitude_sin", "month_sin"],
        k_values=[8, 16, 32, 64],
        radii_km=[25, 50, 100, 200],
    ).fit(train)
    support = index.query(held)
    assert len(support) == 3
    assert support.loc[1, "environment_k16"] > support.loc[0, "environment_k16"]
    assert bool(support.loc[2, "missing_predictor"])
    assert {
        "unique_cruises",
        "effective_groups",
        "sampled_years",
        "sampled_months",
        "support_entropy",
    } <= set(support)
    for radius in (25, 50, 100, 200):
        assert f"count_within_{radius}km" in support


@pytest.mark.l1
def test_risk_curve_rewards_correct_error_ranking() -> None:
    truth = np.zeros(100)
    prediction = np.linspace(0, 10, 100)
    good = risk_coverage_curve(truth, prediction, prediction)
    bad = risk_coverage_curve(truth, prediction, -prediction)
    assert risk_curve_auc(good) < risk_curve_auc(bad)


@pytest.mark.l1
def test_reliability_schema_keeps_independent_support_sealed() -> None:
    schema = reliability_schema()
    assert schema["reason_bits"]["INDEPENDENT_SUPPORT_SEALED"] == int(
        ReasonBit.INDEPENDENT_SUPPORT_SEALED
    )
    frame = pd.DataFrame({column: [np.nan] for column in schema["columns"]})
    frame["applicability_flag"] = "A"
    frame["evidence_stage"] = "development"
    validate_reliability_frame(frame)
    frame["independent_support_km"] = 1.0
    with pytest.raises(ValueError, match="independent-validation"):
        validate_reliability_frame(frame)


@pytest.mark.l3
def test_label_free_scores_cover_every_registered_method() -> None:
    base = pd.DataFrame(
        {
            "geographic_km": [1.0, 100.0],
            "lme_month_count": [100, 0],
            "basin_month_count": [100, 1],
            "regime_month_count": [100, 0],
            "water_connected_km": [2.0, np.inf],
            **{f"environment_k{k}": [0.1, 5.0] for k in (8, 16, 32, 64)},
        }
    )
    scored = add_label_free_risk_scores(base)
    for method in (
        "geographic",
        "region",
        "graph",
        "environment_k8",
        "environment_k16",
        "environment_k32",
        "environment_k64",
        "hybrid",
    ):
        assert f"risk_{method}" in scored
    assert scored.loc[1, "risk_hybrid"] > scored.loc[0, "risk_hybrid"]
