"""L1/L2 tests for the Issue #54 predictor-ablation contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recad.evaluate.predictor_ablation import (
    balanced_sample_weight,
    frozen_splits,
    paired_group_bootstrap,
    qualifies_addition,
    regression_metrics,
)


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_key": [f"g{i // 2}" for i in range(20)],
            "lme_id": np.repeat(np.arange(5), 4),
            "cruise_fold": np.tile(np.arange(5), 4),
            "spatial_fold": np.repeat(np.arange(5), 4),
            "forward_role": ["forward_train"] * 12
            + ["forward_development"] * 4
            + ["late_development"] * 4,
        }
    )


def test_frozen_splits_have_no_group_leakage() -> None:
    frame = synthetic_frame()
    for scheme in ("cruise", "spatial", "whole_lme", "forward"):
        for split in frozen_splits(frame, scheme):
            fit = set(frame.iloc[split.fit_index].group_key)
            held = set(frame.iloc[split.held_index].group_key)
            assert fit.isdisjoint(held)


def test_balanced_weights_equalize_regions() -> None:
    frame = synthetic_frame()
    weights = balanced_sample_weight(frame)
    totals = pd.Series(weights).groupby(frame.lme_id).sum()
    assert totals.max() == pytest.approx(totals.min())


def test_metrics_and_group_bootstrap_detect_improvement() -> None:
    truth = np.arange(20, dtype=float)
    reference = truth + 2
    candidate = truth + 1
    groups = np.asarray([f"g{i // 2}" for i in range(20)])
    metrics = regression_metrics(truth, candidate, groups)
    assert metrics["rmse"] == pytest.approx(1)
    frame = pd.DataFrame(
        {"group_key": groups, "truth": truth, "candidate": candidate, "reference": reference}
    )
    result = paired_group_bootstrap(frame, "candidate", "reference", replicates=200)
    assert result["ci_high_delta_mse"] < 0


@pytest.mark.l2
def test_qualification_gate() -> None:
    effect = pd.DataFrame(
        {
            "scheme": ["cruise", "spatial", "whole_lme", "forward_2019_2021", "forward_2022_2025"],
            "relative_rmse_improvement": [0.03, 0.02, 0.01, 0.01, 0.005],
            "common_relative_rmse_improvement": [0.03, 0.02, 0.01, 0.01, 0.005],
            "bootstrap_ci_high_delta_mse": [-0.1, -0.1, np.nan, np.nan, np.nan],
        }
    )
    passed, reasons = qualifies_addition(effect)
    assert passed
    assert reasons == []
