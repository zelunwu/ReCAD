"""Holdout validation against v1.1 protocol (year-level robustness)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recad.config import Config
from recad.data.features import PreparedData
from recad.data.split import SplitMasks
from recad.model.ensemble import DeepEnsemble
from recad.train.metrics import metrics_summary, per_year_metrics
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


def run_holdout_validation(
    cfg: Config,
    prepared: PreparedData,
    masks: SplitMasks,
    ensemble: DeepEnsemble,
    device,
) -> dict[str, pd.DataFrame]:
    """Validate the ensemble on the held-out (test) years and the val split.

    Mirrors the v1.1 robustness protocol (``RFR_models_test.mlx``): predict
    the test years (default 2004-2005) and report the year-level R2/RMSE table
    against SOCAT. Also reports the val-split overall metrics as a sanity
    check of the training log.
    """
    moments = ensemble.predict_field(
        prepared,
        cfg.model,
        tile_cells=None,
        batch_size=cfg.train.batch_windows,
        use_amp=cfg.train.use_amp,
    )
    y_true = prepared.target_values()
    y_pred = moments.mean

    test_mask = masks.test
    val_mask = masks.val

    test_table = per_year_metrics(
        np.where(test_mask, y_true, np.nan),
        np.where(test_mask, y_pred, np.nan),
        prepared.years,
    )
    test_overall = metrics_summary(y_true[test_mask], y_pred[test_mask])
    val_overall = metrics_summary(y_true[val_mask], y_pred[val_mask])

    _LOG.info("test-year overall: %s", {k: round(v, 3) for k, v in test_overall.items()})
    return {
        "test_years": test_table,
        "test_overall": pd.DataFrame([test_overall]),
        "val_overall": pd.DataFrame([val_overall]),
    }
