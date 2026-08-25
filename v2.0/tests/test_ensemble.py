"""Deep-ensemble moment aggregation and field-scatter tests."""

from __future__ import annotations

import numpy as np
import torch

from recad.data.tensorize import CoastalPatchDataset
from recad.model.ensemble import DeepEnsemble, EnsembleMoments, pick_device
from recad.model.factory import build_model


def _ensemble(prepared_small, model_cfg, n=2):
    shapes = CoastalPatchDataset(
        prepared_small,
        masks=None,
        split=None,
        model_cfg=model_cfg,
        require_target=False,
    ).shapes()
    members = [build_model(shapes, model_cfg, seed=i) for i in range(n)]
    return DeepEnsemble(members, torch.device("cpu"))


def test_predict_field_shapes_and_scatter(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small, n=2)
    moments = ens.predict_field(prepared_small, model_cfg_small, batch_size=2)
    assert isinstance(moments, EnsembleMoments)
    assert moments.mean.shape == prepared_small.shape4d
    assert moments.epistemic_std.shape == prepared_small.shape4d
    assert moments.aleatoric_std.shape == prepared_small.shape4d
    assert moments.model_std.shape == prepared_small.shape4d
    assert moments.n_members == 2
    # predictions only where coastal: non-coastal cells stay NaN
    coast = prepared_small.coastal_mask
    non_coast = ~np.broadcast_to(coast, prepared_small.shape4d)
    assert np.isnan(moments.mean[non_coast]).all()
    # coastal cells with valid predictors should have finite predictions
    coast_sel = np.broadcast_to(coast, prepared_small.shape4d) & ~np.isnan(
        prepared_small.target_values()
    )
    assert np.isfinite(moments.mean[coast_sel]).any()


def test_moment_math(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small, n=3)
    moments = ens.predict_field(prepared_small, model_cfg_small, batch_size=1)
    member_means = np.stack(moments.member_means, axis=0)
    assert member_means.shape == (3, *prepared_small.shape4d)
    assert moments.mean.shape == prepared_small.shape4d  # mean over members
    # mean over members
    sel = ~np.isnan(member_means).all(axis=0)
    assert np.allclose(moments.mean[sel], np.nanmean(member_means, axis=0)[sel], equal_nan=True)
    # epistemic std = std over member means
    assert np.allclose(
        moments.epistemic_std[sel],
        np.nanstd(member_means, axis=0)[sel],
        atol=1e-6,
        equal_nan=True,
    )
    # model_std = sqrt(epistemic^2 + aleatoric^2)
    assert np.allclose(
        moments.model_std[sel],
        np.sqrt(moments.epistemic_std[sel] ** 2 + moments.aleatoric_std[sel] ** 2),
        atol=1e-6,
    )


def test_predict_field_deterministic(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small, n=2)
    a = ens.predict_field(prepared_small, model_cfg_small)
    b = ens.predict_field(prepared_small, model_cfg_small)
    assert np.array_equal(np.nan_to_num(a.mean), np.nan_to_num(b.mean))


def test_pick_device():
    dev = pick_device("cpu")
    assert dev.type == "cpu"
    dev2 = pick_device("auto")
    assert dev2.type in {"cpu", "cuda", "mps"}
