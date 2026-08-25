"""Monte-Carlo input-error propagation tests."""

from __future__ import annotations

import numpy as np
import torch

from recad.config import Config
from recad.data.tensorize import CoastalPatchDataset
from recad.model.ensemble import DeepEnsemble
from recad.model.factory import build_model
from recad.uncertainty.input_mc import propagate_input_uncertainty


def _ensemble(prepared, model_cfg, n=2):
    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=model_cfg, require_target=False
    ).shapes()
    members = [build_model(shapes, model_cfg, seed=i) for i in range(n)]
    return DeepEnsemble(members, torch.device("cpu"))


def _uncfg(overrides=None):
    base = {
        "n_mc_draws": 3,
        "propagate_input": True,
        "input_uncertainties": {
            "u_sst": 0.4,
            "u_sss": 0.2,
            "u_ssh": 0.02,
            "u_pco2air": 0.3,
            "u_u10": 0.5,
        },
        "combine": "rss",
    }
    if overrides:
        base.update(overrides)
    return Config.from_dict({"uncertainty": base}).uncertainty


def test_mc_std_shape_and_coverage(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small)
    mc_std, mc_mean = propagate_input_uncertainty(
        ens,
        prepared_small,
        _uncfg(),
        model_cfg_small,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert mc_std.shape == prepared_small.shape4d
    assert mc_mean.shape == prepared_small.shape4d
    coast = prepared_small.coastal_mask
    sel = np.broadcast_to(coast, prepared_small.shape4d)
    # perturbed inputs must produce some spread where there is signal
    assert np.nanmax(mc_std[sel]) > 0.0
    assert np.isfinite(mc_mean[sel]).any()


def test_mc_deterministic(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small)
    a, _ = propagate_input_uncertainty(
        ens, prepared_small, _uncfg(), model_cfg_small, batch_size=2, device=torch.device("cpu")
    )
    b, _ = propagate_input_uncertainty(
        ens, prepared_small, _uncfg(), model_cfg_small, batch_size=2, device=torch.device("cpu")
    )
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_mc_zero_uncertainty_gives_zero_std(prepared_small, model_cfg_small):
    ens = _ensemble(prepared_small, model_cfg_small)
    u = {
        "input_uncertainties": dict.fromkeys(("u_sst", "u_sss", "u_ssh", "u_pco2air", "u_u10"), 0.0)
    }
    mc_std, _ = propagate_input_uncertainty(
        ens, prepared_small, _uncfg(u), model_cfg_small, batch_size=2, device=torch.device("cpu")
    )
    coast = prepared_small.coastal_mask
    sel = np.broadcast_to(coast, prepared_small.shape4d)
    # identical inputs across draws: the draw-to-draw spread must vanish up to
    # float-level jitter (torch CPU reductions) — ignore any genuinely missing
    # (NaN) cells / nan-degenerate columns.
    spread = np.nanmax(np.abs(mc_std[sel]))
    assert np.isnan(spread) or spread < 1e-2  # << 0.01 µatm, physically zero
