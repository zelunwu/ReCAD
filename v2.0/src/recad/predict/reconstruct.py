"""Full-domain reconstruction with the trained ensemble."""

from __future__ import annotations

import numpy as np

from recad.config import Config
from recad.data.features import PreparedData
from recad.model.ensemble import DeepEnsemble
from recad.model.factory import build_model
from recad.utils.chem import fco2_to_pco2
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


def load_ensemble(cfg: Config, shapes, device) -> DeepEnsemble:
    """Rebuild the DeepEnsemble from the checkpoints on disk."""
    import torch

    members = []
    for i in range(cfg.ensemble.n_members):
        path = f"{cfg.train.checkpoint_dir}/member_{i}.pt"
        state = torch.load(path, map_location=device)
        model = build_model(shapes, cfg.model, seed=None)
        model.load_state_dict(state["model_state"])
        members.append(model)
    return DeepEnsemble(members, device)


def build_ensemble_prediction(
    cfg: Config,
    prepared: PreparedData,
    ensemble: DeepEnsemble,
    device,
) -> dict[str, np.ndarray]:
    """Run the ensemble over the full coastal domain.

    Returns a dict of 4-D fields: ``fco2`` (mean), ``fco2_err``
    (model_std), plus pCO2 variants converted at OISST (using the prepared
    SST field), and the separated aleatoric/epistemic std fields.
    """
    moments = ensemble.predict_field(
        prepared,
        cfg.model,
        tile_cells=None,  # full reconstruction: every coastal cell
        batch_size=cfg.train.batch_windows,
        use_amp=cfg.train.use_amp,
    )
    sst = prepared.require("sst")
    fco2 = moments.mean
    pco2 = fco2_to_pco2(fco2, sst)
    pco2_err = fco2_to_pco2(moments.model_std, sst)

    _LOG.info(
        "reconstruction: mean fCO2 %.1f +/- %.1f µatm over valid coastal cells",
        np.nanmean(fco2),
        np.nanmean(moments.model_std),
    )
    return {
        "fco2": fco2,
        "fco2_err_model": moments.model_std,
        "fco2_err_aleatoric": moments.aleatoric_std,
        "fco2_err_epistemic": moments.epistemic_std,
        "pco2": pco2,
        "pco2_err_model": pco2_err,
        "member_means": moments.member_means,
    }
