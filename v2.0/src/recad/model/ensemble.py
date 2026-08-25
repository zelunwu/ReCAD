"""Deep ensemble for calibrated predictive uncertainty.

Deep ensembles (Lakshminarayanan, Pritzel & Blundell, 2017, NeurIPS) provide a
simple, well-validated route to calibrated predictive uncertainty for deep
regression models, which is exactly what a geoscience reconstruction needs:

  * aleatoric std  = sqrt(mean over members of the Gaussian-head variance),
      capturing irreducible observation/scatter noise;
  * epistemic std  = std over member means,
      capturing model/parametric uncertainty (ensemble disagreement);
  * model std      = sqrt(aleatoric^2 + epistemic^2).

Members differ by random seed and by a year-level bootstrap of the training
samples (see recad/train/trainer.py), matching the v1.1 philosophy of
repeating the pipeline over perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recad.config import ModelConfig
from recad.data.features import PreparedData
from recad.data.tensorize import CoastalPatchDataset, collate_items
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


def pick_device(device: str = "auto") -> torch.device:
    """Resolve the 'auto' device string to cuda/mps/cpu."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EnsembleMoments:
    """Combined predictive moments over the whole field."""

    mean: np.ndarray  # [n_year, 12, n_lat, n_lon] float32
    aleatoric_std: np.ndarray
    epistemic_std: np.ndarray
    model_std: np.ndarray  # sqrt(aleatoric^2 + epistemic^2)
    member_means: list[np.ndarray] = field(default_factory=list)

    @property
    def n_members(self) -> int:
        return len(self.member_means)


class DeepEnsemble:
    """Container of ``n_members`` ST-Transformers with joint inference."""

    def __init__(self, members: list[nn.Module], device: torch.device) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = members
        self.device = device
        for member in members:
            member.to(device)
            member.eval()

    def __len__(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_field(
        self,
        prepared: PreparedData,
        model_cfg: ModelConfig,
        *,
        tile_cells: int | None = None,
        batch_size: int = 1,
        use_amp: bool = True,
        progress: bool = True,
    ) -> EnsembleMoments:
        """Predict the target for every coastal cell and month.

        Runs each member in evaluation mode over the full domain (all years),
        returning per-cell predictive moments scattered back onto the
        ``[n_year, 12, n_lat, n_lon]`` field.
        """
        shape = prepared.shape4d
        member_means: list[np.ndarray] = []
        member_vars: list[np.ndarray] = []

        for member in self.members:
            mean_field = np.full(shape, np.nan, dtype=np.float32)
            var_field = np.full(shape, np.nan, dtype=np.float32)
            self._scatter_predictions(
                member, prepared, model_cfg, tile_cells, batch_size, use_amp, mean_field, var_field
            )
            member_means.append(mean_field)
            member_vars.append(var_field)

        mean = np.mean(member_means, axis=0)
        aleatoric_var = np.mean(member_vars, axis=0)
        epistemic_std = np.std(member_means, axis=0)
        aleatoric_std = np.sqrt(aleatoric_var)
        model_std = np.sqrt(aleatoric_var + epistemic_std**2)

        return EnsembleMoments(
            mean=mean.astype(np.float32),
            aleatoric_std=aleatoric_std.astype(np.float32),
            epistemic_std=epistemic_std.astype(np.float32),
            model_std=model_std.astype(np.float32),
            member_means=member_means,
        )

    # ------------------------------------------------------------------
    def _scatter_predictions(
        self,
        member: nn.Module,
        prepared: PreparedData,
        model_cfg: ModelConfig,
        tile_cells: int | None,
        batch_size: int,
        use_amp: bool,
        mean_field: np.ndarray,
        var_field: np.ndarray,
    ) -> None:
        """Run one member over the domain, scattering outputs into fields."""
        dataset = CoastalPatchDataset(
            prepared,
            masks=None,
            split=None,
            model_cfg=model_cfg,
            tile_cells=tile_cells,
            require_target=False,
        )
        cell_flat = dataset.cell_flat_indices
        t_mean, t_std = dataset.t_mean, dataset.t_std
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_items)
        amp_ctx = torch.autocast("cuda", enabled=use_amp and self.device.type == "cuda")
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with amp_ctx:
                out = member(batch)
                mean = out["mean"].float().cpu().numpy()  # [B, T, C] standardized
                logvar = out.get("logvar")
            # de-standardize back into physical units
            mean = mean * t_std + t_mean
            var = (
                np.exp(logvar.float().cpu().numpy()) * (t_std**2)
                if logvar is not None
                else np.nan * mean
            )
            year_ids = batch["year_id"].cpu().numpy()
            month_ids = batch["month_ids"].cpu().numpy()
            for b in range(mean.shape[0]):
                y = year_ids[b]
                for t, m in enumerate(month_ids[b]):
                    mean_field[y, m - 1, :, :].ravel()[cell_flat] = mean[b, t]
                    var_field[y, m - 1, :, :].ravel()[cell_flat] = var[b, t]
