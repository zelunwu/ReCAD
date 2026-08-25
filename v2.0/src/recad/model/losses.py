"""Loss functions with explicit masking.

Every loss returns ``(loss, n_valid)`` where ``loss`` is masked to the valid
(masked=True) entries only, so training on partially observed months works
naturally and the same code serves train/val/test.
"""

from __future__ import annotations

import torch


def _masked_entries(
    target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        mask = ~torch.isnan(target)
    target = torch.where(mask, target, torch.zeros_like(target))
    return target, mask


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative log-likelihood of a Gaussian with diagonal variance.

    loss = 0.5 * (log(2 pi) + logvar + (y - mu)^2 / exp(logvar)),
    summed over masked entries.
    """
    target, mask = _masked_entries(target, mask)
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.zeros((), dtype=mean.dtype, device=mean.device), n_valid
    var = torch.exp(logvar)
    nll = 0.5 * (
        torch.log(2 * torch.tensor(torch.pi, device=mean.device))
        + logvar
        + (target - mean) ** 2 / var
    )
    nll = nll[mask]
    if reduction == "mean":
        return nll.mean(), n_valid
    return nll.sum(), n_valid


def mse_loss_masked(
    mean: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    target, mask = _masked_entries(target, mask)
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.zeros((), dtype=mean.dtype, device=mean.device), n_valid
    se = (target - mean) ** 2
    se = se[mask]
    if reduction == "mean":
        return se.mean(), n_valid
    return se.sum(), n_valid


def quantile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinball loss for a set of quantile predictions (optional head)."""
    target, mask = _masked_entries(target, mask)
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.zeros((), dtype=pred.dtype, device=pred.device), n_valid
    target = target.unsqueeze(-1).expand_as(pred)
    mask_e = mask.unsqueeze(-1).expand_as(pred)
    losses = torch.stack(
        [
            torch.where(
                target >= q_p,
                q * (target - q_p),
                (q - 1.0) * (target - q_p),
            )
            for q_p, q in zip(pred.unbind(-1), quantiles, strict=False)
        ],
        dim=-1,
    )
    losses = losses[mask_e]
    if reduction == "mean":
        return losses.mean(), n_valid
    return losses.sum(), n_valid


def build_loss(head_name: str):
    """Return the loss callable for a given head name."""
    if head_name == "gaussian":
        return gaussian_nll_loss
    if head_name == "mean":
        return mse_loss_masked
    if head_name == "quantile":
        quantiles = (0.1, 0.5, 0.9)

        def _q_loss(mean, logvar, target, mask=None, reduction="mean"):
            return quantile_loss(mean, target, quantiles, mask, reduction)

        return _q_loss
    raise ValueError(f"unknown head: {head_name}")
