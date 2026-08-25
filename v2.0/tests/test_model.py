"""Model tests: shapes, NaN-safety, attention masking, heads."""

from __future__ import annotations

import torch

from recad.model.factory import build_model
from recad.model.losses import gaussian_nll_loss, mse_loss_masked, quantile_loss
from recad.model.st_transformer import count_parameters


def _model(cfg):
    from recad.data.tensorize import TensorShapes

    shapes = TensorShapes(
        n_tokens=24, n_cells=50, n_token_features=6, n_cell_features=14, n_predictors=5
    )
    return build_model(shapes, cfg, seed=0)


def _dummy_batch(cfg, device, B=2, T=12, P=24, C=50):
    tok = torch.randn(B, T, P, 6, device=device).abs()  # keep positive
    valid = torch.ones(B, T, P, dtype=torch.bool, device=device)
    valid[..., 3:5] = False  # a few padded tokens
    coord = torch.rand(B, T, P, 2, device=device)
    month = torch.rand(B, T, P, 2, device=device)
    cell_feats = torch.randn(B, T, C, 14, device=device)
    cell_feats[..., ::3] = float("nan")  # NaN feature robustness
    cell_patch = torch.randint(0, P, (B, T, C), device=device)
    cell_patch[..., 0] = -1  # one out-of-patch cell
    return {
        "token_feats": tok,
        "token_valid": valid,
        "coord": coord,
        "month": month,
        "cell_feats": cell_feats,
        "cell_patch": cell_patch,
    }


def test_gaussian_forward_shapes(model_cfg_small):
    cfg = model_cfg_small
    m = _model(cfg).eval()
    batch = _dummy_batch(cfg, torch.device("cpu"))
    out = m(batch)
    assert "mean" in out and "logvar" in out
    assert out["mean"].shape == (2, 12, 50)
    assert out["logvar"].shape == (2, 12, 50)
    assert count_parameters(m) > 0


def test_head_variants():
    from recad.config import Config

    for head in ("mean", "quantile", "gaussian"):
        cfg = Config.from_dict({"model": {"head": head, "embed_dim": 16, "n_heads": 2}}).model
        m = _model(cfg)
        out = m(_dummy_batch(cfg, torch.device("cpu")))
        assert out["mean"].shape == (2, 12, 50)
        if head == "quantile":
            assert out["quantiles"].shape == (2, 12, 50, 3)
        if head == "gaussian":
            assert out["logvar"].shape == (2, 12, 50)


def test_attention_masking_isolates_padded_tokens(model_cfg_small):
    """Outputs must not depend on the features of masked (padded) tokens."""
    cfg = model_cfg_small
    m = _model(cfg).eval()
    torch.manual_seed(3)
    batch = _dummy_batch(cfg, torch.device("cpu"))  # tokens 3,4 already padded
    with torch.no_grad():
        base = m(batch)["mean"]
        # Corrupt the input features of a padded token: this must not
        # propagate anywhere because the token is excluded from attention.
        corrupted = {k: v.clone() for k, v in batch.items()}
        corrupted["token_feats"][..., 4, :] = 1e6
        out_corrupted = m(corrupted)["mean"]
    assert torch.allclose(base, out_corrupted, atol=1e-5)


def test_loss_gaussian_nll_hand_masked():
    mean = torch.zeros(3)
    logvar = torch.zeros(3)
    target = torch.tensor([1.0, 5.0, 3.0])
    mask = torch.tensor([True, False, True])
    loss, n = gaussian_nll_loss(mean, logvar, target, mask)
    assert n.item() == 2
    # mean over the masked entries of 0.5*(log2pi + y^2): y=1 and y=3
    two_pi = torch.tensor(2 * torch.pi)
    expected = 0.5 * (torch.log(two_pi).item() + 1.0) + 0.5 * (torch.log(two_pi).item() + 9.0)
    assert loss.item() == pytest_appx(expected / 2.0)


def test_loss_gaussian_logvar_clamped_by_head(model_cfg_small):
    cfg = model_cfg_small
    m = _model(cfg).eval()
    batch = _dummy_batch(cfg, torch.device("cpu"))
    out = m(batch)
    assert out["logvar"].min() >= -8.0 and out["logvar"].max() <= 5.0


def test_loss_mse_masked():
    mean = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([2.0, 2.0, 2.0])
    mask = torch.tensor([True, False, True])
    loss, n = mse_loss_masked(mean, target, mask)
    assert n.item() == 2
    assert loss.item() == 4.0


def test_loss_quantile_pinball():
    pred = torch.tensor([[0.5, 0.5, 0.5]])  # [n_samples=1, 3 quantiles]
    target = torch.tensor([0.0])  # [n_samples]
    mask = torch.ones(1, dtype=torch.bool)
    loss, n = quantile_loss(pred, target, (0.1, 0.5, 0.9), mask)
    assert n.item() == 1
    # pinball at tau=0.1 for y=0 p=0.5: (0.1-1)*(0-0.5)=0.45; tau=0.5: 0.25;
    # tau=0.9: 0.05 -> mean 0.25
    assert loss.item() == pytest_appx(0.25)


def pytest_appx(x):
    return __import__("pytest").approx(x, abs=1e-6)
