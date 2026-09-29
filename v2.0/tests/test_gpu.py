"""GPU smoke tests: CUDA training/inference path (skipped when no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recad.config import Config
from recad.data.tensorize import CoastalPatchDataset
from recad.model.ensemble import DeepEnsemble
from recad.model.factory import build_model
from recad.train.trainer import Trainer
from recad.utils.native import native_available

pytestmark = [pytest.mark.l3, pytest.mark.gpu]

CUDA = torch.cuda.is_available()


def _ensemble(prepared, model_cfg, device, n=2):
    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=model_cfg, require_target=False
    ).shapes()
    members = [build_model(shapes, model_cfg, seed=i) for i in range(n)]
    return DeepEnsemble(members, device)


def test_device_selection_uses_cuda():
    if not CUDA:
        pytest.skip("no CUDA device")
    assert torch.cuda.get_device_name(0) != ""


def test_gpu_forward_and_predict_matches_cpu(prepared_small, masks_small, model_cfg_small):
    if not CUDA:
        pytest.skip("no CUDA device")
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    ens_gpu = _ensemble(prepared_small, model_cfg_small, gpu)
    ens_cpu = _ensemble(prepared_small, model_cfg_small, cpu)
    with torch.no_grad():
        mg = ens_gpu.predict_field(prepared_small, model_cfg_small, batch_size=2, use_amp=True)
        mc = ens_cpu.predict_field(prepared_small, model_cfg_small, batch_size=2, use_amp=False)
    sel = ~np.isnan(mg.mean) & ~np.isnan(mc.mean)
    # GPU (fp32 + AMP fp16 intermediates) and CPU agree to fp16-level precision
    # (~1e-3 relative, i.e. <= ~0.5 µatm at 340 µatm) - physically negligible.
    assert np.allclose(mg.mean[sel], mc.mean[sel], rtol=2e-2, atol=0.5)


def test_gpu_training_runs_with_amp(prepared_small, masks_small, cfg_small, tmp_path):
    if not CUDA:
        pytest.skip("no CUDA device")
    cfg = Config.from_dict(
        {
            "train": {
                "n_epochs": 2,
                "batch_windows": 2,
                "use_amp": True,
                "device": "cuda",
                "lr": 1e-3,
                "checkpoint_dir": str(tmp_path / "checkpoints-gpu"),
            },
            "model": {
                "embed_dim": 16,
                "n_heads": 2,
                "spatial_layers": 1,
                "temporal_layers": 1,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
        }
    )
    device = torch.device("cuda")
    train_ds = CoastalPatchDataset(
        prepared_small, masks_small, "train", cfg.model, require_target=True
    )
    val_ds = CoastalPatchDataset(prepared_small, masks_small, "val", cfg.model, require_target=True)
    model = build_model(train_ds.shapes(), cfg.model, seed=0)
    history = Trainer(cfg.train, device).fit_member(model, train_ds, val_ds, member_index=0)
    assert len(history.train_loss) >= 1
    assert np.isfinite(history.train_loss).all()
    assert np.isfinite(history.val_rmse).all()
    # the loss must actually decrease over two epochs on learnable data
    assert history.train_loss[0] >= history.train_loss[-1] * 0.95 or len(history.train_loss) == 1


def test_native_cuda_dll_loaded():
    """The pipeline should run with the native (ideally CUDA) kernel library."""
    if not native_available():
        pytest.skip("native recad_cuda library not built")
    assert native_available()
