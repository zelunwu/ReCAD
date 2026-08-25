"""Custom training loop (device-agnostic, AMP-enabled, ensemble-aware).

The loop is intentionally explicit (no framework magic): a plain PyTorch
train/validate loop with automatic mixed precision (CUDA), gradient clipping,
warmup + cosine LR, checkpointing and early stopping via callbacks. The same
Trainer fits every deep-ensemble member; members differ by seed and by a
year-level bootstrap of the training pool (``EnsembleConfig.bootstrap_
fraction``), which is the main source of ensemble diversity (see
docs/design.md §Deep ensemble).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recad.config import Config
from recad.data.features import PreparedData
from recad.data.split import SplitMasks
from recad.data.tensorize import CoastalPatchDataset, collate_items
from recad.model.ensemble import pick_device
from recad.model.factory import build_model
from recad.model.losses import build_loss
from recad.train.callbacks import EarlyStopping, History, ModelCheckpoint
from recad.train.metrics import metrics_summary
from recad.utils.logging import get_logger
from recad.utils.seed import seed_everything

_LOG = get_logger(__name__)


class Trainer:
    """Fits one model instance on a split-gated dataset."""

    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg  # TrainConfig
        self.device = device

    # ------------------------------------------------------------------
    def fit_member(
        self,
        model: nn.Module,
        train_ds: CoastalPatchDataset,
        val_ds: CoastalPatchDataset,
        member_index: int = 0,
    ) -> History:
        cfg = self.cfg
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = self._build_scheduler(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and self.device.type == "cuda")
        loss_fn = build_loss(model.cfg.head)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_windows,
            shuffle=True,
            collate_fn=collate_items,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_windows, shuffle=False, collate_fn=collate_items
        )

        ckpt = ModelCheckpoint(f"{cfg.checkpoint_dir}/member_{member_index}.pt", monitor="rmse")
        early = EarlyStopping(patience=cfg.early_stop_patience, monitor="rmse")
        history = History()

        for epoch in range(cfg.n_epochs):
            model.train()
            train_loss = self._run_epoch(
                model, train_loader, train_ds, optimizer, scaler, loss_fn, training=True
            )

            val_metrics: dict[str, float] | None = None
            last = epoch == cfg.n_epochs - 1
            if epoch % cfg.eval_every_epochs == 0 or last:
                val_metrics = self._evaluate(model, val_loader, val_ds, loss_fn)
                history.log(epoch, train_loss, val_metrics)
                ckpt(model, val_metrics, epoch)
                _LOG.info(
                    "epoch %3d | train_loss %.4f | val_loss %.4f | val_rmse %.3f | val_r2 %.4f",
                    epoch,
                    train_loss,
                    val_metrics.get("loss", float("nan")),
                    val_metrics.get("rmse", float("nan")),
                    val_metrics.get("r2", float("nan")),
                )
                if early(val_metrics, epoch):
                    break
            else:
                history.log(epoch, train_loss, None)

            if cfg.scheduler == "plateau" and val_metrics is not None:
                scheduler.step(val_metrics.get("loss", float("nan")))
            elif cfg.scheduler != "plateau":
                scheduler.step()

        # Restore the best validation checkpoint.
        ckpt.load_best(model)
        return history

    # ------------------------------------------------------------------
    def _build_scheduler(self, optimizer):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            warmup = cfg.warmup_epochs
            total = cfg.n_epochs

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return float(epoch + 1) / max(warmup, 1)
                prog = (epoch - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + np.cos(np.pi * prog))

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _loss_gate(batch: dict[str, torch.Tensor], ds: CoastalPatchDataset) -> torch.Tensor:
        """Split mask gate for a batch: [B, T, C] bool on the model device."""
        split_mask = ds.masks.get(ds.split)  # [n_year, 12, n_lat, n_lon]
        cell_flat = ds.cell_flat_indices
        gates: list[np.ndarray] = []
        year_ids = batch["year_id"].cpu().numpy()
        month_ids = batch["month_ids"].cpu().numpy()
        for b in range(len(year_ids)):
            y = int(year_ids[b])
            for m in month_ids[b]:
                gates.append(split_mask[y, int(m) - 1].ravel()[cell_flat])
        stacked = np.stack(gates, axis=0).reshape(len(year_ids), -1, cell_flat.size)
        return torch.from_numpy(stacked).bool()

    def _run_epoch(self, model, loader, ds, optimizer, scaler, loss_fn, training: bool) -> float:
        running = 0.0
        n_batches = 0
        autocast = torch.autocast("cuda", enabled=self.cfg.use_amp and self.device.type == "cuda")
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            gate = self._loss_gate(batch, ds).to(self.device)
            with autocast:
                out = model(batch)
                loss, _ = loss_fn(
                    out["mean"],
                    out.get("logvar"),
                    batch["targets"],
                    mask=gate & batch["cell_mask"],
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            running += float(loss.detach().cpu())
            n_batches += 1
        return running / max(n_batches, 1)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _evaluate(self, model, loader, ds, loss_fn) -> dict[str, float]:
        model.eval()
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0
        t_mean, t_std = ds.t_mean, ds.t_std
        autocast = torch.autocast("cuda", enabled=self.cfg.use_amp and self.device.type == "cuda")
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            gate = self._loss_gate(batch, ds).to(self.device)
            with autocast:
                out = model(batch)
                loss, _ = loss_fn(
                    out["mean"],
                    out.get("logvar"),
                    batch["targets"],
                    mask=gate & batch["cell_mask"],
                )
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            sel = (gate & batch["cell_mask"]).cpu().numpy()  # [B, T, C]
            # de-standardize back to physical units for the metric report
            y_true_list.append(batch["targets"].cpu().numpy()[sel] * t_std + t_mean)
            y_pred_list.append(out["mean"].float().cpu().numpy()[sel] * t_std + t_mean)
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        metrics = metrics_summary(y_true, y_pred)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics


def train_ensemble(
    cfg: Config,
    prepared: PreparedData,
    masks: SplitMasks,
    device: torch.device | None = None,
) -> tuple[list[nn.Module], list[History]]:
    """Train all deep-ensemble members and return (members, histories)."""
    device = device or pick_device(cfg.train.device)
    seed_everything(cfg.split.seed)

    # Structural shapes (identical across splits and members).
    probe = CoastalPatchDataset(
        prepared,
        masks,
        "train",
        cfg.model,
        tile_cells=cfg.train.tile_cells,
        require_target=True,
    )
    shapes = probe.shapes()

    members: list[nn.Module] = []
    histories: list[History] = []
    for i in range(cfg.ensemble.n_members):
        seed = cfg.split.seed + cfg.ensemble.seed_offset + i
        _LOG.info("training ensemble member %d/%d (seed=%d)", i + 1, cfg.ensemble.n_members, seed)
        model = build_model(shapes, cfg.model, seed=seed)

        year_repeats = _bootstrap_year_repeats(cfg, prepared, masks, i)
        train_ds = CoastalPatchDataset(
            prepared,
            masks,
            "train",
            cfg.model,
            tile_cells=cfg.train.tile_cells,
            require_target=True,
            year_repeats=year_repeats,
        )
        val_ds = CoastalPatchDataset(
            prepared,
            masks,
            "val",
            cfg.model,
            tile_cells=cfg.train.tile_cells,
            require_target=True,
        )
        trainer = Trainer(cfg.train, device)
        history = trainer.fit_member(model, train_ds, val_ds, member_index=i)
        members.append(model)
        histories.append(history)
        _LOG.info("member %d done: final val_rmse %.3f", i + 1, history.val_rmse[-1])
    return members, histories


def _bootstrap_year_repeats(cfg: Config, prepared: PreparedData, masks: SplitMasks, member: int):
    """Year-level bootstrap of the training pool for one ensemble member.

    Returns a dict year-index -> multiplicity, or None when the bootstrap
    fraction is 1.0 (no resampling, v1.1-style single pool per member).
    """
    fraction = cfg.ensemble.bootstrap_fraction
    if fraction >= 1.0 - 1e-9:
        return None
    train_mask = masks.train
    year_ids = [y for y in range(prepared.n_year) if train_mask[y].any()]
    n = max(1, round(fraction * len(year_ids)))
    rng = np.random.default_rng(cfg.split.seed + cfg.ensemble.seed_offset + member)
    sampled = rng.choice(year_ids, size=n, replace=True)
    from collections import Counter

    return dict(Counter(y for y in sampled))
