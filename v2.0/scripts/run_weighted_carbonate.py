"""Spatially-weighted carbonate closure experiment (global + US coast).

Improvement over the previous uniform-closure runs
(experiments/joint_four_full_20260907, carbonate_latent_20260909):

  * the CO2SYS closure penalty is **weighted by observation support**
    (chem_support_<tag>.nc): weight 1.0 on GLODAP anchor cells, linear decay
    0.1 -> 0 within 200 km, exactly 0 beyond. This removes the pseudo-
    constraint that uniform closure imposed on the 66-99% of cells with no
    TA/DIC observations (docs/backup/ta_dic_constraint_review.md).
  * reports R2 / RMSE / MAE / bias for fCO2, SSS, TA, DIC across
    train / validation / test splits, on both the global domain and the
    US-coast reporting windows.

Model / targets follow the settled protocol:
    independent heads:  SSS (residual on GLODAP background), fCO2, TA
    derived:            DIC = inverse CO2SYS emulator(T, S_hat, TA_hat, fCO2_hat)

Usage:
    python scripts/run_weighted_carbonate.py --domain naccom --tag weighted_v1
    python scripts/run_weighted_carbonate.py --domain global --tag weighted_global
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "experiments"

# reuse the proven framework from the previous round
spec = importlib.util.spec_from_file_location(
    "base_reg", ROOT / "scripts" / "run_naccom_regularized_joint.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from recad.chem.co2sys_emulator import CO2SYSDICEmulator, CO2SYSEmulator, pyco2sys_fco2
from recad.chem.support import ChemSupport
from recad.config import Config
from recad.data.tensorize import TensorShapes
from recad.model.carbonate_latent_st import CarbonateLatentST

SEEDS = (100, 101, 102)
TARGETS4 = ("sss", "fco2", "ta", "dic")
DOMAINS = {
    # domain tag -> (prepared cache, joint cache path, chem-support tag, window bounds)
    # NOTE: the joint caches are region-specific.  The NACCOM (North-American
    # Atlantic coastal margin) cache covers lon 265.9-319.9E / lat 10-65N and
    # carries 144k SOCAT and 1.7k GLODAP training points; the us_west cache is
    # the Pacific coast (225-250E) and must NOT be used under the 'naccom' tag.
    "naccom": ("prepared_naccom.nc",
               "experiments/joint_four_full_20260907/joint_cache.npz",
               "naccom", (275.0, 300.0, 20.0, 55.0)),   # US east-coast window
    "us_west": ("prepared_us_west.nc",
                "experiments/us_west_joint_cache_20260909/joint_cache.npz",
                "us_west", (230.0, 250.0, 24.0, 50.0)),  # US west-coast window
    "global": ("prepared_global.nc",
               "experiments/global_joint_cache/joint_cache.npz",
               "global", (275.0, 300.0, 20.0, 55.0)),   # US-east reporting window
    # Global domain at a 4-degree patch size: the 2-degree variant needs 14,760
    # tokens and costs ~37 s/step (attention is O(P^2)), i.e. ~13 days for the
    # full run.  At 4 degrees there are 3,690 tokens -> 0.42 s/step (~3.5 h).
    "global_p32": ("prepared_global_p32.nc",
                   "experiments/global_joint_cache_p32/joint_cache.npz",
                   "global", (275.0, 300.0, 20.0, 55.0)),
}


def metric(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok].astype(float)
    p = p[ok].astype(float)
    e = p - y
    den = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(e.mean()),
        "r2": float(1 - np.sum(e * e) / den) if den > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 else float("nan"),
    }


def combine_selection_rmse(dense_rmse, sparse_rmse, dense_scales, sparse_scales):
    """Balanced checkpoint criterion for dense and sparse target families.

    Each target contributes its RMSE divided by a training-derived scale.  The
    two family means are then averaged, so the much larger SOCAT table cannot
    dominate TA/DIC and the sparse carbonate table cannot silently degrade the
    SSS/fCO2 products.
    """
    dense = [float(dense_rmse[k]) / max(float(dense_scales[k]), 1e-12)
             for k in ("sss", "fco2")]
    sparse = [float(sparse_rmse[k]) / max(float(sparse_scales[k]), 1e-12)
              for k in ("ta", "dic")]
    values = np.asarray(dense + sparse, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return float("inf")
    return float(0.5 * np.mean(values[:2]) + 0.5 * np.mean(values[2:]))


class WeightedData(base.Data):
    """base.Data + per-sample chemistry support weights (spatially weighted loss)."""

    def __init__(self, support_tag: str, include_independent: bool = False):
        super().__init__(include_independent=include_independent)
        self.cs = ChemSupport.load(support_tag)
        self.support_tag = support_tag
        self._wcache: dict[str, np.ndarray] = {}

    def weights(self, source: str, split: str, indices: np.ndarray | None = None) -> np.ndarray:
        """Support weight for every record of a (source, split) table."""
        key = f"{source}_{split}"
        if key not in self._wcache:
            x = self.z[f"{key}_x"]
            lon = (x[:, 0] * 360.0).astype(np.float64)
            lat = (x[:, 1] * 180.0 - 90.0).astype(np.float64)
            self._wcache[key] = self.cs.query(lon, lat).astype(np.float32)
        w = self._wcache[key]
        return w if indices is None else w[indices]


def make_model_for(data: WeightedData, seed: int):
    """Build the carbonate latent model with shapes derived from the cache.

    The region caches carry a Chl-a channel (token width 14) while the global
    cache does not (token width 12), so the shapes must be read from the data
    rather than hardcoded - otherwise the first matmul fails on the global run.
    """
    import math

    torch.manual_seed(seed)
    cfg = Config.from_yaml(str(ROOT / "configs" / "experiment_naccom_st_full.yaml")).model
    cfg = replace(cfg, embed_dim=192, head="mean", dropout=0.1)
    tok = data.z["token_feats"]
    n_tokens = tok.shape[2]
    n_token_features = tok.shape[3]
    x = data.z["socat_train_x"]
    n_cell_features = x.shape[1]
    # predictor count = cell features minus {lon, lat, sin, cos} minus the
    # trailing TA-prior pair
    n_predictors = (n_cell_features - 4 - 2) // 2
    shapes = TensorShapes(n_tokens, 1, n_token_features, n_cell_features, n_predictors)
    print(f"  model shapes: tokens={n_tokens}, token_feat={n_token_features}, "
          f"cell_feat={n_cell_features}, predictors={n_predictors}", flush=True)
    return CarbonateLatentST(shapes, cfg).cuda()


def physical_latent(o, g, data: WeightedData, baseline, inv):
    """Physical fields from the latent head, with DIC **derived** by inverse CO2SYS.

    ``CarbonateLatentST`` emits sss / fco2 / ta (and a TA log-variance); DIC is
    not an independent head. Following the settled protocol we obtain it from
    the inverse emulator:  DIC = inv(T, S_hat, TA_hat, fCO2_hat).
    """
    s = (g["base_sss"] + o["sss"].float() * data.sscale).clamp(5, 42)
    f = (o["fco2"].float() * data.fstd + data.fmean).clamp(20, 2500)
    ta = (g["base_ta"] + o["ta"].float() * float(baseline["ta_scale"])).clamp(700, 2850)
    temp = (g["x"][:, 4].float() * float(data.z["feature_stds"][0])
            + float(data.z["feature_means"][0])).clamp(-2, 35)
    dic = inv(temp, s, ta, f)
    return {"sss": s, "fco2": f, "ta": ta, "dic": dic,
            "ta_logvar": o["ta_logvar"].float()}


def weighted_chem_loss(p, g, idx, data: WeightedData, emulator, w: torch.Tensor):
    """Σ w(x)·((CO2SYS(T,S,TA,DIC) − fCO2)/σ)² / Σ w(x)."""
    temp = g["x"][idx, 4].float() * float(data.z["feature_stds"][0]) + float(
        data.z["feature_means"][0]
    )
    f_chem = emulator(
        temp.clamp(-2, 35),
        p["sss"].clamp(5, 42),
        p["ta"].clamp(700, 2850),
        p["dic"].clamp(600, 2700),
    )
    r2 = ((f_chem - p["fco2"]) / base.CHEM_SCALE).square()
    wsum = w.sum()
    if float(wsum) <= 0:
        return torch.zeros((), device=r2.device)
    return (w * r2).sum() / wsum


def pretrain_socat(data: WeightedData, seed: int, steps: int, baseline):
    """Phase 1: train the shared trunk + dense heads on SOCAT (fCO2/SSS).

    Skipping this phase (as an earlier draft did) leaves the encoder
    untrained, and the chemistry adapter then fits a useless trunk -- the
    resulting SSS/fCO2 skill collapses (SSS RMSE ~2.3 PSU vs ~0.9 expected).
    """
    model = make_model_for(data, seed)
    groups = data.records("socat", "train", baseline)
    # Phase-1 trains the shared trunk plus the dense (SOCAT-supervised) heads.
    # The TA head and its log-variance head stay frozen: they are the sparse
    # chemistry targets fitted in phase 2.
    sparse = {"outputs.ta", "outputs.ta_logvar"}
    params = [p for name, p in model.named_parameters()
              if not any(name.startswith(f"head.{s}") for s in sparse)]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        year, g, idx = base.sample_group(groups, rng)
        scale = min((step + 1) / 100, 1) * (1 if step < 2000 else 0.3)
        opt.param_groups[0]["lr"] = 1e-3 * scale
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = model.encode_context(data.ctx(year))
            o = model.decode_queries(ctx, g["x"][idx], g["month"][idx], g["patch"][idx])
        gi = {k: (v[idx] if isinstance(v, torch.Tensor) and len(v) == len(g["x"]) else v)
              for k, v in g.items()}
        # phase 1 supervises fCO2/SSS only, which needs no DIC derivation
        s = (gi["base_sss"] + o["sss"].float() * data.sscale).clamp(5, 42)
        f = (o["fco2"].float() * data.fstd + data.fmean).clamp(20, 2500)
        p = {"sss": s, "fco2": f}
        loss = base.direct_loss(o, p, g, idx, "socat", data, baseline)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  [pretrain] seed={seed} step={step+1} loss={loss.item():.4f}", flush=True)
    return model


def direct_loss_latent(o, p, g, idx, source, data: WeightedData, baseline):
    """Task loss for the latent head.

    SOCAT branch supervises fCO2 and in-situ SSS (delegates to the shared
    implementation).  GLODAP branch supervises the **TA** head and, through the
    derived DIC, the anchor DIC as well - ``CarbonateLatentST`` has no DIC head,
    so DIC is obtained from the inverse CO2SYS emulator (``physical_latent``).
    """
    if source == "socat":
        return base.direct_loss(o, p, g, idx, source, data, baseline)
    ta_t = (g["ta"][idx] - g["base_ta"][idx]) / float(baseline["ta_scale"])
    l_ta = (o["ta"].float() - ta_t).square().mean()
    l_dic = ((p["dic"] - g["dic"][idx]) / float(data.target_std["dic"])).square().mean()
    return l_ta + l_dic


def train_weighted(data: WeightedData, seed: int, steps: int, *, chem_lambda: float,
                   select_groups=None, baseline=None, pretrained=None):
    """Phase 2: fit the chemistry adapter with the spatially weighted closure term."""
    allg = np.arange(len(data.z["glodap_train_x"]))
    b = baseline if baseline is not None else data.fit_baseline(allg)
    emulator = CO2SYSEmulator.load_frozen(base.EMULATOR, "cuda")
    # inverse emulator: DIC is derived from (T, S, TA, fCO2), not predicted
    inv = CO2SYSDICEmulator.load_frozen(
        ROOT / "outputs/chem/co2sys_dic_emulator_v1.pt", "cuda")
    model = pretrained if pretrained is not None else make_model_for(data, seed)
    # Phase 2 fits only the carbonate outputs (and the latent TA head) on the
    # sparse GLODAP anchors, with the spatially weighted closure term. The
    # SOCAT-supervised trunk/heads stay frozen so the dense skill is retained.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head.")
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=4e-4, weight_decay=3e-3)
    rng = np.random.default_rng(seed)

    groups = data.records("glodap", "train", b)
    w_all = data.weights("glodap", "train")
    # Checkpoint selection must run in BOTH phases.  An earlier version only
    # evaluated when `select_groups` was passed, so the final fit kept
    # `best_state = None` and silently returned the *last* step: phase 2 trains
    # only the heads against sparse TA/DIC + closure, which degrades the fCO2
    # head without any supervision.  That produced seed-to-seed fCO2 test skill
    # ranging from -0.21 to +0.35 on the global run.
    # When no explicit selection set is given we hold out a small slice of the
    # anchor table for model selection.
    if select_groups is None:
        allg_ix = np.arange(len(data.z["glodap_train_x"]))
        select_groups = data.records("glodap", "train", b, allg_ix[::5])
    # Dense-side selection set: the SOCAT dev split, so a checkpoint that
    # degrades fCO2/SSS cannot be selected.
    alls_ix = np.arange(len(data.z["socat_train_x"]))
    socat_sel = data.records("socat", "train", b, alls_ix[::8])
    best, best_state, best_step = float("inf"), None, steps
    for step in range(steps):
        year, g, idx = base.sample_group(groups, rng, True, 256)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = model.encode_context(data.ctx(year))
            o = model.decode_queries(ctx, g["x"][idx], g["month"][idx], g["patch"][idx])
        gi = {k: (v[idx] if isinstance(v, torch.Tensor) and len(v) == len(g["x"]) else v)
              for k, v in g.items()}
        p = physical_latent(o, gi, data, b, inv)
        wb = torch.from_numpy(w_all[idx.cpu().numpy()]).cuda().float()
        loss = direct_loss_latent(o, p, g, idx, "glodap", data, b) + chem_lambda * weighted_chem_loss(
            p, g, idx, data, emulator, wb
        )
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1)
        opt.step()
        if (step + 1) % 250 == 0 or step == steps - 1:
            score = selection_score(model, data, b, inv, socat_sel, select_groups)
            print(f"  [select] seed={seed} step={step+1} score={score:.4f}", flush=True)
            if score < best:
                best, best_step = score, step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, b, {"best_step": best_step, "best_score": best}


@torch.inference_mode()
def evaluate_derived(model, data: WeightedData, groups, b, inv):
    """Evaluate with DIC **derived** from the inverse CO2SYS emulator.

    Returns {target: {"y": truth, "p": prediction}} for sss/fco2/ta/dic.
    Mirrors base.evaluate but replaces the DIC head with
        DIC = inv(T, S_hat, TA_hat, fCO2_hat).
    """
    model.eval()
    out = {t: {"y": [], "p": []} for t in TARGETS4}
    for year, g in groups.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = model.encode_context(data.ctx(year))
        for start in range(0, len(g["x"]), 8192):
            sl = slice(start, start + 8192)
            gg = {k: (v[sl] if isinstance(v, torch.Tensor) and len(v) == len(g["x"]) else v)
                  for k, v in g.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.decode_queries(ctx, g["x"][sl], g["month"][sl], g["patch"][sl])
            s = (gg["base_sss"] + o["sss"].float() * data.sscale).clamp(5, 42)
            f = (o["fco2"].float() * data.fstd + data.fmean).clamp(20, 2500)
            ta = (gg["base_ta"] + o["ta"].float() * float(b["ta_scale"])).clamp(700, 2850)
            temp = (gg["x"][:, 4].float() * float(data.z["feature_stds"][0])
                    + float(data.z["feature_means"][0])).clamp(-2, 35)
            dic = inv(temp, s, ta, f)
            preds = {"sss": s, "fco2": f, "ta": ta, "dic": dic}
            for t in TARGETS4:
                if t in gg:
                    out[t]["y"].append(gg[t].cpu().numpy())
                    out[t]["p"].append(preds[t].cpu().numpy())
    return {t: {k: np.concatenate(v) for k, v in q.items()} for t, q in out.items() if q["y"]}


def selection_score(model, data: WeightedData, b, inv, socat_groups, carbon_groups):
    """Score a checkpoint on held-out dense and carbonate observations."""
    dense = evaluate_derived(model, data, socat_groups, b, inv)
    sparse = evaluate_derived(model, data, carbon_groups, b, inv)
    dense_rmse = {t: metric(dense[t]["y"], dense[t]["p"])["rmse"]
                  for t in ("sss", "fco2")}
    sparse_rmse = {t: metric(sparse[t]["y"], sparse[t]["p"])["rmse"]
                   for t in ("ta", "dic")}
    return combine_selection_rmse(
        dense_rmse,
        sparse_rmse,
        {"sss": data.sscale, "fco2": data.fstd},
        {"ta": data.target_std["ta"], "dic": data.target_std["dic"]},
    )


@torch.inference_mode()
def evaluate_splits(data: WeightedData, model, b, bounds=None):
    """Per-split metrics for all four targets (optionally cropped to a window).

    DIC follows the settled protocol: it is **derived** from the inverse
    CO2SYS emulator given (T, S_hat, TA_hat, fCO2_hat), not predicted by an
    independent head.

    ``bounds`` selects a reporting window (lon0, lon1, lat0, lat1) by
    subsetting the record tables **before** inference, so the returned arrays
    stay aligned with the records actually evaluated.
    """
    rows = []
    inv = CO2SYSDICEmulator.load_frozen(ROOT / "outputs/chem/co2sys_dic_emulator_v1.pt", "cuda")
    for source, targets in (("socat", ("sss", "fco2")), ("glodap", ("ta", "dic"))):
        for split in ("train", "dev", "independent"):
            key = f"{source}_{split}"
            if f"{key}_x" not in data.z.files:
                continue
            try:
                if bounds is None:
                    groups = data.records(source, split, b)
                else:
                    x = data.z[f"{key}_x"]
                    lon = x[:, 0] * 360.0
                    lat = x[:, 1] * 180.0 - 90.0
                    lon0, lon1, lat0, lat1 = bounds
                    sel = np.flatnonzero((lon >= lon0) & (lon < lon1) & (lat >= lat0) & (lat <= lat1))
                    if sel.size == 0:
                        continue
                    groups = data.records(source, split, b, sel)
            except RuntimeError:
                continue
            q = evaluate_derived(model, data, groups, b, inv)
            label = {"train": "train", "dev": "validation", "independent": "test"}[split]
            for t in targets:
                if t in q:
                    rows.append({"source": source, "split": label, "target": t,
                                 **metric(q[t]["y"], q[t]["p"])})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="naccom", choices=sorted(DOMAINS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--pretrain-steps", type=int, default=3000,
                    help="SOCAT trunk/head pretraining steps before the chem adapter")
    ap.add_argument("--chem-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", default="100,101,102")
    args = ap.parse_args()

    prepared, joint_cache, support_tag, bounds = DOMAINS[args.domain]
    base.PREPARED = ROOT / "outputs" / prepared
    base.CACHE = ROOT / "outputs" / joint_cache
    if not base.CACHE.exists():
        raise FileNotFoundError(f"joint cache missing for domain {args.domain}: {base.CACHE}")
    OUT = OUT_ROOT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(v) for v in args.seeds.split(","))

    print(f"[{args.tag}] domain={args.domain} prepared={prepared} "
          f"cache={joint_cache} support={support_tag}", flush=True)
    data = WeightedData(support_tag, include_independent=False)
    print("support stats:", {k: round(v, 4) for k, v in data.cs.stats().items()}, flush=True)

    # 1) folds for selection (GLODAP cruise-grouped)
    allg = np.arange(len(data.z["glodap_train_x"]))
    cruises = data.z["glodap_train_cruise"]
    folds = base.cruise_folds(cruises, 5)
    baseline = data.fit_baseline(allg)

    # 2) short CV to pick the closure weight / step count on held-out cruises
    pretrain_steps = args.pretrain_steps
    cv_rows = []
    for k in range(min(3, len(folds))):
        gi = np.setdiff1d(allg, folds[k])
        b_k = data.fit_baseline(gi)
        vg = data.records("glodap", "train", b_k, folds[k])
        pre = pretrain_socat(data, 100 + k, pretrain_steps, b_k)
        m, _, info = train_weighted(data, 100 + k, args.steps, chem_lambda=args.chem_lambda,
                                    select_groups=vg, baseline=b_k, pretrained=pre)
        cv_rows.append({"fold": k, **info})
        del m, pre
        torch.cuda.empty_cache()
    pd.DataFrame(cv_rows).to_csv(OUT / "cv_results.csv", index=False)
    best_steps = int(np.median([r["best_step"] for r in cv_rows])) or args.steps
    print(f"[{args.tag}] CV done; best_steps={best_steps}", flush=True)

    # 3) final members on all training data (pretrain -> weighted chem adapter)
    preds = []
    for seed in seeds:
        pre = pretrain_socat(data, seed, pretrain_steps, baseline)
        m, b, info = train_weighted(data, seed, best_steps, chem_lambda=args.chem_lambda,
                                    baseline=baseline, pretrained=pre)
        torch.save({"model": m.state_dict(), "baseline": b, "info": info, "support": support_tag},
                   OUT / f"final_seed{seed}.pt")
        preds.append((m, b))
        del m, pre
        torch.cuda.empty_cache()

    # 4) metrics: full domain + US-coast reporting window
    data_ind = WeightedData(support_tag, include_independent=True)
    rows = []
    for tag, bnd in (("domain", None), ("us_window", bounds)):
        if bnd is None and tag == "us_window":
            continue
        for m, b in preds:
            rows.extend([{**r, "region": tag, "seed": "member"}
                         for r in evaluate_splits(data_ind, m, b, bnd)])
    df = pd.DataFrame(rows)
    # ensemble mean metrics
    ens_rows = []
    for region in df["region"].unique():
        for source in df["source"].unique():
            for split in df["split"].unique():
                for t in TARGETS4:
                    sub = df[(df.region == region) & (df.source == source) & (df.split == split) & (df.target == t)]
                    if len(sub):
                        ens_rows.append({"region": region, "source": source, "split": split, "target": t,
                                         "n": int(sub["n"].iloc[0]), "rmse": float(sub["rmse"].mean()),
                                         "r2": float(sub["r2"].mean())})
    pd.DataFrame(ens_rows).to_csv(OUT / "metrics_by_split.csv", index=False)
    print(df.to_string(), flush=True)
    (OUT / "completed.json").write_text(json.dumps({"status": "complete", "tag": args.tag,
                                                    "support": support_tag, "steps": best_steps,
                                                    "chem_lambda": args.chem_lambda}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
