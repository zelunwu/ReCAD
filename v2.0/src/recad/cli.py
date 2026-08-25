"""ReCAD v2.0 command-line interface.

Subcommands (all reproducible, all config-driven):

    describe     print the registered data-source metadata
    ingest       normalise raw sources onto the target grid (standardised cache)
    preprocess   build PreparedData + split masks + feature statistics
    train        train the deep ensemble (GPU/CPU)
    predict      reconstruct the full product field with the trained ensemble
    uncertainty  Monte-Carlo input-error propagation + combined uncertainty
    validate     holdout-year validation table (v1.1-compatible protocol)
    e2e          full pipeline on synthetic data (CI smoke test)

Example:
    recad e2e --workdir outputs/smoke
    recad train --config configs/test_smoke.yaml --prepared outputs/prepared.nc \\
           --masks outputs/masks.nc --device auto
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

from recad.config import Config
from recad.data.pipeline import (
    build_prepared,
    ingest_variable,
    load_masks,
    load_prepared,
    prepare_split,
    save_masks,
    save_prepared,
)
from recad.data.specs import describe_sources
from recad.model.ensemble import DeepEnsemble, pick_device
from recad.predict.export import product_dataset, to_netcdf
from recad.predict.reconstruct import build_ensemble_prediction, load_ensemble
from recad.train.metrics import per_year_metrics
from recad.train.trainer import train_ensemble
from recad.uncertainty.decompose import combine_uncertainties
from recad.uncertainty.input_mc import propagate_input_uncertainty
from recad.utils.chem import fco2_to_pco2
from recad.utils.io import dump_json, touch_dir
from recad.utils.logging import get_logger

_LOG = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML config file (optional)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. train.n_epochs=3",
    )


def _parse_overrides(args) -> dict[str, object] | None:
    overrides: dict[str, object] = {}
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"--override expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key] = _coerce(value)
    return overrides or None


def _coerce(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_describe(args) -> int:
    for spec in describe_sources():
        print(f"[{spec.variable}] {spec.kind} ({spec.units}) - {spec.temporal_sampling}")
        print(f"    v1.1 provenance: {spec.v11_provenance}")
        print(f"    notes: {spec.notes}\n")
    return 0


def cmd_ingest(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    cache = Path(args.cache)
    touch_dir(cache)
    for name in ("sst", "sss", "adt", "wspd", "xco2air", "fco2", "mask"):
        template = cfg.data.templates.get(name)
        if not template:
            continue
        ingest_variable(cfg, name, cache)
    return 0


def cmd_preprocess(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    prepared = build_prepared(cfg, args.cache)
    masks = prepare_split(cfg, prepared)
    save_prepared(prepared, args.prepared)
    save_masks(masks, args.masks, prepared.years, prepared.grid)
    stats = {
        "predictor_means": prepared.feature_stats()[0].tolist(),
        "predictor_stds": prepared.feature_stats()[1].tolist(),
        "split_counts": masks.counts(),
        "coastal_cells": int(prepared.coastal_mask.sum()),
    }
    dump_json(stats, args.stats)
    _LOG.info("preprocessed -> %s, masks -> %s", args.prepared, args.masks)
    return 0


def cmd_train(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    prepared = load_prepared(args.prepared)
    masks = load_masks(args.masks)
    if not _try_load_feature_stats(prepared, args.stats):
        _LOG.warning("no usable feature-stats file; re-fitting on the train split")
        prepared.fit_feature_stats(masks.train)
    device = pick_device(cfg.train.device)
    _LOG.info("device: %s", device)
    members, histories = train_ensemble(cfg, prepared, masks, device)
    out_dir = touch_dir(cfg.output.out_dir)
    for i, history in enumerate(histories):
        dump_json(history.to_dict(), Path(out_dir) / f"history_member_{i}.json")
    _LOG.info("trained %d members; checkpoints in %s", len(members), cfg.train.checkpoint_dir)
    return 0


def _try_load_feature_stats(prepared, stats_path: str) -> bool:
    """Reload per-predictor normalisation stats; False if the file is unusable."""
    if not stats_path or not Path(stats_path).is_file():
        return False
    try:
        stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        prepared._feature_stats = (
            np.asarray(stats["predictor_means"], dtype=float),
            np.asarray(stats["predictor_stds"], dtype=float),
        )
        return True
    except (KeyError, ValueError, json.JSONDecodeError):
        return False


def cmd_predict(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    prepared = load_prepared(args.prepared)
    device = pick_device(cfg.train.device)
    from recad.data.tensorize import CoastalPatchDataset

    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=cfg.model, require_target=False
    ).shapes()
    ensemble = load_ensemble(cfg, shapes, device)
    fields = build_ensemble_prediction(cfg, prepared, ensemble, device)
    ds = product_dataset(cfg, prepared.grid, prepared.years, fields)
    out = to_netcdf(ds, Path(cfg.output.out_dir) / f"{cfg.output.product_name}.nc")
    _LOG.info("product written -> %s", out)
    return 0


def cmd_uncertainty(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    prepared = load_prepared(args.prepared)
    device = pick_device(cfg.train.device)
    from recad.data.tensorize import CoastalPatchDataset

    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=cfg.model, require_target=False
    ).shapes()
    ensemble = load_ensemble(cfg, shapes, device)
    base = build_ensemble_prediction(cfg, prepared, ensemble, device)

    alg = base["fco2_err_aleatoric"]
    epi = base["fco2_err_epistemic"]
    if cfg.uncertainty.propagate_input:
        mc_std, _ = propagate_input_uncertainty(
            ensemble,
            prepared,
            cfg.uncertainty,
            cfg.model,
            tile_cells=args.tile_cells,
            batch_size=cfg.train.batch_windows,
            device=device,
        )
    else:
        mc_std = np.zeros_like(alg)

    total = combine_uncertainties(alg, epi, mc_std, mode=cfg.uncertainty.combine)
    pco2_err_total = fco2_to_pco2(total, prepared.require("sst"))
    fields = {
        "fco2": base["fco2"],
        "fco2_err_total": total,
        "fco2_err_aleatoric": alg,
        "fco2_err_epistemic": epi,
        "fco2_err_input": mc_std,
        "pco2": base["pco2"],
        "pco2_err_total": pco2_err_total,
    }
    ds = product_dataset(cfg, prepared.grid, prepared.years, fields)
    out = to_netcdf(ds, Path(cfg.output.out_dir) / f"{cfg.output.product_name}_with_uncertainty.nc")
    _LOG.info("uncertainty product written -> %s", out)
    return 0


def cmd_validate(args) -> int:
    cfg = Config.from_yaml(args.config, _parse_overrides(args))
    prepared = load_prepared(args.prepared)
    masks = load_masks(args.masks)
    device = pick_device(cfg.train.device)
    from recad.data.tensorize import CoastalPatchDataset

    shapes = CoastalPatchDataset(
        prepared, masks=None, split=None, model_cfg=cfg.model, require_target=False
    ).shapes()
    ensemble = load_ensemble(cfg, shapes, device)
    moments = ensemble.predict_field(
        prepared,
        cfg.model,
        tile_cells=None,
        batch_size=cfg.train.batch_windows,
        use_amp=cfg.train.use_amp,
    )
    y_true = prepared.target_values()
    y_pred = moments.mean
    test = per_year_metrics(
        np.where(masks.test, y_true, np.nan),
        np.where(masks.test, y_pred, np.nan),
        prepared.years,
    )
    out_dir = touch_dir(cfg.output.out_dir)
    test.to_csv(Path(out_dir) / "test_year_metrics.csv", index=False)
    _LOG.info("test-year table written; tail:\n%s", test.tail(6).to_string(index=False))
    return 0


def cmd_plot(args) -> int:
    """Regenerate the v1.1-style figure set from product + data + masks."""
    from recad.data.pipeline import load_masks, load_prepared
    from recad.viz.report import render_figure_set

    prepared = load_prepared(args.prepared)
    masks = load_masks(args.masks)
    outdir = args.outdir or "figures"
    outputs = render_figure_set(
        args.product,
        prepared,
        masks,
        outdir,
        fmts=tuple(args.fmt),
        target_var=args.var,
    )
    for name in ("product_mean_map", "density_scatter_ttv", "trend_map"):
        _LOG.info("wrote %s -> %s", name, outputs.get(name, "?"))
    _LOG.info("plot set complete (%d outputs)", len(outputs))
    return 0


def cmd_download(args) -> int:
    """Download the latest data stack into data/raw (git-ignored)."""
    from recad.data.download import SOURCES
    from recad.data.download import main as download_main

    if args.list:
        for name, spec in SOURCES.items():
            auth = f"  [auth: {spec.requires_auth[:50]}...]" if spec.requires_auth else ""
            _LOG.info("%-12s %s%s", name, spec.title, auth)
        return 0
    mapped: list[str] = []
    if args.doc:
        mapped.append("--doc")
    if args.only:
        mapped += ["--only", args.only]
    if args.region != "-100,-40,10,65":
        mapped += [f"--region={args.region}"]
    if args.years != "1993,2021":
        mapped += [f"--years={args.years}"]
    if args.dry_run:
        mapped.append("--dry-run")
    return download_main(mapped)


def cmd_e2e(args) -> int:
    """Full pipeline on synthetic data inside a workdir (CI smoke test)."""
    from recad.testing.synthetic import make_synthetic_workdir

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config_path, cache, *_ = make_synthetic_workdir(workdir, n_years=args.n_years)
    cfg = Config.from_yaml(str(config_path))
    _LOG.info("e2e: ingesting synthetic data")
    for name in ("sst", "sss", "adt", "wspd", "xco2air", "fco2", "mask"):
        if name in cfg.data.templates:
            ingest_variable(cfg, name, cache)
    prepared = build_prepared(cfg, cache)
    masks = prepare_split(cfg, prepared)
    save_prepared(prepared, workdir / "prepared.nc")
    save_masks(masks, workdir / "masks.nc", prepared.years, prepared.grid)

    _LOG.info("e2e: training (tiny config)")
    device = pick_device(cfg.train.device)
    members, histories = train_ensemble(cfg, prepared, masks, device)

    _LOG.info("e2e: predicting + validating")
    ensemble = DeepEnsemble(members, device)
    fields = build_ensemble_prediction(cfg, prepared, ensemble, device)
    ds = product_dataset(cfg, prepared.grid, prepared.years, fields)
    out = to_netcdf(ds, workdir / "product.nc")
    for i, history in enumerate(histories):
        dump_json(history.to_dict(), Path(cfg.output.out_dir) / f"history_member_{i}.json")
    _LOG.info("e2e: product written -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    _LOG.info("e2e: OK")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recad",
        description="ReCAD v2.0 - global coastal pCO2/fCO2 reconstruction "
        "(spatio-temporal transformer deep ensemble, GPU-accelerated).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("describe", help="print registered data sources")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("ingest", help="standardise raw sources onto the grid")
    _add_common(p)
    p.add_argument("--cache", default="outputs/cache", help="standardised cache dir")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("preprocess", help="build PreparedData + split masks")
    _add_common(p)
    p.add_argument("--cache", default="outputs/cache")
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.add_argument("--masks", default="outputs/masks.nc")
    p.add_argument("--stats", default="outputs/feature_stats.json")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("train", help="train the deep ensemble")
    _add_common(p)
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.add_argument("--masks", default="outputs/masks.nc")
    p.add_argument("--stats", default="outputs/feature_stats.json")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="reconstruct the product field")
    _add_common(p)
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("uncertainty", help="input-error MC + combined uncertainty")
    _add_common(p)
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.add_argument("--tile-cells", type=int, default=None)
    p.set_defaults(func=cmd_uncertainty)

    p = sub.add_parser("validate", help="holdout-year validation table")
    _add_common(p)
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.add_argument("--masks", default="outputs/masks.nc")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("e2e", help="end-to-end smoke test on synthetic data")
    p.add_argument("--workdir", default=None, help="workdir (default: temp)")
    p.add_argument("--n-years", type=int, default=4)
    p.set_defaults(func=cmd_e2e)

    p = sub.add_parser("plot", help="regenerate the v1.1-style figure set")
    _add_common(p)
    p.add_argument(
        "--product",
        default="outputs/ReCAD-v2.0-pCO2.nc",
        help="product NetCDF from 'recad predict/uncertainty'",
    )
    p.add_argument("--prepared", default="outputs/prepared.nc")
    p.add_argument("--masks", default="outputs/masks.nc")
    p.add_argument("--outdir", default=None, help="figure output dir")
    p.add_argument("--var", default="fco2", help="product variable to plot")
    p.add_argument("--fmt", action="append", default=["png"], choices=["png", "pdf", "jpg"])
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("download", help="download the latest data stack into data/raw")
    p.add_argument("--list", action="store_true", help="print the download manifest")
    p.add_argument("--doc", action="store_true", help="print per-source instructions")
    p.add_argument(
        "--only",
        default=None,
        help="download one source (xco2air|socat|gshhg|sst|sss|adt|wspd|bathymetry)",
    )
    p.add_argument("--region", default="-100,-40,10,65", help="lon0,lon1,lat0,lat1 (SOCAT subset)")
    p.add_argument("--years", default="1993,2021", help="year0,year1 (SOCAT subset)")
    p.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    p.set_defaults(func=cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        _LOG.error("%s: %s", type(exc).__name__, exc)
        return 1


def _configure_stdio() -> None:
    """Force UTF-8 output so µatm/°C-style characters never crash on
    legacy consoles (e.g. cp936 on Windows)."""
    with contextlib.suppress(AttributeError, ValueError):
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
