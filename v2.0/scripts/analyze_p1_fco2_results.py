"""Create the P1.2 gate and fCO2-specific support-distance audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256

ROOT = Path(__file__).resolve().parents[1]


def xyz(frame: pd.DataFrame) -> np.ndarray:
    lat = np.deg2rad(frame.latitude.to_numpy(float))
    lon = np.deg2rad(frame.longitude.to_numpy(float))
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def nearest_training_km(train: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    sites = train[["latitude", "longitude"]].drop_duplicates()
    chord, _ = cKDTree(xyz(sites)).query(xyz(query), k=1, workers=-1)
    return 6371.0088 * 2 * np.arcsin(np.clip(chord / 2, 0, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "outputs/experiments/p1_fco2_viability_v2.2")
    args = parser.parse_args()
    manifest = FrozenManifest.load(ROOT / "configs/frozen/data_manifest_v2.2.json")
    gateway = P1DataGateway(manifest)
    train = gateway.load_labels("fco2", Purpose.TRAIN)
    development = gateway.load_labels("fco2", Purpose.SELECTION).reset_index(drop=True)
    development["record_id"] = (
        development.group_key.astype(str) + ":" + development.year.astype(str) + ":" +
        development.month.astype(str) + ":" + development.latitude.round(4).astype(str) + ":" +
        development.longitude.round(4).astype(str)
    )
    development["fco2_support_distance_km"] = nearest_training_km(train, development)
    predictions = pd.read_parquet(args.experiment / "candidate_predictions.parquet")
    summary = pd.read_csv(args.experiment / "summary.csv")
    metrics = pd.read_csv(args.experiment / "metrics_by_seed.csv")
    forward = pd.read_csv(args.experiment / "forward_metrics.csv")
    strata = pd.read_csv(args.experiment / "stratified_metrics.csv")
    selection = json.loads((args.experiment / "selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]
    ensemble = predictions.loc[predictions.model.eq(selected)].groupby("record_id").agg(
        truth=("truth", "first"), background=("background", "first"), prediction=("prediction", "mean")
    ).reset_index()
    ensemble = ensemble.merge(
        development[["record_id", "fco2_support_distance_km"]], on="record_id", how="left", validate="one_to_one"
    )
    bins = [0, 1, 5, 25, 50, 100, 250, np.inf]
    ensemble["support_bin_km"] = pd.cut(ensemble.fco2_support_distance_km, bins, right=False)
    rows = []
    for group, part in ensemble.groupby("support_bin_km", observed=True):
        model_rmse = float(np.sqrt(np.mean((part.prediction - part.truth) ** 2)))
        background_rmse = float(np.sqrt(np.mean((part.background - part.truth) ** 2)))
        rows.append({"support_bin_km": str(group), "n": len(part),
                     "model_rmse": model_rmse, "background_rmse": background_rmse,
                     "skill_vs_background": 1 - model_rmse**2 / background_rmse**2})
    support = pd.DataFrame(rows)
    support.to_csv(args.experiment / "fco2_support_distance_metrics.csv", index=False)

    base = summary.loc[summary.model.eq("seasonal_climatology")].iloc[0]
    candidate = summary.loc[summary.model.eq(selected)].iloc[0]
    lme = strata.loc[(strata.model.eq(selected)) & strata.stratum.eq("lme")].groupby("group").agg(
        n=("n", "max"), skill=("skill_vs_background", "mean")).reset_index()
    eligible_lme = lme.loc[lme.n >= 100]
    fco2_bands = strata.loc[(strata.model.eq(selected)) & strata.stratum.eq("fco2_band")]
    selected_seed_metrics = metrics.loc[metrics.model.eq(selected)]
    forward_candidate = forward.loc[forward.model.eq(selected)]
    checks = {
        "cruise_equal_improvement_ge_5pct": bool(1 - candidate.cruise_equal_rmse_mean / base.cruise_equal_rmse_mean >= 0.05),
        "lme_macro_improvement_ge_5pct": bool(1 - candidate.lme_macro_rmse_mean / base.lme_macro_rmse_mean >= 0.05),
        "positive_skill_lme_fraction_ge_70pct": bool(float((eligible_lme.skill > 0).mean()) >= 0.70),
        "worst_lme_degradation_le_10pct": bool(candidate.worst_lme_rmse_mean / base.worst_lme_rmse_mean - 1 <= 0.10),
        "forward_skill_positive_all_seeds": bool((forward_candidate.skill_vs_background > 0).all()),
        "all_fco2_bands_skill_positive": bool((fco2_bands.skill_vs_background > 0).all()),
        "primary_skill_positive_all_seeds": bool((selected_seed_metrics.skill_vs_background > 0).all()),
        "support_bin_skill_positive": bool((support.skill_vs_background > 0).all()),
    }
    gate = {
        "selected_model": selected,
        "development_gate_passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "pooled_rmse": float(candidate.pooled_rmse_mean),
            "cruise_equal_rmse": float(candidate.cruise_equal_rmse_mean),
            "lme_macro_rmse": float(candidate.lme_macro_rmse_mean),
            "worst_lme_rmse": float(candidate.worst_lme_rmse_mean),
            "cruise_equal_improvement": float(1 - candidate.cruise_equal_rmse_mean / base.cruise_equal_rmse_mean),
            "lme_macro_improvement": float(1 - candidate.lme_macro_rmse_mean / base.lme_macro_rmse_mean),
            "worst_lme_improvement": float(1 - candidate.worst_lme_rmse_mean / base.worst_lme_rmse_mean),
            "positive_skill_lme_fraction": float((eligible_lme.skill > 0).mean()),
            "forward_pooled_skill_mean": float(forward_candidate.skill_vs_background.mean()),
        },
        "decision": "nominate_for_issue_11_locked_gate",
        "scope": "North-American-adjacent frozen cache; no global claim",
        "locked_test_opened": False,
        "external_opened": False,
    }
    (args.experiment / "development_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    figure_dir = args.experiment / "figures"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(support.support_bin_km, support.skill_vs_background, color="#2A9D8F")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill versus seasonal climatology")
    ax.set_xlabel("Nearest fCO2 training location (km)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figure_dir / "fco2_support_distance_skill.png", dpi=180)
    plt.close(fig)
    hashes = {str(path.relative_to(args.experiment)): sha256(path)
              for path in args.experiment.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"}
    (args.experiment / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
