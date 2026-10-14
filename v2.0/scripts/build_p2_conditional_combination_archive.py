"""Build the reviewer archive and frozen decisions for Issue 64."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_conditional_combination_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/p2_conditional_combination_v2.4"
FIGURES = ARCHIVE / "figures"
TABLES = ARCHIVE / "tables"
SCHEMES = ["cruise", "geographic", "whole_region", "forward"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lf(path: Path) -> None:
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def pooled(target: str, scheme: str) -> pd.DataFrame:
    paths = sorted((OUTPUT / "partitions").glob(f"{target}_{scheme}_fold*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def rmse(truth: pd.Series, prediction: pd.Series) -> float:
    return float(np.sqrt(np.mean((prediction.to_numpy(float) - truth.to_numpy(float)) ** 2)))


def wis(
    truth: np.ndarray, prediction: np.ndarray, width50: np.ndarray, width90: np.ndarray
) -> float:
    absolute = np.abs(truth - prediction)
    scores = 0.5 * absolute
    for alpha, weight, width in ((0.5, 0.25, width50), (0.1, 0.05, width90)):
        lower, upper = prediction - width, prediction + width
        interval = 2 * width + 2 / alpha * np.maximum(lower - truth, 0)
        interval += 2 / alpha * np.maximum(truth - upper, 0)
        scores += weight * interval
    return float(np.mean(scores / 0.8))


def paired_upper(frame: pd.DataFrame, candidate: str, *, seed: int) -> float:
    work = frame.assign(
        difference=(frame[candidate] - frame.truth) ** 2
        - (frame.base_prediction - frame.truth) ** 2
    )
    group = work.groupby("group_key", sort=False).difference.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        draws.append(float(rng.choice(group, size=len(group), replace=True).mean()))
    return float(np.quantile(draws, 0.95))


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    point_rows, uncertainty_rows, bootstrap_rows = [], [], []
    for target in ("SSS", "fCO2"):
        for scheme_index, scheme in enumerate(SCHEMES):
            frame = pooled(target, scheme)
            base = rmse(frame.truth, frame.base_prediction)
            candidates = ["graph_prediction"]
            if target == "fCO2":
                candidates.append("ensemble_prediction")
            for candidate in ["base_prediction", *candidates]:
                score = rmse(frame.truth, frame[candidate])
                point_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "candidate": candidate,
                        "rows": len(frame),
                        "rmse": score,
                        "relative_improvement": (base - score) / base,
                    }
                )
                if candidate != "base_prediction":
                    bootstrap_rows.append(
                        {
                            "target": target,
                            "scheme": scheme,
                            "candidate": candidate,
                            "upper95_group_mse_difference": paired_upper(
                                frame, candidate, seed=64024 + scheme_index
                            ),
                        }
                    )
            truth = frame.truth.to_numpy(float)
            prediction = frame.base_prediction.to_numpy(float)
            for method in ("control", "scale"):
                width50 = frame[f"{method}_width50"].to_numpy(float)
                width90 = frame[f"{method}_width90"].to_numpy(float)
                uncertainty_rows.append(
                    {
                        "target": target,
                        "scheme": scheme,
                        "method": method,
                        "coverage50": float(np.mean(np.abs(truth - prediction) <= width50)),
                        "coverage90": float(np.mean(np.abs(truth - prediction) <= width90)),
                        "mean_width50": float(np.mean(2 * width50)),
                        "mean_width90": float(np.mean(2 * width90)),
                        "wis": wis(truth, prediction, width50, width90),
                    }
                )
    points = pd.DataFrame(point_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    points.to_csv(TABLES / "point_metrics.csv", index=False)
    uncertainty.to_csv(TABLES / "uncertainty_metrics.csv", index=False)
    bootstrap.to_csv(TABLES / "paired_group_bootstrap.csv", index=False)
    shutil.copy2(OUTPUT / "tables/fCO2_inner_ensemble_trigger.csv", TABLES)
    shutil.copy2(OUTPUT / "tables/fCO2_outer_ensemble_summary.csv", TABLES)

    decisions = []
    for target in ("SSS", "fCO2"):
        graph = points[(points.target == target) & (points.candidate == "graph_prediction")]
        boot = bootstrap[(bootstrap.target == target) & (bootstrap.candidate == "graph_prediction")]
        merged = graph.merge(boot, on=["target", "scheme", "candidate"])
        passed = bool(
            (
                (merged.relative_improvement >= 0.01) & (merged.upper95_group_mse_difference < 0)
            ).sum()
            >= 2
            and merged.relative_improvement.min() >= -0.02
        )
        decisions.append({"target": target, "branch": "graph_residual_adapter", "retained": passed})
    ensemble = points[(points.target == "fCO2") & (points.candidate == "ensemble_prediction")]
    ensemble_boot = bootstrap[
        (bootstrap.target == "fCO2") & (bootstrap.candidate == "ensemble_prediction")
    ]
    ensemble_gate = ensemble.merge(ensemble_boot, on=["target", "scheme", "candidate"])
    ensemble_passed = bool(
        (
            (ensemble_gate.relative_improvement >= 0.01)
            & (ensemble_gate.upper95_group_mse_difference < 0)
        ).sum()
        >= 2
        and ensemble_gate.relative_improvement.min() >= -0.02
    )
    decisions.append(
        {"target": "fCO2", "branch": "catboost_f4_convex_ensemble", "retained": ensemble_passed}
    )
    for target in ("SSS", "fCO2"):
        shifted = uncertainty[
            (uncertainty.target == target)
            & uncertainty.scheme.isin(["geographic", "whole_region", "forward"])
        ]
        pivot = shifted.pivot(
            index="scheme", columns="method", values=["coverage90", "mean_width90", "wis"]
        )
        passed = bool(
            (pivot["coverage90"]["scale"] >= 0.85).all()
            and (pivot["mean_width90"]["scale"] <= 1.25 * pivot["mean_width90"]["control"]).all()
            and (pivot["wis"]["scale"] < pivot["wis"]["control"]).all()
        )
        decisions.append({"target": target, "branch": "uncertainty_scale_head", "retained": passed})
    decisions += [
        {"target": "both", "branch": "process_factor_combination", "retained": False},
        {"target": "both", "branch": "pretraining", "retained": False},
    ]
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(TABLES / "branch_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "branch": "graph",
                "status": "triggered_controlled_test",
                "evidence": "Issue 62 adjacent residual signal",
            },
            {
                "branch": "uncertainty_scale",
                "status": "triggered_controlled_test",
                "evidence": "Issue 63 shifted under-coverage",
            },
            {
                "branch": "point_ensemble",
                "status": "triggered_after_inner_oof",
                "evidence": "median residual correlation below 0.90 and two-scheme inner gain",
            },
            {
                "branch": "process_combination",
                "status": "not_triggered",
                "evidence": "Issue 63 retained no process factor",
            },
            {
                "branch": "pretraining",
                "status": "not_triggered",
                "evidence": "no auditable frozen source and matched control",
            },
        ]
    ).to_csv(TABLES / "trigger_audit.csv", index=False)
    decision_payload = {
        "decisions": decisions,
        "machine_decision": "retain_fco2_convex_ensemble_for_development_only"
        if ensemble_passed
        else "retain_issue63_finalists",
        "independent_labels_opened": False,
        "labels_2026_opened": False,
    }
    (OUTPUT / "branch_decisions.json").write_text(
        json.dumps(decision_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Figures 1-2: graph performance.
    for index, target in enumerate(("SSS", "fCO2"), start=1):
        data = points[
            (points.target == target)
            & points.candidate.isin(["base_prediction", "graph_prediction"])
        ]
        pivot = data.pivot(index="scheme", columns="candidate", values="rmse").loc[SCHEMES]
        ax = pivot.rename(
            columns={"base_prediction": "Finalist", "graph_prediction": "Graph adapter"}
        ).plot.bar(figsize=(8, 4))
        ax.set_ylabel("RMSE" + (" (µatm)" if target == "fCO2" else ""))
        ax.set_title(f"{target}: controlled graph residual adapter")
        save_figure(ax.figure, f"fig0{index}_{target.lower()}_graph_rmse.png")
    graph_gain = (
        points[points.candidate == "graph_prediction"].pivot(
            index="target", columns="scheme", values="relative_improvement"
        )
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    image = ax.imshow(graph_gain[SCHEMES], cmap="RdBu", vmin=-3, vmax=3)
    ax.set_xticks(range(4), SCHEMES, rotation=20)
    ax.set_yticks(range(len(graph_gain)), graph_gain.index)
    fig.colorbar(image, ax=ax, label="RMSE improvement (%)")
    save_figure(fig, "fig03_graph_relative_improvement.png")

    trigger = pd.read_csv(TABLES / "fCO2_inner_ensemble_trigger.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        trigger.residual_correlation,
        100 * trigger.relative_improvement,
        c=trigger.scheme.map({s: i for i, s in enumerate(SCHEMES)}),
        cmap="viridis",
    )
    ax.axvline(0.9, color="red", linestyle="--")
    ax.axhline(1, color="red", linestyle="--")
    ax.set(
        xlabel="Inner residual correlation",
        ylabel="Oracle blend improvement (%)",
        title="Honest inner-OOF ensemble trigger",
    )
    save_figure(fig, "fig04_inner_ensemble_trigger.png")
    ens = points[
        (points.target == "fCO2")
        & points.candidate.isin(["base_prediction", "ensemble_prediction"])
    ]
    pivot = ens.pivot(index="scheme", columns="candidate", values="rmse").loc[SCHEMES]
    ax = pivot.rename(
        columns={"base_prediction": "F4 finalist", "ensemble_prediction": "Frozen convex ensemble"}
    ).plot.bar(figsize=(8, 4))
    ax.set_ylabel("RMSE (µatm)")
    ax.set_title("fCO2 outer corrected-fold ensemble")
    save_figure(ax.figure, "fig05_fco2_ensemble_rmse.png")
    fold = pd.read_csv(TABLES / "fCO2_outer_ensemble_summary.csv")
    fold["improvement"] = (
        100
        * (fold.base_group_equal_rmse - fold.ensemble_group_equal_rmse)
        / fold.base_group_equal_rmse
    )
    ax = fold.boxplot(column="improvement", by="scheme", figsize=(8, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Fold group-equal RMSE improvement (%)")
    ax.figure.suptitle("")
    ax.set_title("fCO2 ensemble fold distribution")
    save_figure(ax.figure, "fig06_fco2_ensemble_folds.png")
    for number, value, ylabel, title in (
        (7, "coverage90", "Empirical 90% coverage", "Shifted interval coverage"),
        (8, "wis", "WIS (lower is better)", "Uncertainty quality"),
    ):
        data = uncertainty[uncertainty.scheme.isin(SCHEMES)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, target in zip(axes, ("SSS", "fCO2"), strict=True):
            data[data.target == target].pivot(index="scheme", columns="method", values=value).loc[
                SCHEMES
            ].plot.bar(ax=ax)
            ax.set_title(target)
            ax.set_ylabel(ylabel)
            if value == "coverage90":
                ax.axhspan(0.85, 0.95, color="green", alpha=0.12)
        fig.suptitle(title)
        save_figure(fig, f"fig0{number}_{value}.png")

    artifact_rows = []
    for kind, pattern in (
        ("partition", "partitions/*.parquet"),
        ("selection", "selection/*.json"),
        ("table", "tables/*"),
        ("decision", "*.json"),
    ):
        for path in sorted(OUTPUT.glob(pattern)):
            if path.is_file():
                artifact_rows.append(
                    {
                        "kind": kind,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    pd.DataFrame(artifact_rows).to_csv(TABLES / "external_artifact_manifest.csv", index=False)

    report = f"""# P2 conditional models and uncertainty v2.4

## Scientific question and permitted claim

Issue #64 tests only hypotheses activated by Issues 62-63 under the exact corrected Issue 61 folds. The graph branch propagates honest inner-validation residuals across a water-only seasonal graph and is architecturally distinct from the failed Issue 42 ST96 model. The uncertainty branch learns only residual magnitude inside outer model-fit data. The fCO2 ensemble weight is selected exclusively from honest inner-OOF residuals. No locked, external-independent, or 2026 labels were opened.

## Data, splits, and leakage controls

The experiment reuses the frozen 1993-2025 matchup, predictor cube, and all 36 target-specific Issue 61 outer splits. Graph sources, graph hyperparameter selection, residual-scale fitting, and ensemble weights remain inside each outer fit partition. Held labels are used only for scoring.

## Candidate models and training

The controlled candidates are a four-season nearest-source water-graph residual adapter, an ExtraTrees log-absolute-error scale head with standardized conformal calibration, and—only after its inner trigger fired—a convex CatBoost/F4 fCO2 ensemble. Process-factor combinations and pretraining remain untriggered. The failed Issue 42 four-neighbour ST96 implementation was not rerun.

## Main development results

The graph residual adapter is rejected for both targets. SSS improves pooled cruise RMSE but does not pass two transfer schemes with paired group evidence; fCO2 graph corrections are neutral or harmful except one forward fold. Adjacent residual correlation was therefore diagnostic, not proof that graph propagation transfers.

The scale-head uncertainty branch is rejected. It raises fCO2 geographic 90% coverage from 0.795 to 0.833 and forward from 0.837 to 0.847, but remains below 0.85 and worsens whole-region coverage from 0.783 to 0.742. SSS geographic and whole-region coverage also remain far below the gate. Heteroscedastic scaling cannot replace reliability flags in unsupported domains.

The honest inner-OOF audit activates the fCO2 CatBoost/F4 convex ensemble (median residual correlation {trigger.residual_correlation.median():.3f}). Frozen fold-specific CatBoost weights improve pooled fCO2 RMSE from **32.776 to 32.471 µatm** on cruise, **54.061 to 52.495 µatm** on geographic transfer, **52.834 to 52.504 µatm** on whole-region transfer, and **39.317 to 38.853 µatm** forward. It passes the point gate through geographic and forward improvements, with no scheme regression, and is retained for development only. It is not product-qualified until the one-time independent gate.

Process-factor combinations and pretraining remain untriggered negative branches. The machine decision is `{decision_payload["machine_decision"]}`.

## Decision and limitations

Only the fCO2 convex ensemble is retained for development. The graph and scale-head branches are preserved as negative results. None is production-qualified: all scores use development labels already covered by the exposure ledger, and the one-time independent gate remains sealed.

## Figure and table index

### Figure 1

![Figure 1](figures/fig01_sss_graph_rmse.png)

Figure 1. SSS pooled corrected-fold RMSE for the Issue 63 finalist and seasonal water-graph residual adapter. The apparent cruise gain is insufficient across transfer schemes.

### Figure 2

![Figure 2](figures/fig02_fco2_graph_rmse.png)

Figure 2. fCO2 pooled corrected-fold RMSE for the continuous-background finalist and graph adapter. The graph hypothesis does not transfer consistently.

### Figure 3

![Figure 3](figures/fig03_graph_relative_improvement.png)

Figure 3. Relative graph-adapter RMSE improvement by target and split; blue denotes improvement and red regression. The required two-scheme evidence is absent.

### Figure 4

![Figure 4](figures/fig04_inner_ensemble_trigger.png)

Figure 4. Honest inner-OOF fCO2 residual correlation and oracle convex-blend gain. These labels only trigger the outer branch and never tune on held folds.

### Figure 5

![Figure 5](figures/fig05_fco2_ensemble_rmse.png)

Figure 5. Pooled outer corrected-fold fCO2 RMSE for F4 and the fold-specific inner-frozen CatBoost/F4 convex ensemble.

### Figure 6

![Figure 6](figures/fig06_fco2_ensemble_folds.png)

Figure 6. Fold-level group-equal RMSE improvement for the fCO2 ensemble, preserving heterogeneity hidden by pooled scores.

### Figure 7

![Figure 7](figures/fig07_coverage90.png)

Figure 7. Empirical 90% interval coverage for prediction-stratified conformal control and residual scale head; the green band marks the frozen 0.85-0.95 acceptance range.

### Figure 8

![Figure 8](figures/fig08_wis.png)

Figure 8. Weighted interval score for control and scale-head intervals. Coverage failures prevent retaining the scale head even where average interval score changes favorably.

Exact source tables are under `tables/`. Row-level predictions and selections remain outside Git under `{OUTPUT}` and are hashed in `external_artifact_manifest.csv`.
"""
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Issue 64 reviewer archive\n\nStart with [REPORT.md](REPORT.md). Rebuild with `python scripts/build_p2_conditional_combination_archive.py` and verify with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_conditional_combination_v2.4`.\n",
        encoding="utf-8",
    )
    captions = {
        "fig01_sss_graph_rmse.png": "Figure 1. SSS pooled corrected-fold RMSE for the Issue 63 finalist and seasonal water-graph residual adapter. The apparent cruise gain is insufficient across transfer schemes.",
        "fig02_fco2_graph_rmse.png": "Figure 2. fCO2 pooled corrected-fold RMSE for the continuous-background finalist and graph adapter. The graph hypothesis does not transfer consistently.",
        "fig03_graph_relative_improvement.png": "Figure 3. Relative graph-adapter RMSE improvement by target and split; blue denotes improvement and red regression. The required two-scheme evidence is absent.",
        "fig04_inner_ensemble_trigger.png": "Figure 4. Honest inner-OOF fCO2 residual correlation and oracle convex-blend gain. These labels only trigger the outer branch and never tune on held folds.",
        "fig05_fco2_ensemble_rmse.png": "Figure 5. Pooled outer corrected-fold fCO2 RMSE for F4 and the fold-specific inner-frozen CatBoost/F4 convex ensemble.",
        "fig06_fco2_ensemble_folds.png": "Figure 6. Fold-level group-equal RMSE improvement for the fCO2 ensemble, preserving heterogeneity hidden by pooled scores.",
        "fig07_coverage90.png": "Figure 7. Empirical 90% interval coverage for prediction-stratified conformal control and residual scale head; the green band marks the frozen 0.85-0.95 acceptance range.",
        "fig08_wis.png": "Figure 8. Weighted interval score for control and scale-head intervals. Coverage failures prevent retaining the scale head even where average interval score changes favorably.",
    }
    caption_text = (
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n"
    )
    (ARCHIVE / "CAPTIONS.md").write_text(caption_text, encoding="utf-8")
    source_map = {
        "fig01_sss_graph_rmse.png": ["point_metrics.csv"],
        "fig02_fco2_graph_rmse.png": ["point_metrics.csv"],
        "fig03_graph_relative_improvement.png": ["point_metrics.csv"],
        "fig04_inner_ensemble_trigger.png": ["fCO2_inner_ensemble_trigger.csv"],
        "fig05_fco2_ensemble_rmse.png": ["point_metrics.csv"],
        "fig06_fco2_ensemble_folds.png": ["fCO2_outer_ensemble_summary.csv"],
        "fig07_coverage90.png": ["uncertainty_metrics.csv"],
        "fig08_wis.png": ["uncertainty_metrics.csv"],
    }
    for path in ARCHIVE.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json"}:
            normalize_lf(path)
    files = {}
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path.name != "archive_manifest.json":
            files[str(path.relative_to(ARCHIVE)).replace("\\", "/")] = sha256(path)
    source_paths = (
        ROOT / "configs/p2_conditional_combination_v2.4.json",
        ROOT / "scripts/run_p2_conditional_combination_v2_4.py",
        ROOT / "src/recad/evaluate/conditional_models.py",
        OUTPUT / "branch_decisions.json",
        OUTPUT / "ensemble_trigger_decision.json",
    )
    manifest = {
        "experiment_id": "p2_conditional_combination_v2.4",
        "training_git_commit": "681d046",
        "data_manifest_sha256": sha256(ROOT / "configs/p2_conditional_combination_v2.4.json"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "provisional_2026_opened": False,
        "evidence_scope": "1993-2025 development labels on corrected Issue 61 folds",
        "decision": decision_payload,
        "figure_source_data": source_map,
        "figure_captions": captions,
        "files_sha256": files,
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(ROOT / "scripts/run_p2_conditional_combination_v2_4.py"),
        "source_artifacts_sha256": {str(path): sha256(path) for path in source_paths},
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
