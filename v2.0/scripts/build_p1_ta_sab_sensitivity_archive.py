"""Build the reviewer-ready SAB TA latitude-boundary sensitivity archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_ta_sab_latitude_sensitivity_v2.2"
VARIANTS = {"26": 26.0, "27": 27.0, "28p45": 28.45, "30p5": 30.5}
MODEL_LABELS = {
    "carter_esper_prior": "Carter/ESPER",
    "carter_esper_oracle_insitu_sss": "Carter oracle",
    "global_ta_sss": "SAB TA-SSS",
    "lme_ta_sss": "Latitude-band TA-SSS",
    "hierarchical_ta_sss": "Hierarchical TA-SSS",
    "hierarchical_residual": "Hierarchical residual MLP",
    "carter_region_season_corrected": "Carter + band/season correction",
}
COLORS = {26.0: "#0072B2", 27.0: "#009E73", 28.45: "#E69F00", 30.5: "#D55E00"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def skill(model_rmse: float, baseline_rmse: float) -> float:
    return 1.0 - (model_rmse**2 / baseline_rmse**2)


def load_inputs(source: Path) -> dict[str, object]:
    rows, cv_rows, strata, scopes, gates, uncertainties, availability = [], [], [], [], [], [], []
    predictions: dict[float, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    protocols: dict[float, dict] = {}
    for tag, boundary in VARIANTS.items():
        folder = source / tag
        protocol = read_json(folder / "protocol.json")
        gate = read_json(folder / "development_gate.json")
        selection = read_json(folder / "selection.json")
        protocols[boundary] = protocol
        summary = pd.read_csv(folder / "summary.csv")
        summary.insert(0, "boundary_deg_n", boundary)
        summary["model_label"] = summary.model.map(MODEL_LABELS)
        rows.append(summary)
        cv = (
            pd.read_csv(folder / "cv_metrics.csv")
            .groupby("model", as_index=False)
            .agg(
                folds=("fold", "nunique"),
                pooled_rmse_mean=("pooled_rmse", "mean"),
                pooled_rmse_sd=("pooled_rmse", "std"),
                cruise_equal_rmse_mean=("cruise_equal_rmse", "mean"),
                band_macro_rmse_mean=("lme_macro_rmse", "mean"),
                band_macro_rmse_sd=("lme_macro_rmse", "std"),
            )
        )
        cv.insert(0, "boundary_deg_n", boundary)
        cv["model_label"] = cv.model.map(MODEL_LABELS)
        cv_rows.append(cv)
        selected = gate["selected_model"]
        strat = pd.read_csv(folder / "stratified_metrics.csv")
        selected_strata = (
            strat.loc[strat.model.eq(selected)]
            .groupby(["stratum", "group"], as_index=False)
            .agg(
                n=("n", "max"),
                cruises=("cruises", "max"),
                model_rmse=("model_rmse", "mean"),
                carter_rmse=("carter_rmse", "mean"),
                skill_vs_carter=("skill_vs_carter", "mean"),
            )
        )
        selected_strata.insert(0, "boundary_deg_n", boundary)
        strata.append(selected_strata)
        scopes.append(
            {
                "boundary_deg_n": boundary,
                "train_rows": protocol["train_rows"],
                "train_cruises": protocol["train_cruises"],
                "development_rows": protocol["development_rows"],
                "development_cruises": protocol["development_cruises"],
                "populated_cv_folds": len(protocol["populated_cv_folds"]),
                "safe_forward_development_rows": protocol["safe_forward_development_rows"],
                "selected_model": selected,
                "decision": gate["decision"],
            }
        )
        gates.extend(
            {"boundary_deg_n": boundary, "check": name, "passed": passed}
            for name, passed in gate["checks"].items()
        )
        uncertainties.append(
            {
                "boundary_deg_n": boundary,
                "nominal_coverage": 0.90,
                "development_coverage": selection["development_coverage_90"],
                "oof_absolute_error_q90": selection["conformal_absolute_error_q90"],
            }
        )
        availability.append(
            {
                "boundary_deg_n": boundary,
                "a3_nonlinear_triggered": selection["a3_stopping_rule_triggered"],
                "selected_model": selected,
                "candidate_count": int(summary.model.nunique()),
            }
        )
        predictions[boundary] = pd.read_parquet(folder / "selected_development_predictions.parquet")
        for name in (
            "summary.csv",
            "cv_metrics.csv",
            "stratified_metrics.csv",
            "development_gate.json",
            "selection.json",
            "protocol.json",
            "selected_development_predictions.parquet",
        ):
            source_hashes[f"{tag}/{name}"] = sha256(folder / name)
    return {
        "summary": pd.concat(rows, ignore_index=True),
        "cv": pd.concat(cv_rows, ignore_index=True),
        "strata": pd.concat(strata, ignore_index=True),
        "scope": pd.DataFrame(scopes),
        "gates": pd.DataFrame(gates),
        "uncertainty": pd.DataFrame(uncertainties),
        "availability": pd.DataFrame(availability),
        "predictions": predictions,
        "source_hashes": source_hashes,
        "protocols": protocols,
    }


def selected_tables(data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = data["summary"]
    cv = data["cv"]
    scope = data["scope"]
    selected_rows, selected_cv = [], []
    for row in scope.itertuples():
        part = summary[
            (summary.boundary_deg_n == row.boundary_deg_n) & (summary.model == row.selected_model)
        ]
        carter = summary[
            (summary.boundary_deg_n == row.boundary_deg_n) & (summary.model == "carter_esper_prior")
        ]
        out = part.iloc[0].to_dict()
        out["skill_vs_carter_pooled"] = skill(
            out["pooled_rmse_mean"], carter.iloc[0].pooled_rmse_mean
        )
        out["skill_vs_carter_cruise_equal"] = skill(
            out["cruise_equal_rmse_mean"], carter.iloc[0].cruise_equal_rmse_mean
        )
        out["skill_vs_carter_band_macro"] = skill(
            out["lme_macro_rmse_mean"], carter.iloc[0].lme_macro_rmse_mean
        )
        selected_rows.append(out)
        q = (
            cv[(cv.boundary_deg_n == row.boundary_deg_n) & (cv.model == row.selected_model)]
            .iloc[0]
            .to_dict()
        )
        c = cv[(cv.boundary_deg_n == row.boundary_deg_n) & (cv.model == "carter_esper_prior")].iloc[
            0
        ]
        q["skill_vs_carter_band_macro"] = skill(q["band_macro_rmse_mean"], c.band_macro_rmse_mean)
        selected_cv.append(q)
    selected = pd.DataFrame(selected_rows).sort_values("boundary_deg_n")
    selected_cv_frame = pd.DataFrame(selected_cv).sort_values("boundary_deg_n")
    contrasts = []
    for added_boundary, restricted_boundary, added_band in (
        (26.0, 27.0, "26-27°N"),
        (27.0, 28.45, "27-28.45°N"),
        (28.45, 30.5, "28.45-30.5°N"),
    ):
        a = selected.set_index("boundary_deg_n").loc[added_boundary]
        r = selected.set_index("boundary_deg_n").loc[restricted_boundary]
        acv = selected_cv_frame.set_index("boundary_deg_n").loc[added_boundary]
        rcv = selected_cv_frame.set_index("boundary_deg_n").loc[restricted_boundary]
        dev_change = a.cruise_equal_rmse_mean / r.cruise_equal_rmse_mean - 1
        cv_change = acv.band_macro_rmse_mean / rcv.band_macro_rmse_mean - 1
        contrasts.append(
            {
                "added_band": added_band,
                "from_boundary_deg_n": restricted_boundary,
                "to_boundary_deg_n": added_boundary,
                "development_cruise_equal_change_fraction": dev_change,
                "cv_band_macro_change_fraction": cv_change,
                "harmful_by_preregistered_rule": bool(dev_change > 0.05 and cv_change > 0.0),
                "model_family_changed": a.model != r.model,
            }
        )
    return selected, selected_cv_frame, pd.DataFrame(contrasts)


def line_figure(
    table: pd.DataFrame, columns: list[tuple[str, str]], path: Path, ylabel: str
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for column, label in columns:
        ax.plot(table.boundary_deg_n, table[column], marker="o", linewidth=2, label=label)
    ax.set_xlabel("SAB southern boundary (°N)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(table.boundary_deg_n)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/experiments" / EXPERIMENT_ID)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "docs/experiment_archive" / EXPERIMENT_ID
    )
    args = parser.parse_args()
    figures, tables = args.archive / "figures", args.archive / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.source)
    selected, selected_cv, contrasts = selected_tables(data)
    table_map = {
        "table01_all_development_models.csv": data["summary"],
        "table02_selected_development_metrics.csv": selected,
        "table03_grouped_cv_metrics.csv": selected_cv,
        "table04_nested_boundary_contrasts.csv": contrasts,
        "table05_latitude_band_skill.csv": data["strata"].query("stratum == 'lme'"),
        "table06_support_distance_skill.csv": data["strata"].query("stratum == 'support_distance'"),
        "table07_salinity_band_skill.csv": data["strata"].query("stratum == 'salinity_band'"),
        "table08_uncertainty.csv": data["uncertainty"],
        "table09_gate_checks.csv": data["gates"],
        "table10_data_scope_and_model_availability.csv": data["scope"].merge(data["availability"]),
    }
    for name, frame in table_map.items():
        write_csv(frame, tables / name)

    line_figure(
        selected,
        [
            ("pooled_rmse_mean", "pooled"),
            ("cruise_equal_rmse_mean", "cruise-equal"),
            ("lme_macro_rmse_mean", "latitude-band macro"),
        ],
        figures / "fig01_development_rmse.png",
        "TA RMSE (µmol kg⁻¹)",
    )
    line_figure(
        selected_cv,
        [
            ("pooled_rmse_mean", "pooled"),
            ("cruise_equal_rmse_mean", "cruise-equal"),
            ("band_macro_rmse_mean", "latitude-band macro"),
        ],
        figures / "fig02_grouped_cv_rmse.png",
        "Five-fold grouped-CV TA RMSE (µmol kg⁻¹)",
    )
    line_figure(
        selected,
        [
            ("skill_vs_carter_pooled", "pooled"),
            ("skill_vs_carter_cruise_equal", "cruise-equal"),
            ("skill_vs_carter_band_macro", "latitude-band macro"),
        ],
        figures / "fig03_skill_vs_carter.png",
        "Skill versus Carter/ESPER",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(contrasts))
    ax.bar(
        x - 0.18,
        100 * contrasts.development_cruise_equal_change_fraction,
        0.36,
        label="development cruise-equal",
    )
    ax.bar(
        x + 0.18, 100 * contrasts.cv_band_macro_change_fraction, 0.36, label="grouped-CV band macro"
    )
    ax.axhline(0, color="black", lw=1)
    ax.axhline(5, color="grey", lw=1, ls="--")
    ax.set_xticks(x, contrasts.added_band)
    ax.set_ylabel("RMSE change after adding band (%)")
    ax.legend()
    fig.tight_layout()
    save(fig, figures / "fig04_nested_contrasts.png")

    band = table_map["table05_latitude_band_skill.csv"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for boundary, part in band.groupby("boundary_deg_n"):
        ax.plot(part.group.astype(str), part.skill_vs_carter, marker="o", label=f"≥{boundary:g}°N")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill versus Carter/ESPER")
    ax.set_xlabel("Fixed latitude band ID (1=26-27, …, 5=33-35.3°N)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, figures / "fig05_latitude_band_skill.png")

    support = table_map["table06_support_distance_skill.csv"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for boundary, part in support.groupby("boundary_deg_n"):
        ax.plot(part.group.astype(str), part.skill_vs_carter, marker="o", label=f"≥{boundary:g}°N")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill versus Carter/ESPER")
    ax.set_xlabel("Distance-to-training-support bin")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, figures / "fig06_support_distance_skill.png")

    salinity = table_map["table07_salinity_band_skill.csv"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for boundary, part in salinity.groupby("boundary_deg_n"):
        ax.plot(part.group.astype(str), part.skill_vs_carter, marker="o", label=f"≥{boundary:g}°N")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Skill versus Carter/ESPER")
    ax.set_xlabel("Observed salinity band (diagnostic only)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, figures / "fig07_salinity_band_skill.png")

    uncertainty = data["uncertainty"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(uncertainty.boundary_deg_n, uncertainty.development_coverage, marker="o")
    axes[0].axhspan(0.85, 0.95, alpha=0.15, color="#009E73")
    axes[0].axhline(0.90, color="black", ls="--")
    axes[0].set_ylabel("90% interval empirical coverage")
    axes[1].plot(
        uncertainty.boundary_deg_n, uncertainty.oof_absolute_error_q90, marker="o", color="#D55E00"
    )
    axes[1].set_ylabel("OOF absolute-error q90 (µmol kg⁻¹)")
    for ax in axes:
        ax.set_xlabel("SAB southern boundary (°N)")
        ax.set_xticks(uncertainty.boundary_deg_n)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, figures / "fig08_uncertainty.png")

    scope = table_map["table10_data_scope_and_model_availability.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(scope.boundary_deg_n, scope.train_rows, marker="o", label="train rows")
    axes[0].plot(scope.boundary_deg_n, scope.development_rows, marker="o", label="development rows")
    axes[0].legend()
    axes[0].set_ylabel("Surface TA observations")
    axes[1].plot(scope.boundary_deg_n, scope.train_cruises, marker="o", label="train cruises")
    axes[1].plot(
        scope.boundary_deg_n, scope.development_cruises, marker="o", label="development cruises"
    )
    axes[1].legend()
    axes[1].set_ylabel("Cruises")
    for ax in axes:
        ax.set_xlabel("SAB southern boundary (°N)")
        ax.set_xticks(scope.boundary_deg_n)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, figures / "fig09_data_support.png")

    gate = (
        data["gates"].pivot(index="check", columns="boundary_deg_n", values="passed").astype(float)
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    image = ax.imshow(gate, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(gate.columns)), [f"≥{x:g}°N" for x in gate.columns])
    ax.set_yticks(range(len(gate.index)), [x.replace("_", " ") for x in gate.index])
    for y in range(len(gate.index)):
        for x in range(len(gate.columns)):
            ax.text(
                x, y, "PASS" if gate.iloc[y, x] else "FAIL", ha="center", va="center", fontsize=8
            )
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    save(fig, figures / "fig10_gate_checks.png")

    captions = {
        "fig01_development_rmse.png": "Figure 1. Selected-model development RMSE across nested SAB southern boundaries. The apparent improvement at 30.5°N coincides with nonlinear-model activation and removal of the 28.45-30.5°N observations, so it is not a clean water-mass effect.",
        "fig02_grouped_cv_rmse.png": "Figure 2. Frozen cruise-grouped cross-validation RMSE across boundaries. The 30.5°N variant has four populated folds because frozen fold 0 contains no eligible cruise; folds were not reassigned.",
        "fig03_skill_vs_carter.png": "Figure 3. Development skill of each selected model relative to Carter/ESPER, where positive values favor the model. Pooled and cruise-equal skill are negative through 28.45°N despite positive latitude-band macro skill.",
        "fig04_nested_contrasts.png": "Figure 4. Percent RMSE change caused by adding each southern latitude band to the next restricted dataset. The preregistered harmful-band rule requires development cruise-equal degradation above 5% and grouped-CV band-macro degradation; only 28.45-30.5°N meets both, with a model-family change.",
        "fig05_latitude_band_skill.png": "Figure 5. Selected-model skill versus Carter/ESPER within fixed latitude bands. Sparse bands contain only one or two cruises, and the central 30.5-33°N band dominates the development sample.",
        "fig06_support_distance_skill.png": "Figure 6. Skill versus Carter/ESPER by distance from training TA support. Negative worst-bin skill remains for every boundary, showing that boundary restriction does not solve extrapolation risk.",
        "fig07_salinity_band_skill.png": "Figure 7. Skill versus Carter/ESPER by observed-salinity band, used only for diagnosis. Small low-salinity cells are unstable and must not be treated as evidence for operational coastal-estuarine performance.",
        "fig08_uncertainty.png": "Figure 8. Conformal interval coverage and out-of-fold absolute-error 90th percentile. The 30.5°N variant undercovers at 83.4%, outside the frozen 85-95% acceptance range.",
        "fig09_data_support.png": "Figure 9. Observation and cruise support retained by each nested boundary. All development variants contain only four cruises, while the 30.5°N training set has eight cruises and one empty frozen CV fold.",
        "fig10_gate_checks.png": "Figure 10. Frozen development-gate checks by boundary. No variant passes the full gate; safe-forward evaluation is unavailable because no SAB development record satisfies the frozen forward-time intersection.",
    }
    report = build_report(selected, selected_cv, contrasts, scope, captions)
    (args.archive / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    caption_text = "# Figure captions\n\n" + "".join(
        f"## {name}\n\n{caption}\n\n" for name, caption in captions.items()
    )
    (args.archive / "CAPTIONS.md").write_text(
        caption_text.rstrip() + "\n", encoding="utf-8", newline="\n"
    )
    (args.archive / "README.md").write_text(
        "# P1.3c SAB TA latitude sensitivity reviewer archive\n\nDevelopment and grouped-CV sensitivity evidence for Issue #17. Locked-test and external-independent labels remain sealed.\n",
        encoding="utf-8",
        newline="\n",
    )
    figure_sources = {
        "fig01_development_rmse.png": ["table02_selected_development_metrics.csv"],
        "fig02_grouped_cv_rmse.png": ["table03_grouped_cv_metrics.csv"],
        "fig03_skill_vs_carter.png": ["table02_selected_development_metrics.csv"],
        "fig04_nested_contrasts.png": ["table04_nested_boundary_contrasts.csv"],
        "fig05_latitude_band_skill.png": ["table05_latitude_band_skill.csv"],
        "fig06_support_distance_skill.png": ["table06_support_distance_skill.csv"],
        "fig07_salinity_band_skill.png": ["table07_salinity_band_skill.csv"],
        "fig08_uncertainty.png": ["table08_uncertainty.csv"],
        "fig09_data_support.png": ["table10_data_scope_and_model_availability.csv"],
        "fig10_gate_checks.png": ["table09_gate_checks.csv"],
    }
    tracked = [
        args.archive / "README.md",
        args.archive / "REPORT.md",
        args.archive / "CAPTIONS.md",
        *sorted(figures.glob("*.png")),
        *sorted(tables.glob("*.csv")),
    ]
    protocol = data["protocols"][26.0]
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": protocol["run_git_commit"],
        "preregistration_git_commit": protocol["preregistration_commit"],
        "data_manifest_sha256": protocol["manifest_validation"]["manifest_sha256"],
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "Nested SAB surface-TA development and frozen cruise-grouped-CV latitude sensitivity only",
        "decision": "27N hypothesis not supported; 28.45-30.5N is hypothesis-generating because model availability changes",
        "archive_builder": "scripts/build_p1_ta_sab_sensitivity_archive.py",
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(ROOT / "scripts/run_p1_ta_viability.py"),
        "source_artifacts_sha256": data["source_hashes"],
        "figure_source_data": figure_sources,
        "figure_captions": captions,
        "files_sha256": {
            str(p.relative_to(args.archive)).replace("\\", "/"): sha256(p) for p in tracked
        },
    }
    (args.archive / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "archive": str(args.archive),
                "figures": len(captions),
                "tables": len(table_map),
                "decision": manifest["decision"],
            },
            indent=2,
        )
    )
    return 0


def build_report(
    selected: pd.DataFrame,
    selected_cv: pd.DataFrame,
    contrasts: pd.DataFrame,
    scope: pd.DataFrame,
    captions: dict[str, str],
) -> str:
    row = selected.set_index("boundary_deg_n")
    cv = selected_cv.set_index("boundary_deg_n")
    contrast_lines = "\n".join(
        f"- Adding {r.added_band}: development cruise-equal RMSE {100 * r.development_cruise_equal_change_fraction:+.1f}%; grouped-CV band-macro RMSE {100 * r.cv_band_macro_change_fraction:+.1f}%; preregistered harmful={str(r.harmful_by_preregistered_rule).lower()}; model family changed={str(r.model_family_changed).lower()}."
        for r in contrasts.itertuples()
    )
    figure_blocks = "\n\n".join(
        f"![{name}](figures/{name})\n\n{caption}" for name, caption in captions.items()
    )
    return f"""# SAB TA latitude-boundary sensitivity (P1.3c)

## Scientific question and permitted claim

This experiment tests whether adding southern SAB surface observations degrades TA reconstruction, especially the prior hypothesis that observations south of 27°N mix a different water mass into the relation. It supports only development/grouped-CV sensitivity claims. It does not validate a TA product or identify a causal water-mass boundary.

## Data, splits, and leakage controls

The four nested masks retain LME 6 observations from 26.0, 27.0, 28.45, or 30.5°N to 35.3°N. Only primary, TA-QC=2 observations at 0-5 m are used. Frozen cruise assignments, duplicate groups, CV folds, preprocessing, and evaluation bands are unchanged. The locked test and external-independent labels were not opened. Each development set contains {int(scope.development_cruises.min())} cruises; eligible development rows range from {int(scope.development_rows.min())} to {int(scope.development_rows.max())}. The 30.5°N mask has four populated frozen CV folds because fold 0 has no eligible cruise; no samples were reassigned.

## Candidate models and training

Each boundary independently refits the frozen P1.3 candidates with 4,000 optimizer steps and seeds 100-102 for a nonlinear candidate when the preregistered A3 trigger fires. Selection uses development latitude-band macro RMSE. Boundaries through 28.45°N select hierarchical TA-SSS; 30.5°N triggers and selects the hierarchical residual MLP. This model-family discontinuity is a material confounder in the northern contrast.

## Main development results

The 27°N hypothesis is not supported by the preregistered rule. Adding 27-28.45°N changes development cruise-equal RMSE from {row.loc[28.45].cruise_equal_rmse_mean:.2f} to {row.loc[27.0].cruise_equal_rmse_mean:.2f} µmol kg⁻¹, while grouped-CV band-macro RMSE improves from {cv.loc[28.45].band_macro_rmse_mean:.2f} to {cv.loc[27.0].band_macro_rmse_mean:.2f}. Adding 26-27°N similarly changes development cruise-equal RMSE from {row.loc[27.0].cruise_equal_rmse_mean:.2f} to {row.loc[26.0].cruise_equal_rmse_mean:.2f}, while CV improves from {cv.loc[27.0].band_macro_rmse_mean:.2f} to {cv.loc[26.0].band_macro_rmse_mean:.2f}.

{contrast_lines}

The only band satisfying the frozen harmful-band rule is 28.45-30.5°N. However, the restricted 30.5°N experiment also activates a nonlinear model unavailable in the other three runs. Its development pooled RMSE is {row.loc[30.5].pooled_rmse_mean:.2f}, R² is {row.loc[30.5].pooled_r2_mean:.3f}, and grouped-CV band-macro RMSE is {cv.loc[30.5].band_macro_rmse_mean:.2f}; uncertainty coverage is below gate and worst support-bin skill is strongly negative. This is a hypothesis for a controlled follow-up, not evidence that the latitude band or a specific water mass causes the loss.

## Decision and limitations

No boundary passes the full P1.3 development gate, and no safe-forward SAB development observations are available. We therefore retain SAB TA as non-product evidence. The specific claim that data south of 27°N are responsible for degradation is rejected under the registered rule. A follow-up should force the same candidate family at every boundary and use cruise bootstrap or leave-one-cruise-out contrasts to separate geography from model selection and the four-cruise development composition. Depth sensitivity remains out of scope because this experiment uses only the project-defined 0-5 m surface layer.

## Figure and table index

All figure source data are the CSV files in `tables/`; exact source hashes are in `archive_manifest.json`. The corresponding large prediction files remain in the ignored experiment output directory.

{figure_blocks}
"""


if __name__ == "__main__":
    raise SystemExit(main())
