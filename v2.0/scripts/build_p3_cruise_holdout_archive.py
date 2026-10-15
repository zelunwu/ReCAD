"""Build the reviewer archive for the Issue #5 complete-cruise validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p3_cruise_holdout_v2.4")
ARCHIVE = ROOT / "docs/experiment_archive/p3_cruise_holdout_v2.4"
BASIN_NAMES = {
    0: "Arctic",
    1: "Southern",
    2: "Atlantic",
    3: "Pacific",
    4: "Indian",
    5: "Mediterranean",
}
UNITS = {"SSS": "PSU", "fCO2": "µatm"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    error = frame.prediction.to_numpy(float) - frame.truth.to_numpy(float)
    absolute = np.abs(error)
    background_error = frame.background_prediction.to_numpy(float) - frame.truth.to_numpy(float)
    cruise_mse = frame.assign(squared=error**2).groupby("group_key").squared.mean()
    rmse = float(np.sqrt(np.mean(error**2)))
    background_rmse = float(np.sqrt(np.mean(background_error**2)))
    denominator = float(np.sum((frame.truth - frame.truth.mean()) ** 2))
    return {
        "rows": len(frame),
        "cruises": int(frame.group_key.nunique()),
        "rmse": rmse,
        "mae": float(absolute.mean()),
        "bias": float(error.mean()),
        "r2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else np.nan,
        "cruise_equal_rmse": float(np.sqrt(cruise_mse.mean())),
        "median_cruise_rmse": float(np.sqrt(cruise_mse).median()),
        "worst_cruise_rmse": float(np.sqrt(cruise_mse).max()),
        "coverage50": float(np.mean(absolute <= frame.width50)),
        "coverage90": float(np.mean(absolute <= frame.width90)),
        "median_width50": float(np.median(2 * frame.width50)),
        "median_width90": float(np.median(2 * frame.width90)),
        "background_rmse": background_rmse,
        "background_skill": float(1 - rmse**2 / background_rmse**2),
    }


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(ARCHIVE / "figures" / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for directory in (ARCHIVE / "figures", ARCHIVE / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    frames = {
        target: pd.read_parquet(OUTPUT / "predictions" / f"{target}_held_predictions.parquet")
        for target in ("SSS", "fCO2")
    }
    overall, strata, trusted, cruises = [], [], [], []
    for target, frame in frames.items():
        overall.append({"target": target, **metric_row(frame)})
        for view, values in (
            ("selection_reason", frame.selection_reason.astype(str)),
            ("basin", frame.basin_id.map(BASIN_NAMES)),
            ("grade", frame.grade.astype(str)),
            (
                "support_distance_km",
                pd.cut(
                    frame.geographic_km,
                    [-np.inf, 25, 100, 200, np.inf],
                    labels=["0-25", "25-100", "100-200", ">200"],
                ).astype(str),
            ),
            (
                "time",
                pd.cut(
                    frame.yr,
                    [1992, 2010, 2021, 2025],
                    labels=["1993-2010", "2011-2021", "2022-2025"],
                ).astype(str),
            ),
        ):
            for level in sorted(values.unique()):
                strata.append(
                    {
                        "target": target,
                        "view": view,
                        "level": level,
                        **metric_row(frame.loc[values.eq(level)]),
                    }
                )
        ab = frame.loc[frame.grade.isin(["A", "B"])]
        trusted.append({"target": target, "region": "all", **metric_row(ab)})
        for basin, part in ab.groupby("basin_id", observed=True):
            trusted.append(
                {"target": target, "region": BASIN_NAMES[int(basin)], **metric_row(part)}
            )
        for group, part in frame.groupby("group_key", observed=True):
            cruises.append(
                {
                    "target": target,
                    "group_key": group,
                    "selection_reason": part.selection_reason.iloc[0],
                    "basin": BASIN_NAMES[int(part.basin_id.mode().iloc[0])],
                    "grade_mode": part.grade.mode().iloc[0],
                    **metric_row(part),
                }
            )
    overall_frame = pd.DataFrame(overall)
    strata_frame = pd.DataFrame(strata)
    trusted_frame = pd.DataFrame(trusted)
    cruise_frame = pd.DataFrame(cruises)
    overall_frame.to_csv(ARCHIVE / "tables/table01_overall_metrics.csv", index=False)
    strata_frame.to_csv(ARCHIVE / "tables/table02_stratified_metrics.csv", index=False)
    trusted_frame.to_csv(ARCHIVE / "tables/table03_trusted_ab_metrics.csv", index=False)
    cruise_frame.sort_values(["target", "rmse"], ascending=[True, False]).to_csv(
        ARCHIVE / "tables/table04_per_cruise_metrics.csv", index=False
    )

    decisions = pd.DataFrame(
        [
            {
                "target": "SSS",
                "scope": "A/B all basins",
                "status": "qualified_same_source",
                "reason": "coverage90=0.927 and positive background skill",
            },
            {
                "target": "SSS",
                "scope": "C",
                "status": "diagnostic",
                "reason": "outside trusted contract",
            },
            {
                "target": "SSS",
                "scope": "D",
                "status": "suppressed",
                "reason": "coverage90=0.841; distant-support coverage reaches 0.529",
            },
            {
                "target": "fCO2",
                "scope": "A/B outside Pacific",
                "status": "qualified_same_source",
                "reason": "pooled coverage90=0.904 and positive background skill",
            },
            {
                "target": "fCO2",
                "scope": "Pacific A/B",
                "status": "diagnostic",
                "reason": "coverage90=0.845 below frozen 0.85 gate",
            },
            {"target": "fCO2", "scope": "C", "status": "diagnostic", "reason": "coverage90=0.829"},
            {
                "target": "fCO2",
                "scope": "D",
                "status": "suppressed",
                "reason": "large errors and unsupported extreme cruises",
            },
        ]
    )
    decisions.to_csv(ARCHIVE / "tables/table05_release_decision.csv", index=False)
    contracts = []
    for target in ("SSS", "fCO2"):
        contract = json.loads(
            (OUTPUT / f"{target}_calibration_contract.json").read_text(encoding="utf-8")
        )
        contracts.append(contract)
    pd.DataFrame(contracts).to_csv(
        ARCHIVE / "tables/table07_calibration_contracts.csv", index=False
    )
    pd.DataFrame([json.loads((OUTPUT / "label_access.json").read_text(encoding="utf-8"))]).to_csv(
        ARCHIVE / "tables/table08_label_access.csv", index=False
    )

    for target, frame in frames.items():
        unit = UNITS[target]
        figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        plot = axis.hexbin(
            frame.truth, frame.prediction, gridsize=55, mincnt=1, bins="log", cmap="viridis"
        )
        low = float(min(frame.truth.min(), frame.prediction.min()))
        high = float(max(frame.truth.max(), frame.prediction.max()))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set(xlabel=f"Observed {target} ({unit})", ylabel=f"Predicted {target} ({unit})")
        figure.colorbar(plot, ax=axis, label="log density")
        save(figure, f"fig_{target.lower()}_01_observed_predicted.png")

        group = cruise_frame.loc[cruise_frame.target.eq(target)].sort_values("rmse")
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        colors = np.where(group.grade_mode.eq("D"), "#d73027", "#4575b4")
        axis.bar(np.arange(len(group)), group.rmse, color=colors)
        axis.set(xlabel="Held cruise ranked by RMSE", ylabel=f"RMSE ({unit})", yscale="log")
        save(figure, f"fig_{target.lower()}_02_cruise_rmse.png")

        grade = (
            strata_frame.loc[strata_frame.target.eq(target) & strata_frame.view.eq("grade")]
            .set_index("level")
            .reindex(list("ABCD"))
        )
        figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        axis.bar(grade.index, grade.coverage90, color=["#1a9850", "#91cf60", "#fee08b", "#d73027"])
        axis.axhline(0.85, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Applicability grade", ylabel="90% interval coverage", ylim=(0, 1))
        save(figure, f"fig_{target.lower()}_03_grade_coverage.png")

        support = (
            strata_frame.loc[
                strata_frame.target.eq(target) & strata_frame.view.eq("support_distance_km")
            ]
            .set_index("level")
            .reindex(["0-25", "25-100", "100-200", ">200"])
        )
        figure, axis = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        axis.plot(support.index, support.rmse, marker="o", linewidth=2)
        axis.set(xlabel="Nearest training support distance (km)", ylabel=f"RMSE ({unit})")
        save(figure, f"fig_{target.lower()}_04_support_rmse.png")

        map_frame = frame.assign(absolute_error=np.abs(frame.prediction - frame.truth))
        figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
        plot = axis.scatter(
            map_frame.longitude,
            map_frame.latitude,
            c=map_frame.absolute_error,
            s=12,
            cmap="magma",
            norm="log",
            alpha=0.75,
        )
        axis.set(xlabel="Longitude", ylabel="Latitude", xlim=(-180, 360), ylim=(-90, 90))
        figure.colorbar(plot, ax=axis, label=f"Absolute error ({unit})")
        save(figure, f"fig_{target.lower()}_05_error_map.png")

    artifact_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            artifact_rows.append(
                {
                    "path": str(path.relative_to(OUTPUT)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(
        ARCHIVE / "tables/table06_external_artifact_hashes.csv", index=False
    )

    captions = {
        "fig_sss_01_observed_predicted.png": "Figure 1. Held-cruise SSS observations against frozen-candidate predictions. The one-to-one line exposes both the dominant agreement and a small set of strong freshwater failures.",
        "fig_sss_02_cruise_rmse.png": "Figure 2. SSS RMSE for every held cruise, ranked on a logarithmic axis. Red bars have modal grade D; the extreme tail explains the gap between pooled and cruise-equal RMSE.",
        "fig_sss_03_grade_coverage.png": "Figure 3. SSS 90% interval coverage by frozen applicability grade. The dashed line is the preregistered 0.85 minimum; grade D falls below it and remains suppressed.",
        "fig_sss_04_support_rmse.png": "Figure 4. SSS RMSE by distance to the remaining training cruises. Errors increase sharply away from dense support, so distance remains a required product flag.",
        "fig_sss_05_error_map.png": "Figure 5. Absolute SSS error at held-cruise grid months. The logarithmic color scale preserves ordinary errors while exposing geographically concentrated freshwater failures.",
        "fig_fco2_01_observed_predicted.png": "Figure 6. Held-cruise fCO2 observations against frozen-candidate predictions. Most rows follow the one-to-one line, while a small extreme-estuary tail remains unresolved.",
        "fig_fco2_02_cruise_rmse.png": "Figure 7. fCO2 RMSE for every held cruise, ranked on a logarithmic axis. A few complete cruises dominate cruise-equal risk despite acceptable pooled performance.",
        "fig_fco2_03_grade_coverage.png": "Figure 8. fCO2 90% interval coverage by frozen applicability grade. A/B meet the pooled gate, while C under-covers and D remains unsuitable for trusted publication.",
        "fig_fco2_04_support_rmse.png": "Figure 9. fCO2 RMSE by distance to the remaining training cruises. The 25-100 km bin is especially difficult, and all distant bins retain weak interval coverage.",
        "fig_fco2_05_error_map.png": "Figure 10. Absolute fCO2 error at held-cruise grid months. The logarithmic color scale shows that the largest errors are localized rather than uniformly global.",
    }
    report = [
        "# P3 complete-cruise validation",
        "",
        "## Scientific question and permitted claim",
        "",
        "This one-time evaluation tests frozen SSS and fCO2 candidates on 50 complete SOCAT v2026 cruises selected from metadata before their target values were read. Every held group was excluded from point-model fitting, calibration, continuous backgrounds, upstream SSS, thresholds, and support indices. The permitted claim is same-source transfer to unseen cruises, sparse places, and recent time. No cross-dataset generalization is claimed.",
        "",
        "## Data, splits, and leakage controls",
        "",
        "The frozen manifest contains 8,469 SSS and 8,470 fCO2 rows. Twenty-three cruises stress 2022-2025 transfer and 27 stress sparse 5° blocks. Remaining cruises were split by complete group into model fit and calibration. fCO2 never receives observed SSS: its training SSS is two-fold complete-cruise cross-fitted, while calibration and held SSS are predicted from non-held SSS observations. The label access ledger records exactly one access after PR #75 merged.",
        "",
        "## Candidate models and training",
        "",
        "SSS is the Issue-62 three-seed CatBoost background-residual ensemble. fCO2 is the Issue-64 continuous-background ExtraTrees plus CatBoost convex ensemble with frozen CatBoost weight 0.30. Risk models and interval calibration use only non-held complete groups. The 2.57 GB fCO2 ExtraTrees checkpoint and all other large artifacts remain outside Git with SHA256 hashes in Table 6.",
        "",
        "## Main development results",
        "",
        "This section reports final held-cruise evidence rather than additional development selection. SSS has RMSE 1.185 PSU, R² 0.773, 90% coverage 0.910, and background skill 0.700. fCO2 has RMSE 30.780 µatm, R² 0.592, 90% coverage 0.883, and background skill 0.622. Cruise-equal RMSE is materially worse at 2.445 PSU and 45.942 µatm because a small number of complete cruises fail strongly.",
        "",
        "The frozen A/B layer is much more reliable: SSS RMSE 0.598 PSU with 0.927 coverage and fCO2 RMSE 26.619 µatm with 0.904 coverage. Sparse-block RMSE is 1.487 PSU and 35.005 µatm, worse than recent-cruise RMSE of 0.738 PSU and 25.498 µatm. Pacific fCO2 A/B coverage is 0.845, just below the frozen 0.85 gate, so it cannot enter the trusted layer.",
        "",
    ]
    for name, caption in captions.items():
        report.extend([f"![{caption}](figures/{name})", "", caption, ""])
    report.extend(
        [
            "## Decision and limitations",
            "",
            "Decision: **`pass_flagged_same_source_release_with_regional_suppression`**. SSS A/B may proceed as same-source qualified across represented basins; SSS C remains diagnostic and D suppressed. fCO2 A/B may proceed outside the Pacific; Pacific A/B and all C remain diagnostic, while D is suppressed. Complete geographic coverage must never be described as complete reliability.",
            "",
            "The clearest failure is cruise `187F20250215`, a strong freshwater system with SSS RMSE 15.947 PSU and fCO2 RMSE 266.792 µatm. It is grade D, showing that the flag catches the setting, but its intervals still fail badly. The experiment is within SOCAT v2026 and does not test a different sensor, QC system, or observation product. Fifty cruises provide a much stronger transfer audit than random rows, but basin-level samples remain modest and some cruises cross row-level basin boundaries.",
            "",
            "## Figure and table index",
            "",
            "Figures 1-5 report SSS agreement, cruise risk, grade calibration, support distance, and geography. Figures 6-10 provide the same views for fCO2. Table 1 contains overall metrics; Table 2 all frozen strata; Table 3 trusted A/B regional evidence; Table 4 every cruise; Table 5 the release decision; Table 6 hashes every external output; Tables 7-8 store calibration and the one-time label-access ledger.",
            "",
        ]
    )
    (ARCHIVE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# P3 complete-cruise validation archive\n\nReviewer-ready Issue #5 evidence. Large predictions and checkpoints remain under `C:\\backup\\phd\\data\\processed\\recad_v2_4\\p3_cruise_holdout_v2.4`; Table 6 records their hashes.\n",
        encoding="utf-8",
    )
    caption_lines = ["# Figure captions", ""]
    for name, caption in captions.items():
        caption_lines.extend([f"## {name}", "", caption, ""])
    (ARCHIVE / "CAPTIONS.md").write_text("\n".join(caption_lines), encoding="utf-8")

    tracked = ["README.md", "REPORT.md", "CAPTIONS.md"]
    tracked += [
        str(path.relative_to(ARCHIVE)).replace("\\", "/")
        for path in sorted((ARCHIVE / "figures").glob("*"))
    ]
    tracked += [
        str(path.relative_to(ARCHIVE)).replace("\\", "/")
        for path in sorted((ARCHIVE / "tables").glob("*.csv"))
    ]
    builder_hash = sha256(Path(__file__))
    manifest = {
        "experiment_id": "p3_cruise_holdout_v2.4",
        "training_git_commit": "c39713bdbdd87daa24a00f77c95fb254d37661d6",
        "data_manifest_sha256": sha256(ROOT / "configs/frozen/p3_cruise_holdout_manifest_v2.4.csv"),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "one-time same-source complete-cruise, sparse-place, and recent-time transfer within SOCAT v2026; no cross-dataset claim",
        "decision": "pass_flagged_same_source_release_with_regional_suppression",
        "figure_source_data": {
            name: [
                "table01_overall_metrics.csv",
                "table02_stratified_metrics.csv",
                "table04_per_cruise_metrics.csv",
            ]
            for name in captions
        },
        "figure_captions": captions,
        "files_sha256": {relative: sha256(ARCHIVE / relative) for relative in tracked},
        "archive_builder_sha256": builder_hash,
        "analysis_script_sha256": sha256(ROOT / "scripts/run_p3_cruise_holdout_v2_4.py"),
        "source_artifacts_sha256": {
            "external_artifact_hash_table": sha256(
                ARCHIVE / "tables/table06_external_artifact_hashes.csv"
            ),
            "frozen_manifest": sha256(ROOT / "configs/frozen/p3_cruise_holdout_manifest_v2.4.csv"),
        },
        "label_access_count": 1,
        "target_labels_opened": True,
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
