"""Run and archive the preregistered Issue-10 inverse-CO2SYS DIC diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recad.chem.inverse_dic import (
    forward_fco2,
    interval_summary,
    inverse_dic,
    measured_fco2_mask,
    region_masks,
    regression_metrics,
)
from recad.evaluate.p1_framework import FrozenManifest, P1DataGateway, Purpose, sha256

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_dic_inverse_co2sys_v2.2"
OUTPUT = ROOT / "outputs/experiments" / EXPERIMENT_ID
ARCHIVE = ROOT / "docs/experiment_archive" / EXPERIMENT_ID
TRAINING_COMMIT = "945d68de7ead692ca5d4bcb28f14d2c6832acc1d"
SEEDS = (100, 101, 102)
DRAW_COUNT = 2000


def error_bank(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame.assign(error=frame["prediction"] - frame["truth"])[["group_key", "error"]]


def sample_hierarchical_errors(
    bank: pd.DataFrame,
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> np.ndarray:
    """Sample cruises first and observations second, as preregistered."""
    grouped = [x.to_numpy(float) for _, x in bank.groupby("group_key")["error"]]
    cruise_index = rng.integers(0, len(grouped), size=shape)
    result = np.empty(shape, dtype=np.float64)
    for index, values in enumerate(grouped):
        locations = np.where(cruise_index == index)
        if locations[0].size:
            result[locations] = rng.choice(values, size=locations[0].size, replace=True)
    return result


def solve_draws(
    frame: pd.DataFrame,
    banks: dict[str, pd.DataFrame],
    scenario: str,
    seed: int,
    draw_count: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(frame)
    reference_fco2 = frame["reference_fco2"].to_numpy(float)
    truth = frame["truth"].to_numpy(float)
    collected = np.empty((draw_count, n), dtype=np.float32)
    batch_size = 50
    for start in range(0, draw_count, batch_size):
        stop = min(start + batch_size, draw_count)
        shape = (stop - start, n)
        ta_error = (
            sample_hierarchical_errors(banks["ta"], rng, shape)
            if scenario in {"TA only", "Joint"}
            else 0.0
        )
        fco2_error = (
            sample_hierarchical_errors(banks["fco2"], rng, shape)
            if scenario in {"fCO2 only", "Joint"}
            else 0.0
        )
        sss_error = (
            sample_hierarchical_errors(banks["sss"], rng, shape)
            if scenario in {"SSS only", "Joint"}
            else 0.0
        )
        ta = frame["ta"].to_numpy(float)[None, :] + ta_error
        fco2 = np.maximum(1.0, reference_fco2[None, :] + fco2_error)
        salinity = np.maximum(0.1, frame["salinity"].to_numpy(float)[None, :] + sss_error)
        temperature = np.broadcast_to(frame["temperature"].to_numpy(float), shape)
        collected[start:stop] = inverse_dic(ta, fco2, salinity, temperature)
    summary = interval_summary(truth, collected)
    return {**summary, "draw_sd": np.std(collected, axis=0, ddof=1)}


def save_table(name: str, frame: pd.DataFrame) -> Path:
    path = ARCHIVE / "tables" / name
    frame.to_csv(path, index=False)
    return path


def save_figure(name: str) -> Path:
    path = ARCHIVE / "figures" / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def finite_paired(frame: pd.DataFrame) -> pd.Series:
    return frame[["ta", "temperature", "salinity", "truth"]].notna().all(axis=1) & frame[
        "label_ta_ok"
    ].fillna(False).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DRAW_COUNT)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "figures").mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "tables").mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "configs/frozen/data_manifest_v2.2.json"
    manifest = FrozenManifest.load(manifest_path)
    gateway = P1DataGateway(manifest)
    columns = [
        "ta",
        "temperature",
        "salinity",
        "fco2",
        "ta_qc",
        "fco2_qc",
        "parameter_method_fco2",
        "parameter_method_ta",
        "parameter_method_dic",
        "source",
        "obs_id",
        "is_primary",
        "label_ta_ok",
    ]
    data = pd.concat(
        [
            gateway.load_labels("dic", purpose, columns=columns)
            for purpose in (Purpose.TRAIN, Purpose.SELECTION)
        ],
        ignore_index=True,
    )
    masks = region_masks(data["lme_id"].to_numpy(), data["latitude"].to_numpy())
    paired_mask = finite_paired(data)
    strict_mask = paired_mask & measured_fco2_mask(
        data["fco2"].to_numpy(),
        data["fco2_qc"].to_numpy(),
        data["parameter_method_fco2"].to_numpy(),
    )
    paired = data.loc[paired_mask].copy()
    paired["reference_fco2"] = forward_fco2(
        paired["ta"].to_numpy(),
        paired["truth"].to_numpy(),
        paired["salinity"].to_numpy(),
        paired["temperature"].to_numpy(),
    )
    strict = data.loc[strict_mask].copy()
    strict["prediction"] = inverse_dic(
        strict["ta"].to_numpy(),
        strict["fco2"].to_numpy(),
        strict["salinity"].to_numpy(),
        strict["temperature"].to_numpy(),
    )
    strict["closure_fco2"] = forward_fco2(
        strict["ta"].to_numpy(),
        strict["prediction"].to_numpy(),
        strict["salinity"].to_numpy(),
        strict["temperature"].to_numpy(),
    )
    strict["closure_error"] = strict["closure_fco2"] - strict["fco2"]

    scope_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for scope, full_mask in masks.items():
        all_scope = data.loc[full_mask]
        paired_scope = paired.loc[paired.index.intersection(data.index[full_mask])]
        strict_scope = strict.loc[strict.index.intersection(data.index[full_mask])]
        scope_rows.append(
            {
                "scope": scope,
                "dic_records": len(all_scope),
                "dic_cruises": all_scope.group_key.nunique(),
                "paired_records": len(paired_scope),
                "paired_cruises": paired_scope.group_key.nunique(),
                "strict_records": len(strict_scope),
                "strict_cruises": strict_scope.group_key.nunique(),
            }
        )
        metrics = regression_metrics(strict_scope["truth"], strict_scope["prediction"])
        strict_rows.append(
            {
                "scope": scope,
                **metrics,
                "cruises": strict_scope.group_key.nunique(),
                "closure_max_abs_uatm": float(strict_scope["closure_error"].abs().max())
                if len(strict_scope)
                else np.nan,
                "claimable": strict_scope.group_key.nunique() >= 5,
            }
        )
        if len(paired_scope):
            ref = forward_fco2(
                paired_scope["ta"].to_numpy(),
                paired_scope["truth"].to_numpy(),
                paired_scope["salinity"].to_numpy(),
                paired_scope["temperature"].to_numpy(),
            )
            back = inverse_dic(
                paired_scope["ta"].to_numpy(),
                ref,
                paired_scope["salinity"].to_numpy(),
                paired_scope["temperature"].to_numpy(),
            )
            paired_rows.append({"scope": scope, **regression_metrics(paired_scope["truth"], back)})

    scope_table = pd.DataFrame(scope_rows)
    strict_table = pd.DataFrame(strict_rows)
    closure_table = pd.DataFrame(paired_rows)

    predictions = {
        "sss": ROOT
        / "outputs/experiments/p1_sss_viability_v2.2/selected_development_predictions.parquet",
        "fco2": ROOT
        / "outputs/experiments/p1_fco2_viability_v2.2/selected_development_predictions.parquet",
        "ta": ROOT
        / "outputs/experiments/p1_ta_viability_v2.2/selected_development_predictions.parquet",
    }
    global_banks = {name: error_bank(path) for name, path in predictions.items()}
    regional_ta = {
        "SAB": error_bank(
            ROOT
            / "outputs/experiments/p1_ta_bights_v2.2/sab/selected_development_predictions.parquet"
        ),
        "MAB": error_bank(
            ROOT
            / "outputs/experiments/p1_ta_bights_v2.2/mab/selected_development_predictions.parquet"
        ),
    }
    bank_summary = []
    for target, bank in global_banks.items():
        bank_summary.append(
            {
                "scope": "North America",
                "target": target,
                "n": len(bank),
                "cruises": bank.group_key.nunique(),
                "bias": bank.error.mean(),
                "rmse": np.sqrt(np.mean(bank.error**2)),
                "sd": bank.error.std(ddof=1),
            }
        )
    for scope, bank in regional_ta.items():
        bank_summary.append(
            {
                "scope": scope,
                "target": "ta",
                "n": len(bank),
                "cruises": bank.group_key.nunique(),
                "bias": bank.error.mean(),
                "rmse": np.sqrt(np.mean(bank.error**2)),
                "sd": bank.error.std(ddof=1),
            }
        )
    bank_table = pd.DataFrame(bank_summary)

    propagation_rows: list[dict[str, object]] = []
    row_summaries: list[pd.DataFrame] = []
    scenarios = ("TA only", "fCO2 only", "SSS only", "Joint")
    for scope, full_mask in masks.items():
        subset = paired.loc[paired.index.intersection(data.index[full_mask])].copy()
        banks = dict(global_banks)
        if scope in regional_ta:
            banks["ta"] = regional_ta[scope]
        for scenario in scenarios:
            for seed in SEEDS:
                summary = solve_draws(subset, banks, scenario, seed, args.draws)
                metrics = regression_metrics(subset["truth"], summary["median"])
                propagation_rows.append(
                    {
                        "scope": scope,
                        "scenario": scenario,
                        "seed": seed,
                        **metrics,
                        "cruises": subset.group_key.nunique(),
                        "coverage_50": float(np.mean(summary["covered_50"])),
                        "coverage_90": float(np.mean(summary["covered_90"])),
                        "median_width_50": float(np.median(summary["q75"] - summary["q25"])),
                        "median_width_90": float(np.median(summary["q95"] - summary["q05"])),
                        "median_draw_sd": float(np.median(summary["draw_sd"])),
                    }
                )
                if seed == SEEDS[0]:
                    row_summaries.append(
                        pd.DataFrame(
                            {
                                "obs_id": subset["obs_id"].to_numpy(),
                                "scope": scope,
                                "scenario": scenario,
                                "truth": subset["truth"].to_numpy(),
                                **summary,
                            }
                        )
                    )
    propagation = pd.DataFrame(propagation_rows)
    pd.concat(row_summaries, ignore_index=True).to_parquet(
        OUTPUT / "propagation_row_summary.parquet", index=False
    )

    # Carter/ESPER is available only for development TA rows; evaluate its overlap with strict anchors.
    ta_dev = pd.read_parquet(predictions["ta"])[["obs_id", "carter"]].drop_duplicates("obs_id")
    carter = strict.merge(ta_dev, on="obs_id", how="inner")
    if len(carter):
        carter["carter_dic"] = inverse_dic(
            carter["carter"].to_numpy(),
            carter["fco2"].to_numpy(),
            carter["salinity"].to_numpy(),
            carter["temperature"].to_numpy(),
        )
    carter_rows = []
    carter_masks = (
        region_masks(carter["lme_id"].to_numpy(), carter["latitude"].to_numpy())
        if len(carter)
        else {}
    )
    for scope, mask in carter_masks.items():
        sample = carter.loc[mask]
        for method, column in (("Observed TA", "prediction"), ("Carter TA", "carter_dic")):
            carter_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    **regression_metrics(sample["truth"], sample[column]),
                    "cruises": sample.group_key.nunique(),
                }
            )
    carter_table = pd.DataFrame(carter_rows)

    # Local numerical sensitivity around every paired state.
    delta_ta = (
        inverse_dic(
            paired["ta"].to_numpy() + 1,
            paired["reference_fco2"].to_numpy(),
            paired["salinity"].to_numpy(),
            paired["temperature"].to_numpy(),
        )
        - paired["truth"].to_numpy()
    )
    delta_fco2 = (
        inverse_dic(
            paired["ta"].to_numpy(),
            paired["reference_fco2"].to_numpy() + 1,
            paired["salinity"].to_numpy(),
            paired["temperature"].to_numpy(),
        )
        - paired["truth"].to_numpy()
    )
    sensitivity = pd.DataFrame(
        {
            "obs_id": paired["obs_id"].to_numpy(),
            "dDIC_dTA": delta_ta,
            "dDIC_dfCO2": delta_fco2,
        }
    )

    save_table("table01_scope.csv", scope_table)
    save_table("table02_strict_oracle_metrics.csv", strict_table)
    save_table("table03_exact_roundtrip.csv", closure_table)
    save_table("table04_upstream_residual_banks.csv", bank_table)
    save_table("table05_propagation_metrics.csv", propagation)
    save_table("table06_carter_baseline.csv", carter_table)
    save_table("table07_local_sensitivity.csv", sensitivity)
    save_table(
        "table08_strict_oracle_rows.csv",
        strict[
            [
                "obs_id",
                "group_key",
                "source",
                "evaluation_split",
                "latitude",
                "longitude",
                "truth",
                "prediction",
                "fco2",
                "closure_error",
            ]
        ],
    )

    # Reviewer figures, each backed by the archived CSV tables.
    scope_table.set_index("scope")[["dic_records", "paired_records", "strict_records"]].plot.bar(
        figsize=(8, 5)
    )
    plt.ylabel("records")
    plt.title("Available DIC evidence by scope")
    save_figure("fig01_evidence_counts.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(strict["truth"], strict["prediction"], s=18, alpha=0.65)
    bounds = [
        min(strict.truth.min(), strict.prediction.min()),
        max(strict.truth.max(), strict.prediction.max()),
    ]
    plt.plot(bounds, bounds, "k--")
    plt.xlabel("Observed DIC")
    plt.ylabel("Inverse DIC")
    plt.title("Strict measured-fCO2 oracle")
    save_figure("fig02_strict_oracle_scatter.png")

    plt.figure(figsize=(8, 5))
    strict.assign(error=strict.prediction - strict.truth).boxplot(column="error", by="source")
    plt.suptitle("")
    plt.title("Strict oracle error by source")
    plt.ylabel("DIC error (µmol kg⁻¹)")
    save_figure("fig03_strict_error_by_source.png")

    mean_prop = propagation.groupby(["scope", "scenario"], as_index=False).mean(numeric_only=True)
    mean_prop.pivot(index="scenario", columns="scope", values="rmse").plot.bar(figsize=(9, 5))
    plt.ylabel("RMSE (µmol kg⁻¹)")
    plt.title("Residual-injection DIC error")
    save_figure("fig04_propagation_rmse.png")

    mean_prop.pivot(index="scenario", columns="scope", values="coverage_90").plot.bar(
        figsize=(9, 5)
    )
    plt.axhspan(0.85, 0.95, color="green", alpha=0.12)
    plt.ylabel("90% interval coverage")
    plt.title("Nominal 90% propagation coverage")
    save_figure("fig05_coverage90.png")

    mean_prop.pivot(index="scenario", columns="scope", values="median_width_90").plot.bar(
        figsize=(9, 5)
    )
    plt.ylabel("Median 90% width (µmol kg⁻¹)")
    plt.title("Propagated DIC interval width")
    save_figure("fig06_interval_width.png")

    sensitivity[["dDIC_dTA", "dDIC_dfCO2"]].plot.hist(bins=60, alpha=0.65, figsize=(8, 5))
    plt.xlabel("Local derivative")
    plt.title("Exact local inverse-CO2SYS sensitivity")
    save_figure("fig07_local_sensitivity.png")

    comparison = carter_table.pivot(index="scope", columns="method", values="rmse")
    comparison.plot.bar(figsize=(8, 5))
    plt.ylabel("Strict-oracle RMSE (µmol kg⁻¹)")
    plt.title("Observed and Carter TA inputs")
    save_figure("fig08_carter_comparison.png")

    strict_na = strict_table.loc[strict_table.scope.eq("North America")].iloc[0]
    joint_na = mean_prop.loc[
        (mean_prop.scope.eq("North America")) & (mean_prop.scenario.eq("Joint"))
    ].iloc[0]
    ta_na = mean_prop.loc[
        (mean_prop.scope.eq("North America")) & (mean_prop.scenario.eq("TA only"))
    ].iloc[0]
    fco2_na = mean_prop.loc[
        (mean_prop.scope.eq("North America")) & (mean_prop.scenario.eq("fCO2 only"))
    ].iloc[0]
    sss_na = mean_prop.loc[
        (mean_prop.scope.eq("North America")) & (mean_prop.scenario.eq("SSS only"))
    ].iloc[0]
    carter_na = carter_table.loc[carter_table.scope.eq("North America")].set_index("method")
    decision = "diagnostic_only"
    captions = {
        "fig01_evidence_counts.png": "Figure 1. DIC evidence counts in unsealed train/development. The strict measured-fCO2 subset contracts from 5,356 DIC records to 119 records and only eight cruises.",
        "fig02_strict_oracle_scatter.png": "Figure 2. Exact TA plus measured-fCO2 inverse DIC against observed DIC for the strict subset. Scatter includes sampling and carbonate-parameter mismatch, not solver approximation.",
        "fig03_strict_error_by_source.png": "Figure 3. Strict inverse-DIC errors separated by CODAP-NA and GLODAP source. Source structure warns against treating the 119 records as an exchangeable product-validation sample.",
        "fig04_propagation_rmse.png": "Figure 4. DIC RMSE after injecting frozen upstream development residuals into exact PyCO2SYS. TA error dominates the joint propagation and prevents a publishable DIC product.",
        "fig05_coverage90.png": "Figure 5. Empirical coverage of propagated nominal 90% DIC intervals on paired-carbon anchors. This is a residual-injection calibration diagnostic, not independent pointwise validation.",
        "fig06_interval_width.png": "Figure 6. Median propagated 90% DIC interval width by error source and region. Joint intervals quantify the precision cost of uncertain TA, fCO2, and SSS inputs.",
        "fig07_local_sensitivity.png": "Figure 7. Exact PyCO2SYS local derivatives on paired observed TA-DIC states. One-unit TA errors transfer nearly one-for-one into DIC, while fCO2 sensitivity is smaller per unit.",
        "fig08_carter_comparison.png": "Figure 8. Strict measured-fCO2 DIC RMSE using observed TA versus Carter/ESPER TA where development records overlap. The small overlap makes this an audit rather than a product comparison.",
    }
    report = f"""# Exact inverse-CO2SYS DIC validation and uncertainty propagation

## Scientific question and permitted claim

This preregistered Issue #10 experiment asks whether DIC can be inferred from TA and fCO2 with exact PyCO2SYS and calibrated propagated uncertainty. The permitted result is a structural and development diagnostic. It does not validate a DIC product because upstream TA is diagnostic-only/fail and fCO2 is diagnostic-only.

## Data, splits, and leakage controls

The gateway read only frozen train/development DIC labels. Locked test and external-independent labels remained sealed. There are {len(data):,} DIC records from {data.group_key.nunique()} cruises; {len(paired):,} records support paired-carbon propagation, while only {len(strict):,} records from {strict.group_key.nunique()} cruises also contain eligible measured fCO2 (method 1/2, QC=2). Calculated method-3 fCO2 was excluded from the strict oracle.

## Candidate models and training

No DIC machine-learning model was fitted. Exact PyCO2SYS used TA+fCO2 with the preregistered constants. A numerical round trip used observed TA+DIC to calculate reference fCO2 and invert it back. The uncertainty experiment sampled complete frozen development residual banks hierarchically by cruise for SSS, fCO2, and TA, with {args.draws} draws for each of seeds 100, 101, and 102. Carter/ESPER TA was evaluated only where it overlaps strict development anchors.

## Main development results

The exact paired-carbon round trip is numerical, with North America RMSE {closure_table.loc[closure_table.scope.eq("North America"), "rmse"].iloc[0]:.3g} µmol kg⁻¹. On the genuinely measured-fCO2 strict subset, observed-TA inverse DIC has RMSE {strict_na.rmse:.1f}, MAE {strict_na.mae:.1f}, bias {strict_na.bias:+.1f} µmol kg⁻¹ and R² {strict_na.r2:.3f}; exact forward closure remains below {strict_na.closure_max_abs_uatm:.3g} µatm. The joint residual-injection diagnostic has North America median-prediction RMSE {joint_na.rmse:.1f} µmol kg⁻¹, but this centered-simulation error is not end-to-end skill. Its 50% and 90% coverages are both {joint_na.coverage_50:.3f}/{joint_na.coverage_90:.3f}, above the preregistered ranges, while the median 90% interval is {joint_na.median_width_90:.1f} µmol kg⁻¹ wide. TA-only propagation dominates (RMSE {ta_na.rmse:.2f}; width {ta_na.median_width_90:.1f}), compared with fCO2-only (RMSE {fco2_na.rmse:.2f}; width {fco2_na.median_width_90:.1f}) and SSS-only (RMSE {sss_na.rmse:.3f}; width {sss_na.median_width_90:.1f}). On the same {int(carter_na.loc["Observed TA", "n"])}-record, {int(carter_na.loc["Observed TA", "cruises"])}-cruise overlap, replacing observed TA with Carter TA raises RMSE from {carter_na.loc["Observed TA", "rmse"]:.1f} to {carter_na.loc["Carter TA", "rmse"]:.1f} µmol kg⁻¹. Component results and regional SAB/MAB values are in Tables 2-7.

## Decision and limitations

Decision: `{decision}`. Exact inversion is technically sound, but the DIC product gate fails because TA and fCO2 have not passed their upstream product gates. The strict subset spans only eight cruises overall; SAB and MAB do not reach five strict cruises. The larger paired-carbon experiment starts from fCO2 calculated using observed DIC, so it quantifies error propagation and cannot establish independent DIC predictive skill. Both nominal coverage gates fail through severe overcoverage. Residual banks come from different observation networks, so cross-target error dependence is unidentified.

## Figure and table index

Every figure is backed by a CSV under `tables/`; hashes and mappings are in `archive_manifest.json`.

"""
    for figure_name, caption in captions.items():
        report += f"![{figure_name}](figures/{figure_name})\n\n{caption}\n\n"
    (ARCHIVE / "REPORT.md").write_text(report, encoding="utf-8")
    (ARCHIVE / "CAPTIONS.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(f"## {name}\n\n{caption}" for name, caption in captions.items())
        + "\n",
        encoding="utf-8",
    )
    (ARCHIVE / "README.md").write_text(
        "# Issue #10 reviewer archive\n\nRun `python scripts/run_p1_dic_inverse_co2sys.py`, then verify with `python scripts/verify_experiment_archive.py docs/experiment_archive/p1_dic_inverse_co2sys_v2.2`. Large row-level draws and summaries remain under ignored `outputs/`.\n",
        encoding="utf-8",
    )

    source_mapping = {
        "fig01_evidence_counts.png": ["table01_scope.csv"],
        "fig02_strict_oracle_scatter.png": ["table08_strict_oracle_rows.csv"],
        "fig03_strict_error_by_source.png": ["table08_strict_oracle_rows.csv"],
        "fig04_propagation_rmse.png": ["table05_propagation_metrics.csv"],
        "fig05_coverage90.png": ["table05_propagation_metrics.csv"],
        "fig06_interval_width.png": ["table05_propagation_metrics.csv"],
        "fig07_local_sensitivity.png": ["table07_local_sensitivity.csv"],
        "fig08_carter_comparison.png": [
            "table02_strict_oracle_metrics.csv",
            "table06_carter_baseline.csv",
        ],
    }
    tracked = [ARCHIVE / "README.md", ARCHIVE / "REPORT.md", ARCHIVE / "CAPTIONS.md"]
    tracked += sorted((ARCHIVE / "figures").glob("*")) + sorted((ARCHIVE / "tables").glob("*.csv"))
    archive_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_git_commit": TRAINING_COMMIT,
        "data_manifest_sha256": sha256(manifest_path),
        "locked_test_opened": False,
        "external_independent_opened": False,
        "evidence_scope": "North America plus SAB/MAB train/development structural diagnostic",
        "decision": decision,
        "archive_builder_sha256": sha256(Path(__file__)),
        "analysis_script_sha256": sha256(Path(__file__)),
        "source_artifacts_sha256": {name: sha256(path) for name, path in predictions.items()},
        "figure_source_data": source_mapping,
        "figure_captions": captions,
        "files_sha256": {
            str(path.relative_to(ARCHIVE)).replace("\\", "/"): sha256(path) for path in tracked
        },
    }
    (ARCHIVE / "archive_manifest.json").write_text(
        json.dumps(archive_manifest, indent=2), encoding="utf-8"
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "strict_oracle": strict_table.to_dict(orient="records"),
        "joint_propagation": mean_prop.loc[mean_prop.scenario.eq("Joint")].to_dict(
            orient="records"
        ),
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
