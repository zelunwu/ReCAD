# Figure captions

## fig01_sss_rmse.png

Figure 1. Held-out SSS RMSE for all six preregistered point-model families across cruise, true geographic-block plus 100 km buffer, whole-region, and rolling-forward validation; every bar pools the complete corrected outer predictions rather than an inner tuning subset.

## fig02_fco2_rmse.png

Figure 2. Held-out fCO2 RMSE for the same six families and repaired validation designs, using outer-nested predicted SSS rather than observed salinity as an input; the strict transfer columns determine whether ordinary cruise skill generalizes.

## fig03_sss_relative_improvement.png

Figure 3. Percent SSS RMSE improvement relative to the Issue 61 CatBoost background-residual baseline; positive values indicate improvement, while the four columns expose any gain obtained by sacrificing geographic, regional, or future transfer.

## fig04_fco2_relative_improvement.png

Figure 4. Percent fCO2 RMSE improvement relative to the Issue 61 modernized random forest; the comparison uses identical outer folds and nested upstream SSS, so differences isolate model family and its fixed small search allowance.

## fig05_group_equal_rmse.png

Figure 5. Group-equal RMSE assigns every held cruise or platform group equal influence and therefore checks whether dense observations dominate pooled RMSE; lower values and consistent color across columns indicate more stable transfer.

## fig06_finalist_seed_stability.png

Figure 6. Three-seed RMSE mean and standard deviation for the preregistered Pareto or fallback finalists (SSS: CatBoost; fCO2: ExtraTrees, CatBoost); error bars quantify stochastic training sensitivity after configuration selection.

## fig07_finalist_coverage90.png

Figure 7. Empirical coverage of nominal 90% prediction intervals for the finalists; the green band marks the frozen 0.85-0.95 acceptance range and reveals where calibration fitted inside outer training fails under distribution shift.

## fig08_geographic_risk_coverage.png

Figure 8. Geographic-holdout RMSE after retaining observations with the narrowest prediction intervals; a monotone curve supports interval width as a risk-ranking device but does not validate unobserved coastal grid cells.

## fig09_compute_performance.png

Figure 9. Complete screening compute time versus mean held-out rank, including all six inner configurations and final outer fits; the lower-left region identifies families that combine transfer skill with practical computational cost.

## fig10_inner_search_distribution.png

Figure 10. Distribution of inner group-equal RMSE across every family configuration and outer fold; broad or shifted boxes show whether the equal six-configuration allowance yielded a robust family or a configuration-sensitive result.

## fig11_late_bias_worst_region.png

Figure 11. Finalist late-period bias for 2020-2025 and worst RMSE among LMEs with at least 200 held observations; these diagnostics prevent good global averages from hiding temporal drift or severe supported-region failures.

## fig12_graph_residual_signal.png

Figure 12. Same-month four-neighbour residual correlation after subtracting a within-month permutation control, averaged over finalist seeds; the frozen graph trigger is 0.20 and the resulting statuses are SSS=triggered, fCO2=triggered.

## fig13_leakage_audit.png

Figure 13. Summed group, record, calibration-group, and forbidden-future overlap counts inherited from the repaired Issue 61 folds; every displayed zero is required before model-family comparisons can be interpreted.
