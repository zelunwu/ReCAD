# Figure captions

## fig01_native_rmse_matrix.png

Figure 1. Native held-out fCO2 RMSE for seven model families and four validation designs; the honest v1.x random forest leads cruise and spatial tests, while no model leads every transfer regime.

## fig02_cruise_equal_rmse.png

Figure 2. Cruise-equal RMSE gives each cruise equal influence and confirms that none of the modern candidates beats the honest random-forest baseline by the frozen five-percent margin.

## fig03_common_grid_rmse.png

Figure 3. Monthly common-grid RMSE preserves the same ranking as native records, showing that repeated sampling within grid months does not explain the random forest advantage.

## fig04_controlled_factors.png

Figure 4. Controlled CatBoost effects isolate target residualization from adding cross-fitted SSS; SSS gives small consistent gains, while residualization alone does not improve ordinary held-cruise skill.

## fig05_forward_periods.png

Figure 5. Strict forward-time RMSE in 2019-2021 and 2022-2025; residual sharing reduces the late-period error and bias but every candidate remains above the 40 microatmosphere minimum target.

## fig06_lme_distributions.png

Figure 6. Per-LME RMSE distributions reveal a long geographic error tail and the tradeoff between random-forest interpolation and residual-model temporal transfer.

## fig07_failure_map.png

Figure 7. Whole-LME GMM-minus-random-forest RMSE difference at observation-weighted LME centers; positive red points mark regions where soft experts transfer less reliably.

## fig08_caribbean.png

Figure 8. Caribbean Sea LME 12 RMSE against the prior Issue 26 regional candidate; no global candidate preserves its cruise, spatial, and forward skill simultaneously.

## fig09_qualified_lmes.png

Figure 9. Development-only LME qualification counts; GMM meets simple absolute/positive-skill thresholds in more LMEs, but this does not override failed global, calibration, and paired-improvement gates.

## fig10_extremes.png

Figure 10. Cruise-held-out RMSE in low, central, and high observed-fCO2 bands; extreme values remain substantially harder than the central distribution.

## fig11_annual_bias.png

Figure 11. Annual cruise-OOF bias for the principal models; the random forest underestimates the observed temporal rise more strongly than residual-sharing candidates.

## fig12_interval_coverage.png

Figure 12. Leakage-safe 50 and 90 percent interval coverage; forward-time coverage collapses below the frozen calibration ranges for every leading model.

## fig13_risk_coverage.png

Figure 13. Forward RMSE after retaining cases with the narrowest empirical 90 percent intervals; ranking risk helps triage cases but does not establish unsampled-grid reliability.

## fig14_bootstrap.png

Figure 14. Cruise-level paired MSE bootstrap contrasts against the honest v1.x random forest; every modern candidate has a positive confidence interval in cruise and spatial validation.

## fig15_selection_gate.png

Figure 15. Frozen selection-gate totals for each modern candidate; none passes more than three of ten checks, so no architecture is promoted.
