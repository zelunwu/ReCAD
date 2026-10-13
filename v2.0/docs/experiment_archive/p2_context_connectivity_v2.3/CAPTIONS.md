# Figure captions

## fig01_native_rmse_matrix.png

Figure 1. Native held-out fCO2 RMSE across point, lag, recurrent, local-transformer, and water-connected transformer candidates; the random forest leads three transfer schemes and lag CatBoost gives only a small forward gain.

## fig02_cruise_equal_rmse.png

Figure 2. Cruise-equal RMSE gives every independent cruise equal influence and confirms that no context candidate reaches the frozen five-percent improvement threshold.

## fig03_relative_improvement.png

Figure 3. Relative cruise-equal improvement against the honest random forest; positive values indicate improvement, and only lag CatBoost is slightly positive in forward validation.

## fig04_forward_periods.png

Figure 4. Strict forward RMSE in 2019-2021 and 2022-2025; lag CatBoost reduces temporal bias but remains above the 40 microatmosphere ceiling in both periods.

## fig05_learning_curve_rmse.png

Figure 5. Cruise-fold-0 pooled learning curves under nested LME-stratified fCO2 cruise-label budgets; random-forest error continues to fall as coverage increases, lag CatBoost plateaus, and ST96 is unstable.

## fig06_learning_curve_cruise_equal.png

Figure 6. Cruise-equal learning curves show that the apparent row count overstates independent information and that ST96 does not unlock additional capacity with the available cruise coverage.

## fig07_graph_residual_structure.png

Figure 7. Water-adjacent residual correlation remains high for every model and is larger after the context networks, so the graph input does not remove the spatially connected error structure that triggered it.

## fig08_front_preservation.png

Figure 8. Correlation between observed and predicted adjacent-grid fCO2 gradients; all candidates smooth fronts strongly and context networks preserve less gradient structure than the point random forest.

## fig09_boundary_discontinuity.png

Figure 9. Mean predicted gradient within and across LME boundaries; residual models inherit artificial cross-LME jumps from the fold-fitted LME-month background while the direct random forest remains smoother.

## fig10_lme_distributions.png

Figure 10. Per-LME RMSE distributions for cruise, spatial, and whole-LME validation expose a long geographic error tail that additional context does not reduce.

## fig11_low_support.png

Figure 11. Cruise-held-out RMSE by the number of training cruises in the target LME; sparse support remains difficult and no complex model supplies a reliable low-support advantage.

## fig12_annual_bias.png

Figure 12. Annual cruise-OOF bias; lag CatBoost follows the long-term signal better than the random forest in some years but pays for it with higher ordinary held-cruise error.

## fig13_interval_coverage.png

Figure 13. Leakage-safe 50 and 90 percent interval coverage; forward coverage remains below the frozen ranges for every candidate despite wider context-model intervals.

## fig14_risk_coverage.png

Figure 14. Forward RMSE after retaining cases with the narrowest empirical intervals; uncertainty ranking cannot reduce any candidate to the release ceiling at full coverage.

## fig15_bootstrap.png

Figure 15. Cruise-level paired MSE bootstrap contrasts against the random forest; every context candidate has confidence intervals above zero in cruise and spatial validation.

## fig16_selection_gate.png

Figure 16. Frozen promotion-gate totals; no context candidate passes the complete contract, so the current STTransformer, GRU, and graph families stop scaling.

## fig17_compute_cost.png

Figure 17. Median outer-fold training time and peak GPU memory; added neural complexity has measurable cost without a generalization return.
