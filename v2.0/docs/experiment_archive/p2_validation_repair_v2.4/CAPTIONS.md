# Figure captions

## fig01_sss_rmse.png

Figure 1. Pooled held-out SSS RMSE across four leakage-repaired validation designs. CatBoost correction improves on raw GLORYS for cruise, geographic, and forward validation, but degrades under complete-region transfer and misses the 1.2 psu geographic and forward gate.

## fig02_fco2_rmse.png

Figure 2. Pooled held-out fCO2 RMSE across the strict v1 RF replica, modernized RF, and direct CatBoost. Modern RF leads cruise and forward validation, while direct CatBoost is least poor for geographic and whole-region extrapolation; every preregistered RMSE gate fails.

## fig03_r2.png

Figure 3. Held-out R-squared for SSS and fCO2 under the four repaired validation designs. Ordinary cruise validation remains optimistic relative to geographic and whole-region transfer, especially for fCO2 where geographic R-squared falls to about 0.31 for the best model.

## fig04_cruise_equal_rmse.png

Figure 4. Cruise-equal RMSE gives every held group equal weight instead of allowing dense cruises to dominate. The ranking broadly confirms the pooled metrics while exposing the larger error experienced by sparsely sampled groups.

## fig05_coverage50.png

Figure 5. Empirical coverage of nominal 50% prediction intervals calibrated only inside each outer training partition. Cruise coverage is near nominal, but geographic, whole-region, and forward transfer frequently under-cover, so these intervals are not yet deployable uncertainty estimates.

## fig06_coverage90.png

Figure 6. Empirical coverage of nominal 90% prediction intervals calibrated without held-out labels. Several strict-transfer settings fall outside the preregistered 0.85-0.95 acceptance range, demonstrating distribution shift that internal calibration cannot repair by itself.

## fig07_fco2_interval_width.png

Figure 7. Mean width of the nested 90% fCO2 intervals for each model and validation design. Width must be read together with Figure 6: a narrower interval is useful only when its held-out coverage remains calibrated.

## fig08_risk_coverage.png

Figure 8. Geographic-holdout RMSE after retaining observations with the narrowest predicted 90% intervals. The curves test whether interval width ranks error risk; they do not establish accuracy at coastal grid cells with no observations.

## fig09_prediction_coverage.png

Figure 9. Fraction of held-out observations receiving an eligible prediction. The strict v1 RF loses roughly 2.8-2.9% of fCO2 rows through its complete-case feature contract, whereas the modern models and residual SSS path retain all rows.

## fig10_split_sizes.png

Figure 10. Mean fit and held-out row counts per fold after complete-group exclusion and, for geographic folds, the 100 km spatial buffer. Smaller geographic training sets are a deliberate consequence of leakage control rather than failed data loading.

## fig11_leakage_audit.png

Figure 11. Summed leakage diagnostics across every target and validation design. All recorded group overlap, record overlap, fit-calibration group overlap, and forbidden future-time counts are exactly zero.

## fig12_gate_matrix.png

Figure 12. Count of passed applicable preregistered gates for each target, validation design, and model; whole-region has two applicable coverage gates because no absolute RMSE ceiling was frozen for that scheme. Passing a subset does not promote a model; Issue 61 unblocks fair model-family screening while retaining all baselines as development evidence only.
