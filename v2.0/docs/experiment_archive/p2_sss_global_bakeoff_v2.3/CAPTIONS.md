# Figure captions

## fig01_rmse_matrix.png

Figure 1. Held-out SSS RMSE for all six model families and four validation designs; GMM experts lead ordinary cruise, spatial, and forward tests, while raw GLORYS is best under unseen whole-LME transfer.

## fig02_relative_effects.png

Figure 2. Relative RMSE change from the frozen global CatBoost baseline; positive values mean improvement and expose the GMM model's small supported-domain gain alongside its whole-LME penalty.

## fig03_forward_periods.png

Figure 3. Strict forward-time RMSE separated into 2019-2021 and 2022-2025; every learned model remains above the 1.2 PSU late-forward minimum target.

## fig04_lme_distribution.png

Figure 4. Per-LME RMSE distributions for cruise, spatial, and whole-LME validation; pooled gains do not remove the long regional error tail.

## fig05_salinity_strata.png

Figure 5. Cruise-held-out RMSE by observed salinity band; the largest errors occur at 20-30 PSU and the GMM improvement there is modest.

## fig06_coast_strata.png

Figure 6. Cruise-held-out RMSE across distance-to-land bands, showing how performance changes from nearshore cells to the 200-400 km product margin.

## fig07_front_strata.png

Figure 7. Cruise-held-out RMSE across local monthly GLORYS SSS-gradient bands; sharper background fronts remain harder and are diagnosed without using target labels as predictors.

## fig08_lme17.png

Figure 8. LME 17 RMSE under all validation designs; GMM experts do not improve spatial, whole-LME, or forward transfer in this previously identified low-salinity failure region.

## fig09_interval_calibration.png

Figure 9. Leakage-safe empirical 50% and 90% interval coverage; forward 90% coverage falls below nominal for both CatBoost and GMM, so uncertainty is not release-ready.

## fig10_risk_coverage.png

Figure 10. RMSE after retaining observations with the narrowest predicted 90% intervals; uncertainty ranking separates easier from harder cases but does not establish unsampled-grid reliability.

## fig11_bootstrap.png

Figure 11. Cruise-level paired bootstrap MSE contrasts against global CatBoost; GMM gains are statistically consistent in cruise and spatial tests, whereas ST96 and top-two experts degrade.

## fig12_gate.png

Figure 12. Frozen GMM selection gate outcome; failure of the 5% cruise threshold and whole-LME non-degradation rule prevents model promotion despite several positive diagnostics.
