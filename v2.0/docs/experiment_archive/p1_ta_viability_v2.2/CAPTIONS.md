# Figure captions

## fig01_development_model_comparison.png

Development TA RMSE under pooled, cruise-equal, and LME-macro aggregation. All deployable models use background SSS; the purple Carter oracle uses collocated in-situ salinity only to show the information ceiling and is excluded from selection. Lower is better.

## fig02_five_fold_cv.png

Five-fold cruise-grouped training cross-validation LME-macro RMSE. Error bars show one standard deviation across folds. Hierarchical TA-SSS is more stable than the nonlinear residual model, which contrasts with the development ranking and limits the nonlinear claim.

## fig03_lme_skill.png

Development MSE skill of the three-seed hierarchical residual ensemble relative to Carter/ESPER for each LME. Labels show records and cruises. Negative skill in LME 3 and LME 54 is retained; LME 54 has only one development cruise.

## fig04_salinity_band_skill.png

Development skill relative to Carter/ESPER stratified by collocated observed salinity. Observed salinity is used only for evaluation strata, never as a deployable model input; labels give record counts. Skill is negative in the 33-36 and >=36 bands, showing a high-salinity regime failure.

## fig05_support_distance_skill.png

Development skill relative to Carter/ESPER by haversine distance to the nearest training TA observation. Skill is negative in the two most distant populated bins; the worst bin skill is -16.33. Sparse counts do not permit removing these preregistered failures.

## fig06_safe_forward.png

Safe forward-time evidence using only primary-train groups assigned to the frozen <=2018 training period and primary-development groups assigned to 2019-2021. No primary locked groups are materialized.

## fig07_leave_lme_out.png

Leave-one-LME-out transfer skill for hierarchical TA-SSS relative to Carter/ESPER. Positive mean skill is marginal and several complete held-out LMEs degrade, showing that regional transfer remains weak.

## fig08_observed_vs_predicted.png

Development observation-prediction density for Carter with background SSS, the selected three-seed ensemble, and the in-situ-SSS Carter oracle. The selected ensemble pooled RMSE is 90.1 µmol kg⁻¹. The oracle gap isolates the cost of deployable SSS inputs.

## fig09_uncertainty_calibration.png

Development absolute-error distributions. The selected interval half-width (120.6 µmol kg⁻¹) was calibrated only from five-fold training OOF residuals and attains 0.945 development coverage.

## fig10_development_gates.png

Frozen P1.3 development gates. Cruise-equal improvement over both comparators and positive skill in every populated support-distance bin fail, so the locked test remains sealed and TA is diagnostic-only.
