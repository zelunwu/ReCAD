# Figure captions

## fig01_development_rmse.png

Figure 1. Selected-model development RMSE across nested SAB southern boundaries. The apparent improvement at 30.5°N coincides with nonlinear-model activation and removal of the 28.45-30.5°N observations, so it is not a clean water-mass effect.

## fig02_grouped_cv_rmse.png

Figure 2. Frozen cruise-grouped cross-validation RMSE across boundaries. The 30.5°N variant has four populated folds because frozen fold 0 contains no eligible cruise; folds were not reassigned.

## fig03_skill_vs_carter.png

Figure 3. Development skill of each selected model relative to Carter/ESPER, where positive values favor the model. Pooled and cruise-equal skill are negative through 28.45°N despite positive latitude-band macro skill.

## fig04_nested_contrasts.png

Figure 4. Percent RMSE change caused by adding each southern latitude band to the next restricted dataset. The preregistered harmful-band rule requires development cruise-equal degradation above 5% and grouped-CV band-macro degradation; only 28.45-30.5°N meets both, with a model-family change.

## fig05_latitude_band_skill.png

Figure 5. Selected-model skill versus Carter/ESPER within fixed latitude bands. Sparse bands contain only one or two cruises, and the central 30.5-33°N band dominates the development sample.

## fig06_support_distance_skill.png

Figure 6. Skill versus Carter/ESPER by distance from training TA support. Negative worst-bin skill remains for every boundary, showing that boundary restriction does not solve extrapolation risk.

## fig07_salinity_band_skill.png

Figure 7. Skill versus Carter/ESPER by observed-salinity band, used only for diagnosis. Small low-salinity cells are unstable and must not be treated as evidence for operational coastal-estuarine performance.

## fig08_uncertainty.png

Figure 8. Conformal interval coverage and out-of-fold absolute-error 90th percentile. The 30.5°N variant undercovers at 83.4%, outside the frozen 85-95% acceptance range.

## fig09_data_support.png

Figure 9. Observation and cruise support retained by each nested boundary. All development variants contain only four cruises, while the 30.5°N training set has eight cruises and one empty frozen CV fold.

## fig10_gate_checks.png

Figure 10. Frozen development-gate checks by boundary. No variant passes the full gate; safe-forward evaluation is unavailable because no SAB development record satisfies the frozen forward-time intersection.
