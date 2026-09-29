# Figure captions

## fig01_development_model_comparison.png

Development-set fCO2 RMSE for the seasonal-trend climatology and four residual models under pooled, cruise-equal, and LME-macro aggregation. Bars are means across available seeds and error bars show one standard deviation; lower is better. CatBoost wins the preregistered LME-macro criterion as well as pooled and cruise-equal RMSE.

## fig02_five_fold_cv.png

Five-fold cruise-grouped cross-validation for the same candidate families. Bars show mean LME-macro RMSE and pooled skill relative to the seasonal-trend climatology across folds; error bars show one standard deviation. CatBoost is also strongest in grouped cross-validation.

## fig03_lme_skill_sensitivity.png

Development skill by Large Marine Ecosystem (LME), comparing selected CatBoost with soft experts; positive skill indicates lower MSE than the seasonal-trend climatology. Labels give development record counts. CatBoost degrades in LME 55 (n=30), LME 8, and LME 17.

## fig04_fco2_band_skill.png

Development skill of CatBoost and soft experts across observed fCO2 bands, with record counts. CatBoost retains positive skill in every band; sparse concentration extremes remain less certain.

## fig05_forward_chain.png

Forward-chain development skill for seeds 100-102: training cruises end by 2018 and evaluation uses cruises assigned to 2019-2021. All CatBoost seeds retain positive pooled skill relative to the seasonal-trend climatology; the horizontal line marks zero skill.

## fig06_fco2_support_distance.png

Development skill of selected CatBoost by distance to the nearest fCO2 training support, with record counts by bin. Skill turns negative beyond 100 km and is strongly negative beyond 250 km, where only 62 records are available; this failure blocks the development gate.

## fig07_observed_vs_predicted.png

Hexbin density of development observations against the seasonal-trend climatology and the three-seed mean CatBoost residual prediction on identical 0-1000 µatm axes. The white line is 1:1 and color is log10 record count. Displayed RMSE is the row-pooled ensemble RMSE, not the across-seed mean reported in Table 1.

## fig08_absolute_error_calibration.png

Development absolute-error empirical distributions for the seasonal-trend climatology and CatBoost residual. The dotted line is the development-calibrated 90th-percentile error threshold (49.46 µatm); its 0.900 development coverage is not independent calibration evidence.
