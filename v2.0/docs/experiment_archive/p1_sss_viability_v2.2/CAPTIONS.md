# Figure captions

## fig01_development_model_comparison.png

Development-set SSS RMSE for GLORYS and five residual-correction candidates under pooled, cruise-equal, and LME-macro aggregation. Points are means across seeds 100-102 and error bars show one standard deviation; lower is better. Soft experts win the preregistered LME-macro criterion, while CatBoost has the lowest pooled and cruise-equal RMSE.

## fig02_five_fold_cv.png

Five-fold cruise-grouped cross-validation for the same candidate families. Bars show mean LME-macro RMSE and pooled skill relative to GLORYS across folds; error bars show one standard deviation. CatBoost is the strongest cross-validation sensitivity comparator.

## fig03_lme_skill_sensitivity.png

Development skill by Large Marine Ecosystem (LME), comparing the selected soft-expert ensemble with CatBoost; positive skill indicates lower MSE than GLORYS. Labels give development record counts. The soft-expert selection advantage is sensitive to sparse LME 55 (n=30).

## fig04_salinity_band_skill.png

Development skill of the selected soft-expert ensemble across observed-SSS bands, with record counts shown separately. Skill remains positive in every reported band, including low-salinity coastal observations, but the freshest bands have much smaller support.

## fig05_forward_chain.png

Forward-chain development skill for seeds 100-102: training cruises end by 2018 and evaluation uses cruises assigned to 2019-2021. All seeds retain positive pooled skill relative to GLORYS; the horizontal line marks zero skill.

## fig06_sss_support_distance.png

Development skill of the selected soft-expert ensemble by distance to the nearest SSS training support, with record counts by bin. Positive skill persists in all populated bins, but distant support bins contain fewer observations and remain an extrapolation risk.

## fig07_observed_vs_predicted.png

Hexbin density of development observations against GLORYS and the three-seed mean soft-expert prediction on identical 0-40 PSU axes. The white line is 1:1 and color is log10 record count. Displayed RMSE is the row-pooled ensemble RMSE, not the across-seed mean reported in Table 1.

## fig08_absolute_error_calibration.png

Development absolute-error empirical distributions for GLORYS and the selected soft-expert ensemble. The dashed line is the development-calibrated 90th-percentile error threshold (1.156 PSU); its 0.900 development coverage is not independent calibration evidence.
