# Figure captions

## fig01_development_model_comparison.png

Development TA RMSE for independently trained SAB and MAB candidates under pooled, cruise-equal, and frozen subregion-macro aggregation. The in-situ-SSS Carter oracle is diagnostic only and excluded from selection; lower is better.

## fig02_grouped_cv.png

Five-fold cruise-grouped training cross-validation subregion-macro RMSE. Error bars show one standard deviation across folds. SAB estimates are unstable because one frozen fold contains only one training record, while MAB favors the simpler hierarchical linear relation over the nonlinear residual model in CV.

## fig03_observed_vs_predicted.png

Development observation-prediction density for Carter/ESPER and each region's selected model. The selected SAB model has RMSE 99.3 µmol kg⁻¹ and fails to improve pooled Carter error; the selected MAB ensemble reaches 39.9 µmol kg⁻¹.

## fig04_subregion_skill.png

Development MSE skill relative to Carter/ESPER in the preregistered south, central, and north latitude subregions. Labels give record counts. SAB central skill is negative and dominates the regional sample, whereas all three populated MAB subregions show positive skill.

## fig05_support_distance_skill.png

Development MSE skill relative to Carter/ESPER by haversine distance to the nearest training TA observation. Every populated bin is retained regardless of sample size; SAB fails in its dominant 0-25 km bin, while all populated MAB bins are positive.

## fig06_salinity_band_skill.png

Development skill relative to Carter/ESPER stratified by collocated observed salinity, which is used only for evaluation. Sparse low-salinity SAB records have very large errors, and the main 33-36 band also degrades; MAB remains positive but has weak improvement in the 33-36 band.

## fig07_safe_forward.png

Safe forward-time evaluation uses only primary-train groups assigned to forward train and primary-development groups assigned to forward development. SAB has 0 eligible development records, so no forward claim is possible; MAB has 31 records from one cruise.

## fig08_leave_subregion_out.png

Leave-one-subregion-out transfer skill for the hierarchical linear TA-SSS relation relative to Carter/ESPER. Only subregions meeting the frozen minimum of 30 rows and three cruises are evaluated; sparse SAB edge subregions cannot support a complete transfer test.

## fig09_uncertainty_calibration.png

Development coverage of symmetric conformal intervals calibrated from five-fold training OOF absolute residuals. SAB coverage falls inside the frozen 85-95% band, while MAB coverage is 98.5%, showing that its 90% interval is materially overconservative.

## fig10_development_gates.png

Frozen development gates evaluated separately by bight. SAB fails multiple prediction and evidence gates and is classified fail. MAB passes every skill gate but fails uncertainty calibration, so it remains diagnostic-only and the locked test stays sealed.
