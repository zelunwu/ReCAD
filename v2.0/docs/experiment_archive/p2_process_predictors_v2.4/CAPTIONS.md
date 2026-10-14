# Figure captions

## fig01_sss_factor_rmse.png

Figure 1. SSS held-out RMSE for the CatBoost baseline and each deployable process factor on the exact corrected cruise, geographic plus 100 km buffer, whole-region, and past-only forward folds; none of the three factors passes the preregistered two-scheme retention gate.

## fig02_fco2_factor_rmse.png

Figure 2. fCO2 held-out RMSE for the ExtraTrees baseline, three deployable process proxies, and the continuous spherical-seasonal-trend background residual; only the continuous background passes the frozen screen retention rule.

## fig03_relative_improvement.png

Figure 3. Single-seed RMSE improvement relative to each target's identical-fold baseline; red cells are regressions and blue cells are improvements, exposing split-specific gains that cannot justify unrestricted feature combinations.

## fig04_background_seed_stability.png

Figure 4. Three-seed mean and standard deviation for the retained fCO2 continuous-background residual and its ExtraTrees baseline; the gain is small and stable, while geographic and whole-region error remain far above publication gates.

## fig05_background_coverage.png

Figure 5. Empirical coverage of nominal 90% intervals for the retained fCO2 branch and baseline; the green band is the 0.85-0.95 target and shows that continuous-background gains do not repair spatial-shift under-coverage.

## fig06_product_domain_coverage.png

Figure 6. Monthly area-weighted availability over all 838,661 frozen coastal product nodes from 1993-2025; SST and wind are complete, background SSS has explicit sparse gaps, and optional ADT has the largest coverage loss.

## fig07_all_common_support.png

Figure 7. Geographic and whole-region RMSE improvement on all rows with deployable fallbacks and on factor-specific common support; reporting both views prevents apparent gains obtained by deleting difficult missing-field cases.

## fig08_process_regimes.png

Figure 8. Cruise-holdout RMSE across quintiles of the tested front, freshwater, and circulation proxies; errors vary strongly by regime, but the added proxy does not deliver the required transferable aggregate improvement.

## fig09_boundary_diagnostic.png

Figure 9. Cruise-holdout RMSE on observation-supported coastal nodes adjacent to a node assigned to another majority LME; the continuous fCO2 background introduces no discrete LME correction and does not create a new hard boundary artifact.

## fig10_source_audit.png

Figure 10. Frozen-cube source-era and preregistered source-availability audit; all deployed cube fields use one standardized source era, while Chl-a, mixed-layer depth, and river discharge remain explicit unavailable negative results rather than fabricated proxies.
