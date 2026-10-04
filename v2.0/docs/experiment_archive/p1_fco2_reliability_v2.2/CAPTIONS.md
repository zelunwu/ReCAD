# Figure captions

## fig01_candidate_comparison.png

Figure 1. Mean outer-scheme RMSE normalized by the training-only seasonal-trend background for every preregistered model and support-shrinkage pair. Lower is better; the dashed line is background parity. CatBoost with strictly cross-fitted SSS and linear 2-5 environmental shrinkage ranked first before regional specialization.

## fig02_core_selective_gate.png

Figure 2. RMSE and retained fraction for the global support-selective candidate within grade A/B development rows. Cruise and spatial-block subsets are accurate, while whole-LME has no retained A/B rows and forward retention is too small, motivating the preregistered regional-expert branch rather than a global claim.

## fig03_regional_lme_comparison.png

Figure 3. Regional-expert RMSE against the seasonal-trend background for the two LMEs nominated using cruise OOF only. Caribbean Sea (LME 12) improves cruise, spatial-block, and forward partitions; Southeast U.S. Continental Shelf (LME 6) fails spatial confirmation and is excluded.

## fig04_lme12_interval_coverage.png

Figure 4. Empirical 50% and 90% interval coverage for Caribbean Sea (LME 12). Cruise OOF supplies base conformal widths; a nested 2016-2018 pseudo-forward calibration fitted only inside the training era inflates future intervals, bringing the untouched 2019-2021 forward coverage to the preregistered acceptance range.

## fig05_chla_ablation.png

Figure 5. CatBoost RMSE with and without log10 MODIS chlorophyll-a on exactly the same finite-Chl-a observations. Chl-a improves all four outer schemes, but the local cache ends in December 2020, so it is evidence for a future refreshed input rather than a valid 2025 product feature.

## fig06_atlas_grade_coverage.png

Figure 6. Grid-month counts in the 2025 core and available 2026 provisional atlas. Only confirmed Caribbean Sea cells with adequate environmental, cruise, effective-group, and SSS support receive grade A; every other grid-month is retained as an auditable grade-D background fallback and is excluded from publication claims.

## fig07_atlas_map_2025_07.png

Figure 7. July 2025 selective fCO2 atlas status. Blue points are grade-A Caribbean Sea regional predictions; light-gray points are grade-D background fallbacks outside the confirmed regional domain or without required support. The map is a development applicability product, not independent validation.

## fig08_lme12_risk_coverage.png

Figure 8. Caribbean Sea RMSE as progressively higher environmental-k64 risk observations are retained. Curves are shown for cruise, spatial-block, and forward partitions; the ordering diagnoses support sensitivity while the formal regional gate remains based on the frozen grade and interval rules.
