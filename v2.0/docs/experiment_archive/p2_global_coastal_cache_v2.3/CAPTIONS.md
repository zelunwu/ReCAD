# Figure captions

### fig01_global_observation_map.png

Figure 1. Global two-degree audit view of SOCATv2026 Coastal cruise-node-month observations on the complete configured 1/8-degree grid; color and marker size encode retained density, and the map is evidence of observation coverage rather than model skill.

### fig02_processing_funnel.png

Figure 2. Row counts at each ordered processing stage, from source observations through identity/time validation, exact duplicate removal, target QC, complete-grid matching, and cruise-node-month aggregation; the legacy v2.2 mask row is a diagnostic subset rather than a filter.

### fig03_basin_coverage.png

Figure 3. Retained fCO2 and in-situ SSS cruise-node-month coverage across the six frozen coarse basins; this exposes geographic imbalance that later macro-region scoring must prevent from being hidden by observation-weighted metrics.

### fig04_lme_support_status.png

Figure 4. Counts of the 66 LMEs classified as supported, under-supported, or empty for each target using the frozen audit thresholds; empty categories remain explicit and cannot inherit a global validation claim from covariate availability.

### fig05_season_coverage.png

Figure 5. Seasonal target coverage after global QC and coastal matching; the counts diagnose temporal sampling imbalance and define strata required in later model evaluation rather than constituting a performance result.

### fig06_decade_coverage.png

Figure 6. Retained target coverage by decade, including the partial 2020s represented in SOCATv2026; the temporal expansion motivates forward-development evaluation and prevents random splits from being the sole evidence.

### fig07_platform_coverage.png

Figure 7. The fifteen platforms contributing the most compact grid-month rows; concentration in a small platform subset motivates platform and cruise grouped uncertainty analyses in subsequent issues.

### fig08_split_balance.png

Figure 8. Cruise counts in the five deterministic cruise-grouped outer folds; every normalized Expocode belongs to exactly one fold, and balance is descriptive rather than optimized using target values.

### fig09_nearest_support.png

Figure 9. Distance from every occupied observation node to its nearest training-role fCO2 or SSS support node, binned in kilometres; long-distance tails quantify where grouped development depends on spatial transfer and may require abstention.

### fig10_non_na_split_coverage.png

Figure 10. Retained rows inside and outside the exact P1 North-American-adjacent window, separated into train and grouped-development roles; non-NA observations occur in both roles, satisfying the central global-cache acceptance gate.
