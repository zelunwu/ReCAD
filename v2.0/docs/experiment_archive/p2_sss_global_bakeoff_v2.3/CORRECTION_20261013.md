# Validation correction (2026-10-13)

This archive remains an immutable record of the Issue #40 development run. The later exposure and validation audit found that its so-called spatial folds held dominant-cruise blocks rather than unseen geographic blocks, most held grid cells and 5-degree blocks were also represented in training, and SSS interval widths used bands derived from observed salinity. Its point predictions remain useful development diagnostics, but its strict-spatial, sealed-label, deployable-uncertainty, and model-promotion interpretations are withdrawn.

The replacement baseline evidence is archived in [`p2_validation_repair_v2.4`](../p2_validation_repair_v2.4/REPORT.md). It uses actual 5-degree block holdout with a 100 km buffer, complete-group nested calibration, prediction-only interval strata, and rolling past-only windows. Broader model-family results must be rerun on that contract in Issue #62.
