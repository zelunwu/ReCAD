# Validation correction (2026-10-13)

This archive remains an immutable record of the Issue #42 development run. The later audit showed that its spatial and calibration contracts were insufficient for claims about unseen-grid transfer or deployable uncertainty. The negative result is therefore limited to the exact short-budget GRU, ST96, and four-neighbour graph-ST96 implementations that were run; it does not reject Transformer, temporal, or graph families in general.

The repaired validation contract and corrected RF/CatBoost baselines are archived in [`p2_validation_repair_v2.4`](../p2_validation_repair_v2.4/REPORT.md). Any renewed context model must be compared under the same actual geographic blocks, 100 km buffer, past-only forward windows, nested upstream lineage, and training-only calibration in Issue #62.
