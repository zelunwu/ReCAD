# Validation correction (2026-10-13)

This archive remains an immutable record of the Issue #41 development run. The later exposure and validation audit found that legacy locked/external groups had entered the v2.3 development lineage, the spatial folds did not represent unseen geographic blocks, and calibration residuals were not generated wholly inside each outer-training partition. Its point rankings remain development diagnostics, but its sealed-label, strict-spatial, leakage-safe-uncertainty, and exact-v1 identity interpretations are withdrawn.

The replacement baseline evidence is archived in [`p2_validation_repair_v2.4`](../p2_validation_repair_v2.4/REPORT.md). It separates an exact frozen v1-style RF from the modernized RF, nests the upstream SSS and calibration lineage, uses actual geographic blocks plus a 100 km buffer, and applies past-only forward windows. Model-family comparisons must be rerun on that contract in Issue #62.
