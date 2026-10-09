# P2 global coastal mask v2.3

Reviewer-ready evidence for Issue #50. Install `.[coastal-mask]`, fetch the three pinned 0.05-degree landmetrics distance grids, and prepare GEBCO with `python scripts/prepare_gebco_2025_regrid.py <external-output-directory>`. Run `python scripts/build_global_coastal_mask_v2_3.py` with the external landmetrics, GEBCO extracts, and frozen Issue #38 cache. Verify this directory with `python scripts/verify_experiment_archive.py docs/experiment_archive/p2_global_coastal_mask_v2.3`.

The large NetCDF product remains outside Git; its path and SHA256 are frozen in the report and manifest.
