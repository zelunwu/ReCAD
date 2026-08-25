# Migrating v1.1 → v2.0

This document maps every piece of the v1.1 pipeline (the published NACCOM
product, Wu et al. 2024 ESSD) to its v2.0 implementation, so the migration is
auditable *line by line*.

## Where v1.1 lived

v1.1 was a set of notebooks and MATLAB live scripts in one flat directory:

| v1.1 artifact | Purpose | v2.0 home |
|---|---|---|
| `Data_readalldata.ipynb` | read SOCAT/OISST/SODA/CMEMS/CCMP/NOAA, QC, fCO2→pCO2, xCO2→pCO2air, climatology/anomaly, export `satellite_socat_NACCOM.nc` | `recad/data/ingest.py`, `pipeline.py`, `utils/chem.py` |
| `RFR_models.mlx` | build 4-D meshes, hold out 2004–2005, 80/20 split (`create4DSplitIndices`), train RF, reconstruct | `recad/data/grid.py`, `split.py`, `train/trainer.py`, `predict/reconstruct.py` |
| `trainRFER7.m` / `trainRFER8.m` | bagged RF (300 trees) training + 10-fold/leave-year validation | replaced by the ST-Transformer + deep ensemble (`recad/model/`) |
| `RFR_models_test.mlx` | leave-years-out robustness (2004–2005 test) R²/RMSE table | `evaluate/validate.py` (same table format via `per_year_metrics`) |
| `u_inputs_Monte_Carlo.mlx` | 100-draw MC input-error propagation, RSS combination | `uncertainty/input_mc.py`, `decompose.py` (defaults preserved) |
| `calculateR2RMSE.m` | R² = corr², RMSE = sqrt(mse) | `train/metrics.py` (definitions preserved exactly) |
| `Data_product_subset.ipynb` | subset + restructure product NetCDF | `predict/export.py` (`subset_product`) |
| hand-drawn regional masks (GoMe, SS, SAB, …) | coastal domain | GSHHG coastline-distance mask (`coastal_mask.method: distance`), configurable |

## Predictor mapping (unchanged by design)

v1.1 `T_train` columns: `[lon, lat, month(sin), sss, sst, adt, pco2air]`
(+ `wspd` in the leave-year-out variant). v2.0 keeps the exact same predictor
set (`FEATURE_NAMES = sst, sss, adt, pco2air, wspd` + lon/lat/month
features) so head-to-head benchmarks against the published RFR product are
meaningful.

## Behavioural changes (all deliberate, all documented)

1. **Detrending**: v1.1 `calc_clim_anom` used `scipy.signal.detrend`, which
   propagates NaN; v2.0 fits the linear trend on valid samples only.
2. **Splits**: v1.1's global random 80/20 + 2004-2005 hold-out is available
   verbatim (`split.scheme: random_80_20`, `test_holdout_years: [2004, 2005]`);
   the *default* is the more rigorous blocked spatio-temporal split.
3. **Uncertainty**: v1.1 reported only the input-error MC term; v2.0 adds
   aleatoric + epistemic terms and combines all three in quadrature (RSS),
   which subsumes the v1.1 result.
4. **Model**: RFR → ST-Transformer deep ensemble (rationale:
   `docs/design.md` §2). For benchmarking, the RFR recipe itself is *not*
   re-implemented in v2.0; the v1.1 MATLAB code remains the reference and
   `configs/naccom_1over8.yaml` makes the v2.0 side of the comparison as
   close as possible (same domain, predictors, QC, split).
5. **Grid/resolution**: 0.25° regional → 1/8° global (config-driven).

## Reproducing the v1.1 product with v2.0 tooling

```bash
recad ingest   --config configs/naccom_1over8.yaml   # after staging NACCOM raw files per data_sources.md layout
recad preprocess --config configs/naccom_1over8.yaml
recad train    --config configs/naccom_1over8.yaml
recad predict  --config configs/naccom_1over8.yaml
recad validate --config configs/naccom_1over8.yaml   # 2004-2005 table, v1.1 format
```