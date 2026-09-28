# ReCAD v2.0 — Scientific & Engineering Design

**Reconstructed Coastal Acidification Database, version 2.0**

- **v1.1 baseline**: Wu, Z., Lu, W., Roobaert, A., Song, L., Yan, X.-H., & Cai, W.-J.
  (2024). *A machine-learning reconstruction of sea surface pCO2 in the North
  American Atlantic Coastal Ocean Margin from 1993 to 2021.* ESSD.
  https://doi.org/10.5194/essd-2024-309
- **v2.0 goal**: global coastal 1/8-degree monthly reconstruction of sea
  surface fCO2/pCO2 (1993–2021), built on a spatio-temporal transformer deep
  ensemble, GPU-accelerated, with quantitative uncertainty decomposition.
- **Data stack**: see [`data_sources.md`](data_sources.md) (recommended
  sources: SOCAT, OISST v2.1, GLORYS12v1, CMEMS SSH, CCMP v3.1, NOAA GML xCO2,
  GSHHG coastal mask, GEBCO 2025 bathymetry).

---

## 1. Problem statement and why v1.1's model is being replaced

v1.1 reconstructed fCO2/pCO2 in the North American Atlantic Coastal Margin
(NACCOM, 0.25°, monthly, 1993–2021) with a bagged random forest
(`fitrensemble`, 300 trees) over 7 predictors: longitude, latitude, cyclic
month, SSS, SST, ADT (SSH), pCO2(air). The approach is sound and the product
is published, but three scientific limitations motivate the v2.0 upgrade:

1. **No spatial structure.** Each (month, cell) sample is predicted
   independently. Coastal cells covary strongly through shared water masses,
   along-shelf transport and shelf-break exchange; a model that cannot exploit
   spatial context leaves information on the table and inflates small-scale
   noise.
2. **No temporal structure beyond a cyclic month feature.** The seasonal
   cycle and interannual variability (e.g., El Niño teleconnections, river
   plume years) are encoded only through the sin/cos month coordinate and the
   predictor fields themselves. A model with an explicit temporal attention
   stage can learn seasonal *co-variation* among distant coastal regions and
   anomaly propagation.
3. **No principled uncertainty decomposition.** v1.1 estimated uncertainty by
   Monte-Carlo propagation of input measurement errors only
   (`u_inputs_Monte_Carlo.mlx`). It could not separate the aleatoric
   (irreducible scatter) and epistemic (model/parametric) contributions.

v2.0 keeps everything that made v1.1 scientifically defensible — the exact
predictor set, QC thresholds, v1.1-compatible benchmark split, and the
v1.1-style input-error Monte-Carlo — and replaces the regression core with an
architecture designed for spatial+temporal context plus a deep ensemble for
calibrated uncertainty.

---

## 2. Model: spatio-temporal transformer (ST-Transformer) deep ensemble

### 2.1 Why a transformer, and why not alternatives

| Approach | Verdict for this problem |
|---|---|
| **Random forest / GBDT (v1.1)** | Strong tabular baseline; no spatial/temporal context; O(N) inferences make global 1/8° runs slow; no calibrated uncertainty. Kept in v2.0 *only* as a benchmark (see §5). |
| **FNN / SOM-FFN (Laruelle et al. style)** | Good coastal pCO2 precedent (SOM-FFN coastal product); still per-sample; no attention-based context. |
| **Conv nets (U-Net etc.)** | Excellent local spatial context, but fixed receptive field; global coastal margins are long and discontinuous — a U-Net on the global ocean or on coastal strips is awkward; and temporal context requires 3-D convs that are memory-heavy at 1/8°. |
| **Gaussian processes / GP+NN hybrids (Gloege et al. 2021)** | Rigorous UQ, but cubic scaling; global 1/8° monthly is far beyond practical GP size; approximations (SVGP) lose fidelity. |
| **Transformers (this work)** | Self-attention gives *arbitrary-range* spatial context (Gulf of Mexico ↔ Scotian Shelf covariation) and a natural second attention axis over time. Token/patch design bounds memory (see §2.2). Publication precedent is strong and growing in Earth-system ML (FourCastNet, Pangu-Weather, GraphCast-era transformer literature). |
| **Denoising diffusion** | Excellent generative prior and sampling-based UQ, but (i) far heavier training/inference (multiple denoising steps per sample × millions of cells), (ii) its strength — full generative modelling of the field — is not needed for pointwise regression with calibrated error bars, (iii) less established for this exact reconstruction problem. A diffusion head is a *reserved* v2.x extension (see §7). |

The transformer is therefore the "more advanced, more appropriate" model:
- **(a)** it is a regression model with a pointwise decoder (interpretable in
  the pCO2-mapping sense), not a black-box generator;
- **(b)** it naturally handles missing predictors (NaN) through mask
  embeddings and missing-flag channels, mirroring v1.1's NaN handling;
- **(c)** tokenization at the patch level bounds the attention cost (see
  below), making global 1/8° tractable on a single GPU;
- **(d)** a deep ensemble (Lakshminarayanan et al., 2017) wraps it in
  calibrated uncertainty — the standard, peer-reviewed way to get UQ out of
  deep regressors.

### 2.2 Architecture (implemented in `recad/model/st_transformer.py`)

```
per-cell features            patch tokens (per month)
 [B, T, C, F_cell]            [B, T, P, F_token]
        │                              │
        │                     token linear embed (+ mask token) + pos-enc
        │                     (lon/lat sinusoidal) + month enc
        │                              ▼
        │                  SPATIAL stage: transformer encoder over P
        │                  patches (shared weights across months)
        │                              ▼
        │                     ┌→ [B, T, P, D] per-patch embeddings
        │                     │
        │                     ▼
        │                  TEMPORAL stage: transformer encoder over the
        │                  12-month window per patch (seasonal context)
        │                              ▼
        │                     [B, T, P, D] contextual patch states
        │                              │
        │          gather per cell: cell_patch -> its patch state
        │                              ▼
        └──────────────► concat(cell embed, patch context) ──► head
                                                                 │
                                      mean (+ logvar | quantiles)
```

- **Spatial stage**: coastal cells are grouped into square patches
  (default 16×16 cells at 1/8° = 2°×2°). Every patch that intersects the
  coastal mask is a token; attention runs over the (≤ ~10k) active patch
  tokens per (year, month) with a padded-attention mask. Patch *features* are
  the per-patch means of the z-scored predictors plus a coverage channel —
  both computed with the native CUDA `PatchAggregate` kernel
  (`recad/utils/native.py` ↔ `csrc/`). Tokens are **year-specific**, so the
  temporal stage can learn interannual patch-level anomalies, not just the
  climatological seasonal cycle. Patches communicate over arbitrary
  distances, which is the point of the attention stage.
- **Temporal stage**: for each patch, the 12 monthly embeddings form the
  sequence; a second transformer encoder (full, non-causal attention —
  seasonal context is bidirectional) learns the annual co-variation. The
  output is the per-cell *context vector* used by the decoder.
- **Decoder/head**: per cell, the head concatenates (i) the cell's own
  feature vector (lon/lat, cyclic month, z-scored predictors with explicit
  missing flags) and (ii) the temporal context of its patch. The default head
  is a **heteroscedastic Gaussian head** firing (mean, log-variance) — the
  log-variance is the *aleatoric* uncertainty emitted by the model itself
  (clamped to [-8, 5] for stability).
- **NaN handling**: missing predictors are zero-filled with a dedicated
  missing-flag channel; patches with no valid data in a month are masked in
  attention (their embedding is replaced by a learned `mask_token`). This is
  the v1.1 NaN philosophy, tensorised.

### 2.3 Deep ensemble (implemented in `recad/model/ensemble.py`)

A deep ensemble of `n_members` (default 5) ST-transformers is trained; members
differ by random seed and by a **year-level bootstrap** of the training pool
(`EnsembleConfig.bootstrap_fraction`, default 0.8 — the year is the natural
exchangeable unit for pCO2 training data, matching v1.1's yearly block
design). Inference combines members (independent sources):

```
mean        = mean over members of member means
aleatoric²  = mean over members of the Gaussian-head variances
epistemic²  = variance over member means             (model disagreement)
model_std   = sqrt(aleatoric² + epistemic²)
```

This is the Lakshminarayanan et al. (2017) recipe, which is *calibration-
friendly* (empirically better calibrated than MC-dropout for regression) and
simple to reason about in a geoscience manuscript.

---

## 3. Uncertainty decomposition (implemented in `recad/uncertainty/`)

The final product uncertainty is the quadrature (RSS) combination of three
*independent, physics-meaningful* components:

```
σ_total = sqrt( σ_aleatoric² + σ_epistemic² + σ_input² )
```

1. **Aleatoric** — from the Gaussian-head variance (averaged over members):
   the irreducible scatter of fCO2 given the predictors.
2. **Epistemic** — ensemble spread: how much the answer would change with a
   different training draw.
3. **Input** — Monte-Carlo propagation of the observation errors of the
   *input* fields through the frozen ensemble, exactly as v1.1 did
   (`u_inputs_Monte_Carlo.mlx`, 100 draws; v1.1 defaults u_SST = 0.23 °C,
   u_SSS = 0.6 PSU, u_SSH = 0.018 m, u_pCO2air = 0.22 µatm, u_wind = 0.901
   m/s), with optional per-pixel SST/SSH error fields. v2.0 re-tokenizes the
   perturbed fields for every draw, so perturbations propagate into the
   spatial context — strictly more rigorous than perturbing rows of a flat
   table (the v1.1 approach), at an N-draw compute cost that is configurable.

The RSS rule is the same variance-sum principle v1.1 used to combine its
per-input contributions; the sources are genuinely independent, so the
quadrature is defensible (documented in the paper's uncertainty section).

---

## 4. Data & preprocessing (implemented in `recad/data/`)

| Step | v1.1 | v2.0 |
|---|---|---|
| Target | SOCAT v2023 quarter-deg coastal fCO2 | SOCAT (v2025/v2026, PMEL ERDDAP) coastal fCO2 |
| Grid | 0.25° regional, hand-drawn masks | 1/8° global, GSHHG coastline-distance mask (≤200 km, reproducible) |
| QC | scattered literals (`fco2<1`, `sla<-10`, `pco2air<200`, 3σ) | same thresholds, centralised in `constants.py`, unit-tested |
| Chemistry | PyCO2SYS for xCO2→pCO2air; Wanninkhof(1992) fCO2→pCO2 | identical conversions, wrapped in `utils/chem.py` with tests |
| Climatology/anomaly | detrend-then-average (NaN-blind) | NaN-aware linear detrend (fits only valid samples) |
| Split | random 80/20 + hold-out 2004–2005 | default: blocked spatio-temporal + same hold-out; v1.1 scheme reproducible via config |

Longitudes use the `[0, 360)` convention throughout (SOCAT and the v1.1
satellite handling both used it). Regridding is xarray-based (linear for
continuous fields, nearest for masks) against the target mesh; the heavy
masked-reduction kernels (patch aggregation) run in CUDA through the native
library, and the exact same Python code runs without the native DLL via NumPy
fallbacks (CI-smoke tested).

---

## 5. Validation strategy (implemented in `recad/evaluate/`)

Three complementary spokes, mirroring and extending v1.1:

1. **v1.1-comparable hold-out (production benchmark)**:
   `split.test_holdout_years = [2004, 2005]` are held out of training entirely
   (exactly as v1.1's `RFR_models.mlx`); the 20%-of-pool random split inside
   the training years monitors training. Test-year per-year R2/RMSE table is
   output (same table v1.1 printed in `RFR_models_test.mlx`).
2. **Blocked spatio-temporal split (v2.0 default)**:
   train/validation samples are separated in *space* (2° blocks) and *time*
   (6-month blocks) so autocorrelated neighbours never straddle the
   train/validation boundary — the rigorous alternative to a naive random
   split, which leaks through correlation.
3. **Synthetic end-to-end (CI)**:
   `recad e2e` trains the whole pipeline on deterministic synthetic coastal
   data and asserts physical sanity of the product (finite fCO2 in a
   plausible range, structure present, no shape regressions).

Metrics are byte-for-byte the v1.1 definitions
(`calculateR2RMSE.m`: R² = squared Pearson correlation, RMSE = sqrt(mean
squared error)).

---

## 6. GPU execution & scaling (implemented in `recad/train/` & `recad/model/`)

- **Device**: `TrainConfig.device = auto` resolves cuda → mps → cpu; every
  tensor op goes through the selected device; AMP (`torch.autocast` +
  `GradScaler`) is on by default for CUDA.
- **Memory**: the per-item tensors are `[T=12, P, F_token]` (tokens) and
  `[T=12, C, F_cell]` (cells). `TrainConfig.tile_cells` optionally subsamples
  coastal cells deterministically per item, bounding the cell-block memory
  while keeping the patch/token structure intact; `batch_windows` groups
  year-windows per optimizer step. A config file is the only thing that
  changes between a laptop smoke run and a global run
  (`configs/test_smoke.yaml` vs `configs/global_1over8.yaml`).
- **Native kernels**: patch aggregation, masked moments and z-scoring are
  CUDA kernels (`csrc/`, LLVM style), loaded through ctypes with NumPy
  fallbacks; CTest verifies CPU↔CUDA equivalence.

---

## 7. Reproducibility, reserved extensions, and caveats

- **Reproducibility**: one YAML config per experiment (all scientific knobs),
  global seeding (`split.seed = 100` keeps v1.1's convention), member seeds
  derived deterministically, feature-normalisation fitted on the training
  split only (no leakage), every persisted artifact (prepared data, masks,
  checkpoints, histories, products) loadable and auditable (`config_sha256`
  embedded in products).
- **Reserved v2.x extensions** (designed for, not blocking v2.0):
  a diffusion head, explicit SLA/wind channels, climatology-anomaly features,
  per-pixel SST/SSH error fields in the MC step, and a 3-D spatio-temporal
  attention over (patch × month) jointly.
- **Caveats**: (1) the global 1/8° product needs the data-staging plan in
  `data_sources.md` (single download → standardized zarr); (2) the MC
  input-error map is the dominant compute at global scale — the CLI exposes a
  `tile_cells` subsample for that pass; (3) SOCAT coastal coverage is
  heterogeneous; the coastal mask and coverage channel are deliberate,
  documented modeling choices, not silent assumptions.

---

## References

- Wu, Z. et al. (2024). ESSD. doi:10.5194/essd-2024-309 (v1.1 product)
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles. NeurIPS.
- Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian
  deep learning for computer vision? NeurIPS.
- Gloege, L. et al. (2021). Quantifying errors in observationally based
  estimates of ocean carbon air-sea flux. Global Biogeochemical Cycles.
- Wanninkhof, R. (1992). Relationship between wind speed and gas exchange
  over the ocean. JGR.
- Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
- Laruelle, G. G., et al. (2017). Global coastal pCO2 product (SOM-FFN).
  Global Biogeochemical Cycles.