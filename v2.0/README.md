# ReCAD v2.0

**Reconstructed Coastal Acidification Database** — global-coastal sea-surface
pCO2/fCO2 reconstruction with a spatio-temporal transformer deep ensemble,
GPU-accelerated, engineered for reproducible science.

| | v1.x (published) | v2.0 (this repository) |
|---|---|---|
| Domain | North American Atlantic Coastal Margin | Global coastal ocean |
| Resolution | 0.25° | 1/8° (config-selectable, incl. 1/12°) |
| Model | Bagged random forest (300 trees) | Spatio-temporal transformer + deep ensemble |
| Uncertainty | Input-error Monte-Carlo only | Aleatoric + epistemic + input-error, combined in quadrature |
| Compute | CPU (MATLAB) | GPU (PyTorch/CUDA), CPU fallback |
| Code | Single-folder notebooks/scripts | Modular package + native CUDA kernels + tests |

**Citation (v1.1 product):** Wu, Z., Lu, W., Roobaert, A., Song, L., Yan,
X.-H., & Cai, W.-J. (2024). *A machine-learning reconstruction of sea surface
pCO2 in the North American Atlantic Coastal Ocean Margin from 1993 to 2021.*
Earth System Science Data. https://doi.org/10.5194/essd-2024-309

**Scientific design:** [`docs/design.md`](docs/design.md) — model choice,
uncertainty decomposition, validation strategy.
**Data stack:** [`docs/data_sources.md`](docs/data_sources.md).
**v1.1 → v2.0 mapping:** [`docs/migration_v1_to_v2.md`](docs/migration_v1_to_v2.md).
**Coding standards:** [`docs/coding_standards.md`](docs/coding_standards.md).

---

## Quick start

```bash
# 1. environment (Python >= 3.10; GPU optional but recommended)
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA GPU

# 2. end-to-end smoke test on deterministic synthetic data (no downloads)
recad e2e --workdir outputs/smoke --n-years 2

# 3. unit + integration tests
pytest

# 4. native CUDA kernels (LLVM-style C++/CUDA), see csrc/
cmake -S csrc -B csrc/build_cuda -G "Ninja" -DCMAKE_CUDA_COMPILER=<nvcc> ...
cmake --build csrc/build_cuda && ctest --test-dir csrc/build_cuda
```

## Full pipeline (config-driven, reproducible)

```bash
recad describe                                    # registered data sources
recad ingest      --config configs/global_1over8.yaml      # stage raw -> standardised cache
recad preprocess  --config configs/global_1over8.yaml      # PreparedData + split masks + feature stats
recad train       --config configs/global_1over8.yaml      # deep ensemble (GPU)
recad predict     --config configs/global_1over8.yaml      # full product field (fCO2 + pCO2 + model std)
recad uncertainty --config configs/global_1over8.yaml      # input-error MC + combined sigma_total
recad validate    --config configs/global_1over8.yaml      # hold-out-year R2/RMSE table (v1.1 format)
recad plot        --product outputs/ReCAD-v2.0-pCO2.nc     # v1.1-style figure set + summary tables
                   --prepared outputs/prepared.nc --masks outputs/masks.nc
```

Every knob of an experiment — domain, resolution, coastal mask, data paths,
QC, split scheme, model hyper-parameters, ensemble size, training, MC draws —
lives in one YAML config; see [`configs/base.yaml`](configs/base.yaml) for the
documented defaults, and `recad --help` for CLI options. Any config can be
tuned from the command line: `recad train --config ... --override train.n_epochs=60`.

## Repository layout

```
v2.0/
├── pyproject.toml            # package, deps, ruff/mypy/pytest config
├── configs/                  # base / global_1over8 / naccom benchmark / smoke
├── src/recad/                # the package
│   ├── config.py             # validated, YAML-driven configuration
│   ├── constants.py          # physics + v1.1 QC thresholds (provenanced)
│   ├── data/                 # grid, specs, ingest, features, split, tensorize, pipeline
│   ├── model/                # attention, st_transformer, heads, losses, ensemble
│   ├── train/                # trainer (AMP), callbacks, metrics (v1.1 definitions)
│   ├── uncertainty/          # input-error MC (v1.1 protocol) + RSS combination
│   ├── predict/              # reconstruction + CF-compliant NetCDF export
│   ├── evaluate/             # hold-out validation tables
│   ├── utils/                # logging, io, seeding, carbonate chemistry, native bridge
│   ├── testing/              # deterministic synthetic data generator
│   └── cli.py                # recad <subcommand>
├── csrc/                     # C++/CUDA kernels (LLVM style) + CTest
├── tests/                    # pytest suite (unit + integration + e2e)
├── scripts/                  # developer helpers
└── docs/                     # design, data sources, standards, migration
```

## Requirements

- **Python** ≥ 3.10 (deps: numpy, scipy, pandas, xarray, netCDF4, PyYAML,
  tqdm, matplotlib, torch ≥ 2.2, PyCO2SYS, dask)
- **GPU** (optional): any CUDA-capable GPU; tested on RTX-class hardware with
  CUDA 12.8+ / PyTorch cu128 builds; CPU and Apple-MPS fallbacks included.
- **Native kernels** (optional): CMake ≥ 3.20, a C++17 compiler, and (for the
  CUDA path) the CUDA toolkit; without them the pipeline transparently falls
  back to NumPy.

## Contact

Zelun Wu (zelunwu@outlook.com), Wei-Jun Cai (wcai@udel.edu) —
University of Delaware / Xiamen University. Discussion and cooperation
welcome.