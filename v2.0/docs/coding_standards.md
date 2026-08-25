# Coding standards & tooling (ReCAD v2.0)

Two languages, two standard bodies — enforced automatically.

## Python — PEP 8 (via Ruff)

- **Formatter/linter**: [Ruff](https://docs.astral.sh/ruff/) (ruff + ruff-format),
  configured in `pyproject.toml` (`[tool.ruff]`), 100-column line length, double
  quotes, import sorting (I), modern typing (UP), bugbear (B), simplify (SIM)
  and comprehensions (C4) rule sets. Run:
  ```bash
  ruff check .          # lint
  ruff format .         # format
  ```
- **Mypy**: strict-ish config in `pyproject.toml` (`[tool.mypy]`); type hints
  are used throughout (`from __future__ import annotations`).
- **Pre-commit** (`.pre-commit-config.yaml`): ruff lint+format, trailing
  whitespace, EOF fixer, YAML checks, large-file guard, and `clang-format` for
  the C++/CUDA sources.
  ```bash
  pre-commit install && pre-commit run --all-files
  ```

## C++/CUDA — LLVM style

- **`.clang-format`** in `csrc/` is `BasedOnStyle: LLVM` (2-space indent,
  camelCase functions, PascalCase types, 80 columns, include ordering:
  own header first). Enforced by the pre-commit `clang-format` hook and
  directly with:
  ```bash
  clang-format -i csrc/src/*.cpp csrc/src/*.cu csrc/include/recad/*.hpp
  ```
- Conventions in `csrc/`: `extern "C"` ABI surface (stable for ctypes),
  CPU reference implementations + CUDA kernels side-by-side, defensive
  bounds checks returning explicit status codes (0 = success), no global
  mutable state, tests in `csrc/tests/` (dependency-free harness, CTest).

## Git hygiene

- `v2.0/.gitignore` excludes environments, build dirs (`csrc/build*/`),
  data/products/checkpoints, caches and IDE noise.
- No large scientific artifacts are committed; everything reproducible is
  regenerated from config + stage scripts.

## Testing

- **Python**: `pytest` in `v2.0/tests/` (see `pyproject.toml`). Run from
  `v2.0/`:
  ```
  pytest                 # unit + fast integration
  pytest -m slow         # e2e pipeline smoke (also in CI)
  pytest --cov=recad     # coverage
  ```
- **C++/CUDA**: `cmake -S csrc -B csrc/build && cmake --build ... && ctest ...`
  (see `csrc/README` in the CMake header). The test harness checks
  hand-computed properties and, when CUDA is enabled, CPU↔CUDA equivalence
  on seeded random inputs.