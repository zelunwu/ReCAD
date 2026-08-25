#!/usr/bin/env python
"""Generate a synthetic ReCAD v2.0 workspace (raw data + config).

Usage:
    python scripts/make_synthetic_data.py --workdir outputs/synthetic --n-years 4

This is the developer-facing entry point; ``recad e2e`` uses the same
generator internally (``recad.testing.synthetic``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running from source without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recad.testing.synthetic import make_synthetic_workdir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="outputs/synthetic")
    parser.add_argument("--n-years", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = make_synthetic_workdir(args.workdir, n_years=args.n_years, seed=args.seed)
    print("synthetic workspace written:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
