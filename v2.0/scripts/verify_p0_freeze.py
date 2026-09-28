"""Verify the frozen v2.2 P0 assets without opening sealed labels for scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(16 << 20): h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-input-hashes", action="store_true")
    args = ap.parse_args()
    manifest_path = ROOT / "configs/frozen/data_manifest_v2.2.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    for raw, expected in m["artifacts_sha256"].items():
        path = Path(raw); assert path.exists(), path
        assert digest(path) == expected, f"artifact hash mismatch: {path}"
    for raw, expected in m["inputs_sha256"].items():
        path = Path(raw); assert path.exists(), path
        if args.full_input_hashes:
            assert digest(path) == expected, f"input hash mismatch: {path}"
    ext = ROOT / m["independent_validation"]["path"]
    assert digest(ext) == m["independent_validation"]["sha256"]

    split = pd.read_parquet(Path(m["split_manifest"]))
    assert split.groupby("group_key").split.nunique().max() == 1
    assert split.groupby("group_key").cv_fold.nunique().max() == 1
    assert split.groupby("group_key").forward_split.nunique().max() == 1
    external = set(split.loc[split.split == "external_independent", "group_key"])
    assert len(external) == 41
    provisional = set(split.loc[split.split == "provisional_2026", "group_key"])
    out_dir = Path(m["split_manifest"]).parent
    carbon = pd.read_parquet(out_dir / "na_carbon_cache_v2.2.parquet",
                             columns=["group_key", "split", "year"])
    socat = pd.read_parquet(out_dir / "na_socat_cache_v2.2.parquet",
                            columns=["group_key", "split", "yr"])
    assert carbon.year.max() <= 2024 and socat.yr.max() == 2025
    for table in (carbon, socat):
        assert not table.loc[table.group_key.isin(external), "split"].ne("external_independent").any()
        assert not table.loc[~table.group_key.isin(external), "split"].eq("external_independent").any()
    assert set(socat.loc[socat.yr == 2025, "split"]).issubset(
        {"train", "development", "locked_test", "provisional_2026", "external_independent"})
    assert set(socat.loc[socat.yr == 2025, "split"]) & {"train", "development", "locked_test"}
    availability = out_dir / "inference_availability_v2.2.nc"
    with xr.open_dataset(availability) as ds:
        assert list(ds.year.values) == [2025, 2026]
        ready_2025 = ds.strict_all_inputs_ready.sel(year=2025).sum("node").values
        ready_2026 = ds.strict_all_inputs_ready.sel(year=2026).sum("node").values
        assert (ready_2025 > 0).all()
        assert ready_2026[0] > 0 and not (ready_2026[1:] > 0).any()
    print(json.dumps({"status": "pass", "groups": int(split.group_key.nunique()),
                      "external_groups": len(external),
                      "provisional_2026_groups": len(provisional),
                      "manifest_sha256": digest(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
