"""Materialize deployable Issue 63 process proxies and source audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from recad.evaluate.process_predictors import (
    cardinal_water_neighbours,
    local_contrast_subset,
)

MATCHUP = Path(
    r"C:\backup\phd\data\processed\recad_v2_3\p2_predictor_ablation_v2.3"
    r"\socat_predictor_matchup_v2.3.parquet"
)
CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\p2_process_predictors_v2.4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_features(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    matchup = pd.read_parquet(
        MATCHUP, columns=["record_id", "time_index", "coastal_node", "yr", "mon"]
    )
    dataset = xr.open_dataset(CUBE, chunks=None, cache=False)
    neighbours = cardinal_water_neighbours(
        dataset.grid_lat_index.values, dataset.grid_lon_index.values
    )
    rows: list[pd.DataFrame] = []
    coverage: list[dict[str, float | int]] = []
    fields = ("sst", "wspd", "sss_bg", "adt")
    area = dataset.cell_area_km2.values.astype(float)
    area_total = float(np.nansum(area))
    for time_index in range(dataset.sizes["time"]):
        observed = matchup.loc[matchup.time_index.eq(time_index)]
        nodes = observed.coastal_node.to_numpy(np.int64)
        part = observed[["record_id"]].copy()
        loaded: dict[str, np.ndarray] = {}
        for field in fields:
            values = dataset[field].isel(time=time_index).values.astype("float32")
            loaded[field] = values
            contrast = local_contrast_subset(values, neighbours, nodes)
            prefix = {
                "sst": "sst_front",
                "wspd": "wspd_front",
                "sss_bg": "sss_local",
                "adt": "adt_local",
            }[field]
            part[f"{prefix}_mad"] = contrast["mean_absolute_difference"]
            part[f"{prefix}_range"] = contrast["range"]
            part[f"{prefix}_neighbour_count"] = contrast["valid_neighbour_count"]
        part["freshwater_deficit"] = np.maximum(0.0, 35.0 - loaded["sss_bg"][nodes]).astype(
            "float32"
        )
        part["sss_process_available"] = np.isfinite(loaded["sss_bg"][nodes]).astype("uint8")
        part["adt_process_available"] = np.isfinite(loaded["adt"][nodes]).astype("uint8")
        part["time_index"] = time_index
        rows.append(part)
        coverage.append(
            {
                "time_index": time_index,
                "year": int(dataset.year.values[time_index]),
                "month": int(dataset.month.values[time_index]),
                **{
                    f"{field}_area_fraction": float(
                        np.nansum(area[np.isfinite(loaded[field])]) / area_total
                    )
                    for field in fields
                },
            }
        )
        if time_index % 24 == 0:
            print(f"features {time_index + 1}/{dataset.sizes['time']}", flush=True)
    cache = pd.concat(rows, ignore_index=True).sort_values("record_id")
    if len(cache) != len(matchup) or cache.record_id.duplicated().any():
        raise RuntimeError("feature cache does not preserve one row per matchup record")
    cache.to_parquet(output / "process_features.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_csv(tables / "product_domain_monthly_coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "factor": "dense_front",
                "source": "frozen SST and wind cube",
                "period": "1993-2025",
                "status": "available",
                "fallback": "zero contrast plus neighbour count",
            },
            {
                "factor": "freshwater_proxy",
                "source": "frozen background SSS cube",
                "period": "1993-2025",
                "status": "available_with_missingness",
                "fallback": "zero plus availability and neighbour count",
            },
            {
                "factor": "circulation_proxy",
                "source": "frozen ADT cube",
                "period": "1993-2025",
                "status": "optional_available_with_missingness",
                "fallback": "zero plus availability and neighbour count",
            },
            {
                "factor": "chlorophyll_a",
                "source": "MODIS Aqua monthly",
                "period": "expected 2002-07 to 2020-12",
                "status": "unavailable",
                "fallback": "no training branch",
            },
            {
                "factor": "mixed_layer_depth",
                "source": "no frozen artifact",
                "period": "none",
                "status": "unavailable",
                "fallback": "no training branch",
            },
            {
                "factor": "river_discharge",
                "source": "no frozen product",
                "period": "none",
                "status": "unavailable",
                "fallback": "no training branch",
            },
        ]
    ).to_csv(tables / "source_lineage_audit.csv", index=False)
    manifest = {
        "matchup": str(MATCHUP),
        "matchup_sha256": sha256(MATCHUP),
        "cube": str(CUBE),
        "cube_sha256": sha256(CUBE),
        "rows": len(cache),
        "months": int(dataset.sizes["time"]),
        "new_labels_opened": False,
        "complete": True,
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "features.complete").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build_features(args.output)


if __name__ == "__main__":
    main()
