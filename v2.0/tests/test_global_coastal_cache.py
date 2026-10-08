from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from recad.data.global_cache import (
    CoastalGrid,
    build_global_socat_cache,
    build_group_manifest,
    nearest_indices,
    normalize_group,
    stable_fold,
)


@pytest.mark.l1
def test_identity_and_fold_are_normalized_and_deterministic() -> None:
    assert normalize_group(" ab-12 / c ") == "AB12C"
    assert normalize_group(None) == "UNKNOWN"
    assert stable_fold("AB12C", "test", folds=5) == stable_fold("AB12C", "test", folds=5)
    assert 0 <= stable_fold("AB12C", "test", folds=5) < 5
    indices = nearest_indices(np.array([0.02, 0.24]), np.array([0.0, 0.125, 0.25]))
    np.testing.assert_array_equal(indices, np.array([0, 2]))


def _write_grid(tmp_path: Path) -> tuple[Path, Path]:
    prepared_path = tmp_path / "prepared.nc"
    spatial_path = tmp_path / "spatial.nc"
    lat = np.array([-0.125, 0.0, 0.125])
    lon = np.array([0.0, 0.125, 0.25])
    mask = np.array([[False, True, False], [True, True, True], [False, True, False]], dtype=np.int8)
    xr.Dataset({"coastal_mask": (("lat", "lon"), mask)}, coords={"lat": lat, "lon": lon}).to_netcdf(
        prepared_path
    )
    flats = np.flatnonzero(mask.ravel())
    li, oi = flats // len(lon), flats % len(lon)
    xr.Dataset(
        {
            "grid_flat": ("node", flats.astype(np.int32)),
            "latitude": ("node", lat[li].astype(np.float32)),
            "longitude": ("node", lon[oi].astype(np.float32)),
            "lme_id": ("node", np.array([1, 1, -1, 2, 2], dtype=np.int16)),
            "basin_id": ("node", np.array([2, 2, 2, 3, 3], dtype=np.int8)),
            "regime_id": ("node", np.array([0, 0, 1, 1, 1], dtype=np.int16)),
        }
    ).to_netcdf(spatial_path)
    return prepared_path, spatial_path


def _write_socat(tmp_path: Path) -> Path:
    path = tmp_path / "SOCAT.tsv"
    metadata = (
        "SOCAT synthetic\n"
        "Expocode\tPlatform Name\n"
        "AA-2020\tResearch Vessel A\n"
        "BB-2023\tMooring B\n"
        "CC-2018\tResearch Vessel C\n"
        "metadata end\n"
    )
    columns = [
        "Expocode",
        "version",
        "Source_DOI",
        "QC_Flag",
        "yr",
        "mon",
        "day",
        "hh",
        "mm",
        "ss",
        "longitude [dec.deg.E]",
        "latitude [dec.deg.N]",
        "sal",
        "fCO2rec [uatm]",
        "fCO2rec_flag",
    ]
    rows = [
        ["AA-2020", "2026", "doi:a", "A", 2020, 1, 1, 0, 0, 0, 0.125, -0.12, 34.0, 400.0, 2],
        ["AA-2020", "2026", "doi:a", "A", 2020, 1, 1, 0, 0, 0, 0.125, -0.12, 34.0, 400.0, 2],
        ["AA-2020", "2026", "doi:a", "A", 2020, 1, 2, 0, 0, 0, 0.125, -0.12, 35.0, 410.0, 2],
        ["BB-2023", "2026", "doi:b", "B", 2023, 7, 1, 0, 0, 0, 0.25, 0.0, 33.0, 1200.0, 2],
        ["BB-2023", "2026", "doi:b", "B", 2023, 7, 2, 0, 0, 0, 0.25, 0.0, 60.0, 420.0, 2],
        ["CC-2018", "2026", "doi:c", "A", 2018, 5, 1, 0, 0, 0, 0.0, -0.125, 32.0, 430.0, 2],
    ]
    frame = pd.DataFrame(rows, columns=columns)
    path.write_text(metadata + frame.to_csv(sep="\t", index=False), encoding="utf-8")
    return path


@pytest.mark.l2
def test_synthetic_global_cache_assigns_groups_before_qc_and_deduplicates(
    tmp_path: Path,
) -> None:
    prepared, spatial = _write_grid(tmp_path)
    source = _write_socat(tmp_path)
    grid = CoastalGrid.load(prepared, spatial)

    cache, funnel = build_global_socat_cache(source, grid, chunksize=3)
    groups = build_group_manifest(cache)

    assert set(cache["group_key"]) == {"AA2020", "BB2023", "CC2018"}
    aa = cache.loc[cache["group_key"] == "AA2020"].iloc[0]
    assert aa["sample_n"] == 2
    assert aa["fco2"] == pytest.approx(405.0)
    assert aa["salinity"] == pytest.approx(34.5)
    bb = cache.loc[cache["group_key"] == "BB2023"].iloc[0]
    assert bb["fco2"] == pytest.approx(420.0)
    assert bb["salinity"] == pytest.approx(33.0)
    cc = cache.loc[cache["group_key"] == "CC2018"].iloc[0]
    assert not cc["on_v2_2_product_mask"]
    assert funnel.set_index("stage").loc["after_exact_deduplication", "rows"] == 5
    assert groups.groupby("group_key")["cruise_fold"].nunique().max() == 1
    assert groups.groupby("group_key")["spatial_fold"].nunique().max() == 1
