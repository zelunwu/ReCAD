"""Build the conditional 1993-2025 ReCAD v2.4 release products."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from recad.data.global_cache import coarse_basin

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"C:\backup\phd\data\processed\recad_v2_4\p2_global_reliability_contract_v2.4"
    r"\products"
)
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_4\recad_v2_release_v2.4")
PRODUCTS = OUTPUT / "products"
TABLES = OUTPUT / "tables"
P3_LOCK = ROOT / "configs/frozen/p3_cruise_holdout_lock_v2.4.json"
BASIN_NAMES = {
    0: "Arctic",
    1: "Southern",
    2: "Atlantic",
    3: "Pacific",
    4: "Indian",
    5: "Mediterranean",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def release_status(target: str, grade: np.ndarray, basin: np.ndarray) -> np.ndarray:
    grade = np.asarray(grade)
    basin = np.asarray(basin)
    if target == "SSS":
        return np.select([grade <= 2, grade == 3], [1, 2], default=3).astype("uint8")
    trusted = (grade <= 2) & (basin != 3)
    diagnostic = (grade <= 3) & ~trusted
    return np.select([trusted, diagnostic], [1, 2], default=3).astype("uint8")


def add_release_layers(path: Path, target: str) -> list[dict[str, object]]:
    rows = []
    with netCDF4.Dataset(path, "r+") as dataset:
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
        longitude_180 = (longitude + 180) % 360 - 180
        basin = coarse_basin(longitude_180, latitude).astype("int8")
        if "basin_id" not in dataset.variables:
            basin_variable = dataset.createVariable("basin_id", "i1", ("node",))
            basin_variable[:] = basin
            basin_variable.long_name = "coarse ocean basin used by the release qualification rule"
            basin_variable.flag_values = np.arange(6, dtype="int8")
            basin_variable.flag_meanings = "Arctic Southern Atlantic Pacific Indian Mediterranean"
        chunks = (1, 65536)
        for name, long_name in (
            ("trusted_prediction", "prediction where the P3 release status is trusted"),
            ("diagnostic_prediction", "prediction where the P3 release status is diagnostic"),
        ):
            if name not in dataset.variables:
                variable = dataset.createVariable(
                    name,
                    "f4",
                    ("time", "node"),
                    zlib=True,
                    complevel=2,
                    chunksizes=chunks,
                    fill_value=np.nan,
                )
                variable.units = "uatm" if target == "fCO2" else "1"
                variable.long_name = long_name
        area = np.asarray(dataset.variables["cell_area_km2"][:], dtype=float)
        for time_index in range(len(dataset.dimensions["time"])):
            grade = np.asarray(dataset.variables["applicability_grade"][time_index, :])
            status = release_status(target, grade, basin)
            prediction = np.asarray(
                dataset.variables["prediction"][time_index, :].filled(np.nan), dtype="float32"
            )
            reason = np.asarray(dataset.variables["reason_bits"][time_index, :], dtype="uint16")
            if target == "fCO2":
                restricted = (basin == 3) & (grade <= 2)
                reason[restricted] |= np.uint16(2048)
            dataset.variables["status"][time_index, :] = status
            dataset.variables["reason_bits"][time_index, :] = reason
            dataset.variables["trusted_prediction"][time_index, :] = np.where(
                status == 1, prediction, np.nan
            )
            dataset.variables["diagnostic_prediction"][time_index, :] = np.where(
                status == 2, prediction, np.nan
            )
            for code, label in ((1, "trusted"), (2, "diagnostic"), (3, "suppressed")):
                selected = status == code
                rows.append(
                    {
                        "target": target,
                        "time_index": time_index,
                        "status": label,
                        "nodes": int(selected.sum()),
                        "area_km2": float(area[selected].sum()),
                    }
                )
        dataset.model_version = "ReCAD-v2.4-release"
        dataset.release_version = "2.4.0"
        dataset.release_period = "1993-2025"
        dataset.evidence_scope = (
            "same-source complete-cruise transfer within SOCAT v2026; no cross-dataset claim"
        )
        dataset.release_decision = "pass_flagged_same_source_release_with_regional_suppression"
        dataset.release_status_codes = "1=trusted, 2=diagnostic, 3=suppressed"
        dataset.reason_bit_2048 = "regional release restriction; Pacific fCO2 A/B is diagnostic"
        dataset.ta_dic_policy = "not included; Issue 45 untriggered"
    return rows


def verify_product(path: Path, target: str) -> dict[str, object]:
    with netCDF4.Dataset(path) as dataset:
        required = {
            "prediction",
            "width50",
            "width90",
            "applicability_grade",
            "status",
            "reason_bits",
            "trusted_prediction",
            "diagnostic_prediction",
            "basin_id",
        }
        missing = required - set(dataset.variables)
        if missing:
            raise RuntimeError(f"{target}: missing release variables {sorted(missing)}")
        times = netCDF4.num2date(
            dataset.variables["time"][:],
            dataset.variables["time"].units,
            calendar=getattr(dataset.variables["time"], "calendar", "standard"),
        )
        years = np.asarray([value.year for value in times])
        if years.min() != 1993 or years.max() != 2025 or len(years) != 396:
            raise RuntimeError(f"{target}: release time axis drift")
        mismatch = 0
        trusted_nonfinite = 0
        diagnostic_nonfinite = 0
        for time_index in range(len(times)):
            status = np.asarray(dataset.variables["status"][time_index, :])
            trusted = np.asarray(
                dataset.variables["trusted_prediction"][time_index, :].filled(np.nan)
            )
            diagnostic = np.asarray(
                dataset.variables["diagnostic_prediction"][time_index, :].filled(np.nan)
            )
            mismatch += int(np.isfinite(trusted[status != 1]).sum())
            mismatch += int(np.isfinite(diagnostic[status != 2]).sum())
            trusted_nonfinite += int((~np.isfinite(trusted[status == 1])).sum())
            diagnostic_nonfinite += int((~np.isfinite(diagnostic[status == 2])).sum())
        if mismatch or trusted_nonfinite or diagnostic_nonfinite:
            raise RuntimeError(f"{target}: release-layer mismatch")
        return {
            "target": target,
            "time_steps": len(times),
            "nodes": len(dataset.dimensions["node"]),
            "start_year": int(years.min()),
            "end_year": int(years.max()),
            "release_layer_mismatch": mismatch,
            "trusted_nonfinite": trusted_nonfinite,
            "diagnostic_nonfinite": diagnostic_nonfinite,
        }


def main() -> None:
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    lock = json.loads(P3_LOCK.read_text(encoding="utf-8"))
    if lock["result_decision"] != "pass_flagged_same_source_release_with_regional_suppression":
        raise RuntimeError("P3 release decision is not qualified")
    coverage_rows = []
    checks = []
    product_hashes = {}
    for target in ("SSS", "fCO2"):
        source = SOURCE / f"{target}_reliability_v2.4.nc"
        destination = PRODUCTS / f"ReCAD_v2.4_{target}_1993_2025.nc"
        shutil.copy2(source, destination)
        coverage_rows.extend(add_release_layers(destination, target))
        checks.append(verify_product(destination, target))
        product_hashes[destination.name] = sha256(destination)
    coverage = pd.DataFrame(coverage_rows)
    total = coverage.groupby(["target", "time_index"], observed=True).area_km2.transform("sum")
    coverage["area_fraction"] = coverage.area_km2 / total
    coverage.to_csv(TABLES / "release_status_by_month.csv", index=False)
    summary = (
        coverage.groupby(["target", "status"], as_index=False)
        .agg(
            mean_area_fraction=("area_fraction", "mean"),
            minimum_area_fraction=("area_fraction", "min"),
            maximum_area_fraction=("area_fraction", "max"),
        )
        .sort_values(["target", "status"])
    )
    summary.to_csv(TABLES / "release_status_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLES / "release_checks.csv", index=False)
    release = {
        "release": "ReCAD v2.4.0",
        "period": "1993-2025",
        "products": product_hashes,
        "p3_lock_sha256": sha256(P3_LOCK),
        "decision": lock["result_decision"],
        "trusted_scope": {
            "SSS": "grades A/B in all basins represented by the P3 holdout",
            "fCO2": "grades A/B outside coarse Pacific basin 3",
        },
        "diagnostic_scope": {
            "SSS": "grade C",
            "fCO2": "grade C plus Pacific grades A/B",
        },
        "suppressed_scope": {"SSS": "grade D", "fCO2": "grade D"},
        "TA_DIC": "not released; Issue 45 untriggered",
        "provisional_2026": "not included in this release",
        "cross_dataset_claim": False,
    }
    (OUTPUT / "release_manifest.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "pipeline.complete").write_text(
        json.dumps({"status": "complete", "products": product_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
