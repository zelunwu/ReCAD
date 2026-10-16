"""Build the external read-only patch cache for Issue 89 neural families."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np

from recad.evaluate.coastal_patches import DYNAMIC_VARIABLES, STATIC_VARIABLES

CUBE = Path(r"C:\backup\phd\data\processed\recad_v2_3\global_coastal_predictor_cube_v2.3.nc")
OUTPUT = Path(r"C:\backup\phd\data\processed\recad_v2_5\p2_model_family_benchmark_v2.5\patch_cache")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(CUBE) as dataset:
        n_time = len(dataset.dimensions["time"])
        n_node = len(dataset.dimensions["node"])
        latitude = np.asarray(dataset.variables["grid_lat_index"][:], dtype=np.int32)
        longitude = np.asarray(dataset.variables["grid_lon_index"][:], dtype=np.int32)
        n_latitude = int(latitude.max()) + 1
        n_longitude = int(longitude.max()) + 1
        lookup = np.full((n_latitude, n_longitude), -1, dtype=np.int32)
        lookup[latitude, longitude] = np.arange(n_node, dtype=np.int32)
        np.save(OUTPUT / "node_lookup.int32.npy", lookup, allow_pickle=False)
        np.save(OUTPUT / "node_latitude_index.int32.npy", latitude, allow_pickle=False)
        np.save(OUTPUT / "node_longitude_index.int32.npy", longitude, allow_pickle=False)
        static = np.column_stack(
            [np.ma.filled(dataset.variables[name][:], np.nan) for name in STATIC_VARIABLES]
        ).astype(np.float32)
        np.save(OUTPUT / "static.float32.npy", static, allow_pickle=False)
        dynamic_path = OUTPUT / "dynamic.float16.dat"
        progress = OUTPUT / "dynamic.progress.json"
        expected_bytes = n_time * n_node * len(DYNAMIC_VARIABLES) * np.dtype(np.float16).itemsize
        resumable = dynamic_path.is_file() and dynamic_path.stat().st_size == expected_bytes
        start = 0
        if resumable and progress.is_file():
            state = json.loads(progress.read_text(encoding="utf-8"))
            start = int(state["last_complete_time_index"]) + 1
        dynamic = np.memmap(
            dynamic_path,
            dtype=np.float16,
            mode="r+" if resumable else "w+",
            shape=(n_time, n_node, len(DYNAMIC_VARIABLES)),
        )
        for time_index in range(start, n_time):
            dynamic[time_index] = np.column_stack(
                [
                    np.ma.filled(dataset.variables[name][time_index, :], np.nan)
                    for name in DYNAMIC_VARIABLES
                ]
            ).astype(np.float16)
            dynamic.flush()
            progress.write_text(
                json.dumps({"last_complete_time_index": time_index}) + "\n", encoding="utf-8"
            )
        del dynamic
    files = sorted(path for path in OUTPUT.iterdir() if path.name != "metadata.json")
    metadata = {
        "schema_version": "2.5",
        "source_cube": str(CUBE),
        "source_cube_sha256": sha256(CUBE),
        "n_time": n_time,
        "n_node": n_node,
        "n_latitude": n_latitude,
        "n_longitude": n_longitude,
        "dynamic_variables": list(DYNAMIC_VARIABLES),
        "static_variables": list(STATIC_VARIABLES),
        "dtype": "float16 dynamic, float32 static, int32 topology",
        "target_values_included": False,
        "files": {path.name: sha256(path) for path in files},
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "patch_cache.complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
