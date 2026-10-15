"""Select a metadata-only final SOCAT cruise holdout for P3 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_COLUMNS = (
    "group_key",
    "yr",
    "mon",
    "latitude",
    "longitude",
    "basin_id",
    "fco2_n",
    "sss_n",
)


def stable_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def metadata_groups(matchup: Path) -> pd.DataFrame:
    frame = pd.read_parquet(matchup, columns=list(METADATA_COLUMNS))
    frame = frame.loc[frame.yr.le(2025)].copy()
    groups = (
        frame.groupby("group_key", observed=True)
        .agg(
            rows=("group_key", "size"),
            year_min=("yr", "min"),
            year_max=("yr", "max"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            basin_id=("basin_id", lambda values: values.mode().iloc[0]),
            fco2_rows=("fco2_n", lambda values: int(values.gt(0).sum())),
            sss_rows=("sss_n", lambda values: int(values.gt(0).sum())),
        )
        .reset_index()
    )
    groups = groups.loc[groups.fco2_rows.ge(10) & groups.sss_rows.ge(10)].copy()
    groups["block_lat"] = np.floor(groups.latitude / 5).astype(int)
    groups["block_lon"] = np.floor(((groups.longitude + 180) % 360) / 5).astype(int)
    groups["block_id"] = groups.block_lat.astype(str) + ":" + groups.block_lon.astype(str)
    block_count = groups.groupby(["basin_id", "block_id"], observed=True).size()
    groups["block_group_count"] = [
        block_count.loc[(row.basin_id, row.block_id)] for row in groups.itertuples()
    ]
    return groups.reset_index(drop=True)


def allocate_quotas(groups: pd.DataFrame, total: int, minimum: int = 6) -> dict[int, int]:
    counts = groups.groupby("basin_id", observed=True).size().sort_index()
    if total < minimum * len(counts):
        raise ValueError("total holdout is too small for the basin minimum")
    remaining = total - minimum * len(counts)
    weight = counts / counts.sum()
    raw = weight * remaining
    quota = {int(basin): minimum + math.floor(value) for basin, value in raw.items()}
    for basin in (raw - np.floor(raw)).sort_values(ascending=False).index:
        if sum(quota.values()) >= total:
            break
        quota[int(basin)] += 1
    return quota


def select_holdout(groups: pd.DataFrame, *, total: int = 50, salt: str) -> pd.DataFrame:
    quota = allocate_quotas(groups, total)
    selected = []
    for basin, basin_quota in quota.items():
        candidates = groups.loc[groups.basin_id.eq(basin)].copy()
        candidates["selection_hash"] = candidates.group_key.astype(str).map(
            lambda value: stable_hash(value, salt)
        )
        recent_quota = basin_quota // 2
        recent_pool = candidates.loc[candidates.year_max.ge(2022)].sort_values(
            ["year_max", "selection_hash"], ascending=[False, True]
        )
        recent = recent_pool.head(recent_quota).assign(selection_reason="recent_2022_2025")
        remaining = candidates.loc[~candidates.group_key.isin(recent.group_key)].sort_values(
            ["block_group_count", "selection_hash"], ascending=[True, True]
        )
        sparse = remaining.head(basin_quota - len(recent)).assign(
            selection_reason="sparse_5deg_block"
        )
        chosen = pd.concat([recent, sparse], ignore_index=True)
        if len(chosen) != basin_quota:
            raise RuntimeError(f"basin {basin} cannot satisfy quota {basin_quota}")
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True).sort_values(
        ["basin_id", "selection_reason", "group_key"]
    )
    if len(result) != total or result.group_key.nunique() != total:
        raise RuntimeError("holdout selection is not unique or complete")
    return result.reset_index(drop=True)


def write_figures(selected: pd.DataFrame, output: Path) -> None:
    """Write target-free diagnostics for the frozen cruise selection."""
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = {
        "recent_2022_2025": "#d95f02",
        "sparse_5deg_block": "#1b9e77",
    }

    figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    for reason, frame in selected.groupby("selection_reason", observed=True):
        axis.scatter(
            frame.longitude,
            frame.latitude,
            s=30,
            alpha=0.85,
            label=reason.replace("_", " "),
            color=colors[str(reason)],
        )
    axis.set(xlabel="Longitude", ylabel="Latitude", xlim=(-180, 180), ylim=(-90, 90))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Frozen metadata-only complete-cruise holdout")
    figure.savefig(output / "fig01_holdout_map.png", dpi=180)
    plt.close(figure)

    counts = (
        selected.groupby(["basin_id", "selection_reason"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    bottom = np.zeros(len(counts))
    for reason in ("recent_2022_2025", "sparse_5deg_block"):
        values = counts.get(reason, pd.Series(0, index=counts.index)).to_numpy()
        axis.bar(
            counts.index.astype(str),
            values,
            bottom=bottom,
            label=reason.replace("_", " "),
            color=colors[reason],
        )
        bottom += values
    axis.set(xlabel="Basin ID", ylabel="Complete cruises")
    axis.legend(frameon=False)
    axis.set_title("Basin balance of the frozen holdout")
    figure.savefig(output / "fig02_basin_balance.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bins = np.arange(int(selected.year_max.min()), int(selected.year_max.max()) + 2)
    for reason, frame in selected.groupby("selection_reason", observed=True):
        axis.hist(
            frame.year_max,
            bins=bins,
            alpha=0.72,
            label=reason.replace("_", " "),
            color=colors[str(reason)],
        )
    axis.set(xlabel="Latest year represented by cruise", ylabel="Complete cruises")
    axis.legend(frameon=False)
    axis.set_title("Time distribution of the frozen holdout")
    figure.savefig(output / "fig03_time_distribution.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matchup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--total", type=int, default=50)
    parser.add_argument("--salt", default="recad-p3-final-cruise-holdout-v2.4")
    args = parser.parse_args()
    groups = metadata_groups(args.matchup)
    selected = select_holdout(groups, total=args.total, salt=args.salt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "schema_version": "2.4",
        "selection": "metadata_only_complete_cruises",
        "target_values_read": False,
        "source": str(args.matchup),
        "salt": args.salt,
        "total_groups": len(selected),
        "basin_groups": {str(k): int(v) for k, v in selected.groupby("basin_id").size().items()},
        "reason_groups": {
            str(k): int(v) for k, v in selected.groupby("selection_reason").size().items()
        },
        "fco2_rows": int(selected.fco2_rows.sum()),
        "sss_rows": int(selected.sss_rows.sum()),
        "manifest_sha256": digest,
        "training_exclusion": "remove every selected group from SSS/fCO2 fit, calibration, upstream SSS, background, thresholds, and support indices",
        "evaluation_views": [
            "complete_cruise_holdout",
            "support-distance strata measured against the remaining training cruises",
            "recent 2022-2025 subset",
            "sparse 5-degree-block subset",
        ],
        "claim_scope": "same-source cruise/spatial/time transfer; no cross-dataset claim",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.figures_dir is not None:
        write_figures(selected, args.figures_dir)


if __name__ == "__main__":
    main()
