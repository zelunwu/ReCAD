"""Shared, leakage-aware data and evaluation framework for P1 experiments.

This module is deliberately model agnostic.  All P1 models consume records
through :class:`P1DataGateway`, use the split helpers and report metrics through
``evaluate_predictions``.  Sealed rows are filtered by the parquet reader, so
their target columns are never materialised in a training or selection process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml


class AccessViolation(RuntimeError):
    """Raised when code attempts to cross a frozen data boundary."""


class Purpose(str, Enum):
    TRAIN = "train"
    SELECTION = "selection"
    LOCKED_TEST = "locked_test"
    PREDICTION = "prediction"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    cache: str
    label: str
    valid_column: str
    year_column: str
    month_column: str
    last_label_year: int
    provenance: str


TARGETS: dict[str, TargetSpec] = {
    "sss": TargetSpec("sss", "socat", "salinity", "sal_n", "yr", "mon", 2025, "SOCAT_in_situ_salinity"),
    "fco2": TargetSpec("fco2", "socat", "fco2", "f_n", "yr", "mon", 2025, "SOCAT_fCO2rec"),
    "ta": TargetSpec("ta", "carbon", "ta", "label_ta_ok", "year", "month", 2024, "CODAP_GLODAP_observed_TA"),
    "dic": TargetSpec("dic", "carbon", "dic", "label_dic_ok", "year", "month", 2024, "CODAP_GLODAP_observed_DIC"),
}


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    content: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path) -> FrozenManifest:
        resolved = Path(path).resolve()
        return cls(resolved, json.loads(resolved.read_text(encoding="utf-8")))

    @property
    def split_path(self) -> Path:
        return Path(str(self.content["split_manifest"]))

    @property
    def data_dir(self) -> Path:
        return self.split_path.parent

    def cache_path(self, cache: str) -> Path:
        names = {"socat": "na_socat_cache_v2.2.parquet", "carbon": "na_carbon_cache_v2.2.parquet"}
        return self.data_dir / names[cache]

    def validate(self, *, hash_mode: str = "artifacts") -> dict[str, object]:
        """Validate paths and hashes before a run.

        ``full`` hashes both raw inputs and frozen artifacts. ``artifacts``
        hashes every derived asset and only checks raw-input existence. Formal
        runs must use ``full``; the cheaper mode is intended for smoke tests.
        """
        if hash_mode not in {"full", "artifacts", "existence"}:
            raise ValueError("hash_mode must be full, artifacts, or existence")
        checked, hashed = 0, 0
        sections = ("inputs_sha256", "artifacts_sha256")
        for section in sections:
            entries = self.content.get(section, {})
            for raw, expected in entries.items():
                path = Path(raw)
                if not path.exists():
                    raise FileNotFoundError(path)
                checked += 1
                should_hash = hash_mode == "full" or (
                    hash_mode == "artifacts" and section == "artifacts_sha256"
                )
                if should_hash:
                    actual = sha256(path)
                    hashed += 1
                    if actual != expected:
                        raise ValueError(f"SHA256 mismatch: {path}")
        external = self.content["independent_validation"]
        external_path = (self.path.parents[2] / str(external["path"])).resolve()
        if not external_path.exists():
            raise FileNotFoundError(external_path)
        if hash_mode != "existence" and sha256(external_path) != external["sha256"]:
            raise ValueError(f"SHA256 mismatch: {external_path}")
        return {
            "status": "pass",
            "manifest": str(self.path),
            "manifest_sha256": sha256(self.path),
            "hash_mode": hash_mode,
            "paths_checked": checked + 1,
            "files_hashed": hashed + (hash_mode != "existence"),
        }


@dataclass(frozen=True)
class LockedTestGrant:
    """Explicit capability required to materialise locked-test labels."""

    reason: str
    audit_path: Path

    @classmethod
    def create(cls, reason: str, audit_path: str | Path) -> LockedTestGrant:
        if len(reason.strip()) < 12:
            raise ValueError("locked-test reason must be specific (at least 12 characters)")
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "locked_test_opened",
            "reason": reason.strip(),
            "opened_utc": datetime.now(timezone.utc).isoformat(),
        }
        if path.exists():
            raise AccessViolation(f"locked-test grant already recorded at {path}")
        path.write_text(json.dumps(event, indent=2), encoding="utf-8")
        return cls(reason.strip(), path.resolve())


def load_framework_config(path: str | Path) -> dict[str, object]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "data_manifest", "seeds", "support_distance_bins_km"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"framework config is missing: {sorted(missing)}")
    if len(config["seeds"]) != 3:
        raise ValueError("P1 framework requires exactly three predeclared seeds")
    return config


class P1DataGateway:
    """Only supported label-loading path for P1 models."""

    BASE_COLUMNS = (
        "group_key", "latitude", "longitude", "lme_id", "basin_id", "regime_id",
        "nearest_carbon_km", "split", "cv_fold", "record_status",
    )

    def __init__(self, manifest: FrozenManifest):
        self.manifest = manifest

    @staticmethod
    def _split_for(purpose: Purpose) -> str:
        return {
            Purpose.TRAIN: "train",
            Purpose.SELECTION: "development",
            Purpose.LOCKED_TEST: "locked_test",
        }[purpose]

    def load_labels(
        self,
        target: str,
        purpose: Purpose | str,
        *,
        columns: Sequence[str] = (),
        grant: LockedTestGrant | None = None,
        split_scheme: str = "primary",
    ) -> pd.DataFrame:
        """Load one legal label partition using parquet predicate pushdown."""
        purpose = Purpose(purpose)
        if purpose is Purpose.PREDICTION:
            raise AccessViolation("prediction frames do not contain target labels")
        if purpose is Purpose.LOCKED_TEST:
            self._validate_grant(grant)
        if split_scheme not in {"primary", "forward"}:
            raise ValueError("split_scheme must be primary or forward")
        spec = TARGETS[target.lower()]
        split = self._split_for(purpose)
        requested = list(dict.fromkeys((*self.BASE_COLUMNS, spec.year_column, spec.month_column,
                                        spec.label, spec.valid_column, *columns)))
        # The split filter is passed to the parquet engine. Sealed rows never
        # enter this process, even transiently.
        if split_scheme == "primary":
            filters = [("split", "==", split)]
        else:
            groups = self.split_manifest()
            groups = groups.loc[groups.forward_split.eq(split), "group_key"].tolist()
            filters = [("group_key", "in", groups)]
        frame = pd.read_parquet(
            self.manifest.cache_path(spec.cache), columns=requested,
            filters=filters,
        )
        # One cruise may occur in both SOCAT and carbon caches. P0 guarantees
        # identical assignments across those duplicate dataset rows.
        protocol = self.split_manifest()[["group_key", "forward_split", "max_year"]].drop_duplicates("group_key")
        frame = frame.merge(protocol, on="group_key", how="left", validate="many_to_one")
        valid = frame[spec.valid_column].fillna(0).astype(bool)
        frame = frame.loc[valid & frame[spec.label].notna()].copy()
        if (frame[spec.year_column] > spec.last_label_year).any():
            raise AccessViolation(f"{target} labels extend past frozen year {spec.last_label_year}")
        evaluation_split = frame["split"] if split_scheme == "primary" else frame["forward_split"]
        if purpose is Purpose.TRAIN and not evaluation_split.eq("train").all():
            raise AccessViolation("non-training row entered training")
        if purpose is Purpose.SELECTION and not evaluation_split.eq("development").all():
            raise AccessViolation("non-development row entered model selection")
        frame = frame.rename(columns={spec.year_column: "year", spec.month_column: "month", spec.label: "truth"})
        frame["evaluation_split"] = evaluation_split.to_numpy()
        frame["split_scheme"] = split_scheme
        frame["target"] = spec.name
        frame["target_provenance"] = spec.provenance
        return frame.reset_index(drop=True)

    @staticmethod
    def _validate_grant(grant: LockedTestGrant | None) -> None:
        if grant is None or not grant.audit_path.is_file():
            raise AccessViolation("locked-test labels require a recorded LockedTestGrant")
        try:
            event = json.loads(grant.audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AccessViolation("locked-test audit record is unreadable") from exc
        if event.get("event") != "locked_test_opened" or event.get("reason") != grant.reason:
            raise AccessViolation("locked-test audit record does not match the grant")

    def load_external_labels(self, *_args, **_kwargs) -> pd.DataFrame:
        raise AccessViolation("external-independent labels remain sealed until P3")

    def cv_fold(self, target: str, held_out_fold: int, *, columns: Sequence[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
        if held_out_fold not in range(5):
            raise ValueError("held_out_fold must be in 0..4")
        frame = self.load_labels(target, Purpose.TRAIN, columns=columns)
        fit = frame.loc[frame.cv_fold.ne(held_out_fold)].copy()
        validation = frame.loc[frame.cv_fold.eq(held_out_fold)].copy()
        if set(fit.group_key) & set(validation.group_key):
            raise AccessViolation("a cruise crosses the CV fold boundary")
        return fit.reset_index(drop=True), validation.reset_index(drop=True)

    def prediction_availability(self, years: Iterable[int] = (2025, 2026)) -> pd.DataFrame:
        path = self.manifest.data_dir / "inference_availability_v2.2.nc"
        wanted = {int(y) for y in years}
        with xr.open_dataset(path) as ds:
            subset = ds[["strict_all_inputs_ready"]].sel(year=sorted(wanted)).load()
        frame = subset.to_dataframe().reset_index()
        frame = frame.loc[frame.strict_all_inputs_ready.astype(bool)].copy()
        frame["prediction_status"] = np.where(frame.year.eq(2026), "provisional", "core")
        if frame.loc[frame.year.eq(2026), "prediction_status"].ne("provisional").any():
            raise AccessViolation("2026 output must be provisional")
        return frame.reset_index(drop=True)

    def split_manifest(self) -> pd.DataFrame:
        return pd.read_parquet(self.manifest.split_path)


def forward_chain(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Select a frozen forward-chain stage without changing group assignment."""
    if stage not in {"train", "development", "locked_test"}:
        raise ValueError(stage)
    if "forward_split" not in frame:
        raise ValueError("forward_split column is required")
    return frame.loc[frame.forward_split.eq(stage)].copy()


def leave_group_out(frame: pd.DataFrame, column: str) -> Iterator[tuple[object, pd.DataFrame, pd.DataFrame]]:
    """Yield deterministic leave-one-LME/regime-out partitions."""
    if column not in {"lme_id", "regime_id"}:
        raise ValueError("only lme_id and regime_id are supported")
    for value in sorted(frame[column].dropna().unique().tolist()):
        held = frame[column].eq(value)
        yield value, frame.loc[~held].copy(), frame.loc[held].copy()


def _metric_row(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                "r2": np.nan, "pearson_r": np.nan}
    error = p - y
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(error**2) / denom if denom > 0 else np.nan
    corr = np.corrcoef(y, p)[0, 1] if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "r2": float(r2), "pearson_r": float(corr)}


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    truth: str = "truth",
    prediction: str = "prediction",
    support_bins: Sequence[float] = (0, 25, 50, 100, 200, 500, np.inf),
) -> pd.DataFrame:
    """Return the frozen pooled, macro and support-stratified metric table."""
    required = {truth, prediction, "group_key", "lme_id", "regime_id", "nearest_carbon_km"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"prediction table is missing {sorted(missing)}")
    rows: list[dict[str, object]] = []

    def add(kind: str, name: object, part: pd.DataFrame) -> None:
        metric = _metric_row(part[truth].to_numpy(float), part[prediction].to_numpy(float))
        rows.append({"aggregation": kind, "group": str(name), **metric})

    add("pooled", "all", predictions)
    grouped_metrics: dict[str, list[dict[str, float | int]]] = {}
    for kind, column in (("cruise", "group_key"), ("lme", "lme_id"), ("regime", "regime_id")):
        grouped_metrics[kind] = []
        for name, part in predictions.groupby(column, dropna=False, sort=True):
            metric = _metric_row(part[truth].to_numpy(float), part[prediction].to_numpy(float))
            grouped_metrics[kind].append(metric)
            add(f"{kind}_detail", name, part)
        valid = [m for m in grouped_metrics[kind] if m["n"]]
        macro = {key: float(np.nanmean([m[key] for m in valid])) for key in ("rmse", "mae", "bias", "r2", "pearson_r")}
        macro["n"] = int(sum(int(m["n"]) for m in valid))
        rows.append({"aggregation": f"{kind}_macro", "group": "all", **macro})
    cruise = grouped_metrics["cruise"]
    equal_rmse = float(np.sqrt(np.nanmean([float(m["rmse"]) ** 2 for m in cruise])))
    rows.append({"aggregation": "cruise_equal", "group": "all", "n": len(cruise),
                 "rmse": equal_rmse, "mae": np.nan, "bias": np.nan, "r2": np.nan, "pearson_r": np.nan})
    lme_details = [r for r in rows if r["aggregation"] == "lme_detail" and np.isfinite(r["rmse"])]
    if lme_details:
        worst = max(lme_details, key=lambda r: r["rmse"])
        rows.append({**worst, "aggregation": "worst_lme"})
    labels = [f"[{support_bins[i]},{support_bins[i + 1]})" for i in range(len(support_bins) - 1)]
    support = pd.cut(predictions.nearest_carbon_km, bins=support_bins, right=False, labels=labels)
    for name, part in predictions.groupby(support, observed=True):
        add("support_distance", name, part)
    return pd.DataFrame(rows)


class BalancedBatchSampler:
    """Deterministic region -> cruise -> month -> record sampler."""

    def __init__(self, frame: pd.DataFrame, *, seed: int = 100, region_column: str = "lme_id"):
        required = {region_column, "group_key", "month"}
        if required - set(frame):
            raise ValueError(f"sampler needs {sorted(required)}")
        self.frame = frame.reset_index(drop=True)
        self.seed = int(seed)
        self.region_column = region_column
        self.tree: dict[object, dict[object, dict[int, np.ndarray]]] = {}
        for (region, cruise, month), part in self.frame.groupby([region_column, "group_key", "month"], sort=True):
            self.tree.setdefault(region, {}).setdefault(cruise, {})[int(month)] = part.index.to_numpy()

    def sample(self, batch_size: int, *, step: int = 0) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step)]))
        regions = sorted(self.tree, key=str)
        result = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            region = regions[int(rng.integers(len(regions)))]
            cruises = sorted(self.tree[region], key=str)
            cruise = cruises[int(rng.integers(len(cruises)))]
            months = sorted(self.tree[region][cruise])
            month = months[int(rng.integers(len(months)))]
            indices = self.tree[region][cruise][month]
            result[i] = indices[int(rng.integers(len(indices)))]
        return result


def select_checkpoint(records: pd.DataFrame, *, metric: str = "lme_macro_rmse", mode: str = "min") -> pd.Series:
    required = {"checkpoint", metric}
    if required - set(records):
        raise ValueError(f"checkpoint records need {sorted(required)}")
    eligible = records.loc[records[metric].notna()].copy()
    if "eligible" in eligible:
        eligible = eligible.loc[eligible.eligible.astype(bool)]
    if eligible.empty:
        raise ValueError("no eligible checkpoint")
    index = eligible[metric].idxmin() if mode == "min" else eligible[metric].idxmax()
    return eligible.loc[index]


def aggregate_seeds(records: pd.DataFrame, *, group_columns: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    if "seed" not in records:
        raise ValueError("seed column is required")
    if records.seed.nunique() != 3:
        raise ValueError("exactly three seeds are required")
    result = records.groupby(list(group_columns), dropna=False)[list(metrics)].agg(["mean", "std", "median"])
    result.columns = [f"{name}_{stat}" for name, stat in result.columns]
    return result.reset_index()


PREDICTION_COLUMNS = (
    "record_id", "target", "prediction", "uncertainty", "lower", "upper",
    "target_provenance", "model_provenance", "group_key", "year", "month",
    "latitude", "longitude", "split", "cv_fold", "lme_id", "regime_id",
    "nearest_carbon_km", "ood", "prediction_status", "seed",
)


def standardize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(PREDICTION_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"standard prediction table is missing {sorted(missing)}")
    result = frame.copy()
    is_2026 = result.year.eq(2026)
    if result.loc[is_2026, "prediction_status"].ne("provisional").any():
        raise AccessViolation("every 2026 prediction must be marked provisional")
    if "truth" in result and result.loc[is_2026, "truth"].notna().any():
        raise AccessViolation("2026 prediction rows cannot carry truth labels")
    return result[[*PREDICTION_COLUMNS, *(c for c in ("truth",) if c in result)]].copy()


def build_audit(gateway: P1DataGateway, validation: Mapping[str, object]) -> dict[str, object]:
    split = gateway.split_manifest()
    group_splits = split.groupby("group_key").split.nunique()
    group_folds = split.groupby("group_key").cv_fold.nunique()
    if group_splits.max() != 1 or group_folds.max() != 1:
        raise AccessViolation("cruise leakage detected in split manifest")
    targets: dict[str, object] = {}
    for name, spec in TARGETS.items():
        counts: dict[str, int] = {}
        years: dict[str, list[int] | None] = {}
        for purpose in (Purpose.TRAIN, Purpose.SELECTION):
            frame = gateway.load_labels(name, purpose)
            counts[purpose.value] = len(frame)
            years[purpose.value] = [int(frame.year.min()), int(frame.year.max())] if len(frame) else None
        targets[name] = {"rows": counts, "year_ranges": years, "last_label_year": spec.last_label_year}
    availability = gateway.prediction_availability((2025, 2026))
    ready = availability.groupby(["year", "prediction_status"]).size().to_dict()
    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation": dict(validation),
        "split": {
            "groups": int(split.group_key.nunique()),
            "counts": {str(k): int(v) for k, v in split.split.value_counts().items()},
            "counts_by_dataset": {
                f"{partition}:{dataset}": int(n)
                for (partition, dataset), n in split.groupby(["split", "dataset"]).size().items()
            },
            "forward_counts": {str(k): int(v) for k, v in split.forward_split.value_counts().items()},
            "cv_folds": sorted(int(v) for v in split.cv_fold.dropna().unique()),
            "no_group_split_leakage": True,
            "external_labels_opened": False,
            "locked_test_labels_opened": False,
        },
        "targets": targets,
        "prediction_availability": {f"{year}:{status}": int(n) for (year, status), n in ready.items()},
    }
