"""Configuration system for ReCAD v2.0.

Every scientific and engineering knob of the pipeline is declared as a
dataclass and validated on load. Configurations are authored as YAML files
(see ``v2.0/configs/*.yaml``) and parsed by :func:`Config.from_yaml`, so a
full experiment (domain, data, split, model, ensemble, training, uncertainty,
output) is a single, reviewable, version-controlled artifact.

Nothing here depends on PyTorch or xarray, keeping the config layer unit
testable in isolation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

from recad.constants import (
    DEFAULT_ENSEMBLE_MEMBERS,
    DEFAULT_INPUT_UNCERTAINTIES,
    DEFAULT_MC_DRAWS,
    RESOLUTIONS_SUPPORTED,
    YEAR_MAX_DEFAULT,
    YEAR_MIN_DEFAULT,
)


class ConfigError(ValueError):
    """Raised when a configuration is invalid or inconsistent."""


# ---------------------------------------------------------------------------
# Grid / domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    """Spatial reconstruction grid.

    ``lon`` is handled in ``[0, 360)`` throughout the data pipeline (matches
    the SOCAT gridded product and the v1.1 convention of adding 360 to
    satellite longitudes). ``domain`` selects a named preset or ``custom``.
    """

    domain: str = "global"  # global | naccom | custom
    resolution_deg: float = 0.125  # 1/8 deg
    lon_min: float = 0.0
    lon_max: float = 360.0
    lat_min: float = -78.0  # exclude the Antarctic ice margin cells
    lat_max: float = 84.0
    patched: bool = True
    patch_size_cells: int = 16  # side length of one spatial patch, in cells

    def __post_init__(self) -> None:
        if self.resolution_deg not in RESOLUTIONS_SUPPORTED:
            raise ConfigError(
                f"resolution {self.resolution_deg} not in supported set {RESOLUTIONS_SUPPORTED}"
            )
        if self.lon_min < 0 or self.lon_max > 360 or self.lon_min >= self.lon_max:
            raise ConfigError(
                f"invalid longitude range [{self.lon_min}, {self.lon_max}); use [0, 360) convention"
            )
        if self.lat_min < -90 or self.lat_max > 90 or self.lat_min >= self.lat_max:
            raise ConfigError(f"invalid latitude range [{self.lat_min}, {self.lat_max}]")
        if self.patch_size_cells < 1:
            raise ConfigError("patch_size_cells must be >= 1")

    @property
    def n_lon(self) -> int:
        """Number of longitude cells (half-open interval: min..max-res)."""
        return round((self.lon_max - self.lon_min) / self.resolution_deg)

    @property
    def n_lat(self) -> int:
        """Number of latitude rows (closed interval, matching the builder)."""
        return round((self.lat_max - self.lat_min) / self.resolution_deg) + 1


# ---------------------------------------------------------------------------
# Coastal mask
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoastalMaskConfig:
    """Definition of the 'coastal ocean' domain.

    - ``distance_km``: keep every ocean cell whose centre is within this
      distance of the coastline (GSHHG-derived, reproducible).
    - ``source_file``: optional precomputed NetCDF mask (lat, lon) of 1/0.
    - ``socat_coastal_flag``: if True, intersect with the SOCAT coastal flag.
    """

    method: str = "distance"  # distance | file | socat_flag
    distance_km: float = 200.0
    source_file: str | None = None
    socat_coastal_flag: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"distance", "file", "socat_flag"}:
            raise ConfigError(f"unsupported coastal mask method: {self.method}")
        if self.method == "file" and not self.source_file:
            raise ConfigError("mask method 'file' requires source_file")
        if self.distance_km <= 0:
            raise ConfigError("coastal distance_km must be positive")


# ---------------------------------------------------------------------------
# Data sources / raw inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    """Where raw forcing/observation files live and which variables they map to.

    ``templates`` maps a logical variable name (one of ``fco2``, ``sst``,
    ``sss``, ``adt``, ``sla``, ``wspd``, ``chla``, ``xco2air``, ``mask``) to a
    path template. Templates may contain ``{year}`` / ``{month:02d}``
    placeholders for per-file sources; a plain path is used for monolithic
    files. See ``recad/data/specs.py`` for the per-variable metadata registry.
    """

    root: str = "data/raw"
    templates: dict[str, str] = field(default_factory=dict)
    year_min: int = YEAR_MIN_DEFAULT
    year_max: int = YEAR_MAX_DEFAULT
    variable_name: str = "fco2"  # target reconstructed variable (fco2 | pco2)
    longitude_shift: float = 0.0  # add this to source longitudes to reach [0,360)

    def __post_init__(self) -> None:
        if self.year_min > self.year_max:
            raise ConfigError("year_min must be <= year_max")
        allowed = {"fco2", "pco2"}
        if self.variable_name not in allowed:
            raise ConfigError(f"variable_name must be one of {allowed}, got {self.variable_name}")


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Train / validation / test partitioning of the matched samples.

    v1.1 held 2004-2005 out entirely as the independent *test* years and split
    the remaining years randomly 80/20 into train/validation
    (``create4DSplitIndices.m``, ``RFR_models.mlx``). The v2.0 default
    ``scheme=blocked_spatiotemporal`` is more rigorous: within the training
    pool, train and validation samples are separated both in space (blocks of
    ``spatial_block_deg``) and time (blocks of ``temporal_block_months``),
    avoiding leakage along spatial/temporal autocorrelation. ``scheme=
    random_80_20`` reproduces the v1.1 protocol exactly for benchmark
    comparability, and ``test_holdout_years`` (default (2004, 2005), matching
    v1.1) is always held out as the external test set.
    """

    scheme: str = "blocked_spatiotemporal"  # random_80_20 | blocked_spatiotemporal
    train_fraction: float = 0.8
    seed: int = 100  # v1.1 seeding (rng(100))
    test_holdout_years: tuple[int, ...] = (2004, 2005)
    spatial_block_deg: float = 2.0  # blocked scheme: side of each spatial block
    temporal_block_months: int = 6  # blocked scheme: length of each time block

    def __post_init__(self) -> None:
        if self.scheme not in {"random_80_20", "blocked_spatiotemporal"}:
            raise ConfigError(f"unknown split scheme: {self.scheme}")
        if not 0.0 < self.train_fraction < 1.0:
            raise ConfigError("train_fraction must be in (0, 1)")
        if self.scheme == "blocked_spatiotemporal" and (
            self.spatial_block_deg <= 0 or self.temporal_block_months < 1
        ):
            raise ConfigError("blocked split needs positive block sizes")
        if len(set(self.test_holdout_years)) != len(self.test_holdout_years):
            raise ConfigError("test_holdout_years must not contain duplicates")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Spatio-temporal transformer (ST-Transformer) hyper-parameters.

    Architecture (see docs/backup/design.md): coastal cells are grouped into spatial
    patches; a spatial transformer encoder fuses context within each month;
    a temporal transformer encoder fuses context across months for each patch;
    a per-cell MLP head emits the predictive distribution.

    ``head`` selects the output head:
      - ``gaussian``: mean + heteroscedastic log-variance (Gaussian NLL loss);
      - ``mean``: point prediction (MSE loss);
      - ``quantile``: [q0.1, q0.5, q0.9] for quantile regression (optional).
    """

    name: str = "st_transformer"
    embed_dim: int = 64
    n_heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # sinusoidal | learned
    head: str = "gaussian"  # gaussian | mean | quantile
    temporal_window_months: int = 12  # months attended by the temporal stage
    add_lat_lon_features: bool = True
    add_month_features: bool = True

    def __post_init__(self) -> None:
        if self.name != "st_transformer":
            raise ConfigError(f"unknown model name: {self.name}")
        if self.embed_dim < 8 or self.embed_dim % self.n_heads != 0:
            raise ConfigError("embed_dim must be >= 8 and divisible by n_heads")
        if self.head not in {"gaussian", "mean", "quantile"}:
            raise ConfigError(f"unknown head: {self.head}")
        if self.temporal_window_months < 1:
            raise ConfigError("temporal_window_months must be >= 1")


# ---------------------------------------------------------------------------
# Deep ensemble
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleConfig:
    """Deep-ensemble configuration (Lakshminarayanan et al., 2017).

    Each member is trained from a distinct seed with bootstrap resampling of
    the training samples (``bootstrap_fraction`` new samples drawn with
    replacement). Predictive mean = mean of member means; epistemic std =
    std of member means; aleatoric std = mean of member variances
    (see recad/model/ensemble.py).
    """

    n_members: int = DEFAULT_ENSEMBLE_MEMBERS
    seed_offset: int = 0  # member seeds: seed + seed_offset + i
    bootstrap_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.n_members < 1:
            raise ConfigError("n_members must be >= 1")
        if not 0.0 < self.bootstrap_fraction <= 1.0:
            raise ConfigError("bootstrap_fraction must be in (0, 1]")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters (custom Trainer, see recad/train/trainer.py)."""

    n_epochs: int = 50
    tile_cells: int | None = None  # per-item cell subsample (None = all coastal cells)
    batch_windows: int = 2  # year-windows per optimizer step
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    scheduler: str = "cosine"  # cosine | plateau | none
    grad_clip_norm: float = 1.0
    use_amp: bool = True  # automatic mixed precision on CUDA
    early_stop_patience: int = 10
    eval_every_epochs: int = 1
    device: str = "auto"  # auto | cuda | cpu | mps
    checkpoint_dir: str = "outputs/checkpoints"
    log_every_batches: int = 50

    def __post_init__(self) -> None:
        if self.n_epochs < 1:
            raise ConfigError("n_epochs must be >= 1")
        if self.lr <= 0:
            raise ConfigError("lr must be positive")
        if self.batch_windows < 1:
            raise ConfigError("batch_windows must be >= 1")
        if self.scheduler not in {"cosine", "plateau", "none"}:
            raise ConfigError(f"unknown scheduler: {self.scheduler}")


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UncertaintyConfig:
    """Uncertainty-quantification configuration.

    v2.0 combines three, physics-aware sources of uncertainty (see
    recad/uncertainty/decompose.py and docs/backup/design.md §Uncertainty):

    1. aleatoric  - mean over members of the Gaussian-NLL variance head;
    2. epistemic  - std over member means (ensemble disagreement);
    3. input      - Monte-Carlo propagation of input measurement errors
                    (v1.1 protocol: u_inputs_Monte_Carlo.mlx) through the
                    frozen ensemble.

    ``combine`` = ``rss`` sums them in quadrature (independent sources),
    matching v1.1's final ``sqrt(sum of squares)``.
    """

    n_mc_draws: int = DEFAULT_MC_DRAWS
    propagate_input: bool = True
    input_uncertainties: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INPUT_UNCERTAINTIES)
    )
    per_pixel_sst_err: bool = False  # use an SST error field if present
    per_pixel_ssh_err: bool = False  # use an SSH error field if present
    combine: str = "rss"  # rss | none

    def __post_init__(self) -> None:
        if self.n_mc_draws < 1:
            raise ConfigError("n_mc_draws must be >= 1")
        if self.combine not in {"rss", "none"}:
            raise ConfigError(f"unknown combine mode: {self.combine}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputConfig:
    """Product and run outputs."""

    out_dir: str = "outputs"
    product_name: str = "ReCAD-v2.0"
    product_attrs: dict[str, str] = field(default_factory=dict)
    write_individual_members: bool = False

    def __post_init__(self) -> None:
        if not self.product_name:
            raise ConfigError("product_name must not be empty")


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Complete, validated configuration of one ReCAD v2.0 experiment."""

    grid: GridConfig
    coastal_mask: CoastalMaskConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    ensemble: EnsembleConfig
    train: TrainConfig
    uncertainty: UncertaintyConfig
    output: OutputConfig

    @staticmethod
    def from_yaml(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
        """Load a config from a YAML file (or defaults), then apply overrides.

        ``overrides`` uses dotted keys, e.g. ``{"train.n_epochs": 3}``,
        allowing the CLI to tune a file-based config without rewriting it.
        """
        raw: dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        if overrides:
            _apply_overrides(raw, overrides)
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> Config:
        """Build and validate a Config from a (possibly partial) dict."""
        sections = {
            "grid": GridConfig,
            "coastal_mask": CoastalMaskConfig,
            "data": DataConfig,
            "split": SplitConfig,
            "model": ModelConfig,
            "ensemble": EnsembleConfig,
            "train": TrainConfig,
            "uncertainty": UncertaintyConfig,
            "output": OutputConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, cls in sections.items():
            section = raw.get(key) or {}
            if isinstance(section, dict):
                # YAML parses tuples as lists; coerce back to tuple fields.
                section = _coerce_tuple_fields(cls, section)
                try:
                    kwargs[key] = cls(**section)
                except TypeError as exc:  # unknown key inside a section
                    raise ConfigError(f"invalid key in [{key}]: {exc}") from exc
            else:
                raise ConfigError(f"section '{key}' must be a mapping")
        cfg = Config(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.grid.__post_init__()
        self.coastal_mask.__post_init__()
        self.split.__post_init__()
        self.model.__post_init__()
        self.ensemble.__post_init__()
        self.train.__post_init__()
        self.uncertainty.__post_init__()

    def to_yaml(self, path: str) -> None:
        """Serialize the full config to YAML (audit trail)."""
        payload = {
            "grid": dataclasses.asdict(self.grid),
            "coastal_mask": dataclasses.asdict(self.coastal_mask),
            "data": dataclasses.asdict(self.data),
            "split": dataclasses.asdict(self.split),
            "model": dataclasses.asdict(self.model),
            "ensemble": dataclasses.asdict(self.ensemble),
            "train": dataclasses.asdict(self.train),
            "uncertainty": dataclasses.asdict(self.uncertainty),
            "output": dataclasses.asdict(self.output),
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Apply dotted-key overrides into a nested dict, creating sections."""
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        section = raw
        for key in keys[:-1]:
            section = section.setdefault(key, {})
            if not isinstance(section, dict):
                raise ConfigError(f"override path '{dotted}' conflicts with a scalar")
        section[keys[-1]] = value


def _coerce_tuple_fields(cls, section: dict[str, Any]) -> dict[str, Any]:
    """Convert list values to tuples where the dataclass field is tuple-typed."""
    import typing

    hints = typing.get_type_hints(cls)
    for key, value in list(section.items()):
        annotation = hints.get(key)
        if (
            annotation is not None
            and typing.get_origin(annotation) is tuple
            and isinstance(value, list)
        ):
            section[key] = tuple(value)
    return section
