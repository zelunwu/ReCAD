"""Frozen fair-model benchmark contract and deterministic advancement rules."""

from __future__ import annotations

import configparser
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FROZEN_FAMILIES = (
    "ridge",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
    "residual_mlp",
    "realmlp",
    "tabm",
    "ft_transformer",
    "tabpfn_3",
    "unet_convlstm",
    "vision_transformer",
)

FROZEN_PROTOCOLS = (
    "whole_cruise",
    "buffered_geographic",
    "whole_region",
    "forward_time",
    "early_era",
)


@dataclass(frozen=True)
class FamilyContract:
    """One frozen model-family column."""

    name: str
    kind: str
    representation: str
    backend: str
    neural: bool
    minimum_trials: int
    maximum_trials: int
    minimum_seeds: int
    search_space: dict[str, dict[str, object]]


@dataclass(frozen=True)
class BenchmarkContract:
    """Machine-readable Issue 88 benchmark contract."""

    contract_id: str
    schema_version: str
    targets: tuple[str, ...]
    families: tuple[str, ...]
    protocols: tuple[str, ...]
    fidelity_rows: tuple[int | str, ...]
    tail_fraction: float
    maximum_tail_relative_improvement: float
    capacity_tail_relative_improvement: float
    advancers_per_target: int
    protocol_weights: dict[str, float]
    tie_breakers: tuple[str, ...]
    required_secondary_metrics: tuple[str, ...]
    family_contracts: dict[str, FamilyContract]


@dataclass(frozen=True)
class ConvergenceDecision:
    """One reproducible empirical-convergence decision."""

    status: str
    converged: bool
    tail_relative_improvement: float
    reasons: tuple[str, ...]


def _required(parser: configparser.ConfigParser, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise ValueError(f"missing required setting [{section}] {key}")
    value = parser.get(section, key).strip()
    if not value:
        raise ValueError(f"empty required setting [{section}] {key}")
    return value


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV settings must contain unique values")
    return values


def _protocol_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        name, raw_weight = item.split(":", maxsplit=1)
        weights[name.strip()] = float(raw_weight)
    if tuple(weights) != FROZEN_PROTOCOLS or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("protocol weights must cover the frozen protocols and sum to one")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("protocol weights must be positive")
    return weights


def _validate_search_space(space: dict[str, dict[str, object]]) -> None:
    if not space:
        raise ValueError("every family needs a non-empty search space")
    for name, specification in space.items():
        if not name or not isinstance(specification, dict):
            raise ValueError("invalid search-space entry")
        kind = specification.get("type")
        if kind == "choice":
            values = specification.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"choice parameter {name} has no values")
        elif kind in {"int", "float", "log_float"}:
            low, high = specification.get("low"), specification.get("high")
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(f"numeric parameter {name} needs numeric bounds")
            if low >= high or (kind == "log_float" and low <= 0):
                raise ValueError(f"invalid bounds for {name}")
        else:
            raise ValueError(f"unsupported search-space type for {name}: {kind}")


def load_benchmark_contract(path: str | Path) -> BenchmarkContract:
    """Load and strictly validate the frozen Issue 88 configuration."""

    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(Path(path), encoding="utf-8"):
        raise FileNotFoundError(path)
    families = _csv(_required(parser, "contract", "families"))
    protocols = _csv(_required(parser, "contract", "protocols"))
    targets = _csv(_required(parser, "contract", "targets"))
    if families != FROZEN_FAMILIES:
        raise ValueError("model-family columns drifted from the frozen set")
    if protocols != FROZEN_PROTOCOLS or targets != ("SSS", "fCO2"):
        raise ValueError("targets or development protocols drifted")

    minimum_seeds = parser.getint("optimization", "minimum_seeds")
    minimum_neural_seeds = parser.getint("optimization", "minimum_neural_seeds")
    family_contracts: dict[str, FamilyContract] = {}
    for family in families:
        minimum_trials, maximum_trials = (
            int(item) for item in _required(parser, "resources", f"{family}_trials").split(",")
        )
        if not 1 <= minimum_trials <= maximum_trials:
            raise ValueError(f"invalid trial budget for {family}")
        section = f"family:{family}"
        neural = parser.getboolean(section, "neural")
        search_space = json.loads(_required(parser, section, "search_space"))
        _validate_search_space(search_space)
        family_contracts[family] = FamilyContract(
            name=family,
            kind=_required(parser, section, "kind"),
            representation=_required(parser, section, "representation"),
            backend=_required(parser, section, "backend"),
            neural=neural,
            minimum_trials=minimum_trials,
            maximum_trials=maximum_trials,
            minimum_seeds=minimum_neural_seeds if neural else minimum_seeds,
            search_space=search_space,
        )

    fidelity_rows: list[int | str] = []
    for item in _csv(_required(parser, "optimization", "fidelity_rows")):
        fidelity_rows.append(item if item == "FULL" else int(item))
    if tuple(fidelity_rows) != (120000, 300000, "FULL"):
        raise ValueError("fidelity schedule drifted")

    prohibited = (
        parser.getboolean("data", "external_pretraining_in_main_ranking")
        or parser.getboolean("data", "external_target_products_in_main_ranking")
        or parser.getboolean("data", "one_time_independent_labels_opened")
        or parser.getboolean("data", "labels_2026_opened")
        or parser.getboolean("policy", "different_rows_between_families")
        or parser.getboolean("policy", "target_specific_outer_folds")
        or parser.getboolean("policy", "external_family_specific_subsample")
        or parser.getboolean("policy", "regional_family_switching")
        or parser.getboolean("policy", "result_dependent_search_space_change")
        or parser.getboolean("policy", "incomplete_model_may_advance")
        or parser.getboolean("policy", "hard_partition_is_product_candidate")
    )
    required = (
        parser.getboolean("data", "same_eligible_rows_within_target")
        and parser.getboolean("optimization", "full_data_required")
        and parser.getboolean("optimization", "group_safe_inner_validation")
        and parser.getboolean("policy", "preserve_negative_results")
    )
    if prohibited or not required:
        raise ValueError("benchmark safety policy was weakened")

    tail_fraction = parser.getfloat("optimization", "tail_fraction")
    tail_limit = parser.getfloat("optimization", "maximum_tail_relative_improvement")
    capacity_limit = parser.getfloat("optimization", "capacity_tail_relative_improvement")
    if not 0 < tail_fraction <= 0.5 or not 0 <= tail_limit <= 0.01:
        raise ValueError("invalid search convergence thresholds")
    if not 0 <= capacity_limit <= 0.01:
        raise ValueError("invalid capacity convergence threshold")

    return BenchmarkContract(
        contract_id=_required(parser, "contract", "contract_id"),
        schema_version=_required(parser, "contract", "schema_version"),
        targets=targets,
        families=families,
        protocols=protocols,
        fidelity_rows=tuple(fidelity_rows),
        tail_fraction=tail_fraction,
        maximum_tail_relative_improvement=tail_limit,
        capacity_tail_relative_improvement=capacity_limit,
        advancers_per_target=parser.getint("selection", "advancers_per_target"),
        protocol_weights=_protocol_weights(_required(parser, "selection", "protocol_weights")),
        tie_breakers=_csv(_required(parser, "selection", "tie_breakers")),
        required_secondary_metrics=_csv(
            _required(parser, "selection", "required_secondary_metrics")
        ),
        family_contracts=family_contracts,
    )


def assess_empirical_convergence(
    contract: BenchmarkContract,
    family: str,
    objective_history: list[float],
    *,
    full_data_completed: bool,
    seeds_completed: int,
    capacity_tail_relative_improvement: float,
) -> ConvergenceDecision:
    """Apply the preregistered trial, seed, fidelity, and plateau gates."""

    profile = contract.family_contracts[family]
    reasons: list[str] = []
    if not objective_history or any(not math.isfinite(value) for value in objective_history):
        return ConvergenceDecision("optimization_incomplete", False, math.inf, ("invalid_trials",))
    tail_count = max(1, math.ceil(len(objective_history) * contract.tail_fraction))
    boundary = max(1, len(objective_history) - tail_count)
    best_before_tail = min(objective_history[:boundary])
    best_final = min(objective_history)
    denominator = max(abs(best_before_tail), 1e-12)
    tail_improvement = max((best_before_tail - best_final) / denominator, 0.0)
    if len(objective_history) < profile.minimum_trials:
        reasons.append("minimum_trials_not_completed")
    if not full_data_completed:
        reasons.append("full_data_not_completed")
    if seeds_completed < profile.minimum_seeds:
        reasons.append("minimum_seeds_not_completed")
    if tail_improvement >= contract.maximum_tail_relative_improvement:
        reasons.append("search_tail_not_flat")
    if capacity_tail_relative_improvement >= contract.capacity_tail_relative_improvement:
        reasons.append("capacity_tail_not_flat")
    if reasons:
        can_continue = len(objective_history) < profile.maximum_trials
        status = "needs_more_optimization" if can_continue else "optimization_incomplete"
        return ConvergenceDecision(status, False, tail_improvement, tuple(reasons))
    return ConvergenceDecision("empirically_converged", True, tail_improvement, ())


def rank_advancers(contract: BenchmarkContract, metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank eligible families using frozen protocol-normalized cruise-equal RMSE."""

    required = {
        "target",
        "family",
        "protocol",
        "cruise_equal_rmse",
        "control_cruise_equal_rmse",
        "whole_region_ratio",
        "worst_region_ratio",
        "wall_seconds",
        "optimization_status",
        *contract.required_secondary_metrics,
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"ranking input misses columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for (target, family), frame in metrics.groupby(["target", "family"], sort=False):
        if target not in contract.targets or family not in contract.families:
            raise ValueError(f"unexpected target/family: {target}/{family}")
        if set(frame.protocol) != set(contract.protocols) or frame.protocol.duplicated().any():
            continue
        if set(frame.optimization_status) != {"empirically_converged"}:
            continue
        if frame[list(contract.required_secondary_metrics)].isna().any().any():
            continue
        ratios = frame.set_index("protocol").cruise_equal_rmse.div(
            frame.set_index("protocol").control_cruise_equal_rmse
        )
        if not all(math.isfinite(value) and value > 0 for value in ratios):
            continue
        log_score = sum(
            contract.protocol_weights[protocol] * math.log(float(ratios[protocol]))
            for protocol in contract.protocols
        )
        rows.append(
            {
                "target": target,
                "family": family,
                "primary_score": math.exp(log_score),
                "whole_region_ratio": float(frame.whole_region_ratio.iloc[0]),
                "worst_region_ratio": float(frame.worst_region_ratio.iloc[0]),
                "wall_seconds": float(frame.wall_seconds.sum()),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking.assign(rank=pd.Series(dtype=int), advances=pd.Series(dtype=bool))
    ranking = ranking.sort_values(
        ["target", "primary_score", *contract.tie_breakers], kind="mergesort"
    ).reset_index(drop=True)
    ranking["rank"] = ranking.groupby("target", sort=False).cumcount() + 1
    ranking["advances"] = ranking["rank"] <= contract.advancers_per_target
    return ranking


def primary_cell_budget(contract: BenchmarkContract) -> pd.DataFrame:
    """Return the frozen upper bound of scientific comparison cells."""

    targets = len(contract.targets)
    global_cells = len(contract.families) * targets
    regional_cells = contract.advancers_per_target * 3 * targets
    representation_cells = 10
    return pd.DataFrame(
        [
            ("global family row", global_cells),
            ("three additional spatial rows", regional_cells),
            ("bounded target-representation comparison", representation_cells),
            ("total upper bound", global_cells + regional_cells + representation_cells),
        ],
        columns=["stage", "primary_cells"],
    )
