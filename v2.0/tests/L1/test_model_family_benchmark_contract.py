from __future__ import annotations

from pathlib import Path

import pandas as pd

from recad.evaluate.benchmark_contract import (
    FROZEN_FAMILIES,
    FROZEN_PROTOCOLS,
    assess_empirical_convergence,
    load_benchmark_contract,
    primary_cell_budget,
    rank_advancers,
)

CONFIG = Path(__file__).parents[2] / "configs" / "frozen" / "model_family_benchmark_v2.5.in"


def test_frozen_benchmark_contract_loads_exact_matrix() -> None:
    contract = load_benchmark_contract(CONFIG)

    assert contract.families == FROZEN_FAMILIES
    assert contract.protocols == FROZEN_PROTOCOLS
    assert contract.targets == ("SSS", "fCO2")
    assert contract.advancers_per_target == 5
    assert sum(contract.protocol_weights.values()) == 1.0
    assert primary_cell_budget(contract).primary_cells.tolist() == [28, 30, 10, 68]
    assert contract.family_contracts["random_forest"].minimum_trials == 48
    assert contract.family_contracts["tabm"].minimum_seeds == 5


def test_convergence_requires_flat_search_capacity_full_data_and_seeds() -> None:
    contract = load_benchmark_contract(CONFIG)
    flat = [1.0] * contract.family_contracts["ridge"].minimum_trials

    accepted = assess_empirical_convergence(
        contract,
        "ridge",
        flat,
        full_data_completed=True,
        seeds_completed=3,
        capacity_tail_relative_improvement=0.0,
    )
    rejected = assess_empirical_convergence(
        contract,
        "ridge",
        [2.0 - index * 0.01 for index in range(len(flat))],
        full_data_completed=False,
        seeds_completed=2,
        capacity_tail_relative_improvement=0.01,
    )

    assert accepted.converged
    assert accepted.status == "empirically_converged"
    assert not rejected.converged
    assert "full_data_not_completed" in rejected.reasons
    assert "minimum_seeds_not_completed" in rejected.reasons
    assert "search_tail_not_flat" in rejected.reasons
    assert "capacity_tail_not_flat" in rejected.reasons


def _ranking_rows() -> pd.DataFrame:
    contract = load_benchmark_contract(CONFIG)
    rows = []
    for target in contract.targets:
        for index, family in enumerate(contract.families[:6]):
            for protocol in contract.protocols:
                rows.append(
                    {
                        "target": target,
                        "family": family,
                        "protocol": protocol,
                        "cruise_equal_rmse": 1.0 + index / 10,
                        "control_cruise_equal_rmse": 2.0,
                        "whole_region_ratio": 0.5 + index / 20,
                        "worst_region_ratio": 0.6 + index / 20,
                        "wall_seconds": 10.0 + index,
                        "optimization_status": "empirically_converged",
                        "observation_equal_rmse": 1.0,
                        "mae": 0.8,
                        "bias": 0.0,
                        "r2": 0.8,
                        "worst_region_rmse": 1.5,
                        "full_domain_input_availability": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_ranking_advances_exactly_five_complete_converged_families() -> None:
    contract = load_benchmark_contract(CONFIG)
    ranking = rank_advancers(contract, _ranking_rows())

    assert ranking.groupby("target").advances.sum().to_dict() == {"SSS": 5, "fCO2": 5}
    assert ranking.groupby("target").first().family.to_dict() == {"SSS": "ridge", "fCO2": "ridge"}


def test_ranking_excludes_missing_protocol_and_incomplete_optimization() -> None:
    contract = load_benchmark_contract(CONFIG)
    rows = _ranking_rows()
    rows = rows[
        ~((rows.target == "SSS") & (rows.family == "ridge") & (rows.protocol == "early_era"))
    ]
    rows.loc[(rows.target == "fCO2") & (rows.family == "ridge"), "optimization_status"] = (
        "optimization_incomplete"
    )

    ranking = rank_advancers(contract, rows)

    assert not ((ranking.target == "SSS") & (ranking.family == "ridge")).any()
    assert not ((ranking.target == "fCO2") & (ranking.family == "ridge")).any()
