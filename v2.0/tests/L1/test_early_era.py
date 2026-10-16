from pathlib import Path

import pandas as pd
import pytest

from recad.evaluate.early_era import EarlyEraFold, complete_group_indices, load_early_era_contract


def test_complete_group_indices_excludes_boundary_cruises() -> None:
    frame = pd.DataFrame(
        {
            "group_key": ["early", "early", "mixed", "mixed", "late", "late"],
            "yr": [1993, 1998, 1998, 2004, 2004, 2005],
        }
    )
    train, held, excluded = complete_group_indices(frame, EarlyEraFold(0, 1993, 1998, 2004))
    assert set(train) == {4, 5}
    assert set(held) == {0, 1}
    assert set(excluded) == {2, 3}


def test_frozen_contract_is_strict() -> None:
    path = Path(__file__).parents[2] / "configs/frozen/early_era_validation_v2.5.in"
    contract = load_early_era_contract(path)
    assert contract.sss_seeds == (62024, 62025, 62026)
    assert contract.folds[0].train_start > contract.folds[0].held_end
    text = path.read_text(encoding="utf-8").replace(
        "later_labels_only = true", "later_labels_only = false"
    )
    temporary = path.with_name("invalid_early_era.in")
    temporary.write_text(text, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="later-label-only"):
            load_early_era_contract(temporary)
    finally:
        temporary.unlink()
