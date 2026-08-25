"""Config layer tests: validation, YAML round-trip, overrides."""

from __future__ import annotations

import pytest

from recad.config import Config, ConfigError


def test_defaults_are_valid():
    cfg = Config.from_dict({})
    cfg.validate()
    assert cfg.grid.resolution_deg == 0.125
    assert cfg.split.test_holdout_years == (2004, 2005)
    assert cfg.ensemble.n_members == 5


def test_invalid_resolution_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"grid": {"resolution_deg": 0.07}})


def test_invalid_split_scheme_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"split": {"scheme": "magic"}})


def test_unknown_section_key_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"train": {"n_epoch": 5}})  # typo: n_epochs


def test_invalid_holdout_duplicates_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"split": {"test_holdout_years": [2004, 2004]}})


def test_yaml_round_trip(tmp_path):
    cfg = Config.from_dict({"train": {"n_epochs": 7}, "model": {"embed_dim": 32}})
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = Config.from_yaml(str(path))
    assert loaded.train.n_epochs == 7
    assert loaded.model.embed_dim == 32
    assert loaded == cfg


def test_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    Config.from_dict({"train": {"n_epochs": 5}}).to_yaml(path)
    cfg = Config.from_yaml(str(path), overrides={"train.n_epochs": 9, "ensemble.n_members": 3})
    assert cfg.train.n_epochs == 9
    assert cfg.ensemble.n_members == 3
    assert cfg.split.test_holdout_years == (2004, 2005)  # untouched


def test_grid_cell_counts():
    cfg = Config.from_dict({"grid": {"domain": "global", "resolution_deg": 0.125}})
    assert cfg.grid.n_lon == 2880
    assert cfg.grid.n_lat == 1297
