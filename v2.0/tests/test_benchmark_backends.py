"""Fast tests for the Issue 89 uniform model adapters."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from recad.evaluate.benchmark_backends import (
    PATCH_FAMILIES,
    TABULAR_FAMILIES,
    fit_patch_dataset_estimator,
    fit_patch_estimator,
    fit_tabular_estimator,
    predict_patch_dataset_estimator,
    predict_patch_estimator,
    predict_tabular_estimator,
)


def regression_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(89)
    features = rng.normal(size=(96, 6)).astype("float32")
    target = (features[:, 0] - 0.5 * features[:, 1] + rng.normal(0, 0.05, 96)).astype("float32")
    return features, target


@pytest.mark.l1
def test_frozen_family_partition_is_complete() -> None:
    assert set(TABULAR_FAMILIES) | set(PATCH_FAMILIES) == {
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
    }


@pytest.mark.l1
@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("ridge", {"alpha": 1.0}),
        (
            "random_forest",
            {
                "n_estimators": 8,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "max_features": 0.8,
            },
        ),
        (
            "extra_trees",
            {
                "n_estimators": 8,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "max_features": 0.8,
            },
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 10,
                "learning_rate": 0.1,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 5,
                "l2_regularization": 0.1,
            },
        ),
        (
            "residual_mlp",
            {
                "width": 16,
                "depth": 1,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 1,
            },
        ),
        (
            "ft_transformer",
            {
                "width": 16,
                "depth": 1,
                "heads": 4,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 1,
            },
        ),
    ],
)
def test_core_tabular_adapters(family: str, parameters: dict[str, object]) -> None:
    features, target = regression_data()
    model = fit_tabular_estimator(family, features, target, parameters, seed=89)
    prediction = predict_tabular_estimator(model, features[:11])
    assert prediction.shape == (11,)
    assert np.isfinite(prediction).all()


@pytest.mark.l1
@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        (
            "unet_convlstm",
            {
                "base_channels": 4,
                "depth": 2,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "epochs": 1,
                "batch_size": 8,
            },
        ),
        (
            "vision_transformer",
            {
                "width": 16,
                "depth": 1,
                "heads": 4,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "epochs": 1,
                "batch_size": 8,
            },
        ),
    ],
)
def test_patch_adapters(family: str, parameters: dict[str, object]) -> None:
    rng = np.random.default_rng(90)
    features = rng.normal(size=(16, 3, 5, 8, 8)).astype("float32")
    target = rng.normal(size=16).astype("float32")
    model = fit_patch_estimator(family, features, target, parameters, seed=90)
    prediction = predict_patch_estimator(model, features[:5])
    assert prediction.shape == (5,)
    assert np.isfinite(prediction).all()


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self) -> None:
        rng = np.random.default_rng(91)
        self.features = rng.normal(size=(12, 3, 5, 8, 8)).astype("float32")
        self.target = rng.normal(size=12).astype("float32")

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[index]), torch.tensor(self.target[index])


@pytest.mark.l1
def test_streaming_patch_adapter() -> None:
    dataset = PatchDataset()
    parameters = {
        "base_channels": 4,
        "depth": 2,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "epochs": 1,
        "batch_size": 4,
    }
    model = fit_patch_dataset_estimator("unet_convlstm", dataset, parameters, seed=91)
    prediction = predict_patch_dataset_estimator(model, dataset, batch_size=4)
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()
