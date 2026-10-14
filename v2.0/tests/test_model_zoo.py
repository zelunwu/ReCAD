"""Tests for the finite Issue 62 model-family implementations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from recad.evaluate.model_zoo import (
    FTTransformer,
    ResidualMLP,
    TabularTransform,
    deterministic_cap,
    fit_model,
    group_equal_rmse,
    predict_model,
    save_model,
)


@pytest.mark.l1
def test_transform_uses_training_statistics_and_returns_finite_values() -> None:
    train = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [2.0, 2.0, 2.0]})
    held = pd.DataFrame({"a": [np.nan, 101.0], "b": [np.nan, 2.0]})

    transform = TabularTransform.fit(train, ("a", "b"))
    values = transform.apply(held)

    assert values.shape == (2, 2)
    assert np.isfinite(values).all()
    assert transform.median == (2.0, 2.0)
    assert transform.scale[1] == 1.0
    assert values[0].tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.l1
def test_group_equal_rmse_and_deterministic_cap() -> None:
    frame = pd.DataFrame(
        {
            "truth": [0.0, 0.0, 0.0, 0.0],
            "group_key": ["large", "large", "large", "small"],
            "record_id": [1, 2, 3, 4],
        }
    )
    prediction = np.array([0.0, 0.0, 0.0, 2.0])

    assert group_equal_rmse(frame, prediction) == pytest.approx(np.sqrt(2.0))
    first = deterministic_cap(frame, 2, "fixed")
    second = deterministic_cap(frame, 2, "fixed")
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2


@pytest.mark.l1
@pytest.mark.parametrize(
    ("network", "features"),
    [
        (ResidualMLP(n_features=5, width=16, depth=2, dropout=0.0), 5),
        (FTTransformer(n_features=5, width=16, depth=2, heads=4, dropout=0.0), 5),
    ],
)
def test_neural_family_shapes_and_gradients(network: torch.nn.Module, features: int) -> None:
    values = torch.randn(7, features)
    output = network(values)
    output.square().mean().backward()

    assert output.shape == (7,)
    assert all(parameter.grad is not None for parameter in network.parameters())


@pytest.mark.l2
def test_ridge_fit_predict_and_checkpoint_round_trip(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "a": np.linspace(-2.0, 2.0, 40),
            "b": np.tile([0.0, 1.0], 20),
        }
    )
    target = 3.0 * frame.a.to_numpy() - 2.0 * frame.b.to_numpy() + 1.0
    fitted = fit_model(
        "ridge",
        [0.01],
        frame,
        ("a", "b"),
        target,
        np.ones(len(frame)),
        seed=62,
    )

    prediction, elapsed = predict_model(fitted, frame)
    files = save_model(fitted, tmp_path / "ridge")
    metadata = json.loads(files[0].read_text(encoding="utf-8"))

    assert np.sqrt(np.mean((prediction - target) ** 2)) < 0.01
    assert elapsed >= 0.0
    assert all(path.is_file() for path in files)
    assert metadata["family"] == "ridge"
    assert metadata["transform"]["features"] == ["a", "b"]
