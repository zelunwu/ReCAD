"""L1 tests for deployable Issue 63 process features."""

import numpy as np
import pandas as pd

from recad.evaluate.process_predictors import (
    ContinuousBackground,
    cardinal_water_neighbours,
    local_contrast,
)


def test_cardinal_neighbours_wrap_and_do_not_cross_missing_water() -> None:
    row = np.array([0, 0, 0, 1])
    column = np.array([0, 3, 1, 0])
    neighbours = cardinal_water_neighbours(row, column, latitude_cells=2, longitude_cells=4)
    assert set(neighbours[0][neighbours[0] >= 0]) == {1, 2, 3}
    assert 2 not in neighbours[3]


def test_local_contrast_has_zero_fallback_and_explicit_count() -> None:
    neighbours = np.array([[1, -1, -1, -1], [0, -1, -1, -1], [-1, -1, -1, -1]])
    result = local_contrast(np.array([1.0, 4.0, np.nan]), neighbours)
    np.testing.assert_allclose(result["mean_absolute_difference"], [3.0, 3.0, 0.0])
    np.testing.assert_allclose(result["range"], [3.0, 3.0, 0.0])
    np.testing.assert_array_equal(result["valid_neighbour_count"], [1, 1, 0])


def test_continuous_background_fits_smooth_training_relation() -> None:
    frame = pd.DataFrame(
        {
            "latitude": np.linspace(-60, 60, 60),
            "longitude": np.linspace(-170, 170, 60),
            "mon": np.tile(np.arange(1, 13), 5),
            "yr": np.repeat(np.arange(2000, 2005), 12),
        }
    )
    truth = 380 + 0.8 * frame.latitude + 2 * np.sin(2 * np.pi * (frame.mon - 1) / 12)
    model = ContinuousBackground.fit(frame, truth.to_numpy(), alpha=0.01)
    prediction = model.predict(frame)
    assert np.sqrt(np.mean((prediction - truth) ** 2)) < 5.0
