from __future__ import annotations

import numpy as np
import pytest

from recad.chem.inverse_dic import forward_fco2, inverse_dic
from recad.evaluate.dic_reliability import (
    covariance_from_sigmas,
    monte_carlo_dic,
)

pytestmark = pytest.mark.l2


def test_seeded_monte_carlo_is_reproducible_and_solver_closes() -> None:
    means = np.array([[2300.0, 400.0, 35.0], [2200.0, 500.0, 32.0]])
    covariance = covariance_from_sigmas(
        np.array([[40.0, 20.0, 0.5], [50.0, 25.0, 0.8]]), mode="independent"
    )
    first = monte_carlo_dic(means, covariance, [20.0, 15.0], draws=64, seed=28)
    second = monte_carlo_dic(means, covariance, [20.0, 15.0], draws=64, seed=28)
    assert np.allclose(first["q50"], second["q50"])
    assert np.all(first["q95"] > first["q05"])

    dic = inverse_dic(means[:, 0], means[:, 1], means[:, 2], [20.0, 15.0])
    recovered = forward_fco2(means[:, 0], dic, means[:, 2], [20.0, 15.0])
    assert np.allclose(recovered, means[:, 1], atol=1e-6)
