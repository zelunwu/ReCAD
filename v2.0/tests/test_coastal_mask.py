from __future__ import annotations

import numpy as np
import pytest

from recad.data.coastal_mask import (
    CoastalMaskSpec,
    build_coastal_mask_dataset,
    mask_area_km2,
    regular_cell_area_km2,
)


@pytest.mark.l1
def test_spec_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="maximum_distance_km"):
        CoastalMaskSpec(maximum_distance_km=0)
    with pytest.raises(ValueError, match="significant_island_area_km2"):
        CoastalMaskSpec(significant_island_area_km2=-1)
    with pytest.raises(ValueError, match="shelf_depth_m"):
        CoastalMaskSpec(shelf_depth_m=0)


@pytest.mark.l1
def test_frozen_spec_defaults() -> None:
    spec = CoastalMaskSpec()
    assert spec.maximum_distance_km == 400.0
    assert spec.significant_island_area_km2 == 1400.0
    assert spec.shelf_depth_m == 200.0


@pytest.mark.l1
def test_cell_area_decreases_toward_poles_and_sums_to_earth_area() -> None:
    latitude = np.arange(-89.5, 90.0, 1.0)
    area = regular_cell_area_km2(latitude, 1.0)
    assert area[89] > area[0]
    global_area = float(area.sum() * 360)
    expected = 4 * np.pi * 6371.0088**2
    assert global_area == pytest.approx(expected, rel=1e-12)


@pytest.mark.l2
def test_masks_keep_geometry_support_and_shelf_concepts_separate() -> None:
    lon = np.array([0.0, 0.125, 0.25, 0.375])
    lat = np.array([-0.125, 0.0])
    all_land = np.array([[-1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, np.nan]])
    significant = np.array([[10.0, 0.0, 400.0, 400.1], [10.0, 25.0, 399.9, 20.0]])
    bathymetry = np.array([[5.0, -10.0, -200.0, -100.0], [1.0, -201.0, np.nan, -20.0]])

    dataset = build_coastal_mask_dataset(
        lon,
        lat,
        all_land,
        significant,
        bathymetry_m=bathymetry,
    )

    np.testing.assert_array_equal(
        dataset.ocean_mask,
        np.array([[0, 1, 1, 1], [0, 1, 1, 0]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        dataset.coastal_product_mask,
        np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        dataset.shelf_core_mask,
        np.array([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        dataset.slope_deep_mask,
        np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int8),
    )
    assert dataset.bathymetry_unclassified_mask.values[1, 2] == 1
    assert "observation_support" not in dataset
    assert "predictor_ready" not in dataset
    assert dataset.attrs["mask_independent_of_target_observations"] == "true"


@pytest.mark.l1
def test_mask_area_checks_shape() -> None:
    lat = np.array([-0.125, 0.0, 0.125])
    mask = np.ones((3, 4), dtype=bool)
    assert mask_area_km2(mask, lat, 0.125) > 0
    with pytest.raises(ValueError, match="latitude dimension"):
        mask_area_km2(np.ones((2, 4)), lat, 0.125)


@pytest.mark.l1
def test_mask_preserves_dateline_endpoint_nodes() -> None:
    lon = np.array([0.0, 359.875])
    lat = np.array([-78.0, 84.0])
    all_land = np.ones((2, 2))
    significant = np.array([[0.0, 400.0], [400.1, 25.0]])

    dataset = build_coastal_mask_dataset(lon, lat, all_land, significant)

    np.testing.assert_array_equal(dataset.lon, lon.astype(np.float32))
    np.testing.assert_array_equal(dataset.lat, lat.astype(np.float32))
    np.testing.assert_array_equal(
        dataset.coastal_product_mask,
        np.array([[1, 1], [0, 1]], dtype=np.int8),
    )
