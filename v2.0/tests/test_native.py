"""Native kernel / NumPy fallback equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest

from recad.utils.native import native_available, recad_native


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    mask = rng.random(shape) < 0.2
    x[mask] = np.nan
    return x


@pytest.mark.skipif(not native_available(), reason="native DLL not built")
def test_native_available_on_this_machine():
    assert native_available()


def test_masked_moments_vs_numpy():
    data = _rand((4, 3, 20))
    # compute expected
    mean_exp = np.empty(3, dtype=np.float32)
    std_exp = np.empty(3, dtype=np.float32)
    for c in range(3):
        v = data[:, c, :].ravel()
        v = v[~np.isnan(v)]
        mean_exp[c] = np.mean(v)
        std_exp[c] = np.std(v)
    mean, std = recad_native.masked_moments(data, eps=1e-6)
    assert np.allclose(mean, mean_exp, atol=1e-5)
    assert np.allclose(std, std_exp, atol=1e-5)


def test_masked_moments_nan_channel():
    data = np.ones((2, 2, 3), dtype=np.float32)
    data[:, 1, :] = np.nan
    means, stds = recad_native.masked_moments(data, eps=1e-6)
    assert np.isnan(means[1]) and np.isnan(stds[1])
    assert means[0] == 1.0


def test_masked_moments_constant_channel_eps_floor():
    data = np.ones((2, 1, 4), dtype=np.float32)
    _, std = recad_native.masked_moments(data, eps=1e-6)
    assert std[0] == pytest.approx(1e-6)


def test_normalize_vs_numpy():
    data = _rand((3, 2, 10))
    mean, std = recad_native.masked_moments(data, eps=1e-6)
    out = recad_native.normalize_zscore(data, mean, std)
    # NaN passthrough
    assert np.isnan(out[np.isnan(data)]).all()
    for c in range(2):
        v = data[:, c, :].ravel()
        sel = ~np.isnan(v)
        assert np.allclose(out[:, c, :].ravel()[sel], (v[sel] - mean[c]) / std[c], atol=1e-5)


def test_patch_aggregate():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0], dtype=np.float32)
    ids = np.array([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int32)
    counts, means = recad_native.patch_aggregate(values, ids, 2)
    assert counts.tolist() == [4, 3]
    assert means[0] == pytest.approx(4.25)
    assert means[1] == pytest.approx(4.0)


def test_patch_aggregate_all_nan():
    values = np.full(2, np.nan, dtype=np.float32)
    ids = np.array([0, 2], dtype=np.int32)  # patch 1 absent
    counts, means = recad_native.patch_aggregate(values, ids, 3)
    assert counts.tolist() == [0, 0, 0]
    assert np.isnan(means).all()


def test_patch_aggregate_oob_rejected():
    values = np.ones(3, dtype=np.float32)
    ids = np.array([0, 5, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        recad_native.patch_aggregate(values, ids, 2)
    ids_neg = np.array([0, -1, 1], dtype=np.int32)
    with pytest.raises(ValueError):
        recad_native.patch_aggregate(values, ids_neg, 2)


def test_gather_nearest():
    src = np.arange(5, dtype=np.float32) * 10
    idx = np.array([0, 4, 2, -1, 7, 1], dtype=np.int64)
    out = recad_native.gather_nearest(src, idx, fill_value=-1.0)
    assert out.tolist() == [0.0, 40.0, 20.0, -1.0, -1.0, 10.0]


def test_fallback_equivalence_when_native_missing(monkeypatch, tmp_path):
    """Force the NumPy path and confirm identical results to the native one."""
    data = _rand((3, 2, 10), seed=5)
    native_mean, native_std = recad_native.masked_moments(data, eps=1e-6)

    import recad.utils.native as native_mod

    def deny(*args, **kwargs):
        return None

    monkeypatch.setattr(native_mod, "_discover_library", deny)
    fallback = native_mod._NativeKernels()
    fb_mean, fb_std = fallback.masked_moments(data, eps=1e-6)
    assert np.allclose(fb_mean, native_mean, atol=1e-6)
    assert np.allclose(fb_std, native_std, atol=1e-6)

    counts_n, means_n = recad_native.patch_aggregate(
        data.ravel(), np.zeros(data.size, dtype=np.int32), 1
    )
    counts_f, means_f = fallback.patch_aggregate(
        data.ravel(), np.zeros(data.size, dtype=np.int32), 1
    )
    assert np.array_equal(counts_n, counts_f)
    assert means_n == pytest.approx(means_f)
