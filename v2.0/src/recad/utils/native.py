"""Bridge to the native ReCAD kernel library (``csrc``) with NumPy fallback.

The native library (built from ``v2.0/csrc`` via CMake) implements four
performance-critical kernels (masked moments, z-score normalisation, patch
aggregation, nearest-neighbour gather) in CUDA with an LLVM-style C++ host
API (``csrc/include/recad/kernels.hpp``). This module loads that DLL through
ctypes and mirrors each kernel with a pure-NumPy reference implementation.

If the DLL is not built/loadable, ``native_available()`` returns False and all
calls transparently fall back to the NumPy implementations: the pipeline is
therefore fully functional without the native build, which is what CI smoke
tests rely on. Equivalence of the native and NumPy paths is verified in
``tests/test_native.py``.
"""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_size_t, c_void_p
from pathlib import Path

import numpy as np

_LIB_CANDIDATES = (
    "recad_cuda.dll",  # on PATH
    "recad_cuda",  # POSIX soname
    "librecad_cuda.so",
    "librecad_cuda.dylib",
)


def _repo_root() -> Path:
    # recad/utils/native.py -> recad/utils -> recad -> src -> repo
    return Path(__file__).resolve().parents[3]


def _discover_library():
    """Try to locate and load the native library; return the CDLL or None."""
    build_dirs = (
        _repo_root() / "csrc" / "build_cuda",  # CUDA-enabled Ninja build
        _repo_root() / "csrc" / "build" / "Release",  # VS build (or CUDA)
        _repo_root() / "csrc" / "build",
        _repo_root() / "csrc" / "build" / "Debug",
        _repo_root() / "csrc" / "out",
        _repo_root() / "csrc" / "out" / "Release",
    )
    for d in build_dirs:
        if not d.is_dir():
            continue
        dll = d / "recad_cuda.dll"
        if dll.is_file():
            try:
                return ctypes.CDLL(str(dll))
            except OSError:
                continue
    for name in _LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class _NativeKernels:
    """ctypes wrapper around the native library (or None → NumPy fallback)."""

    def __init__(self) -> None:
        self._lib = _discover_library()
        self._available = self._lib is not None
        if self._lib is not None:
            self._bind()

    # ------------------------------------------------------------------
    # ctypes plumbing
    # ------------------------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib
        lib.MaskedMoments.restype = ctypes.c_int
        lib.MaskedMoments.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            c_size_t,
            c_float,
            c_void_p,
            c_void_p,
        ]
        lib.NormalizeZscore.restype = ctypes.c_int
        lib.NormalizeZscore.argtypes = [
            c_void_p,
            c_size_t,
            c_size_t,
            c_size_t,
            c_void_p,
            c_void_p,
            c_void_p,
        ]
        lib.PatchAggregate.restype = ctypes.c_int
        lib.PatchAggregate.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            c_size_t,
            c_void_p,
            c_void_p,
        ]
        lib.GatherNearest.restype = ctypes.c_int
        lib.GatherNearest.argtypes = [
            c_void_p,
            c_size_t,
            c_void_p,
            c_size_t,
            c_float,
            c_void_p,
        ]

    @staticmethod
    def _ptr(arr: np.ndarray) -> c_void_p:
        arr = np.ascontiguousarray(arr)
        if arr.size == 0:
            return c_void_p(0)
        return c_void_p(arr.ctypes.data)

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"native kernel {what} failed with code {rc}")

    # ------------------------------------------------------------------
    # Public kernel API (native or NumPy fallback)
    # ------------------------------------------------------------------
    def masked_moments(self, data: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel masked mean and population std of ``data``.

        Args:
            data: float32 array of shape ``[samples, channels, cells]``.
            eps: guard added to the denominator (unused for moments).

        Returns:
            (mean, std): float32 arrays of shape ``[channels]``.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("data must be [samples, channels, cells]")
        samples, channels, cells = data.shape
        mean = np.empty(channels, dtype=np.float32)
        std = np.empty(channels, dtype=np.float32)
        if self._available and data.size:
            rc = self._lib.MaskedMoments(
                self._ptr(data),
                samples,
                channels,
                cells,
                c_float(eps),
                self._ptr(mean),
                self._ptr(std),
            )
            self._check(rc, "MaskedMoments")
        else:
            self._masked_moments_fallback(data, mean, std, eps)
        return mean, std

    @staticmethod
    def _masked_moments_fallback(
        data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float
    ) -> None:
        _, channels, _ = data.shape
        for c in range(channels):
            flat = data[:, c, :].ravel()
            valid = flat[~np.isnan(flat)]
            if valid.size == 0:
                mean[c] = np.nan
                std[c] = np.nan
            else:
                mean[c] = np.mean(valid)
                # eps floors the reported std (native-kernel parity): std =
                # max(true_std, eps), so the z-score never divides by zero.
                std[c] = max(float(np.std(valid)), eps)

    def normalize_zscore(
        self, data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """Z-score ``data`` per channel using precomputed moments."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        mean = np.ascontiguousarray(mean, dtype=np.float32)
        std = np.ascontiguousarray(std, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("data must be [samples, channels, cells]")
        samples, channels, cells = data.shape
        out = np.empty_like(data)
        if self._available and data.size:
            rc = self._lib.NormalizeZscore(
                self._ptr(data),
                samples,
                channels,
                cells,
                self._ptr(mean),
                self._ptr(std),
                self._ptr(out),
            )
            self._check(rc, "NormalizeZscore")
        else:
            out = data.copy()
            for c in range(channels):
                # std is already floor-guarded (>= eps) by masked_moments.
                out[:, c, :] = (data[:, c, :] - mean[c]) / std[c]
        return out

    def patch_aggregate(
        self, values: np.ndarray, patch_ids: np.ndarray, n_patches: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-patch count and mean of valid (non-NaN) values.

        Args:
            values: float32 array of shape ``[n]``.
            patch_ids: int32 array of shape ``[n]``, entries in [0, n_patches).
            n_patches: number of patches.

        Returns:
            (counts int64[n_patches], means float32[n_patches]); a patch with
            no valid entry has count 0 and a NaN mean.
        """
        values = np.ascontiguousarray(values, dtype=np.float32).ravel()
        patch_ids = np.ascontiguousarray(patch_ids, dtype=np.int32).ravel()
        if values.shape != patch_ids.shape:
            raise ValueError("values and patch_ids must have the same length")
        if patch_ids.size and (patch_ids.min() < 0 or patch_ids.max() >= n_patches):
            raise ValueError("patch_ids contains an out-of-range index")
        counts = np.zeros(n_patches, dtype=np.int64)
        means = np.full(n_patches, np.nan, dtype=np.float32)
        if self._available and values.size:
            rc = self._lib.PatchAggregate(
                self._ptr(values),
                self._ptr(patch_ids),
                values.size,
                n_patches,
                self._ptr(counts),
                self._ptr(means),
            )
            self._check(rc, "PatchAggregate")
        else:
            self._patch_aggregate_fallback(values, patch_ids, n_patches, counts, means)
        return counts, means

    @staticmethod
    def _patch_aggregate_fallback(
        values: np.ndarray,
        patch_ids: np.ndarray,
        n_patches: int,
        counts: np.ndarray,
        means: np.ndarray,
    ) -> None:
        for p in range(n_patches):
            sel = values[patch_ids == p]
            valid = sel[~np.isnan(sel)]
            counts[p] = valid.size
            means[p] = np.mean(valid) if valid.size else np.nan

    def gather_nearest(
        self, src: np.ndarray, src_idx: np.ndarray, fill_value: float = np.nan
    ) -> np.ndarray:
        """Gather ``src`` by index map; out-of-range indices become fill."""
        src = np.ascontiguousarray(src, dtype=np.float32).ravel()
        src_idx = np.ascontiguousarray(src_idx, dtype=np.int64).ravel()
        out = np.empty(src_idx.size, dtype=np.float32)
        if self._available and src_idx.size:
            rc = self._lib.GatherNearest(
                self._ptr(src),
                src.size,
                self._ptr(src_idx),
                src_idx.size,
                c_float(fill_value),
                self._ptr(out),
            )
            self._check(rc, "GatherNearest")
        else:
            self._gather_nearest_fallback(src, src_idx, fill_value, out)
        return out

    @staticmethod
    def _gather_nearest_fallback(
        src: np.ndarray, src_idx: np.ndarray, fill_value: float, out: np.ndarray
    ) -> None:
        valid = (src_idx >= 0) & (src_idx < src.size)
        out[:] = fill_value
        out[valid] = src[src_idx[valid]]


recad_native = _NativeKernels()


def native_available() -> bool:
    """True when the native (CUDA/C++) library has been loaded."""
    return recad_native._available
