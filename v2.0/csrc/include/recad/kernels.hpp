//===----------------------------------------------------------------------===//
// ReCAD v2.0 - native GPU/CPU kernel library (public API)
//
// The functions below are the ABI surface of the ReCAD native library
// (librecad_cuda). They are written in the LLVM coding style (2-space
// indent, camelCase functions, PascalCase types, 80-column lines) and are
// exported with C linkage so they can be consumed from C, C++, and Python
// (ctypes) alike. Every function is implemented twice: a CPU reference
// implementation (kernels_host.cpp) and a CUDA implementation
// (kernels_device.cu). The CPU reference is the source of truth for tests;
// the CUDA path must reproduce it bit-for-bit on valid inputs.
//
// Layout convention (row-major, contiguous):
//   data[s][c][k]  ->  index = ((s * channels) + c) * cells + k
// All NaN handling follows IEEE-754 semantics (NaN comparisons are false).
//===----------------------------------------------------------------------===//

#ifndef RECAD_KERNELS_HPP
#define RECAD_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define RECAD_EXPORT __declspec(dllexport)
#else
#define RECAD_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Computes per-channel masked mean and (population) standard deviation over
// valid (non-NaN) samples. A channel with no valid sample yields NaN mean/std.
// eps is a *floor* applied to the reported std (std = max(true_std, eps)), so
// a constant channel reports std == eps instead of 0 and the z-score in
// NormalizeZscore never divides by zero when its moments come from this
// function. Returns 0 on success, nonzero on invalid arguments (null
// pointers or out-of-range patch ids).
RECAD_EXPORT int MaskedMoments(const float* data, std::size_t samples,
                               std::size_t channels, std::size_t cells,
                               float eps, float* outMean, float* outStd);

// Z-scores data with precomputed per-channel moments (in place or to out).
// NaN entries remain NaN. out may alias data.
RECAD_EXPORT int NormalizeZscore(const float* data, std::size_t samples,
                                 std::size_t channels, std::size_t cells,
                                 const float* mean, const float* std,
                                 float* out);

// Aggregates values per patch id: count of valid (non-NaN) entries and their
// mean. patchIds entries must lie in [0, nPatches). A patch with zero valid
// entries gets count 0 and a NaN mean. Both output arrays are fully
// overwritten, so they need not be pre-zeroed by the caller.
RECAD_EXPORT int PatchAggregate(const float* values, const std::int32_t* patchIds,
                                std::size_t n, std::size_t nPatches,
                                std::int64_t* outCounts, float* outMeans);

// Gathers nearest-neighbour values via an index map: out[j] = (srcIdx[j] in
// [0, srcSize)) ? src[srcIdx[j]] : fillValue. Handles repeat indices.
RECAD_EXPORT int GatherNearest(const float* src, std::size_t srcSize,
                               const std::int64_t* srcIdx, std::size_t dstSize,
                               float fillValue, float* out);

} // extern "C"

#endif // RECAD_KERNELS_HPP