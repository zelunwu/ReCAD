//===----------------------------------------------------------------------===//
// ReCAD v2.0 - CPU reference implementations (ground truth for tests).
//
// These mirror the CUDA kernels in kernels_device.cu function-for-function
// and are intentionally simple: every loop is sequential and every
// accumulator is double precision, so rounding is minimal and each formula
// can be audited by eye against the doc comments in recad/kernels.hpp.
//
// When CUDA is enabled (RECAD_HAS_CUDA) the four entry points below are
// renamed to <Name>Host so the DLL can export the CUDA versions under the
// public names declared in the header; the reference copies stay available
// to tests as MaskedMomentsHost et al. In CPU-only builds the public names
// are used directly.
//===----------------------------------------------------------------------===//

#ifdef RECAD_HAS_CUDA
#define MaskedMoments MaskedMomentsHost
#define NormalizeZscore NormalizeZscoreHost
#define PatchAggregate PatchAggregateHost
#define GatherNearest GatherNearestHost
#endif

#include "recad/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

// IEEE quiet NaN. NaN comparisons are false, per the header's IEEE-754 note.
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

} // namespace

extern "C" {

// Computes per-channel masked mean and (population) standard deviation over
// valid (non-NaN) samples. A channel with no valid sample yields NaN mean and
// std. For valid channels the reported std is floored at eps, so a downstream
// z-score whose denominator is the std can never divide by zero (a constant
// channel reports std == eps instead of 0).
RECAD_EXPORT int MaskedMoments(const float* data, std::size_t samples,
                               std::size_t channels, std::size_t cells,
                               float eps, float* outMean, float* outStd) {
  if (data == nullptr || outMean == nullptr || outStd == nullptr ||
      samples == 0 || channels == 0 || cells == 0)
    return 1;

  const std::size_t perChannel = samples * cells;
  for (std::size_t c = 0; c < channels; ++c) {
    std::size_t count = 0;
    double sum = 0.0;
    double sumSq = 0.0;
    for (std::size_t s = 0; s < samples; ++s) {
      const float* row = data + (s * channels + c) * cells;
      for (std::size_t k = 0; k < cells; ++k) {
        const float v = row[k];
        if (!std::isnan(v)) {
          ++count;
          sum += (double)v;
          sumSq += (double)v * (double)v;
        }
      }
    }
    if (count == 0) {
      outMean[c] = kNaN;
      outStd[c] = kNaN;
    } else {
      // Population moments: mean = E[v], variance = E[v^2] - E[v]^2.
      const double mean = sum / (double)count;
      double variance = sumSq / (double)count - mean * mean;
      if (variance < 0.0)
        variance = 0.0; // Round-off can make the closed form slightly negative.
      const float stdDev = (float)std::sqrt(variance);
      outMean[c] = (float)mean;
      outStd[c] = stdDev < eps ? eps : stdDev;
    }
  }
  return 0;
}

// Z-scores data with precomputed per-channel moments. NaN entries pass
// through unchanged; out may alias data (each element is read before write).
RECAD_EXPORT int NormalizeZscore(const float* data, std::size_t samples,
                                 std::size_t channels, std::size_t cells,
                                 const float* mean, const float* std,
                                 float* out) {
  if (data == nullptr || mean == nullptr || std == nullptr || out == nullptr ||
      samples == 0 || channels == 0 || cells == 0)
    return 1;

  const std::size_t total = samples * channels * cells;
  for (std::size_t i = 0; i < total; ++i) {
    const float v = data[i];
    if (std::isnan(v)) {
      out[i] = v;
      continue;
    }
    const std::size_t c = (i / cells) % channels;
    out[i] = (v - mean[c]) / std[c];
  }
  return 0;
}

// Aggregates values per patch id: count of valid entries and their mean.
// patchIds must lie in [0, nPatches); any out-of-range id is rejected with a
// nonzero status before any counting happens. A patch with no valid entries
// gets count 0 and a NaN mean.
RECAD_EXPORT int PatchAggregate(const float* values,
                                const std::int32_t* patchIds, std::size_t n,
                                std::size_t nPatches, std::int64_t* outCounts,
                                float* outMeans) {
  if (values == nullptr || patchIds == nullptr || outCounts == nullptr ||
      outMeans == nullptr || nPatches == 0)
    return 1;

  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t p = patchIds[i];
    if (p < 0 || (std::size_t)p >= nPatches)
      return 2;
  }

  // Counts are written (not accumulated into) the output buffer, matching
  // the device path which zero-initializes it.
  std::memset(outCounts, 0, nPatches * sizeof(std::int64_t));

  std::vector<double> sums(nPatches, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t p = patchIds[i];
    if (!std::isnan(values[i])) {
      ++outCounts[p];
      sums[p] += (double)values[i];
    }
  }
  for (std::size_t p = 0; p < nPatches; ++p) {
    const std::int64_t count = outCounts[p];
    outMeans[p] = count > 0 ? (float)(sums[p] / (double)count) : kNaN;
  }
  return 0;
}

// Gathers nearest-neighbour values through an index map:
// out[j] = srcIdx[j] in [0, srcSize) ? src[srcIdx[j]] : fillValue.
// Repeat indices are handled naturally; NaN fill values propagate.
RECAD_EXPORT int GatherNearest(const float* src, std::size_t srcSize,
                               const std::int64_t* srcIdx, std::size_t dstSize,
                               float fillValue, float* out) {
  if (src == nullptr || srcIdx == nullptr || out == nullptr || dstSize == 0)
    return 1;

  for (std::size_t j = 0; j < dstSize; ++j) {
    const std::int64_t idx = srcIdx[j];
    out[j] = (idx >= 0 && (std::uint64_t)idx < srcSize)
                 ? src[(std::size_t)idx]
                 : fillValue;
  }
  return 0;
}

} // extern "C"