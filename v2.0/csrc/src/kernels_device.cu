//===----------------------------------------------------------------------===//
// ReCAD v2.0 - CUDA implementations of the public kernel API.
//
// Mirrors kernels_host.cpp function-for-function; both paths must agree to
// within 1e-5 relative on float inputs. Design notes:
//  - MaskedMoments uses one block per channel with a two-pass reduction
//    (pass 1: valid count + sample sum -> mean; pass 2: sum of squared
//    deviations from the mean, population variance).
//  - NormalizeZscore and GatherNearest are grid-stride elementwise kernels.
//  - PatchAggregate uses atomics only for the per-patch valid counts; the
//    per-patch sums are accumulated by one block per patch with a plain
//    block reduction (no atomics on the sums; O(n * nPatches) work - see
//    the report if this ever becomes a hot path).
//
// The file compiles to just the header include unless RECAD_HAS_CUDA is
// defined, so the module also builds CPU-only.
//===----------------------------------------------------------------------===//

#include "recad/kernels.hpp"

#ifdef RECAD_HAS_CUDA

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

namespace {

constexpr std::size_t kBlockSize = 256;
// Grids are one block per channel / patch; gridDim.x must stay under 2^31-1.
constexpr std::size_t kMaxGridDim = 0x7FFFFFFFu;

__device__ __forceinline__ float recadNaN() {
  // IEEE quiet NaN (payload 0x7FFFFFFF); NaN comparisons are false, per the
  // header's IEEE-754 note.
  return __int_as_float(0x7FFFFFFFu);
}

// Standard power-of-two block reduction (launches always use kBlockSize=256).
template <typename T>
__device__ T blockReduceSum(T value, T* shared) {
  const unsigned tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  return shared[0];
}

// Number of kBlockSize-blocks that cover `work`, capped at the grid limit.
unsigned launchGrid(std::size_t work) {
  const std::size_t blocks = (work + kBlockSize - 1) / kBlockSize;
  return (unsigned)(blocks < kMaxGridDim ? blocks : kMaxGridDim);
}

// ---- device kernels --------------------------------------------------------

// One block per channel. Pass 1 reduces (count, sum) to the shared mean;
// pass 2 reduces the squared deviations for the population variance.
__global__ void maskedMomentsKernel(const float* data, std::size_t samples,
                                    std::size_t channels, std::size_t cells,
                                    float eps, float* outMean, float* outStd) {
  __shared__ double sCount[kBlockSize];
  __shared__ double sSum[kBlockSize];
  __shared__ double sMean;
  __shared__ double sDeviation[kBlockSize];

  const std::size_t c = blockIdx.x;
  const std::size_t perChannel = samples * cells;

  // Pass 1: valid count and sample sum.
  double count = 0.0;
  double sum = 0.0;
  for (std::size_t i = threadIdx.x; i < perChannel; i += kBlockSize) {
    const std::size_t s = i / cells;
    const std::size_t k = i - s * cells;
    const float v = data[(s * channels + c) * cells + k];
    if (!isnan(v)) {
      count += 1.0;
      sum += (double)v;
    }
  }
  count = blockReduceSum(count, sCount);
  sum = blockReduceSum(sum, sSum);
  if (threadIdx.x == 0)
    sMean = count == 0.0 ? recadNaN() : sum / count;
  __syncthreads();

  // Pass 2: sum of squared deviations from the shared mean.
  double deviation = 0.0;
  for (std::size_t i = threadIdx.x; i < perChannel; i += kBlockSize) {
    const std::size_t s = i / cells;
    const std::size_t k = i - s * cells;
    const float v = data[(s * channels + c) * cells + k];
    if (!isnan(v)) {
      const double d = (double)v - sMean;
      deviation += d * d;
    }
  }
  deviation = blockReduceSum(deviation, sDeviation);

  if (threadIdx.x == 0) {
    if (count == 0.0) {
      outMean[c] = recadNaN();
      outStd[c] = recadNaN();
    } else {
      const double variance = deviation / count; // population variance
      const float stdDev = (float)sqrt(variance);
      outMean[c] = (float)sMean;
      // Floor the std at eps so a downstream z-score never divides by zero.
      outStd[c] = stdDev < eps ? eps : stdDev;
    }
  }
}

// Elementwise z-score; NaN entries pass through. In-place is safe because each
// element is read into a register before its output slot is written.
__global__ void normalizeZscoreKernel(const float* data, std::size_t total,
                                      std::size_t channels, std::size_t cells,
                                      const float* mean, const float* std,
                                      float* out) {
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += stride) {
    const float v = data[i];
    if (isnan(v)) {
      out[i] = v;
      continue;
    }
    const std::size_t c = (i / cells) % channels;
    out[i] = (v - mean[c]) / std[c];
  }
}

// Grid-stride gather through the index map with out-of-range -> fillValue.
__global__ void gatherNearestKernel(const float* src, std::size_t srcSize,
                                    const std::int64_t* srcIdx,
                                    std::size_t dstSize, float fillValue,
                                    float* out) {
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  for (std::size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < dstSize;
       j += stride) {
    const std::int64_t idx = srcIdx[j];
    out[j] = (idx >= 0 && (std::uint64_t)idx < srcSize)
                 ? src[(std::size_t)idx]
                 : fillValue;
  }
}

// Counts valid entries per patch (the only atomics in the module) and flags
// out-of-range patch ids so the host wrapper can reject them.
__global__ void patchCountKernel(const float* values,
                                 const std::int32_t* patchIds, std::size_t n,
                                 std::size_t nPatches, std::int64_t* counts,
                                 int* invalid) {
  __shared__ int sInvalid[kBlockSize];
  int localInvalid = 0;
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const std::int32_t p = patchIds[i];
    if (p < 0 || (std::uint64_t)p >= nPatches) {
      localInvalid = 1;
      continue;
    }
    if (!isnan(values[i]))
      atomicAdd(reinterpret_cast<unsigned long long*>(&counts[p]), 1ULL);
  }
  const int blockInvalid = blockReduceSum(localInvalid, sInvalid);
  if (threadIdx.x == 0 && blockInvalid != 0)
    atomicExch(invalid, 1);
}

// One block per patch: block-reduces the sums of the patch's valid values
// (no atomics), then divides by the count produced by patchCountKernel.
__global__ void patchSumKernel(const float* values,
                               const std::int32_t* patchIds, std::size_t n,
                               const std::int64_t* counts, float* means) {
  __shared__ double sSum[kBlockSize];
  const std::size_t p = blockIdx.x;
  const std::int32_t patch = (std::int32_t)p;
  double sum = 0.0;
  for (std::size_t i = threadIdx.x; i < n; i += kBlockSize) {
    if (!isnan(values[i]) && patchIds[i] == patch)
      sum += (double)values[i];
  }
  sum = blockReduceSum(sum, sSum);
  if (threadIdx.x == 0) {
    const std::int64_t count = counts[p];
    means[p] = count > 0 ? (float)(sum / (double)count) : recadNaN();
  }
}

} // namespace

// ---- host wrappers ---------------------------------------------------------

// Copies inputs to device memory, runs the kernel, copies results back.
// Status: 0 success, 1 invalid arguments, 2 invalid patch ids, 3 CUDA error.

extern "C" {

int MaskedMoments(const float* data, std::size_t samples,
                  std::size_t channels, std::size_t cells,
                  float eps, float* outMean, float* outStd) {
  if (data == nullptr || outMean == nullptr || outStd == nullptr ||
      samples == 0 || channels == 0 || cells == 0 || channels > kMaxGridDim)
    return 1;
  if (samples > std::numeric_limits<std::size_t>::max() / channels / cells)
    return 1;
  const std::size_t total = samples * channels * cells;
  if (total > std::numeric_limits<std::size_t>::max() / sizeof(float))
    return 1;

  float* dData = nullptr;
  float* dMean = nullptr;
  float* dStd = nullptr;
  int result = 0;

  cudaError_t err = cudaMalloc(&dData, total * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dMean, channels * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dStd, channels * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMemcpy(dData, data, total * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    maskedMomentsKernel<<<(unsigned)channels, kBlockSize>>>(
        dData, samples, channels, cells, eps, dMean, dStd);
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(outMean, dMean, channels * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(outStd, dStd, channels * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    result = 3;

  if (dData != nullptr)
    cudaFree(dData);
  if (dMean != nullptr)
    cudaFree(dMean);
  if (dStd != nullptr)
    cudaFree(dStd);
  return result;
}

int NormalizeZscore(const float* data, std::size_t samples,
                    std::size_t channels, std::size_t cells,
                    const float* mean, const float* std, float* out) {
  if (data == nullptr || mean == nullptr || std == nullptr || out == nullptr ||
      samples == 0 || channels == 0 || cells == 0)
    return 1;
  if (samples > std::numeric_limits<std::size_t>::max() / channels / cells)
    return 1;
  const std::size_t total = samples * channels * cells;
  if (total > std::numeric_limits<std::size_t>::max() / sizeof(float))
    return 1;

  float* dData = nullptr;
  float* dMean = nullptr;
  float* dStd = nullptr;
  float* dOut = nullptr;
  int result = 0;

  cudaError_t err = cudaMalloc(&dData, total * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dMean, channels * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dStd, channels * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dOut, total * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMemcpy(dData, data, total * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(dMean, mean, channels * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(dStd, std, channels * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    normalizeZscoreKernel<<<launchGrid(total), kBlockSize>>>(
        dData, total, channels, cells, dMean, dStd, dOut);
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(out, dOut, total * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    result = 3;

  if (dData != nullptr)
    cudaFree(dData);
  if (dMean != nullptr)
    cudaFree(dMean);
  if (dStd != nullptr)
    cudaFree(dStd);
  if (dOut != nullptr)
    cudaFree(dOut);
  return result;
}

int PatchAggregate(const float* values, const std::int32_t* patchIds,
                   std::size_t n, std::size_t nPatches,
                   std::int64_t* outCounts, float* outMeans) {
  if (values == nullptr || patchIds == nullptr || outCounts == nullptr ||
      outMeans == nullptr || nPatches == 0 || nPatches > kMaxGridDim)
    return 1;

  std::int32_t* dPatchIds = nullptr;
  float* dValues = nullptr;
  std::int64_t* dCounts = nullptr;
  float* dMeans = nullptr;
  int* dInvalid = nullptr;
  int result = 0;

  cudaError_t err = cudaMalloc(&dValues, n * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dPatchIds, n * sizeof(std::int32_t));
  if (err == cudaSuccess)
    err = cudaMalloc(&dCounts, nPatches * sizeof(std::int64_t));
  if (err == cudaSuccess)
    err = cudaMalloc(&dMeans, nPatches * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dInvalid, sizeof(int));
  if (err == cudaSuccess)
    err = cudaMemcpy(dValues, values, n * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(dPatchIds, patchIds, n * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemset(dCounts, 0, nPatches * sizeof(std::int64_t));
  if (err == cudaSuccess)
    err = cudaMemset(dInvalid, 0, sizeof(int));
  if (err == cudaSuccess) {
    patchCountKernel<<<launchGrid(n), kBlockSize>>>(dValues, dPatchIds, n,
                                                    nPatches, dCounts,
                                                    dInvalid);
    patchSumKernel<<<(unsigned)nPatches, kBlockSize>>>(dValues, dPatchIds, n,
                                                       dCounts, dMeans);
    err = cudaDeviceSynchronize();
  }
  int invalid = 0;
  if (err == cudaSuccess)
    err = cudaMemcpy(&invalid, dInvalid, sizeof(int), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(outCounts, dCounts, nPatches * sizeof(std::int64_t),
                     cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(outMeans, dMeans, nPatches * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    result = 3;
  else if (invalid != 0)
    result = 2;

  if (dValues != nullptr)
    cudaFree(dValues);
  if (dPatchIds != nullptr)
    cudaFree(dPatchIds);
  if (dCounts != nullptr)
    cudaFree(dCounts);
  if (dMeans != nullptr)
    cudaFree(dMeans);
  if (dInvalid != nullptr)
    cudaFree(dInvalid);
  return result;
}

int GatherNearest(const float* src, std::size_t srcSize,
                  const std::int64_t* srcIdx, std::size_t dstSize,
                  float fillValue, float* out) {
  if (src == nullptr || srcIdx == nullptr || out == nullptr || dstSize == 0)
    return 1;
  if (dstSize > std::numeric_limits<std::size_t>::max() / sizeof(float))
    return 1;

  float* dSrc = nullptr;
  std::int64_t* dIdx = nullptr;
  float* dOut = nullptr;
  int result = 0;

  cudaError_t err = cudaMalloc(&dSrc, srcSize * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMalloc(&dIdx, dstSize * sizeof(std::int64_t));
  if (err == cudaSuccess)
    err = cudaMalloc(&dOut, dstSize * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMemcpy(dSrc, src, srcSize * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(dIdx, srcIdx, dstSize * sizeof(std::int64_t),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    gatherNearestKernel<<<launchGrid(dstSize), kBlockSize>>>(
        dSrc, srcSize, dIdx, dstSize, fillValue, dOut);
    err = cudaDeviceSynchronize();
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(out, dOut, dstSize * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    result = 3;

  if (dSrc != nullptr)
    cudaFree(dSrc);
  if (dIdx != nullptr)
    cudaFree(dIdx);
  if (dOut != nullptr)
    cudaFree(dOut);
  return result;
}

} // extern "C"

#endif // RECAD_HAS_CUDA