//===----------------------------------------------------------------------===//
// ReCAD v2.0 - dependency-free kernel test harness.
//
// Plain asserts behind a tiny CHECK macro and a main() that returns nonzero
// on failure; no external test framework is fetched. The public ABI is
// declared here (rather than by including recad/kernels.hpp) so the test
// links the DLL through its import library without inheriting the header's
// unconditional __declspec(dllexport) on Windows.
//
// When the library was compiled with CUDA (RECAD_HAS_CUDA), the DLL exports
// both the CUDA entry points (the plain names) and the CPU reference copies
// (<Name>Host); the seeded random comparisons below then assert that the two
// paths agree to within 1e-5 relative.
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

extern "C" {
int MaskedMoments(const float* data, std::size_t samples, std::size_t channels,
                  std::size_t cells, float eps, float* outMean, float* outStd);
int NormalizeZscore(const float* data, std::size_t samples,
                    std::size_t channels, std::size_t cells, const float* mean,
                    const float* std, float* out);
int PatchAggregate(const float* values, const std::int32_t* patchIds,
                   std::size_t n, std::size_t nPatches, std::int64_t* outCounts,
                   float* outMeans);
int GatherNearest(const float* src, std::size_t srcSize,
                  const std::int64_t* srcIdx, std::size_t dstSize,
                  float fillValue, float* out);
#ifdef RECAD_HAS_CUDA
// CPU reference copies, exported alongside the CUDA entry points.
int MaskedMomentsHost(const float* data, std::size_t samples,
                      std::size_t channels, std::size_t cells, float eps,
                      float* outMean, float* outStd);
int NormalizeZscoreHost(const float* data, std::size_t samples,
                        std::size_t channels, std::size_t cells,
                        const float* mean, const float* std, float* out);
int PatchAggregateHost(const float* values, const std::int32_t* patchIds,
                       std::size_t n, std::size_t nPatches,
                       std::int64_t* outCounts, float* outMeans);
int GatherNearestHost(const float* src, std::size_t srcSize,
                      const std::int64_t* srcIdx, std::size_t dstSize,
                      float fillValue, float* out);
#endif
}

namespace {

const float kNan = std::numeric_limits<float>::quiet_NaN();

int gFailures = 0;

void check(bool ok, const char* file, int line, const char* expr) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
    ++gFailures;
  }
}

#define CHECK(expr) check(static_cast<bool>(expr), __FILE__, __LINE__, #expr)

bool isNaN(float v) { return std::isnan(v) != 0; }

// Relative near-equality with an absolute floor for tiny/zero expectations;
// NaN expectations require NaN results (IEEE NaN semantics).
bool nearRel(float got, float expected, float tolRel) {
  if (isNaN(expected))
    return isNaN(got);
  if (isNaN(got))
    return false;
  const float scale = std::fabs(expected) > 1.0f ? std::fabs(expected) : 1.0f;
  return std::fabs(got - expected) <= tolRel * scale;
}

// Fixed-seed LCG (Numerical Recipes constants); deterministic, no <random>.
std::uint32_t gLcg = 0x9E3779B9u;
std::uint32_t lcgNext() {
  gLcg = gLcg * 1664525u + 1013904223u;
  return gLcg;
}
float lcgFloat(float lo, float hi) {
  return lo + (hi - lo) * ((float)lcgNext() * (1.0f / 4294967296.0f));
}

// --- hand-computed property tests (run on the exported entry points) ---------

void testMaskedMoments() {
  // 2 samples x 1 channel x 3 cells with one NaN. Valid values {1, 2, 4, 5,
  // 6}: mean = 18/5 = 3.6, E[v^2] = 82/5 = 16.4, var = 16.4 - 3.6^2 = 3.44,
  // std = sqrt(3.44).
  const float data[6] = {1.0f, 2.0f, kNan, 4.0f, 5.0f, 6.0f};
  float mean = 0.0f, std = 0.0f;
  CHECK(MaskedMoments(data, 2, 1, 3, 1e-6f, &mean, &std) == 0);
  CHECK(nearRel(mean, 3.6f, 1e-6f));
  CHECK(nearRel(std, std::sqrt(3.44f), 1e-6f));

  // Invalid arguments (null pointers) are rejected.
  CHECK(MaskedMoments(nullptr, 2, 1, 3, 1e-6f, &mean, &std) != 0);
  CHECK(MaskedMoments(data, 2, 1, 3, 1e-6f, nullptr, &std) != 0);

  // All-NaN channel -> NaN mean and std.
  const float allNaN[3] = {kNan, kNan, kNan};
  CHECK(MaskedMoments(allNaN, 1, 1, 3, 1e-6f, &mean, &std) == 0);
  CHECK(isNaN(mean));
  CHECK(isNaN(std));

  // eps guard: a constant channel has zero variance, so std is floored at eps
  // (and at exactly 0 when eps == 0).
  const float constant[4] = {7.0f, 7.0f, 7.0f, 7.0f};
  CHECK(MaskedMoments(constant, 2, 1, 2, 1e-6f, &mean, &std) == 0);
  CHECK(nearRel(mean, 7.0f, 1e-6f));
  CHECK(nearRel(std, 1e-6f, 1e-6f));
  CHECK(MaskedMoments(constant, 2, 1, 2, 0.0f, &mean, &std) == 0);
  CHECK(nearRel(std, 0.0f, 1e-6f));

  // Channel isolation: 1 sample x 2 channels x 2 cells; the second channel is
  // entirely masked.
  const float twoCh[4] = {1.0f, 3.0f, kNan, kNan};
  float mean2[2] = {0.0f, 0.0f}, std2[2] = {0.0f, 0.0f};
  CHECK(MaskedMoments(twoCh, 1, 2, 2, 1e-6f, mean2, std2) == 0);
  CHECK(nearRel(mean2[0], 2.0f, 1e-6f));
  CHECK(nearRel(std2[0], 1.0f, 1e-6f));
  CHECK(isNaN(mean2[1]));
  CHECK(isNaN(std2[1]));
}

void testNormalizeZscore() {
  // z = (v - mean) / std with hand-computed moments.
  const float data[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  const float mean[1] = {3.0f};
  const float std[1] = {2.0f};
  const float expected[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  float out[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  CHECK(NormalizeZscore(data, 1, 1, 5, mean, std, out) == 0);
  for (int i = 0; i < 5; ++i)
    CHECK(nearRel(out[i], expected[i], 1e-6f));

  // NaN entries pass through unchanged.
  const float dataN[3] = {1.0f, kNan, 3.0f};
  float outN[3] = {0.0f, 0.0f, 0.0f};
  CHECK(NormalizeZscore(dataN, 1, 1, 3, mean, std, outN) == 0);
  CHECK(nearRel(outN[0], -1.0f, 1e-6f));
  CHECK(isNaN(outN[1]));
  CHECK(nearRel(outN[2], 0.0f, 1e-6f));

  // In-place: out aliases data.
  float copy[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  CHECK(NormalizeZscore(copy, 1, 1, 5, mean, std, copy) == 0);
  for (int i = 0; i < 5; ++i)
    CHECK(nearRel(copy[i], expected[i], 1e-6f));

  // Per-channel moments: 1 sample x 2 channels x 2 cells. Layout puts
  // channel c at ((s * channels) + c) * cells + k.
  const float dataM[4] = {1.0f, 3.0f, 2.0f, 4.0f};
  const float meanM[2] = {1.0f, 3.0f};
  const float stdM[2] = {1.0f, 1.0f};
  float outM[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  CHECK(NormalizeZscore(dataM, 1, 2, 2, meanM, stdM, outM) == 0);
  CHECK(nearRel(outM[0], 0.0f, 1e-6f));  // (1 - 1) / 1
  CHECK(nearRel(outM[1], 2.0f, 1e-6f));  // (3 - 1) / 1
  CHECK(nearRel(outM[2], -1.0f, 1e-6f)); // (2 - 3) / 1
  CHECK(nearRel(outM[3], 1.0f, 1e-6f));  // (4 - 3) / 1
}

void testPatchAggregate() {
  // Repeats across patches, with one NaN value skipped.
  const float values[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, kNan, 8.0f};
  const std::int32_t ids[8] = {0, 1, 0, 1, 0, 1, 1, 0};
  std::int64_t counts[2] = {0, 0};
  float means[2] = {0.0f, 0.0f};
  CHECK(PatchAggregate(values, ids, 8, 2, counts, means) == 0);
  CHECK(counts[0] == 4); // {1, 3, 5, 8}
  CHECK(counts[1] == 3); // {2, 4, 6}
  CHECK(nearRel(means[0], 4.25f, 1e-6f)); // 17 / 4
  CHECK(nearRel(means[1], 4.0f, 1e-6f));  // 12 / 3

  // All-NaN patch -> count 0, NaN mean.
  const float nanVals[2] = {kNan, kNan};
  const std::int32_t nanIds[2] = {0, 0};
  CHECK(PatchAggregate(nanVals, nanIds, 2, 2, counts, means) == 0);
  CHECK(counts[0] == 0);
  CHECK(counts[1] == 0);
  CHECK(isNaN(means[0]));
  CHECK(isNaN(means[1]));

  // Out-of-range patch ids must be rejected with a nonzero status.
  const float v[3] = {1.0f, 2.0f, 3.0f};
  const std::int32_t oob[3] = {0, 5, 1};
  const std::int32_t neg[3] = {0, -1, 1};
  CHECK(PatchAggregate(v, oob, 3, 2, counts, means) != 0);
  CHECK(PatchAggregate(v, neg, 3, 2, counts, means) != 0);
}

void testGatherNearest() {
  // Index map with repeats and out-of-range lookups -> fillValue.
  const float src[5] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  const std::int64_t idx[8] = {0, 4, 2, 1, 4, -1, 5, 7};
  const float expected[8] = {10.0f, 50.0f, 30.0f, 20.0f, 50.0f, -1.0f, -1.0f,
                             -1.0f};
  float out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  CHECK(GatherNearest(src, 5, idx, 8, -1.0f, out) == 0);
  for (int j = 0; j < 8; ++j)
    CHECK(out[j] == expected[j]); // bit-exact elementwise gather

  // NaN fill values propagate.
  const std::int64_t idxN[2] = {0, 99};
  float outN[2] = {0.0f, 0.0f};
  CHECK(GatherNearest(src, 5, idxN, 2, kNan, outN) == 0);
  CHECK(outN[0] == 10.0f);
  CHECK(isNaN(outN[1]));
}

// --- seeded random CPU-vs-CUDA comparisons ----------------------------------

#ifdef RECAD_HAS_CUDA
void testCudaMatchesCpu() {
  // MaskedMoments: 4 samples x 3 channels x 5 cells; channel 2 fully masked.
  const std::size_t samples = 4, channels = 3, cells = 5;
  const float eps = 1e-6f;
  std::vector<float> data(samples * channels * cells);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = (i % 7 == 0) ? kNan : lcgFloat(-1.0f, 1.0f);
    if ((i / cells) % channels == 2)
      data[i] = kNan;
  }

  std::vector<float> hMean(channels), hStd(channels);
  std::vector<float> dMean(channels), dStd(channels);
  CHECK(MaskedMomentsHost(data.data(), samples, channels, cells, eps,
                          hMean.data(), hStd.data()) == 0);
  const int deviceRc = MaskedMoments(data.data(), samples, channels, cells, eps,
                                     dMean.data(), dStd.data());
  if (deviceRc != 0) {
    std::fprintf(stderr,
                 "kernels: CUDA device path unavailable (status %d); "
                 "skipping device comparisons\n",
                 deviceRc);
    return;
  }
  for (std::size_t c = 0; c < channels; ++c) {
    CHECK(nearRel(dMean[c], hMean[c], 1e-5f));
    CHECK(nearRel(dStd[c], hStd[c], 1e-5f));
  }

  // NormalizeZscore with the host-computed moments.
  std::vector<float> hZ(data.size()), dZ(data.size());
  CHECK(NormalizeZscoreHost(data.data(), samples, channels, cells, hMean.data(),
                            hStd.data(), hZ.data()) == 0);
  CHECK(NormalizeZscore(data.data(), samples, channels, cells, hMean.data(),
                        hStd.data(), dZ.data()) == 0);
  for (std::size_t i = 0; i < data.size(); ++i)
    CHECK(nearRel(dZ[i], hZ[i], 1e-5f));

  // PatchAggregate: 1000 values across 17 patches, ~11% NaNs.
  const std::size_t n = 1000, nPatches = 17;
  std::vector<float> pv(n);
  std::vector<std::int32_t> pid(n);
  for (std::size_t i = 0; i < n; ++i) {
    pv[i] = (i % 9 == 0) ? kNan : lcgFloat(-1.0f, 1.0f);
    pid[i] = (std::int32_t)(lcgNext() % (std::uint32_t)nPatches);
  }
  std::vector<std::int64_t> hCnt(nPatches), dCnt(nPatches);
  std::vector<float> hM(nPatches), dM(nPatches);
  CHECK(PatchAggregateHost(pv.data(), pid.data(), n, nPatches, hCnt.data(),
                           hM.data()) == 0);
  CHECK(PatchAggregate(pv.data(), pid.data(), n, nPatches, dCnt.data(),
                       dM.data()) == 0);
  for (std::size_t p = 0; p < nPatches; ++p) {
    CHECK(hCnt[p] == dCnt[p]);
    CHECK(nearRel(dM[p], hM[p], 1e-5f));
  }

  // GatherNearest: 50 sources, 200 lookups incl. out-of-range and repeats.
  const std::size_t srcSize = 50, dstSize = 200;
  const float fill = -123.0f;
  std::vector<float> src(srcSize), hOut(dstSize), dOut(dstSize);
  std::vector<std::int64_t> idxMap(dstSize);
  for (std::size_t i = 0; i < srcSize; ++i)
    src[i] = lcgFloat(-1.0f, 1.0f);
  for (std::size_t j = 0; j < dstSize; ++j)
    idxMap[j] = (std::int64_t)(lcgNext() % 80u) - 15; // range [-15, 64]
  CHECK(GatherNearestHost(src.data(), srcSize, idxMap.data(), dstSize, fill,
                          hOut.data()) == 0);
  CHECK(GatherNearest(src.data(), srcSize, idxMap.data(), dstSize, fill,
                      dOut.data()) == 0);
  for (std::size_t j = 0; j < dstSize; ++j)
    CHECK(dOut[j] == hOut[j]);
}
#endif // RECAD_HAS_CUDA

} // namespace

int main() {
  testMaskedMoments();
  testNormalizeZscore();
  testPatchAggregate();
  testGatherNearest();
#ifdef RECAD_HAS_CUDA
  testCudaMatchesCpu();
#endif
  if (gFailures == 0) {
    std::printf("kernels: all tests passed\n");
    return 0;
  }
  std::fprintf(stderr, "kernels: %d CHECK(s) failed\n", gFailures);
  return 1;
}