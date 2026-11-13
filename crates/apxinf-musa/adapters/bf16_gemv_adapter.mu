// Exact M=1 BF16 GEMV kernels. The launch parameters come from the
// standalone exact-shape sweep; unmatched shapes remain on the provider path.
#include <musa_bf16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdint>

#include "adapter_common.h"

namespace {
using bf16 = __mt_bfloat16;

struct alignas(16) Bf16x8 {
  bf16 value[8];
};

struct alignas(16) Float8 {
  float value[8];
};

__device__ __forceinline__ Float8 add_float8(Float8 lhs,
                                              const Float8& rhs) {
#pragma unroll
  for (int i = 0; i < 8; ++i) lhs.value[i] += rhs.value[i];
  return lhs;
}

__device__ __forceinline__ Float8 shfl32_float8(const Float8& source,
                                                 int offset) {
  Float8 result{};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    union Bits {
      float f;
      int32_t i;
    } value{}, shuffled{};
    value.f = source.value[i];
    shuffled.i = __shfl_down_sync(0xffffffffu, value.i, offset, 32);
    result.value[i] = shuffled.f;
  }
  return result;
}

template <int TileN, int LoadCount>
__global__ void gemv_kn(const bf16* __restrict__ activation,
                                const bf16* __restrict__ weight,
                                bf16* __restrict__ output, int n, int k) {
  constexpr int kThreads = 512;
  static_assert(TileN == 64 || TileN == 128, "supported output tile");
  static_assert(LoadCount == 4 || LoadCount == 8, "supported unroll");
  constexpr int kColumnsPerLane = 8;
  constexpr int kThreadsPerRow = TileN / kColumnsPerLane;
  constexpr int kRowCount = kThreads / kThreadsPerRow;
  constexpr int kGroupCount = kThreads / 32;
  constexpr int kRowsPerGroup = 32 / kThreadsPerRow;

  // The routed steady-state shapes have K=132 or K=1024. Keep the capacity
  // explicit so extending the route requires changing this contract too.
  constexpr int kActivationCapacity = 1024;
  __shared__ bf16 staged_activation[kActivationCapacity];
  __shared__ Float8 group_results[kGroupCount * kThreadsPerRow];

  const int tid = int(threadIdx.x);
  for (int kk = tid; kk < k; kk += kThreads) {
    staged_activation[kk] = activation[kk];
  }
  __syncthreads();

  const int row_index = tid / kThreadsPerRow;
  const int col = int(blockIdx.x) * TileN +
                  (tid % kThreadsPerRow) * kColumnsPerLane;
  Float8 result{};
  for (int base = row_index; base < k; base += kRowCount * LoadCount) {
#pragma unroll
    for (int load = 0; load < LoadCount; ++load) {
      const int kk = base + load * kRowCount;
      if (kk < k && col + kColumnsPerLane <= n) {
        const Bf16x8 weights = *reinterpret_cast<const Bf16x8*>(
            weight + size_t(kk) * n + col);
        const float value = __bfloat162float(staged_activation[kk]);
#pragma unroll
        for (int i = 0; i < kColumnsPerLane; ++i) {
          result.value[i] = fmaf(value, __bfloat162float(weights.value[i]),
                                 result.value[i]);
        }
      }
    }
  }

  const int lane = tid & 31;
  const int group = tid / 32;
#pragma unroll
  for (int offset = 16; offset >= kThreadsPerRow; offset /= 2) {
    const Float8 shuffled = shfl32_float8(result, offset);
    if (lane < offset) result = add_float8(result, shuffled);
  }
  if (lane < kThreadsPerRow) {
    group_results[group * kThreadsPerRow + lane] = result;
  }
  __syncthreads();

  if (group == kGroupCount - 1) {
    Float8 total{};
    const int column_lane = lane % kThreadsPerRow;
#pragma unroll
    for (int current_group = lane / kThreadsPerRow;
         current_group < kGroupCount; current_group += kRowsPerGroup) {
      total = add_float8(
          total, group_results[current_group * kThreadsPerRow + column_lane]);
    }
#pragma unroll
    for (int offset = 16; offset >= kThreadsPerRow; offset /= 2) {
      const Float8 shuffled = shfl32_float8(total, offset);
      if (lane < offset) total = add_float8(total, shuffled);
    }
    if (lane < kThreadsPerRow && col + kColumnsPerLane <= n) {
      Bf16x8 packed{};
#pragma unroll
      for (int i = 0; i < kColumnsPerLane; ++i) {
        packed.value[i] = __float2bfloat16_rn(total.value[i]);
      }
      *reinterpret_cast<Bf16x8*>(output + col) = packed;
    }
  }
}

template <int TileN, int LoadCount>
void launch(const bf16* activation, const bf16* weight, bf16* output, int n,
            int k, musaStream_t stream) {
  gemv_kn<TileN, LoadCount>
      <<<(n + TileN - 1) / TileN, 512, 0, stream>>>(activation, weight,
                                                    output, n, k);
}
}  // namespace

extern "C" musaError_t apxinf_musa_bf16_gemv(
    const void* activation, const void* weight, void* output, int n, int k,
    int tile_n, int load_count, void* stream) {
  if (!activation || !weight || !output || n <= 0 || n % 8 != 0 || k <= 0 || k > 1024 ||
      (tile_n != 64 && tile_n != 128) ||
      (load_count != 4 && load_count != 8)) {
    return musaErrorInvalidValue;
  }
  if (!apxinf_adapters::select_device(
          apxinf_adapters::stream_device(stream))) {
    return musaErrorInvalidDevice;
  }
  const auto* input = static_cast<const bf16*>(activation);
  const auto* matrix = static_cast<const bf16*>(weight);
  auto* result = static_cast<bf16*>(output);
  auto musa_stream = static_cast<musaStream_t>(stream);
  if (tile_n == 64 && load_count == 4) {
    launch<64, 4>(input, matrix, result, n, k, musa_stream);
  } else if (tile_n == 64) {
    launch<64, 8>(input, matrix, result, n, k, musa_stream);
  } else if (load_count == 4) {
    launch<128, 4>(input, matrix, result, n, k, musa_stream);
  } else {
    launch<128, 8>(input, matrix, result, n, k, musa_stream);
  }
  return musaGetLastError();
}
