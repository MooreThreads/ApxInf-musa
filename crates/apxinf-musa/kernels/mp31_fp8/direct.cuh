// MP31 register-fragment FP8 leaves for the fixed PI0.5 action shapes.
#pragma once
#include <mute/arch/mma_mp31.hpp>
#include "../custom/rope.cuh"

namespace apxinf_mp31 {
using DirectAtom = mute::MP31_16x16x32_F32E4M3E4M3F32<
    mute::TCE::Major::K, mute::TCE::Major::K>;

// Input is [K,N], matching the immutable native FP8 weight before its TME
// transpose. Each 16x32 atom owns 512 bytes, one int4 per warp lane.
__global__ void pack_direct_weight(const uint8_t *input, uint8_t *packed,
                                   int n, int k) {
#if __MUSA_ARCH__ == 310
  const int nt = blockIdx.x / (k / 32), kc = blockIdx.x % (k / 32);
  for (int d = threadIdx.x; d < 512; d += blockDim.x) {
    const int lane = d / 16, v = d % 16;
    const int row = lane / 4 + 8 * (v / 8);
    const int col = 4 * (lane % 4) + v % 4 + 16 * ((v / 4) % 2);
    packed[(nt * (k / 32) + kc) * 512 + d] =
        input[(kc * 32 + col) * n + nt * 16 + row];
  }
#endif
}

template <int WARPS = 8>
__global__ __launch_bounds__(WARPS * 32, 1) void direct_qkv_rope(
    const uint8_t *a, const uint8_t *packed, const __mt_bfloat16 *bias,
    __mt_bfloat16 *q, __mt_bfloat16 *k, __mt_bfloat16 *v,
    const float *frequency, int position_offset, int kv_output_offset, float alpha) {
#if __MUSA_ARCH__ == 310
  constexpr int K = 1024, CHUNKS = K / 32, PER = (CHUNKS + WARPS - 1) / WARPS;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int head = blockIdx.x / 8, pair_base = (blockIdx.x % 8) * 16;
  const int nt = head * 16 + blockIdx.x % 8;
  const int first = warp * PER, last = min(first + PER, CHUNKS);
  const int row = lane / 4, col = (lane % 4) * 4;
  float accum0[8] = {}, accum1[8] = {};
  for (int kc = first; kc < last; ++kc) {
    const uint8_t *r0 = a + row * K + kc * 32 + col;
    int4 av = make_int4(*(const uint32_t *)r0, *(const uint32_t *)(r0 + 16), 0, 0);
    if (row < 2) {
      const uint8_t *r1 = a + (row + 8) * K + kc * 32 + col;
      av.z = *(const uint32_t *)r1;
      av.w = *(const uint32_t *)(r1 + 16);
    }
    const int4 b0 = __ldg(((const int4 *)(packed + (nt * CHUNKS + kc) * 512)) + lane);
    const int4 b1 = __ldg(((const int4 *)(packed + ((nt + 8) * CHUNKS + kc) * 512)) + lane);
    DirectAtom::fma((int *)accum0, (const int *)&av, (const int *)&b0, (int *)accum0);
    DirectAtom::fma((int *)accum1, (const int *)&av, (const int *)&b1, (int *)accum1);
  }
  __shared__ float partial[WARPS][512];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int x = (lane / 8 + (j / 2) * 4) * 16 + lane % 8 + (j % 2) * 8;
    partial[warp][x] = accum0[j];
    partial[warp][256 + x] = accum1[j];
  }
  __syncthreads();
  for (int x = threadIdx.x; x < 160; x += blockDim.x) {
    float value0 = 0.f, value1 = 0.f;
#pragma unroll
    for (int s = 0; s < WARPS; ++s) {
      value0 += partial[s][x];
      value1 += partial[s][256 + x];
    }
    const int token = x / 16, pair = pair_base + x % 16;
    float first_value = round_bf16(value0 * alpha);
    float second_value = round_bf16(value1 * alpha);
    if (bias != nullptr) {
      first_value = round_bf16(first_value + __bfloat162float(bias[head * 256 + pair]));
      second_value = round_bf16(second_value + __bfloat162float(bias[head * 256 + pair + 128]));
    }
    if (head == 9) {
      auto *dest = v + (kv_output_offset + token) * 256;
      dest[pair] = __float2bfloat16(first_value);
      dest[pair + 128] = __float2bfloat16(second_value);
    } else {
      float sine, cosine;
      __sincosf(float(position_offset + token) * frequency[pair], &sine, &cosine);
      auto *dest = head < 8 ? q + (token * 8 + head) * 256
                            : k + (kv_output_offset + token) * 256;
      rope_pair_bf16(first_value, second_value, round_bf16(cosine), round_bf16(sine),
                    dest, 128, pair);
    }
  }
#endif
}
} // namespace apxinf_mp31
