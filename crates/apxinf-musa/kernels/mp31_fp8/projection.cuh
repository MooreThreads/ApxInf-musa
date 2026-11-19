#pragma once
#include "../custom/math.cuh"
#include "geglu.cuh"

namespace apxinf_mp31 {
template <int BN, int BK, int ST, int SP, bool QUANT>
__global__ __launch_bounds__(256, 1) void partial_gemm(__attribute__((grid_constant))
                                                       const MUtensorDescriptor ad,
                                                       __attribute__((grid_constant))
                                                       const MUtensorDescriptor bd,
                                                       const Bf16 *bf, float *partial, int m, int n,
                                                       int k, float inverse_scale) {
#if __MUSA_ARCH__ == 310
  using Tile = Fp8Tile<16, BN, BK>;
  extern __shared__ __align__(4096) uint8_t sm[];
  float acc[16 * BN / 128];
  __musa_async_bar_record(2 * ST);
  if (threadIdx.x == 0)
    for (int j = 1; j <= 2 * ST; ++j)
      __musa_async_init_arrival(j, 4, 0);
  __syncthreads_lm();
  if (threadIdx.x < 128) {
#pragma unroll
    for (int j = 0; j < 16 * BN / 128; ++j)
      acc[j] = 0;
  }
  const int chunks = k / (BK * SP), base = blockIdx.y * chunks;
  for (int i = 0; i < chunks; ++i) {
    int st = i % ST, ph = (i / ST) & 1;
    auto *as = sm + ST * BN * BK + st * 16 * BK;
    auto *bs = sm + st * BN * BK;
    if (threadIdx.x >= 128) {
      __musa_async_wait(st + ST + 1, ph ^ 1);
      if (threadIdx.x == 128) {
        __musa_async_add_trans(st + 1, (QUANT ? BN : BN + 16) * BK);
        if constexpr (!QUANT)
          load<mute::TME::CacheHint::CACHE_PERSIST, mute::TME::CacheHint::CACHE_PERSIST>(
              ad, st + 1, as, (base + i) * BK, 0, BK, 16);
        load<mute::TME::CacheHint::CACHE_ONCE, mute::TME::CacheHint::CACHE_NONE>(
            bd, st + 1, bs, (base + i) * BK, blockIdx.x * BN, BK, BN);
      }
      if constexpr (QUANT) {
        for (int x = (threadIdx.x - 128) * 8; x < 16 * BK; x += 128 * 8) {
          int row = x / BK, kk = x % BK;
          uint64_t packed = 0;
          if (row < m) {
            uint4 values = *(const uint4 *)(bf + row * k + (base + i) * BK + kk);
            const Bf16 *lanes = (const Bf16 *)&values;
#pragma unroll
            for (int l = 0; l < 8; ++l)
              packed |= uint64_t(quantize_e4m3_value(float(lanes[l]), inverse_scale)) << (8 * l);
          }
          auto target =
              mute::make_tensor(mute::make_smem_ptr((E4m3 *)as), typename Tile::LayoutA{});
          *(uint64_t *)(&target(row, kk, 0)) = packed;
        }
      }
      __musa_async_arrive(st + 1);
    } else {
      __musa_async_wait(st + 1, ph);
      Tile::run((E4m3 *)as, (E4m3 *)bs, acc);
      __musa_async_arrive(st + ST + 1);
    }
  }
  if (threadIdx.x >= 128) {
    int last = chunks - 1;
    __musa_async_wait(last % ST + ST + 1, (last / ST) & 1);
  }
  __syncthreads_lm();
  if (threadIdx.x < 128) {
    auto coords =
        typename Tile::Mma{}
            .get_thread_slice(threadIdx.x)
            .partition_C(mute::make_identity_tensor(mute::Shape<mute::Int<16>, mute::Int<BN>>{}));
#pragma unroll
    for (int j = 0; j < 16 * BN / 128; ++j) {
      int r = mute::get<0>(coords(j)), c = blockIdx.x * BN + mute::get<1>(coords(j));
      if (r < m && c < n)
        partial[(blockIdx.y * m + r) * n + c] = acc[j];
    }
  }
#endif
}
} // namespace apxinf_mp31
