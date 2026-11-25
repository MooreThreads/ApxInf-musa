#pragma once
#include "../custom/math.cuh"
#include "geglu.cuh"

namespace apxinf_mp31 {
// The projection tile geometry. The device schedule and the adapter's dynamic
// shared-memory request are both derived from these, so they cannot drift.
inline constexpr int kProjectionBn = 64;
inline constexpr int kProjectionBk = 256;
constexpr int projection_smem_bytes(int tiles) {
  return tiles * (16 + kProjectionBn) * kProjectionBk;
}

template <class Tile, int BN>
__device__ void store_projection_partial(const float *acc, float *partial, int m, int n) {
  auto coords = typename Tile::Mma{}.get_thread_slice(threadIdx.x).partition_C(
      mute::make_identity_tensor(mute::Shape<mute::Int<16>, mute::Int<BN>>{}));
#pragma unroll
  for (int j = 0; j < 16 * BN / 128; ++j) {
    int r = mute::get<0>(coords(j)), c = blockIdx.x * BN + mute::get<1>(coords(j));
    if (r < m && c < n)
      partial[(blockIdx.y * m + r) * n + c] = acc[j];
  }
}

// Each K partition fits one or two tiles. Issue all descriptor loads up front
// so the SQMMA squad can consume them without a separate producer squad.
template <int Tiles>
__global__ __launch_bounds__(128) void e4m3_projection_partial(
    __attribute__((grid_constant)) const MUtensorDescriptor ad,
    __attribute__((grid_constant)) const MUtensorDescriptor bd, float *partial, int m, int n) {
#if __MUSA_ARCH__ == 310
  constexpr int BN = kProjectionBn, BK = kProjectionBk;
  using Tile = Fp8Tile<16, BN, BK>;
  extern __shared__ __align__(4096) uint8_t storage[];
  auto *weights = (E4m3 *)storage;
  auto *activations = weights + Tiles * BN * BK;
  __musa_async_bar_record(Tiles);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < Tiles; ++t)
      __musa_async_init_arrival(t + 1, 1, 0);
  }
  __syncthreads_lm();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < Tiles; ++t) {
      const int k_offset = (blockIdx.y * Tiles + t) * BK;
      __musa_async_add_trans(t + 1, (16 + BN) * BK);
      load<mute::TME::CacheHint::CACHE_PERSIST, mute::TME::CacheHint::CACHE_PERSIST>(
          ad, t + 1, activations + t * 16 * BK, k_offset, 0, BK, 16);
      load<mute::TME::CacheHint::CACHE_ONCE, mute::TME::CacheHint::CACHE_NONE>(
          bd, t + 1, weights + t * BN * BK, k_offset, blockIdx.x * BN, BK, BN);
      __musa_async_arrive(t + 1);
    }
  }
  float acc[16 * BN / 128] = {};
#pragma unroll
  for (int t = 0; t < Tiles; ++t) {
    __musa_async_wait(t + 1, 0);
    Tile::run(activations + t * 16 * BK, weights + t * BN * BK, acc);
  }
  store_projection_partial<Tile, BN>(acc, partial, m, n);
#endif
}

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
  if (threadIdx.x < 128)
    store_projection_partial<Tile, BN>(acc, partial, m, n);
#endif
}
} // namespace apxinf_mp31
