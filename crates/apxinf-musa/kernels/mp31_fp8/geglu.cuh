#pragma once
// BM16 producer/consumer schedule adapted from muDNN aaa2127d1,
// 3rdparty/tilelang_kern_gen/gen_kernels/benchmark_gemv_cases.py:
// small_m_bm16_sqmma_tme_pc_transb. Uses the in-tree MATE MUTLASS primitives.
// Copyright @2020-2024 Moore Threads Technology Co., Ltd. All rights reserved.
#include <musa.h>
#include <musa_fp8.h>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
namespace apxinf_mp31 {
using E4m3 = mutlass::float_e4m3_t;
using Bf16 = mutlass::bfloat16_t;
template <int BM, int BN, int BK> struct Fp8Tile {
  using Shape = mute::Shape<mute::Int<BM>, mute::Int<BN>, mute::Int<BK>>;
  using Op =
      decltype(mute::MP31::SQMMA::ss_op_selector<E4m3, E4m3, float, Shape, mute::TCE::Major::K,
                                                 mute::TCE::Major::K>());
  using Mma = decltype(mute::make_tiled_mma(Op{}));
  using AtomA = decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<mute::TCE::Major::K,
                                                                               E4m3, Op, Shape>());
  using AtomB = decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<mute::TCE::Major::K,
                                                                               E4m3, Op, Shape>());
  using LayoutA =
      decltype(mute::tile_to_shape(AtomA{}, mute::Shape<mute::Int<BM>, mute::Int<BK>, mute::_1>{}));
  using LayoutB =
      decltype(mute::tile_to_shape(AtomB{}, mute::Shape<mute::Int<BN>, mute::Int<BK>, mute::_1>{}));
  static __device__ __forceinline__ void run(E4m3 *a, E4m3 *b, float *acc) {
    auto sa = mute::make_tensor(mute::make_smem_ptr(a), LayoutA{});
    auto sb = mute::make_tensor(mute::make_smem_ptr(b), LayoutB{});
    Mma mma;
    auto thr = mma.get_thread_slice(threadIdx.x);
    auto fa = thr.make_fragment_A(thr.partition_A(sa));
    auto fb = thr.make_fragment_B(thr.partition_B(sb));
    auto fc = mute::make_tensor(
        mute::make_rmem_ptr(acc),
        mute::partition_shape_C(mma, mute::Shape<mute::Int<BM>, mute::Int<BN>>{}));
#pragma unroll
    for (int kb = 0; kb < mute::size<2>(fa); ++kb)
      mute::gemm(mma, fa(mute::_, mute::_, kb, 0), fb(mute::_, mute::_, kb, 0), fc);
    __musa_sqmma_wait();
  }
};
template <mute::TME::CacheHint INNER, mute::TME::CacheHint OUTER>
__device__ __forceinline__ void load(const MUtensorDescriptor &desc, int bar, void *sm, int x,
                                     int y, int cols, int rows) {
  mute::MP31_TME_LOAD_2D<mute::TME::SmemSwizzleGranularity::B16, mute::TME::SmemSwizzleStride::B256,
                         mute::TME::SmemSwizzleLine::B256, INNER, OUTER,
                         mute::PrefetchSize::NONE>::copy(&desc, bar, sm, x, y, cols, rows);
}
__device__ __forceinline__ uint8_t geglu_value(float gate, float up, float inverse_scale) {
  float g = float(Bf16(gate)), u = float(Bf16(up));
  float v = float(
      Bf16((0.5f * g * (1.f + __musa_tanh_f(0.7978845608028654f * (g + 0.044715f * g * g * g)))) * u));
  return __musa_cvt_float_to_fp8(fminf(448.f, fmaxf(-448.f, v * inverse_scale)), __MT_SATFINITE,
                                 __MT_E4M3);
}
template <int BK = 256, int ST = 1>
__global__ __launch_bounds__(256, 1) void fp8_geglu(
    __attribute__((grid_constant)) const MUtensorDescriptor ad,
    __attribute__((grid_constant)) const MUtensorDescriptor bd, uint8_t *out,
    int m, int n, int k, float alpha, float inverse_scale) {
#if __MUSA_ARCH__ == 310
  using Tile = Fp8Tile<16, 64, BK>;
  extern __shared__ __align__(4096) uint8_t sm[];
  float acc[8];
  __musa_async_bar_record(2 * ST);
  // Initialize from the full first warp, matching the MUTLASS convention.
  if (threadIdx.x < 32)
    for (int j = 1; j <= 2 * ST; ++j)
      __musa_async_init_arrival(j, 4, 0);
  __syncthreads();
  // Keep the producer and consumer loops in independent warp-uniform scopes.
  // Stage lifetime and tail drains stay explicit across all iterations.
  if (threadIdx.x >= 128) {
    for (int i = 0; i < k / BK; ++i) {
      int st = i % ST, ph = (i / ST) & 1;
      __musa_async_wait(st + ST + 1, ph ^ 1);
      if (threadIdx.x == 128) {
        __musa_async_add_trans(st + 1, 80 * BK);
        load<mute::TME::CacheHint::CACHE_PERSIST,
             mute::TME::CacheHint::CACHE_PERSIST>(
            ad, st + 1, sm + ST * 64 * BK + st * 16 * BK, i * BK, 0, BK, 16);
        load<mute::TME::CacheHint::CACHE_ONCE,
             mute::TME::CacheHint::CACHE_NONE>(bd, st + 1, sm + st * 64 * BK,
                                               i * BK, blockIdx.x * 64, BK, 64);
      }
      __musa_async_arrive(st + 1);
    }
    // Drain every used stage before the CTA can release its barrier allocation.
    for (int st = 0; st < ST; ++st) {
      int last = k / BK - 1 - ((k / BK - 1 - st) % ST);
      if (last >= 0)
        __musa_async_wait(st + ST + 1, (last / ST) & 1);
    }
  }
  if (threadIdx.x < 128) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      acc[j] = 0;
    for (int i = 0; i < k / BK; ++i) {
      int st = i % ST, ph = (i / ST) & 1;
      __musa_async_wait(st + 1, ph);
      Tile::run((E4m3 *)(sm + ST * 64 * BK + st * 16 * BK),
                (E4m3 *)(sm + st * 64 * BK), acc);
      __musa_async_arrive(st + ST + 1);
    }
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    auto coords = typename Tile::Mma{}
                      .get_thread_slice(threadIdx.x)
                      .partition_C(mute::make_identity_tensor(
                          mute::Shape<mute::Int<16>, mute::Int<64>>{}));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = mute::get<0>(coords(j)), c = mute::get<1>(coords(j));
      if (r < m)
        out[r * (n / 2) + blockIdx.x * 32 + c] =
            geglu_value(acc[j] * alpha, acc[j ^ 4] * alpha, inverse_scale);
    }
  }
  __syncthreads();
#endif
}

__global__ void transpose_e4m3(const uint8_t *in, uint8_t *out, int k, int n) {
  __shared__ uint8_t tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
#pragma unroll
  for (int j = 0; j < 32; j += 8)
    if (x < n && y + j < k)
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
  __syncthreads_lm();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
#pragma unroll
  for (int j = 0; j < 32; j += 8)
    if (x < k && y + j < n)
      out[(y + j) * k + x] = tile[threadIdx.x][threadIdx.y + j];
}
} // namespace apxinf_mp31
