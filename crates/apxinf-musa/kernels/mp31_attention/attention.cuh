// MP31 D256 BF16 QK/PV with fixed-order split-KV reduction.
// The checked-in source is the validated MP31 path; unsupported shapes keep
// the MATE fallback.
#pragma once
#include <musa_runtime.h>
#include "../custom/math.cuh"
#include <mute/arch/copy_mp31_tme.hpp>
// Keep one MUTLASS revision across the backend. TileLang uses the newer name.
namespace mute {
using MP31_BULK_COPY_G2S = MP31_BLK_COPY_G2S;
using MP31_BULK_COPY_S2G = MP31_BLK_COPY_S2G;
} // namespace mute
#include <tilelang/musa/common.h>
#include <tilelang/musa/gemm.h>
#include <tilelang/musa/copy.h>
#include <tilelang/musa/reduce.h>
#include <tilelang/musa/musa_fp8.h>
#include <musa.h>
namespace apxinf_mp31_attention {
using namespace tl;
template <int BN, int SP>
__global__ __launch_bounds__(256, 1) void attention_partials(
    __attribute__((grid_constant)) const MUtensorDescriptor qd,
    __attribute__((grid_constant)) const MUtensorDescriptor kd,
    __attribute__((grid_constant)) const MUtensorDescriptor vd, bfloat16_t *out,
    float *partial, float *stats, int t, int kv) {
#if __MUSA_ARCH__ == 310
  using QK = mute::tl_sqmma::GemmTensorOp<16, BN, 128, 4, 1, false, true, false,
                                          bfloat16_t, bfloat16_t, float>;
  using PV = mute::tl_sqmma::GemmTensorOp<16, 256, BN, 4, 1, false, false,
                                          false, bfloat16_t, bfloat16_t, float>;
  constexpr int ST = BN == 64 ? 2 : 1;
  extern __shared__ __align__(4096) bfloat16_t sm[];
  bfloat16_t *qs = sm, *ks = sm + 4096, *vs = ks + ST * BN * 256,
             *ps = vs + ST * BN * 256;
  auto pl =
      mute::as_position_independent_swizzle_layout(typename PV::SmemLayoutA{});
  auto vl =
      mute::as_position_independent_swizzle_layout(typename PV::SmemLayoutB{});
  int tid = threadIdx.x, head = blockIdx.x, split = blockIdx.y;
  const int tiles = (kv + BN - 1) / BN, per = (tiles + SP - 1) / SP,
            start = split * per, end = min(tiles, start + per);
  __musa_async_bar_record(2 + 3 * ST); // Include reserved CTA barrier 0.
  // Barrier initialization is a single elected thread operation on MP31.
  // Initializing from every lane of warp 0 races the hardware barrier state;
  // the generated muDNN TileLang kernels use the same election primitive.
  if (tl::tl_shuffle_elect<0>())
    for (int barrier = 1; barrier <= 1 + 3 * ST; ++barrier)
      // A single producer warp owns each DMA barrier; expect_tx and arrive
      // must stay in that warp. Four consumer warps own release barriers.
      __musa_async_init_arrival(barrier, barrier >= 2 + 2 * ST ? 4 : 1, 0);
  __syncthreads_lm();
  if (tid >= 128 && tid < 160) {
    if (tl::tl_shuffle_elect<128>()) {
      __musa_async_add_trans(1, 16 * 256 * 2);
      tl::tma_load<SmemSwizzleGranularity::B16>(qd, 1, qs, head * 256, 0, 128,
                                                16);
      tl::tma_load<SmemSwizzleGranularity::B16>(qd, 1, qs + 16 * 128,
                                                head * 256 + 128, 0, 128, 16);
    }
    __musa_async_arrive(1);
    for (int tile = start; tile < end; ++tile) {
      int i = tile - start, st = i % ST, ph = (i / ST) & 1;
      __musa_async_wait(st + 2 + 2 * ST, ph ^ 1);
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(st + 2, BN * 256 * 2);
        tl::tma_load<SmemSwizzleGranularity::B16>(
            kd, st + 2, ks + st * BN * 256, 0, tile * BN, 128, BN);
        tl::tma_load<SmemSwizzleGranularity::B16>(
            kd, st + 2, ks + st * BN * 256 + BN * 128, 128, tile * BN, 128, BN);
      }
      __musa_async_arrive(st + 2);
      if (tl::tl_shuffle_elect<128>()) {
        __musa_async_add_trans(st + 2 + ST, BN * 256 * 2);
#pragma unroll
        for (int d = 0; d < 256; d += 64)
          tl::tma_load<SmemSwizzleGranularity::B32>(
              vd, st + 2 + ST, vs + st * BN * 256 + vl(d, 0, 0), d, tile * BN,
              64, BN);
      }
      __musa_async_arrive(st + 2 + ST);
    }
    if (end > start)
      for (int st = 0; st < ST && st < end - start; ++st) {
        int last = (end - start) - 1 - ((end - start - 1 - st) % ST);
        if (last >= 0)
          __musa_async_wait(st + 2 + 2 * ST, (last / ST) & 1);
      }
  } else if (tid < 128) {
    auto sc = typename QK::TiledMma{}.get_thread_slice(tid).partition_C(
        mute::make_identity_tensor(
            mute::Shape<mute::Int<16>, mute::Int<BN>>{}));
    auto oc = typename PV::TiledMma{}.get_thread_slice(tid).partition_C(
        mute::make_identity_tensor(
            mute::Shape<mute::Int<16>, mute::Int<256>>{}));
    float acc[32] = {};
    float mx = -INFINITY, den = 0;
    __musa_async_wait(1, 0);
    for (int tile = start; tile < end; ++tile) {
      int i = tile - start, st = i % ST, ph = (i / ST) & 1;
      __musa_async_wait(st + 2, ph);
      float score[16 * BN / 128] = {};
      QK::body(qs, ks + st * BN * 256, score);
      QK::body(qs + 16 * 128, ks + st * BN * 256 + BN * 128, score);
      float m = -INFINITY;
#pragma unroll
      for (int j = 0; j < 16 * BN / 128; ++j) {
        int col = mute::get<1>(sc(j));
        score[j] = tile * BN + col < kv
                       ? score[j] * (0.0625f * 1.4426950408889634f)
                       : -INFINITY;
        m = fmaxf(m, score[j]);
      }
      m = tl::AllReduce<tl::MaxOp, 8, 1, 128>::run(m);
      float next = fmaxf(mx, m), rescale = __musa_exp2_f(mx - next), sum = 0;
#pragma unroll
      for (int j = 0; j < 32; ++j)
        acc[j] *= rescale;
#pragma unroll
      for (int j = 0; j < 16 * BN / 128; ++j) {
        float p = __musa_exp2_f(score[j] - next);
        sum += p;
        ps[pl(mute::get<0>(sc(j)), mute::get<1>(sc(j)), 0)] = bfloat16_t(p);
      }
      sum = tl::AllReduce<tl::SumOp, 8, 1, 128>::run(sum);
      den = den * rescale + sum;
      mx = next;
      __syncwarp();
      __musa_async_wait(st + 2 + ST, ph);
      PV::body(ps, vs + st * BN * 256, acc);
      __musa_async_arrive(st + 2 + 2 * ST);
    }
    float inverse_den = den > 0 ? 1.f / den : 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      int r = mute::get<0>(oc(j)), d = mute::get<1>(oc(j));
      if (r < t) {
        int x = (r * 8 + head) * 256 + d;
        partial[split * t * 8 * 256 + x] = den > 0 ? acc[j] * inverse_den : 0;
      }
    }
    if constexpr (SP > 1) {
      int r = tid / 8;
      if ((tid & 7) == 0 && r < t)
        stats[(split * t + r) * 8 + head] =
            den > 0 ? mx + __musa_log2_f(den) : -INFINITY;
    }
  }
  __syncthreads();

#endif
}
// `inverse_scale` is the next projection's input quantizer, the same one its
// BF16 producer applies to this row, so the E4M3 store here is byte-identical
// to merging to BF16 and letting that producer quantize from memory.
template <int SP, bool QUANT = false>
__global__ void merge_partials(const float *part, const float *stats, void *out,
                               int t, float inverse_scale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= t * 8 * 256)
    return;
  int row = x / 256;
  float weights[SP] = {};
  if ((threadIdx.x & 31) == 0) {
    float maximum = -INFINITY;
    for (int split = 0; split < SP; ++split)
      maximum = fmaxf(maximum, stats[split * t * 8 + row]);
    float sum = 0;
#pragma unroll
    for (int split = 0; split < SP; ++split) {
      float score = stats[split * t * 8 + row];
      weights[split] = isfinite(score) ? __musa_exp2_f(score - maximum) : 0;
      sum += weights[split];
    }
    float inv = 1.f / sum;
#pragma unroll
    for (int split = 0; split < SP; ++split)
      weights[split] *= inv;
  }
  float y = 0;
#pragma unroll
  for (int split = 0; split < SP; ++split) {
    float w = __shfl_sync(0xffffffff, weights[split], 0);
    if (w != 0)
      y += w * part[split * t * 8 * 256 + x];
  }
  bfloat16_t b = bfloat16_t(y);
  if constexpr (QUANT)
    ((uint8_t *)out)[x] = quantize_e4m3_value(float(b), inverse_scale);
  else
    ((bfloat16_t *)out)[x] = b;
}

} // namespace apxinf_mp31_attention
