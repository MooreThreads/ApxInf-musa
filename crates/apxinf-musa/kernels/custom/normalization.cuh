#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

#include "reduction.cuh"
#include "math.cuh"

// Normalization kernels, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/normalization.cuh` this port reaches.
//
// CUDA's file also carries the quantized FP8 outputs and a register-cached
// LayerNorm for widths that divide the block; those arrive with their operators
// rather than in advance, which is also what keeps this file free of the
// aggregate-header shape the CUDA crate's README forbids.

// RMSNorm: one block per row, block-wide sum of squares, then the weight.
//
// **Deliberately not bit-identical to the bridge.** The port started with CUDA's
// shape dispatch here: a single-warp `rms_vector_square_mean_bf16` for
// `rows >= 16 && cols > 128 && cols % 4 == 0`, a block reduction otherwise.
// That reproduced the reference's bits on `[522, 2048]`, at a price measured
// with the port's rms-variant sweep: the reduction was **89% of the
// kernel** (0.686 ms of 0.773), and the same reduction over all 256 threads
// instead of 32 ran it in 0.108 ms -- **6.4x**. On `[522, 2048]` that is
// 10.5 ms of a pass.
//
// The two branches summed in different orders, which is the only reason the
// dispatch existed; the reference is a `mean`, not a hand-rolled tree, so
// neither order is *the* reference order. Collapsing to one block reduction
// costs the bit-identity the ledger recorded for #1 and moves the statistic by
// well under a bf16 ulp -- the same trade `layer_norm_bf16_kernel` and
// `bias_residual_rms_norm_bf16_kernel` already make, and both remain in the
// gate. It is recorded in the port's latency-reduction notes rather than here.
__global__ void rms_norm_bf16_kernel(const __mt_bfloat16* input,
                                     const __mt_bfloat16* weight,
                                     __mt_bfloat16* output, int rows, int cols,
                                     float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const __mt_bfloat16* row_input =
      input + static_cast<int64_t>(row) * cols;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(row_input[col]);
    square_sum += value * value;
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(__bfloat162float(input[index]) *
                                     inverse_rms *
                                     __bfloat162float(weight[col]));
  }
}

// The load width a register-cached RMSNorm uses for a given lanes-per-thread.
// A `std::conditional_t` would pull in `<type_traits>`; this file stays free of
// host headers.
template <int VEC>
struct RmsPack;
template <>
struct RmsPack<4> {
  using Type = uint64_t;
};
template <>
struct RmsPack<8> {
  using Type = uint4;
};

// Vectorized, register-cached RMSNorm: one block per row, each thread loads its
// `VEC`-wide column group once, holds it in registers across the block
// reduction, and writes from those registers -- so the row is read once instead
// of twice. Requires `cols == blockDim.x * PER`, which is what lets the whole
// row live in `PER` registers per thread.
//
// This is the "register-cached LayerNorm for widths that divide the block" the
// header note above says CUDA carries and this port had not brought over. It
// reads the row once and writes it once, against the two reads and one write of
// `rms_norm_bf16_kernel`, and on `[522, 2048]` that is the whole of the gain.
//
// The per-thread summation order is grouped-by-thread rather than strided, so
// like the single block reduction above it is not bit-identical to the earlier
// kernel; it stays inside the same per-tensor gate for the same reason.
template <int VEC, int PER>
__global__ void rms_norm_bf16_vector_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* weight,
                                            __mt_bfloat16* output, int rows,
                                            int cols, float eps) {
  static_assert(PER % VEC == 0, "PER must be a multiple of VEC");
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  constexpr int kChunks = PER / VEC;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[PER];
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT raw = *reinterpret_cast<const LoadT*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      const float value = __bfloat162float(packed[lane]);
      values[chunk * VEC + lane] = value;
      square_sum += value * value;
    }
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT weight_raw =
        *reinterpret_cast<const LoadT*>(weight + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    LoadT out_raw;
    __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      packed_out[lane] = __float2bfloat16(values[chunk * VEC + lane] *
                                          inverse_rms *
                                          __bfloat162float(packed_weight[lane]));
    }
    *reinterpret_cast<LoadT*>(output + base + column) = out_raw;
  }
}
// Adaptive LayerNorm: modulation is [scale, shift], shared by all rows.
// Keep the reduction in FP32 and round only the final affine result, matching
// the recorded reference contract.
__global__ void adaptive_layer_norm_bf16_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* modulation,
    __mt_bfloat16* output, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += __bfloat162float(input[base + col]);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    output[index] = __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}


// Adaptive LayerNorm with both a BF16 and a static-E4M3 result.
//
// The arithmetic, reduction order and BF16 rounding are the BF16 sibling's
// exactly; the second store adds the quantizer's multiply/clamp/convert, so the
// E4M3 bytes match `adaptive_layer_norm_bf16` followed by
// `quantize_fixed_e4m3`. One launch replaces the pair.
__global__ void adaptive_layer_norm_bf16_e4m3_dual_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* modulation,
    __mt_bfloat16* normalized_bf16, uint8_t* quantized, int rows, int cols,
    float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    sum += __bfloat162float(input[base + col]);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __mt_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    normalized_bf16[index] = rounded;
    quantized[index] =
        quantize_e4m3_value(__bfloat162float(rounded), inverse_scale);
  }
}

// Register-cached sibling of `adaptive_layer_norm_bf16_e4m3_dual_kernel` for
// the GR00T DiT width: the row is loaded once into registers, mean and
// centered variance are reduced over those registers in the same two passes as
// the scalar kernel, and the affine plus dual store come from the same values.
// Requires `kThreads * 4 * kChunks == cols`.
template <int kThreads, int kChunks>
__global__ void adaptive_layer_norm_bf16_e4m3_dual_block_t_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* modulation,
    __mt_bfloat16* normalized_bf16, uint8_t* quantized, int cols, float eps,
    float inverse_scale) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0, "kThreads must be a warp multiple");
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float values[kChunks * 4];
  // The modulation is row-independent, so it is prefetched alongside the row:
  // both global round trips are in flight together instead of the modulation
  // arriving only after the reduction. The arithmetic is unchanged.
  uint2 scale_raw[kChunks];
  uint2 shift_raw[kChunks];
  float sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(input + base + column);
    scale_raw[chunk] = *reinterpret_cast<const uint2*>(modulation + column);
    shift_raw[chunk] =
        *reinterpret_cast<const uint2*>(modulation + cols + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * 4 + lane_in] = value;
      sum += value;
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float ssum[kWarps];
  __shared__ float ssq[kWarps];
  sum = warp_sum(sum);
  if (lane == 0) {
    ssum[warp] = sum;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total += ssum[w];
  }
  const float mean = total / cols;
  float variance_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kChunks * 4; ++i) {
    const float centered = values[i] - mean;
    variance_sum += centered * centered;
  }
  variance_sum = warp_sum(variance_sum);
  if (lane == 0) {
    ssq[warp] = variance_sum;
  }
  __syncthreads();
  float total_variance = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total_variance += ssq[w];
  }
  const float inverse_std = rsqrtf(total_variance / cols + eps);
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const __mt_bfloat16* packed_scale =
        reinterpret_cast<const __mt_bfloat16*>(&scale_raw[chunk]);
    const __mt_bfloat16* packed_shift =
        reinterpret_cast<const __mt_bfloat16*>(&shift_raw[chunk]);
    uint2 normalized_raw;
    __mt_bfloat16* packed_normalized =
        reinterpret_cast<__mt_bfloat16*>(&normalized_raw);
    uint32_t packed_out = 0;
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const float scale = __bfloat162float(packed_scale[lane_in]);
      const float shift = __bfloat162float(packed_shift[lane_in]);
      const __mt_bfloat16 rounded = __float2bfloat16(
          (values[chunk * 4 + lane_in] - mean) * inverse_std *
              (1.0f + scale) +
          shift);
      packed_normalized[lane_in] = rounded;
      packed_out |= static_cast<uint32_t>(
                        quantize_e4m3_value(__bfloat162float(rounded),
                                            inverse_scale))
                    << (8 * lane_in);
    }
    *reinterpret_cast<uint2*>(normalized_bf16 + base + column) =
        normalized_raw;
    *reinterpret_cast<uint32_t*>(quantized + base + column) = packed_out;
  }
}

// Static-E4M3 output for the same operator. The arithmetic up to and including
// the BF16 rounding is the sibling's; the store adds `inverse_scale` and the
// E4M3 conversion, which is what makes this one launch instead of `normalized`
// + `quantize_fixed_e4m3`.
template <int VEC, int PER>
__global__ void rms_norm_bf16_e4m3_vector_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight, uint8_t* output,
    int rows, int cols, float eps, float inverse_scale) {
  static_assert(PER % VEC == 0, "PER must be a multiple of VEC");
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  constexpr int kChunks = PER / VEC;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[PER];
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT raw = *reinterpret_cast<const LoadT*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      const float value = __bfloat162float(packed[lane]);
      values[chunk * VEC + lane] = value;
      square_sum += value * value;
    }
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column =
        (static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x) * VEC;
    const LoadT weight_raw = *reinterpret_cast<const LoadT*>(weight + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    uint64_t packed_out = 0;
#pragma unroll
    for (int lane = 0; lane < VEC; ++lane) {
      const __mt_bfloat16 rounded = __float2bfloat16(
          values[chunk * VEC + lane] * inverse_rms *
          __bfloat162float(packed_weight[lane]));
      packed_out |= static_cast<uint64_t>(
                        quantize_e4m3_value(__bfloat162float(rounded),
                                            inverse_scale))
                    << (8 * lane);
    }
    if constexpr (VEC == 4) {
      *reinterpret_cast<uint32_t*>(output + base + column) =
          static_cast<uint32_t>(packed_out);
    } else {
      *reinterpret_cast<uint64_t*>(output + base + column) = packed_out;
    }
  }
}

// Static-E4M3 output for the two-row form. The sibling above launches one block
// per row -- 522 blocks for `[522, 2048]` -- and its per-block prologue plus
// barrier made that 3.2 MB pass cost 9.01 us. Two rows per 512-thread block
// (261 launches, each half reducing into its own scratch behind one barrier)
// costs 7.64 us with a bit-identical output. Measured on an idle S5000.
__global__ void rms_norm_bf16_e4m3_block2_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight, uint8_t* output,
    int rows, int cols, float eps, float inverse_scale) {
  const int half = threadIdx.x >> 8;
  const int local = threadIdx.x & 255;
  const int row = blockIdx.x * 2 + half;
  const bool active = row < rows;
  const int64_t column = static_cast<int64_t>(local) * 8;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  float values[8];
  float square_sum = 0.0f;
  // The weight is row-independent and is prefetched with the row so both global
  // round trips overlap instead of the weight arriving only after the
  // reduction. The arithmetic is unchanged.
  uint4 weight_raw;
  if (active) {
    const uint4 raw = *reinterpret_cast<const uint4*>(input + index);
    weight_raw = *reinterpret_cast<const uint4*>(weight + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < 8; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[lane_in] = value;
      square_sum += value * value;
    }
  }
  const int lane = local & 31;
  const int warp = local >> 5;
  __shared__ float sscratch[2][8];
  square_sum = warp_sum(square_sum);
  if (lane == 0) sscratch[half][warp] = square_sum;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) total += sscratch[half][i];
  const float inverse_rms = rsqrtf(total / cols + eps);
  if (!active) return;
  const __mt_bfloat16* packed_weight =
      reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
  uint64_t packed_out = 0;
#pragma unroll
  for (int lane_in = 0; lane_in < 8; ++lane_in) {
    const __mt_bfloat16 rounded = __float2bfloat16(
        values[lane_in] * inverse_rms * __bfloat162float(packed_weight[lane_in]));
    packed_out |= static_cast<uint64_t>(
                      quantize_e4m3_value(__bfloat162float(rounded),
                                          inverse_scale))
                  << (8 * lane_in);
  }
  *reinterpret_cast<uint64_t*>(output + index) = packed_out;
}
//
// `style` is `[3 * cols]` and only `[0, 2 * cols)` is read -- the third segment
// belongs to the gated variant. That is the model layer's layout, not this
// kernel's choice.
__global__ void ada_rms_norm_bf16_kernel(const __mt_bfloat16* input,
                                         const __mt_bfloat16* style,
                                         __mt_bfloat16* output, int rows,
                                         int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float normalized = __bfloat162float(input[index]) * inverse_rms;
    output[index] = __float2bfloat16(
        normalized * (1.0f + __bfloat162float(style[col])) +
        __bfloat162float(style[cols + col]));
  }
}

// Vectorized, register-cached form of the adaLN RMSNorm above, for widths that
// divide the block exactly (`cols == blockDim.x * VEC`).
//
// This is `[10, 1024]` -- the action expert's own shape, one block per row, so
// ten blocks on an eight-SM device with no occupancy to hide latency behind.
// Each thread loads its `VEC` input columns and both `style` segments in one
// `RmsPack<VEC>::Type` transaction, holds the row in registers across the block
// reduction, and writes the output from those registers: the row is read once
// instead of twice, the same trade
// `ada_gate_residual_rms_norm_bf16_vector_kernel` makes for its sibling.
//
// The per-thread summation is grouped `VEC` columns per thread instead of the
// scalar kernel's `cols / blockDim.x` strided ones, so the accumulation order
// differs slightly; this operator is already gated rather than bit-identical to
// the reference (see the `# Numerics` note on `adaptive_rms_bf16`), and the
// reorder stays inside that gate.
template <int VEC>
__global__ void ada_rms_norm_bf16_vector_kernel(const __mt_bfloat16* input,
                                                const __mt_bfloat16* style,
                                                __mt_bfloat16* output, int rows,
                                                int cols, float eps) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = threadIdx.x * VEC;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  const LoadT input_raw = *reinterpret_cast<const LoadT*>(input + index);
  const __mt_bfloat16* packed_input =
      reinterpret_cast<const __mt_bfloat16*>(&input_raw);
  float values[VEC];
  float square_sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    values[lane] = __bfloat162float(packed_input[lane]);
    square_sum += values[lane] * values[lane];
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  // The norm's scale and shift are the first two `style` segments, as in the
  // scalar kernel.
  const LoadT scale_raw = *reinterpret_cast<const LoadT*>(style + column);
  const LoadT shift_raw =
      *reinterpret_cast<const LoadT*>(style + cols + column);
  const __mt_bfloat16* packed_scale =
      reinterpret_cast<const __mt_bfloat16*>(&scale_raw);
  const __mt_bfloat16* packed_shift =
      reinterpret_cast<const __mt_bfloat16*>(&shift_raw);
  LoadT out_raw;
  __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    packed_out[lane] = __float2bfloat16(
        values[lane] * inverse_rms *
            (1.0f + __bfloat162float(packed_scale[lane])) +
        __bfloat162float(packed_shift[lane]));
  }
  *reinterpret_cast<LoadT*>(output + index) = out_raw;
}

// Static-E4M3 output for the adaLN vector form. Same arithmetic and single
// BF16 rounding as the sibling; the store adds `inverse_scale` and the E4M3
// conversion, which is what makes this one launch instead of `normalized` +
// `quantize_fixed_e4m3`.
template <int VEC>
__global__ void ada_rms_norm_bf16_e4m3_vector_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* style, uint8_t* output,
    int rows, int cols, float eps, float inverse_scale) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = threadIdx.x * VEC;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  const LoadT input_raw = *reinterpret_cast<const LoadT*>(input + index);
  const __mt_bfloat16* packed_input =
      reinterpret_cast<const __mt_bfloat16*>(&input_raw);
  float values[VEC];
  float square_sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    values[lane] = __bfloat162float(packed_input[lane]);
    square_sum += values[lane] * values[lane];
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  const LoadT scale_raw = *reinterpret_cast<const LoadT*>(style + column);
  const LoadT shift_raw =
      *reinterpret_cast<const LoadT*>(style + cols + column);
  const __mt_bfloat16* packed_scale =
      reinterpret_cast<const __mt_bfloat16*>(&scale_raw);
  const __mt_bfloat16* packed_shift =
      reinterpret_cast<const __mt_bfloat16*>(&shift_raw);
  uint64_t packed_out = 0;
#pragma unroll
  for (int lane = 0; lane < VEC; ++lane) {
    const __mt_bfloat16 rounded = __float2bfloat16(
        values[lane] * inverse_rms *
            (1.0f + __bfloat162float(packed_scale[lane])) +
        __bfloat162float(packed_shift[lane]));
    packed_out |= static_cast<uint64_t>(
                      quantize_e4m3_value(__bfloat162float(rounded),
                                          inverse_scale))
                  << (8 * lane);
  }
  if constexpr (VEC == 4) {
    *reinterpret_cast<uint32_t*>(output + index) =
        static_cast<uint32_t>(packed_out);
  } else {
    *reinterpret_cast<uint64_t*>(output + index) = packed_out;
  }
}

// LayerNorm: one block per row, mean and variance over the row, then the
// affine transform.
//
// Three passes over the row and **no cached specialization**. CUDA's file has a
// `cached<2|4|8>` variant that keeps the row in registers and short-circuits
// the reloads, selected for `threads * per_thread == cols` with
// `cols ∈ {512, 1024, 2048}`; PI0.5's `norm.layer_bf16` is `cols = 1152`, which
// selects none of them. A port that skipped the branch because 1152 "does not
// fit anyway" would be right about this model and wrong about the operator's
// contract -- so the branch is not here at all, and adding it later means
// adding a variant whose accumulation order matches CUDA's.
//
// The `__syncthreads()` before the second reduction is CUDA's, and it is
// load-bearing: `block_sum_parallel_unsafe`'s trailing barrier publishes
// `scratch[0]` but does not protect the reads of it, so a fast thread could
// overwrite the mean while a slow one is still reading it.
__global__ void layer_norm_bf16_kernel(const __mt_bfloat16* input,
                                       const __mt_bfloat16* weight,
                                       const __mt_bfloat16* bias,
                                       __mt_bfloat16* output, int rows, int cols,
                                       float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__bfloat162float(input[index]) - mean) * inverse_std *
            __bfloat162float(weight[col]) +
        __bfloat162float(bias[col]);
    output[index] = __float2bfloat16(value);
  }
}

// Register-cached, barrier-free warp-per-row LayerNorm.
//
// A whole row belongs to one warp: each lane walks `NITER` column groups of
// `VEC` elements, holds the row in registers across the statistics, and writes
// the affine result from those registers, so the row is read once and written
// once. The mean and variance come from a single `__shfl_xor_sync` reduction of
// (sum, sum of squares) -- **no shared memory and no `__syncthreads()`**, which
// is what the one-block-per-row shapes pay twice. Requires
// `cols == 32 * VEC * NITER`, which is what keeps every access in range without
// a predicate.
//
// The variance is `E[x^2] - mean^2` rather than the sum of squared deviations
// the three-pass kernel uses: one fused reduction instead of two. On
// `[512, 1152]` this measures 0.0444 ms against the three-pass 0.0832, and the
// two forms agree to 2.4e-4 on the per-tensor gate.
template <int VEC, int NITER, int WARPS>
__global__ void layer_norm_bf16_warp_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* weight,
                                            const __mt_bfloat16* bias,
                                            __mt_bfloat16* output, int rows,
                                            int cols, float eps) {
  static_assert(VEC == 4 || VEC == 8, "VEC must be 4 or 8 bf16 lanes");
  using LoadT = typename RmsPack<VEC>::Type;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * WARPS + warp;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[NITER * VEC];
  float sum = 0.0f;
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < NITER; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * 32 + lane) * VEC;
    const LoadT raw = *reinterpret_cast<const LoadT*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < VEC; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * VEC + lane_in] = value;
      sum += value;
      square_sum += value * value;
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    square_sum += __shfl_xor_sync(0xffffffffu, square_sum, offset);
  }
  const float mean = sum / cols;
  const float inverse_std = rsqrtf(square_sum / cols - mean * mean + eps);
#pragma unroll
  for (int chunk = 0; chunk < NITER; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * 32 + lane) * VEC;
    const LoadT weight_raw = *reinterpret_cast<const LoadT*>(weight + column);
    const LoadT bias_raw = *reinterpret_cast<const LoadT*>(bias + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    LoadT out_raw;
    __mt_bfloat16* packed_out = reinterpret_cast<__mt_bfloat16*>(&out_raw);
#pragma unroll
    for (int lane_in = 0; lane_in < VEC; ++lane_in) {
      packed_out[lane_in] = __float2bfloat16(
          (values[chunk * VEC + lane_in] - mean) * inverse_std *
              __bfloat162float(packed_weight[lane_in]) +
          __bfloat162float(packed_bias[lane_in]));
    }
    *reinterpret_cast<LoadT*>(output + base + column) = out_raw;
  }
}

// Templated sibling of `layer_norm_bf16_e4m3_block_kernel` for the widths
// GR00T's DiT/vl/vision normalizations use. `kThreads * 4 * kChunks == cols`;
// one block per row and one barrier over `kThreads/32` warp partials.
template <int kThreads, int kChunks>
__global__ void layer_norm_bf16_e4m3_block_t_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight,
    const __mt_bfloat16* bias, uint8_t* output, int cols, float eps,
    float inverse_scale) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0, "kThreads must be a warp multiple");
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float values[kChunks * 4];
  // The affine is row-independent and is prefetched with the row so both
  // global round trips overlap instead of the affine arriving only after the
  // reduction. The arithmetic is unchanged.
  uint2 weight_raw[kChunks];
  uint2 bias_raw[kChunks];
  float sum = 0.0f;
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(input + base + column);
    weight_raw[chunk] = *reinterpret_cast<const uint2*>(weight + column);
    bias_raw[chunk] = *reinterpret_cast<const uint2*>(bias + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * 4 + lane_in] = value;
      sum += value;
      square_sum += value * value;
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float ssum[kWarps];
  __shared__ float ssq[kWarps];
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if (lane == 0) {
    ssum[warp] = sum;
    ssq[warp] = square_sum;
  }
  __syncthreads();
  float total = 0.0f;
  float total_sq = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total += ssum[w];
    total_sq += ssq[w];
  }
  const float mean = total / cols;
  const float inverse_std = rsqrtf(total_sq / cols - mean * mean + eps);
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw[chunk]);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw[chunk]);
    uint32_t packed_out = 0;
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const __mt_bfloat16 rounded = __float2bfloat16(
          (values[chunk * 4 + lane_in] - mean) * inverse_std *
              __bfloat162float(packed_weight[lane_in]) +
          __bfloat162float(packed_bias[lane_in]));
      packed_out |= static_cast<uint32_t>(
                        quantize_e4m3_value(__bfloat162float(rounded),
                                            inverse_scale))
                    << (8 * lane_in);
    }
    *reinterpret_cast<uint32_t*>(output + base + column) = packed_out;
  }
}

// Dual-output sibling: writes the BF16 normalized value and its static E4M3
// quantization in one pass. The E4M3 bytes match
// `layer_norm_bf16_e4m3_block_t_kernel`; the BF16 store carries the same
// rounded value the quantizer consumes, so callers that keep both outputs do
// not recompute the statistics.
template <int kThreads, int kChunks>
__global__ void layer_norm_bf16_e4m3_dual_block_t_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight,
    const __mt_bfloat16* bias, __mt_bfloat16* normalized, uint8_t* output,
    int cols, float eps, float inverse_scale) {
  constexpr int kWarps = kThreads / 32;
  static_assert(kThreads % 32 == 0, "kThreads must be a warp multiple");
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float values[kChunks * 4];
  float sum = 0.0f;
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * 4 + lane_in] = value;
      sum += value;
      square_sum += value * value;
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float ssum[kWarps];
  __shared__ float ssq[kWarps];
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if (lane == 0) {
    ssum[warp] = sum;
    ssq[warp] = square_sum;
  }
  __syncthreads();
  float total = 0.0f;
  float total_sq = 0.0f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total += ssum[w];
    total_sq += ssq[w];
  }
  const float mean = total / cols;
  const float inverse_std = rsqrtf(total_sq / cols - mean * mean + eps);
#pragma unroll
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int64_t column = static_cast<int64_t>(chunk * kThreads + tid) * 4;
    const uint2 weight_raw = *reinterpret_cast<const uint2*>(weight + column);
    const uint2 bias_raw = *reinterpret_cast<const uint2*>(bias + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    uint2 normalized_raw;
    __mt_bfloat16* packed_normalized =
        reinterpret_cast<__mt_bfloat16*>(&normalized_raw);
    uint32_t packed_out = 0;
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const __mt_bfloat16 rounded = __float2bfloat16(
          (values[chunk * 4 + lane_in] - mean) * inverse_std *
              __bfloat162float(packed_weight[lane_in]) +
          __bfloat162float(packed_bias[lane_in]));
      packed_normalized[lane_in] = rounded;
      packed_out |= static_cast<uint32_t>(
                        quantize_e4m3_value(__bfloat162float(rounded),
                                            inverse_scale))
                    << (8 * lane_in);
    }
    *reinterpret_cast<uint2*>(normalized + base + column) = normalized_raw;
    *reinterpret_cast<uint32_t*>(output + base + column) = packed_out;
  }
}

// Static-E4M3 output for the LayerNorm. Same arithmetic as the sibling,
// including the single BF16 rounding; the store quantizes instead of storing
// BF16.
//
// One block per row, three warps per block. The recorded call is
// `[512, 1152]`, where the warp-per-row form launched 128 blocks and left each
// SM with 8 warps; three warps give the grid 512 blocks (32 warps per SM), and
// the reduction is one barrier over three warp partials. Measured on an idle
// S5000: 5.25 us against 7.54, bit-identical.
__global__ void layer_norm_bf16_e4m3_block_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight,
    const __mt_bfloat16* bias, uint8_t* output, int cols, float eps,
    float inverse_scale) {
  constexpr int kBlockThreads = 96;  // cols == 3 * kBlockThreads * 4
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float values[12];
  float sum = 0.0f;
  float square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 3; ++chunk) {
    const int64_t column =
        static_cast<int64_t>(chunk * kBlockThreads + tid) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(input + base + column);
    const __mt_bfloat16* packed = reinterpret_cast<const __mt_bfloat16*>(&raw);
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const float value = __bfloat162float(packed[lane_in]);
      values[chunk * 4 + lane_in] = value;
      sum += value;
      square_sum += value * value;
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float ssum[3];
  __shared__ float ssq[3];
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if (lane == 0) {
    ssum[warp] = sum;
    ssq[warp] = square_sum;
  }
  __syncthreads();
  const float total = ssum[0] + ssum[1] + ssum[2];
  const float total_sq = ssq[0] + ssq[1] + ssq[2];
  const float mean = total / cols;
  const float inverse_std = rsqrtf(total_sq / cols - mean * mean + eps);
#pragma unroll
  for (int chunk = 0; chunk < 3; ++chunk) {
    const int64_t column =
        static_cast<int64_t>(chunk * kBlockThreads + tid) * 4;
    const uint2 weight_raw =
        *reinterpret_cast<const uint2*>(weight + column);
    const uint2 bias_raw = *reinterpret_cast<const uint2*>(bias + column);
    const __mt_bfloat16* packed_weight =
        reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    const __mt_bfloat16* packed_bias =
        reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    uint32_t packed_out = 0;
#pragma unroll
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      const __mt_bfloat16 rounded = __float2bfloat16(
          (values[chunk * 4 + lane_in] - mean) * inverse_std *
              __bfloat162float(packed_weight[lane_in]) +
          __bfloat162float(packed_bias[lane_in]));
      packed_out |= static_cast<uint32_t>(
                        quantize_e4m3_value(__bfloat162float(rounded),
                                            inverse_scale))
                    << (8 * lane_in);
    }
    *reinterpret_cast<uint32_t*>(output + base + column) = packed_out;
  }
}
