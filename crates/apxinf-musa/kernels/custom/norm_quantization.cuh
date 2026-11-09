#pragma once
#include "quantization.cuh"

// Match the BF16 producer's eight-adjacent-column reduction and rounding.
template <bool RESIDUAL, bool HAS_BIAS>
__global__ void rms_quantize_bf16_int8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* residual,
    const __mt_bfloat16* projection_bias, const __mt_bfloat16* weight,
    const __mt_bfloat16* norm_bias, __mt_bfloat16* hidden, int8_t* output,
    __half* scales, int rows, int cols, float eps, const float* inv_c) {
  __shared__ float scratch[8];
  const int row = blockIdx.x, column = threadIdx.x * 8;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  PackedBf16x8 source, res, bias, hid, weights;
  source.bits = *reinterpret_cast<const uint4*>(input + index);
  weights.bits = *reinterpret_cast<const uint4*>(weight + column);
  if (RESIDUAL) res.bits = *reinterpret_cast<const uint4*>(residual + index);
  if (HAS_BIAS)
    bias.bits = *reinterpret_cast<const uint4*>(projection_bias + column);
  float values[8], square_sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) {
    float value = __bfloat162float(source.lanes[lane]);
    if (RESIDUAL) {
      value += __bfloat162float(res.lanes[lane]);
      if (HAS_BIAS) value += __bfloat162float(bias.lanes[lane]);
      hid.lanes[lane] = __float2bfloat16(value);
      value = __bfloat162float(hid.lanes[lane]);
    }
    values[lane] = value;
    square_sum += value * value;
  }
  if (RESIDUAL) *reinterpret_cast<uint4*>(hidden + index) = hid.bits;
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / cols;
  const float inverse_rms =
      RESIDUAL ? rsqrtf(mean + eps) : rsqrtf(__fadd_rn(mean, eps));
  // All threads must finish reading the sum before amax reuses shared memory.
  __syncthreads();
  float factors[8];
  if (inv_c) load_inv_c_lane(inv_c, column, factors);
  float maximum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) {
    auto rounded = __float2bfloat16(values[lane] * inverse_rms *
                                    __bfloat162float(weights.lanes[lane]));
    if (inv_c)
      rounded = __float2bfloat16(__bfloat162float(rounded) * factors[lane]);
    values[lane] = __bfloat162float(rounded);
    maximum = fmaxf(maximum, fabsf(values[lane]));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = __float2half_rn(scale);
  if (row == 0)
    for (int pad = threadIdx.x; pad < kW8A8ScalePad; pad += blockDim.x)
      scales[rows + pad] = __float2half_rn(0.0f);
  PackedI8x8 quantized;
#pragma unroll
  for (int lane = 0; lane < 8; ++lane)
    quantized.lanes[lane] = quantize_rows_value(values[lane], scale);
  *reinterpret_cast<uint64_t*>(output + index) = quantized.bits;
}

// One software warp per row, preserving the original LayerNorm xor tree.
template <int WARPS, bool RESIDUAL, bool HAS_BIAS>
__global__ void layer_quantize_bf16_int8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* residual,
    const __mt_bfloat16* projection_bias, const __mt_bfloat16* weight,
    const __mt_bfloat16* norm_bias, __mt_bfloat16* hidden, int8_t* output,
    __half* scales, int rows, int cols, float eps, const float* inv_c) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
  // Include inactive rows in padding initialization for short/partial blocks.
  if (blockIdx.x == 0)
    for (int pad = threadIdx.x; pad < kW8A8ScalePad; pad += blockDim.x)
      scales[rows + pad] = __float2half_rn(0.0f);
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float values[36], sum = 0.0f, square_sum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 9; ++chunk) {
    const int column = (chunk * 32 + lane) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(input + base + column);
    uint2 res_raw, bias_raw, hidden_raw;
    if (RESIDUAL)
      res_raw = *reinterpret_cast<const uint2*>(residual + base + column);
    if (HAS_BIAS)
      bias_raw = *reinterpret_cast<const uint2*>(projection_bias + column);
    const auto* source = reinterpret_cast<const __mt_bfloat16*>(&raw);
    const auto* res = reinterpret_cast<const __mt_bfloat16*>(&res_raw);
    const auto* bias = reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    auto* hid = reinterpret_cast<__mt_bfloat16*>(&hidden_raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float value = __bfloat162float(source[i]);
      if (RESIDUAL) {
        value += __bfloat162float(res[i]);
        if (HAS_BIAS) value += __bfloat162float(bias[i]);
        hid[i] = __float2bfloat16(value);
        value = __bfloat162float(hid[i]);
      }
      values[chunk * 4 + i] = value;
      sum += value;
      square_sum += value * value;
    }
    if (RESIDUAL)
      *reinterpret_cast<uint2*>(hidden + base + column) = hidden_raw;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    square_sum += __shfl_xor_sync(0xffffffffu, square_sum, offset);
  }
  const float mean = sum / cols;
  const float inverse_std = rsqrtf(square_sum / cols - mean * mean + eps);
  float maximum = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 9; ++chunk) {
    const int column = (chunk * 32 + lane) * 4;
    const uint2 weight_raw = *reinterpret_cast<const uint2*>(weight + column);
    const uint2 bias_raw = *reinterpret_cast<const uint2*>(norm_bias + column);
    const auto* w = reinterpret_cast<const __mt_bfloat16*>(&weight_raw);
    const auto* b = reinterpret_cast<const __mt_bfloat16*>(&bias_raw);
    float4 factor_raw;
    if (inv_c) factor_raw = *reinterpret_cast<const float4*>(inv_c + column);
    const auto* factors = reinterpret_cast<const float*>(&factor_raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto rounded = __float2bfloat16((values[chunk * 4 + i] - mean) *
                                          inverse_std * __bfloat162float(w[i]) +
                                      __bfloat162float(b[i]));
      if (inv_c)
        rounded = __float2bfloat16(__bfloat162float(rounded) * factors[i]);
      values[chunk * 4 + i] = __bfloat162float(rounded);
      maximum = fmaxf(maximum, fabsf(values[chunk * 4 + i]));
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
  const float scale = fmaxf(maximum / 127.0f, 1.0e-12f);
  if (lane == 0) scales[row] = __float2half_rn(scale);
#pragma unroll
  for (int chunk = 0; chunk < 9; ++chunk) {
    uint32_t packed;
    auto* bytes = reinterpret_cast<int8_t*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      bytes[i] = quantize_rows_value(values[chunk * 4 + i], scale);
    *reinterpret_cast<uint32_t*>(output + base + (chunk * 32 + lane) * 4) =
        packed;
  }
}
