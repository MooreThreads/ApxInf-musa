#pragma once
#include "../custom/math.cuh"
#include "../custom/reduction.cuh"
#include <musa_bf16.h>
namespace apxinf_mp31 {
// The M=10 projection consumer keeps the FP32 intermediate: the hidden value is
// rounded once for the residual output, while the RMS statistic and the E4M3
// quantization read the unrounded value. This is the measured default path.
template <int SP>
__global__ void partial_norm(const float *partial, const __mt_bfloat16 *residual,
                             const __mt_bfloat16 *gate_style, const __mt_bfloat16 *norm_style,
                             __mt_bfloat16 *hidden, uint8_t *normalized, int rows, int cols,
                             float eps, float inverse_scale, float alpha) {
  using LoadT = uint2;
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  const int column = threadIdx.x * 4;
  const int64_t index = static_cast<int64_t>(row) * cols + column;
  LoadT projection_raw;
  auto *dst = reinterpret_cast<__mt_bfloat16 *>(&projection_raw);
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    float value = 0;
#pragma unroll
    for (int sp = 0; sp < SP; ++sp)
      value += partial[sp * rows * cols + index + lane];
    dst[lane] = __float2bfloat16(alpha == 0.f ? 0.f : value * alpha);
  }
  const LoadT residual_raw = *reinterpret_cast<const LoadT *>(residual + index);
  const LoadT gate_raw = *reinterpret_cast<const LoadT *>(gate_style + 2 * cols + column);
  const __mt_bfloat16 *packed_projection = reinterpret_cast<const __mt_bfloat16 *>(&projection_raw);
  const __mt_bfloat16 *packed_residual = reinterpret_cast<const __mt_bfloat16 *>(&residual_raw);
  const __mt_bfloat16 *packed_gate = reinterpret_cast<const __mt_bfloat16 *>(&gate_raw);
  float values[4];
  float square_sum = 0.0f;
  LoadT hidden_raw;
  __mt_bfloat16 *packed_hidden = reinterpret_cast<__mt_bfloat16 *>(&hidden_raw);
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    const float hidden_value = __bfloat162float(packed_residual[lane]) +
                               __bfloat162float(packed_projection[lane]) *
                                   __bfloat162float(packed_gate[lane]);
    packed_hidden[lane] = __float2bfloat16(hidden_value);
    values[lane] = hidden_value;
    square_sum += values[lane] * values[lane];
  }
  *reinterpret_cast<LoadT *>(hidden + index) = hidden_raw;
  const float inverse_rms = rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  const LoadT scale_raw = *reinterpret_cast<const LoadT *>(norm_style + column);
  const LoadT shift_raw = *reinterpret_cast<const LoadT *>(norm_style + cols + column);
  const __mt_bfloat16 *packed_scale = reinterpret_cast<const __mt_bfloat16 *>(&scale_raw);
  const __mt_bfloat16 *packed_shift = reinterpret_cast<const __mt_bfloat16 *>(&shift_raw);
  uint32_t packed_out = 0;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    const float output_value = values[lane] * inverse_rms *
                                   (1.0f + __bfloat162float(packed_scale[lane])) +
                               __bfloat162float(packed_shift[lane]);
    packed_out |= static_cast<uint32_t>(quantize_e4m3_value(output_value, inverse_scale))
                  << (8 * lane);
  }
  *reinterpret_cast<uint32_t *>(normalized + index) = packed_out;
}
} // namespace apxinf_mp31
