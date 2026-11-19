#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

#include "math.cuh"

// Pure device code, grouped by physical operation. Launch policy and the C ABI
// live under `adapters/`; nothing here is `extern "C"`.
//
// BF16 epilogues retain FP32 activation math and explicit BF16 output
// rounding. GeGLU uses packed4 for aligned widths and scalar tails otherwise.
//
// The activation codes are the CUDA kernel ABI's, so a diagnosable name never
// has to be renumbered across backends: 0 none, 1 GELU (tanh approximation),
// 2 SiLU, 3 ReLU.

enum class Activation : int {
  None = 0,
  Gelu = 1,
  Silu = 2,
  Relu = 3,
};

__global__ void bias_activation_bf16_kernel(const __mt_bfloat16* input,
                                            const __mt_bfloat16* bias,
                                            __mt_bfloat16* output, int rows,
                                            int cols, Activation activation) {
  // Rows on `gridDim.y` and column chunks on `gridDim.x`, so the column is an
  // offset from the block's own base and never a remainder. The flat
  // grid-stride form this replaced took `index % cols` per element, and on
  // M1000 a 64-bit integer division is a software routine: measuring it (the
  // same work with and without) put 6x of this kernel's runtime in the divide
  // measured directly, with and without the division.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    // `bias` is row-broadcast: one value per column, for every row. The bridge
    // reaches the same place through tensor broadcasting, which is why the
    // recorded tensors do not depend on the row count.
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      float value = __bfloat162float(input[base + col]);
      if (bias != nullptr) value += __bfloat162float(bias[col]);
      if (activation == Activation::Gelu) value = gelu_tanh(value);
      if (activation == Activation::Silu) value = value / (1.0f + __musa_exp_f(-value));
      if (activation == Activation::Relu) value = fmaxf(value, 0.0f);
      output[base + col] = __float2bfloat16(value);
    }
  }
}

// Qwen gated MLP epilogue: `silu(gate) * up`, with one BF16 output rounding.
__global__ void silu_mul_bf16_kernel(const __mt_bfloat16* gate,
                                     const __mt_bfloat16* up,
                                     __mt_bfloat16* output, int64_t count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const float gate_value = __bfloat162float(gate[index]);
    const float up_value = __bfloat162float(up[index]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[index] = __float2bfloat16(silu * up_value);
  }
}

// Qwen gated MLP epilogue written as static E4M3: the same BF16 product
// rounding as `silu_mul_bf16_kernel`, followed by the fixed-scale conversion
// `quantize_bf16_e4m3_kernel` applies, in one launch. The composed pair
// measured ~11.9 us/call for the text widths; this removes one launch and the
// [145,6144] BF16 round trip per text layer.
__global__ void silu_mul_bf16_e4m3_kernel(const __mt_bfloat16* gate,
                                          const __mt_bfloat16* up,
                                          uint8_t* output, int64_t count,
                                          float inverse_scale) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const float gate_value = __bfloat162float(gate[index]);
    const float up_value = __bfloat162float(up[index]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    const __mt_bfloat16 rounded = __float2bfloat16(silu * up_value);
    output[index] = quantize_e4m3_value(__bfloat162float(rounded), inverse_scale);
  }
}

union __align__(16) ActivationPacked8 {
  uint4 bits;
  __mt_bfloat16 lanes[8];
};

// Vectorized sibling of `silu_mul_bf16_e4m3_kernel`: eight lanes per thread in
// two 16-byte loads and one 8-byte store. The arithmetic per lane is the
// scalar kernel's, so the output is byte-identical; the launcher falls back
// unless `count % 8 == 0` and the pointers meet the vector alignment.
__global__ void silu_mul_bf16_e4m3_vec8_kernel(const __mt_bfloat16* gate,
                                               const __mt_bfloat16* up,
                                               uint8_t* output, int64_t count,
                                               float inverse_scale) {
  const int64_t vectors = count / 8;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors; index += stride) {
    ActivationPacked8 gate_raw, up_raw;
    gate_raw.bits = *reinterpret_cast<const uint4*>(gate + index * 8);
    up_raw.bits = *reinterpret_cast<const uint4*>(up + index * 8);
    uint64_t packed_out = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gate_value = __bfloat162float(gate_raw.lanes[i]);
      const float up_value = __bfloat162float(up_raw.lanes[i]);
      const float silu = gate_value / (1.0f + expf(-gate_value));
      const __mt_bfloat16 rounded = __float2bfloat16(silu * up_value);
      packed_out |=
          static_cast<uint64_t>(quantize_e4m3_value(
              __bfloat162float(rounded), inverse_scale))
          << (8 * i);
    }
    *reinterpret_cast<uint64_t*>(output + index * 8) = packed_out;
  }
}

// Vectorized bias + activation epilogue: eight columns per thread in one 16-byte
// load and one 16-byte store.
//
// Both branches the scalar kernel above takes per element are **compile-time**
// parameters here -- the bias presence, and which activation -- for the reason
// `fused.cuh`'s register-cached kernels record: a runtime null test or code test
// inside the element loop costs more than the load it guards. The scalar form
// also moves 2 bytes per access, which is what its ceiling is really made of:
// on `[512, 4304]` -- the vision tower's fc1 epilogue -- it measures 0.2163 ms
// (40.8 GB/s) with GELU and 0.1862 (47.4) with the activation off, so the
// `tanh` is 13% and the rest is scalar traffic. Requires `cols % 8 == 0`; the
// launcher checks that and the pointer alignment, and every other width keeps
// the scalar kernel.
//
// The arithmetic and the single BF16 rounding are unchanged, so this is
// bit-identical to the scalar form.
template <Activation ACTIVATION, bool HAS_BIAS>
__global__ void bias_activation_bf16_vector8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* bias,
    __mt_bfloat16* output, int rows, int cols) {
  static_assert(ACTIVATION == Activation::None || ACTIVATION == Activation::Gelu ||
                    ACTIVATION == Activation::Silu || ACTIVATION == Activation::Relu,
                "unknown activation code");
  const int width = cols / 8;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < width;
         lane += stride) {
      const int64_t column = static_cast<int64_t>(lane) * 8;
      ActivationPacked8 input_raw, bias_raw, output_raw;
      input_raw.bits = *reinterpret_cast<const uint4*>(input + base + column);
      if (HAS_BIAS) {
        bias_raw.bits = *reinterpret_cast<const uint4*>(bias + column);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = __bfloat162float(input_raw.lanes[i]);
        if (HAS_BIAS) value += __bfloat162float(bias_raw.lanes[i]);
        if (ACTIVATION == Activation::Gelu) value = gelu_tanh(value);
        if (ACTIVATION == Activation::Silu) value = value / (1.0f + __musa_exp_f(-value));
        if (ACTIVATION == Activation::Relu) value = fmaxf(value, 0.0f);
        output_raw.lanes[i] = __float2bfloat16(value);
      }
      *reinterpret_cast<uint4*>(output + base + column) = output_raw.bits;
    }
  }
}

// Plain SiLU used where Qwen applies the activation before a separately
// expressed multiply.  Keep the f32 activation and single BF16 output rounding
// of both Torch eager and the CUDA backend.
__global__ void silu_bf16_kernel(const __mt_bfloat16* input,
                                 __mt_bfloat16* output, int64_t count) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += stride) {
    const float value = __bfloat162float(input[index]);
    output[index] = __float2bfloat16(value / (1.0f + expf(-value)));
  }
}

// Static-E4M3 output for the vectorized bias + activation epilogue. The
// arithmetic and the single BF16 rounding are the sibling's; the store
// quantizes eight lanes into one 8-byte transaction, which is what makes this
// one launch instead of `bias_activation` + `quantize_fixed_e4m3`.
template <int ACTIVATION, bool HAS_BIAS>
__global__ void bias_activation_bf16_e4m3_vector8_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* bias, uint8_t* output,
    int rows, int cols, float inverse_scale) {
  static_assert(ACTIVATION >= 0 && ACTIVATION <= 2, "unknown activation code");
  const int width = cols / 8;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * cols;
    for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < width;
         lane += stride) {
      const int64_t column = static_cast<int64_t>(lane) * 8;
      ActivationPacked8 input_raw, bias_raw;
      input_raw.bits = *reinterpret_cast<const uint4*>(input + base + column);
      if (HAS_BIAS) {
        bias_raw.bits = *reinterpret_cast<const uint4*>(bias + column);
      }
      uint64_t packed_out = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float value = __bfloat162float(input_raw.lanes[i]);
        if (HAS_BIAS) value += __bfloat162float(bias_raw.lanes[i]);
        if (ACTIVATION == 1) value = gelu_tanh(value);
        if (ACTIVATION == 2) value = value / (1.0f + __musa_exp_f(-value));
        const __mt_bfloat16 rounded = __float2bfloat16(value);
        packed_out |= static_cast<uint64_t>(
                          quantize_e4m3_value(__bfloat162float(rounded),
                                              inverse_scale))
                      << (8 * i);
      }
      *reinterpret_cast<uint64_t*>(output + base + column) = packed_out;
    }
  }
}

// Gated GELU: `gelu_tanh(gate) * up`, both halves of the row.
//
// The gate is the first half of the columns and the up projection the second,
// which is the plain layout; the interleaved packings belong to the fused GEMM
// that consumes this, not here.
//
// Arithmetic in f32 with one rounding on the way out, which is what the bridge
// did (`_geglu_bf16`: `_f32(gate_up)`, `_gelu_tanh`, multiply, `_write`). A
// single multiply has no add to contract into, so unlike the euler step there
// is nothing here for the compiler to fuse away.
__global__ void geglu_bf16_kernel(const __mt_bfloat16* gate_up,
                                  __mt_bfloat16* output, int rows, int inner) {
  // As above: the row/column split comes from the grid rather than from a
  // 64-bit divide and remainder per element, which is where six sevenths of
  // this kernel's time went on M1000.
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const __mt_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    __mt_bfloat16* row_output = output + static_cast<int64_t>(row) * inner;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < inner;
         col += stride) {
      row_output[col] = __float2bfloat16(
          gelu_tanh(__bfloat162float(row_input[col])) *
          __bfloat162float(row_input[inner + col]));
    }
  }
}

union __align__(8) GegluPacked4 {
  uint64_t bits;
  __mt_bfloat16 lanes[4];
};

// Same FP32 GELU and the same BF16 rounding as the scalar path. Vectorized
// gate/up reads and output writes reduce traffic/CTAs without changing math.
__global__ void geglu_bf16_packed4_kernel(const __mt_bfloat16* input,
                                        __mt_bfloat16* output,
                                        int64_t rows, int64_t hidden) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = row * hidden;
    for (int64_t col = 4LL * (blockIdx.x * blockDim.x + threadIdx.x);
         col < hidden; col += 4LL * blockDim.x * gridDim.x) {
      GegluPacked4 gate, up, result;
      gate.bits = *reinterpret_cast<const uint64_t*>(input + 2 * base + col);
      up.bits = *reinterpret_cast<const uint64_t*>(input + 2 * base + hidden + col);
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        result.lanes[lane] = __float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[lane])) *
            __bfloat162float(up.lanes[lane]));
      }
      *reinterpret_cast<uint64_t*>(output + base + col) = result.bits;
    }
  }
}

// Gated GELU whose output is the static-FP8 quantizer's E4M3, one launch
// instead of the composed geglu + quantize pair.
//
// The arithmetic is the composed path's: f32 GELU, multiply, one BF16 rounding,
// then `quantize_e4m3_value`. The rounding before the conversion is what keeps
// this byte-identical to reading the BF16 the other kernel would have written.
__global__ void geglu_bf16_e4m3_packed4_kernel(const __mt_bfloat16* input,
                                               uint8_t* output, int64_t rows,
                                               int64_t hidden,
                                               float inverse_scale) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = row * hidden;
    for (int64_t col = 4LL * (blockIdx.x * blockDim.x + threadIdx.x);
         col < hidden; col += 4LL * blockDim.x * gridDim.x) {
      GegluPacked4 gate, up;
      gate.bits = *reinterpret_cast<const uint64_t*>(input + 2 * base + col);
      up.bits =
          *reinterpret_cast<const uint64_t*>(input + 2 * base + hidden + col);
      uint32_t packed = 0;
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const float value = __bfloat162float(__float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[lane])) *
            __bfloat162float(up.lanes[lane])));
        packed |= static_cast<uint32_t>(
                      quantize_e4m3_value(value, inverse_scale))
                  << (8 * lane);
      }
      *reinterpret_cast<uint32_t*>(output + base + col) = packed;
    }
  }
}

union __align__(16) GegluPacked8 {
  uint4 bits;
  __mt_bfloat16 lanes[8];
};

// Eight lanes per thread. The prefix half-split shape (hidden = 8192) has both
// halves 16-byte aligned, so the packed4 form's two 8-byte loads become one
// 16-byte load on gate and one on up. The per-lane arithmetic, the BF16 round
// and the E4M3 byte order are identical to `..._packed4_kernel`.
__global__ void geglu_bf16_e4m3_packed8_kernel(const __mt_bfloat16* input,
                                               uint8_t* output, int64_t rows,
                                               int64_t hidden,
                                               float inverse_scale) {
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = row * hidden;
    for (int64_t col = 8LL * (blockIdx.x * blockDim.x + threadIdx.x);
         col < hidden; col += 8LL * blockDim.x * gridDim.x) {
      GegluPacked8 gate, up;
      gate.bits = *reinterpret_cast<const uint4*>(input + 2 * base + col);
      up.bits =
          *reinterpret_cast<const uint4*>(input + 2 * base + hidden + col);
      uint64_t packed = 0;
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        const float value = __bfloat162float(__float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[lane])) *
            __bfloat162float(up.lanes[lane])));
        packed |= static_cast<uint64_t>(
                      quantize_e4m3_value(value, inverse_scale))
                  << (8 * lane);
      }
      *reinterpret_cast<uint64_t*>(output + base + col) = packed;
    }
  }
}

__global__ void geglu_bf16_e4m3_kernel(const __mt_bfloat16* gate_up,
                                       uint8_t* output, int rows, int inner,
                                       float inverse_scale) {
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const __mt_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    uint8_t* row_output = output + static_cast<int64_t>(row) * inner;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < inner;
         col += stride) {
      const float value = __bfloat162float(__float2bfloat16(
          gelu_tanh(__bfloat162float(row_input[col])) *
          __bfloat162float(row_input[inner + col])));
      row_output[col] = quantize_e4m3_value(value, inverse_scale);
    }
  }
}

// Pair-interleaved `[g0..g_p, u0..u_p, g_p.., ...]` layout. Each block owns
// one `pair_width` window, so the hot loop contains no integer division:
// MUSA's 64-bit division is a software routine, and a naive `col / pair_width`
// version regressed the model by 2.5 ms. The half-split kernels above are left
// untouched so their path pays no branch for a layout it never sees.
__global__ void geglu_bf16_e4m3_pairs_packed4_kernel(
    const __mt_bfloat16* input, uint8_t* output, int rows, int hidden,
    int pair_width, float inverse_scale) {
  const int block = blockIdx.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * 2 * hidden;
    const int64_t gate_base = base + 2LL * pair_width * block;
    const int64_t out_base =
        static_cast<int64_t>(row) * hidden + pair_width * block;
    for (int within = 4 * threadIdx.x; within < pair_width;
         within += 4 * blockDim.x) {
      GegluPacked4 gate, up;
      gate.bits = *reinterpret_cast<const uint64_t*>(input + gate_base + within);
      up.bits = *reinterpret_cast<const uint64_t*>(
          input + gate_base + pair_width + within);
      uint32_t packed = 0;
#pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        const float value = __bfloat162float(__float2bfloat16(
            gelu_tanh(__bfloat162float(gate.lanes[lane])) *
            __bfloat162float(up.lanes[lane])));
        packed |= static_cast<uint32_t>(
                      quantize_e4m3_value(value, inverse_scale))
                  << (8 * lane);
      }
      *reinterpret_cast<uint32_t*>(output + out_base + within) = packed;
    }
  }
}

__global__ void geglu_bf16_e4m3_pairs_kernel(
    const __mt_bfloat16* input, uint8_t* output, int rows, int hidden,
    int pair_width, float inverse_scale) {
  const int block = blockIdx.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t base = static_cast<int64_t>(row) * 2 * hidden;
    const int64_t gate_base = base + 2LL * pair_width * block;
    const int64_t out_base =
        static_cast<int64_t>(row) * hidden + pair_width * block;
    for (int within = threadIdx.x; within < pair_width;
         within += blockDim.x) {
      const float value = __bfloat162float(__float2bfloat16(
          gelu_tanh(__bfloat162float(input[gate_base + within])) *
          __bfloat162float(input[gate_base + pair_width + within])));
      output[out_base + within] = quantize_e4m3_value(value, inverse_scale);
    }
  }
}
