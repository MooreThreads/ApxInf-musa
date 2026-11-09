#pragma once

// Apply the vision tower's half-split 2-D rotary embedding to
// `[tokens, heads, head_dim]`.  The first half of the rotary pairs uses the
// first position column (height), and the second half uses the second column
// (width). This follows the recorded reference bridge: the
// input is widened to f32, the four arithmetic operations remain f32, and
// only the final stores round to BF16.
//
// `grid.x` covers rotary pairs, `grid.y` heads, and `grid.z` tokens.  The
// kernel is intentionally grid-stride in pairs so the launcher can keep the
// same block size for head dimensions larger than one block.
__global__ void vision_2d_rope_bf16_kernel(const __mt_bfloat16* input,
                                            const uint32_t* positions,
                                            __mt_bfloat16* output, int tokens,
                                            int heads, int head_dim, float theta) {
  const int half_dim = head_dim / 2;
  const int axis_width = half_dim / 2;
  const int token = blockIdx.z;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads || axis_width <= 0) return;

  const int64_t row = (static_cast<int64_t>(token) * heads + head) * head_dim;
  const uint32_t* position = positions + static_cast<int64_t>(token) * 2;
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < half_dim;
       pair += gridDim.x * blockDim.x) {
    const int axis = pair >= axis_width ? 1 : 0;
    const int pair_in_axis = pair % axis_width;
    const float exponent =
        -2.0f * static_cast<float>(pair_in_axis) / static_cast<float>(half_dim);
    const float frequency = powf(theta, exponent);
    const float angle = static_cast<float>(position[axis]) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = __bfloat162float(input[row + pair]);
    const float second = __bfloat162float(input[row + half_dim + pair]);

    // Keep the subtraction/addition separate from the products.  The bridge
    // computes these expressions in f32 and rounds only at the BF16 stores;
    // explicit intrinsics prevent a compiler FMA from changing that contract.
    const float first_cosine = __fmul_rn(first, cosine);
    const float second_sine = __fmul_rn(second, sine);
    const float second_cosine = __fmul_rn(second, cosine);
    const float first_sine = __fmul_rn(first, sine);
    output[row + pair] = __float2bfloat16(__fsub_rn(first_cosine, second_sine));
    output[row + half_dim + pair] =
        __float2bfloat16(__fadd_rn(second_cosine, first_sine));
  }
}

// Build one reusable (cosine, sine) pair per token and rotary pair. The table
// is interleaved f32 `[tokens, half_dim, 2]` so callers can reuse it across all
// layers sharing the same 2-D position grid.
__global__ void build_vision_rotation_table_f32_kernel(
    const uint32_t* positions, float* rotation_table, int tokens,
    int head_dim, float theta) {
  const int half_dim = head_dim / 2;
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(tokens) * half_dim;
  if (linear >= total) return;
  const int pair = static_cast<int>(linear % half_dim);
  const int token = static_cast<int>(linear / half_dim);
  const int axis_width = half_dim / 2;
  const int axis = pair >= axis_width ? 1 : 0;
  const int pair_in_axis = pair % axis_width;
  const uint32_t position = positions[static_cast<int64_t>(token) * 2 + axis];
  const float exponent =
      -2.0f * static_cast<float>(pair_in_axis) / static_cast<float>(half_dim);
  const float frequency = powf(theta, exponent);
  const float angle = static_cast<float>(position) * frequency;
  rotation_table[linear * 2] = cosf(angle);
  rotation_table[linear * 2 + 1] = sinf(angle);
}

// Split packed QKV, add projection bias, apply a prepared 2-D rotary table to
// Q/K, and write V. Explicit BF16 rounding after bias preserves the arithmetic
// boundary of the prior split-kernel followed by RoPE-kernel sequence.
__global__ void qkv_split_bias_vision_rope_precomputed_bf16_kernel(
    const __mt_bfloat16* qkv, const __mt_bfloat16* bias,
    __mt_bfloat16* q_out, __mt_bfloat16* k_out, __mt_bfloat16* v_out,
    const float* rotation_table, int tokens, int heads, int head_dim) {
  const int half_dim = head_dim / 2;
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(tokens) * heads * half_dim;
  if (linear >= total) return;
  const int pair = static_cast<int>(linear % half_dim);
  const int64_t token_head = linear / half_dim;
  const int head = static_cast<int>(token_head % heads);
  const int token = static_cast<int>(token_head / heads);
  const int width = heads * head_dim;
  const int head_col = head * head_dim;
  const int second = pair + half_dim;
  const int64_t qkv_row = static_cast<int64_t>(token) * 3 * width;
  const int64_t out_base = token_head * head_dim;

  const __mt_bfloat16 rounded_q0 = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + head_col + pair]) +
      __bfloat162float(bias[head_col + pair]));
  const __mt_bfloat16 rounded_q1 = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + head_col + second]) +
      __bfloat162float(bias[head_col + second]));
  const __mt_bfloat16 rounded_k0 = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + width + head_col + pair]) +
      __bfloat162float(bias[width + head_col + pair]));
  const __mt_bfloat16 rounded_k1 = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + width + head_col + second]) +
      __bfloat162float(bias[width + head_col + second]));
  const int64_t rotation =
      (static_cast<int64_t>(token) * half_dim + pair) * 2;
  const float cosine = rotation_table[rotation];
  const float sine = rotation_table[rotation + 1];
  const float q0 = __bfloat162float(rounded_q0);
  const float q1 = __bfloat162float(rounded_q1);
  const float k0 = __bfloat162float(rounded_k0);
  const float k1 = __bfloat162float(rounded_k1);
  q_out[out_base + pair] = __float2bfloat16(
      __fsub_rn(__fmul_rn(q0, cosine), __fmul_rn(q1, sine)));
  q_out[out_base + second] = __float2bfloat16(
      __fadd_rn(__fmul_rn(q1, cosine), __fmul_rn(q0, sine)));
  k_out[out_base + pair] = __float2bfloat16(
      __fsub_rn(__fmul_rn(k0, cosine), __fmul_rn(k1, sine)));
  k_out[out_base + second] = __float2bfloat16(
      __fadd_rn(__fmul_rn(k1, cosine), __fmul_rn(k0, sine)));
  v_out[out_base + pair] = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair]) +
      __bfloat162float(bias[2 * width + head_col + pair]));
  v_out[out_base + second] = __float2bfloat16(
      __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
      __bfloat162float(bias[2 * width + head_col + second]));
}

// Apply Qwen's multimodal 3-D RoPE to `[tokens, heads, head_dim]`.  Rotary
// pairs are assigned to time/height/width by their interleaved index: pair 0
// is temporal, pair 1 is height, pair 2 is width, and so on.  The height and
// width sections stop at `section_* * 3`, exactly matching the bridge's masks;
// pairs outside those ranges remain temporal.
__global__ void mrope_bf16_kernel(const __mt_bfloat16* input,
                                  const uint32_t* positions,
                                  __mt_bfloat16* output, int tokens, int heads,
                                  int head_dim, int section_h, int section_w,
                                  float theta) {
  const int half_dim = head_dim / 2;
  const int token = blockIdx.z;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;

  const int64_t row = (static_cast<int64_t>(token) * heads + head) * head_dim;
  const uint32_t* position = positions + static_cast<int64_t>(token) * 3;
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < half_dim;
       pair += gridDim.x * blockDim.x) {
    int axis = 0;  // temporal; this is also the bridge's default.
    if (pair % 3 == 1 && pair < section_h * 3) axis = 1;
    if (pair % 3 == 2 && pair < section_w * 3) axis = 2;

    const float exponent =
        -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
    const float frequency = powf(theta, exponent);
    const float angle = static_cast<float>(position[axis]) * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float first = __bfloat162float(input[row + pair]);
    const float second = __bfloat162float(input[row + half_dim + pair]);
    const float first_cosine = __fmul_rn(first, cosine);
    const float second_sine = __fmul_rn(second, sine);
    const float second_cosine = __fmul_rn(second, cosine);
    const float first_sine = __fmul_rn(first, sine);
    output[row + pair] = __float2bfloat16(__fsub_rn(first_cosine, second_sine));
    output[row + half_dim + pair] =
        __float2bfloat16(__fadd_rn(second_cosine, first_sine));
  }
}

// Build one reusable multimodal (cosine, sine) pair for every token and
// half-split rotary pair. The table is interleaved f32
// `[tokens, head_dim / 2, 2]` and can be shared by every layer with the same
// positions and rotary configuration.
__global__ void build_mrope_rotation_table_f32_kernel(
    const uint32_t* positions, float* rotation_table, int tokens,
    int head_dim, int section_h, int section_w, float theta) {
  const int half_dim = head_dim / 2;
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(tokens) * half_dim;
  if (linear >= total) return;

  const int pair = static_cast<int>(linear % half_dim);
  const int token = static_cast<int>(linear / half_dim);
  int axis = 0;
  if (pair % 3 == 1 && pair < section_h * 3) axis = 1;
  if (pair % 3 == 2 && pair < section_w * 3) axis = 2;
  const uint32_t position = positions[static_cast<int64_t>(token) * 3 + axis];
  const float exponent =
      -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
  const float frequency = powf(theta, exponent);
  const float angle = static_cast<float>(position) * frequency;
  rotation_table[linear * 2] = cosf(angle);
  rotation_table[linear * 2 + 1] = sinf(angle);
}

// Fuse Q/K RMSNorm and multimodal RoPE while reading a prepared FP32 rotation
// table. One block owns one `(token, head)` row. The normalized row is rounded
// to BF16 before rotation so this preserves the fallback's arithmetic boundary.
__global__ void qk_rms_norm_mrope_precomputed_bf16_kernel(
    const __mt_bfloat16* query_input, const __mt_bfloat16* query_weight,
    __mt_bfloat16* query_output, const __mt_bfloat16* key_input,
    const __mt_bfloat16* key_weight, __mt_bfloat16* key_output,
    const float* rotation_table, int tokens, int query_heads, int key_heads,
    int head_dim, float eps) {
  __shared__ float scratch[8];
  __shared__ __mt_bfloat16 rounded[128];

  const int token = blockIdx.x;
  const int combined_head = blockIdx.y;
  const bool is_query = combined_head < query_heads;
  const int heads = is_query ? query_heads : key_heads;
  const int head = is_query ? combined_head : combined_head - query_heads;
  if (token >= tokens || head >= heads) return;

  const __mt_bfloat16* input = is_query ? query_input : key_input;
  const __mt_bfloat16* weight = is_query ? query_weight : key_weight;
  __mt_bfloat16* output = is_query ? query_output : key_output;
  const int64_t base =
      (static_cast<int64_t>(token) * heads + head) * head_dim;

  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const float value = __bfloat162float(input[base + col]);
    square_sum += value * value;
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / head_dim;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    rounded[col] = __float2bfloat16(
        __bfloat162float(input[base + col]) * inverse_rms *
        __bfloat162float(weight[col]));
  }
  __syncthreads();

  const int half_dim = head_dim / 2;
  for (int pair = threadIdx.x; pair < half_dim; pair += blockDim.x) {
    const int64_t rotation =
        (static_cast<int64_t>(token) * half_dim + pair) * 2;
    const float cosine = rotation_table[rotation];
    const float sine = rotation_table[rotation + 1];
    const float first = __bfloat162float(rounded[pair]);
    const float second = __bfloat162float(rounded[half_dim + pair]);
    output[base + pair] = __float2bfloat16(__fsub_rn(
        __fmul_rn(first, cosine), __fmul_rn(second, sine)));
    output[base + half_dim + pair] = __float2bfloat16(__fadd_rn(
        __fmul_rn(second, cosine), __fmul_rn(first, sine)));
  }
}


// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Rotary position embedding, a port of
// `crates/apxinf-cuda/kernels/custom/fused.cuh`'s `qkv_rope_bf16_kernel`.
//
// One kernel serves both entry points: `rope::split_qkv_apply_bf16` writes q, k
// and v into fresh buffers, `rope::apply_q_write_kv_bf16` writes k and v into
// the first `tokens` rows of a caller-owned cache at `kv_output_offset`. They
// differ in where the two destinations point and in nothing else, which is why
// CUDA has one launcher behind both and why this ABI takes the offset.

// The bf16 value `value` rounds to, as an f32.
//
// The round trip is the arithmetic, not a storage detail: the reference rounds
// at each of these points and the recorded tensors depend on it. See the chain
// described on the kernel below.
__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

// The rope frequency table: `powf(theta, -pair / half_dim)` for every pair.
//
// One block is plenty -- `half_dim` is at most 512 -- and it is launched only
// when `(theta, half_dim)` changes, so the cost does not appear in a pass.
__global__ void rope_frequency_kernel(float* out, int half_dim, float theta) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < half_dim) {
    out[pair] = powf(theta, -static_cast<float>(pair) / half_dim);
  }
}

// One half-split pair, rotated with the reference's rounding chain: each
// product rounds to bf16, then the difference or sum rounds again.
//
// `__fmul_rn`/`__fsub_rn`/`__fadd_rn` rather than bare operators, because the
// compiler is otherwise free to contract `first * cosine - second * sine` into
// one FMA -- which skips exactly the rounding the bf16 intermediate exists to
// impose.
__device__ __forceinline__ void rope_pair_bf16(float first, float second,
                                               float cosine, float sine,
                                               __mt_bfloat16* destination,
                                               int half_dim, int pair) {
  const float first_cosine = round_bf16(__fmul_rn(first, cosine));
  const float second_sine = round_bf16(__fmul_rn(second, sine));
  const float second_cosine = round_bf16(__fmul_rn(second, cosine));
  const float first_sine = round_bf16(__fmul_rn(first, sine));
  destination[pair] = __float2bfloat16(__fsub_rn(first_cosine, second_sine));
  destination[half_dim + pair] =
      __float2bfloat16(__fadd_rn(second_cosine, first_sine));
}

// Split a packed `[tokens, (q_heads + 2 * kv_heads) * head_dim]` projection,
// add an optional per-column bias, rotate Q and K, and write all three parts.
//
// # The rounding chain is the contract
//
// This is **not** the arithmetic the CUDA kernel computes, and the difference is
// deliberate. The reference tensors were produced by the eager bridge
// (`_rope_and_split` in the torch bridge that recorded them; it is not part of
// this tree), which follows
// Gemma: the bias is added in f32, the result is rounded to bf16, `cos`/`sin`
// are rounded to bf16 *before* the rotation, and the rotation is then bf16
// arithmetic. CUDA's kernel keeps everything in f32 and rounds once at the end,
// which is more accurate and therefore different -- measured against the
// recorded q/k it disagrees on 35.6% of elements. So the rounds below are not to
// be simplified into a single conversion at the end.
//
// # Layout
//
// Pairs are half-split -- `(i, i + head_dim / 2)` -- not interleaved. The
// frequency is `theta^(-pair / half_dim)`, read from the table described above
// (the CUDA kernel and the bridge evaluate it inline; the values are the same),
// and the
// angle is built in f32 from the integer position just as the bridge builds it
// from an f32 arange. V is not rotated; it only takes the bias.
//
// # Launch
//
// `grid(tokens, q_heads + 2 * kv_heads)`, `block(head_dim / 2)`. One thread per
// pair, so `blockDim.x == half_dim` exactly; the bounds check is for the
// launcher's benefit rather than the kernel's.
__global__ void qkv_rope_bf16_kernel(const __mt_bfloat16* qkv,
                                     const __mt_bfloat16* bias,
                                     __mt_bfloat16* q, __mt_bfloat16* k,
                                     __mt_bfloat16* v, int q_heads, int kv_heads,
                                     int head_dim, const float* frequency,
                                     int position_offset,
                                     int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half_dim) return;

  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  // `frequency[pair]` is `powf(theta, -pair / half_dim)`, evaluated once into a
  // table instead of once per (token, head, pair).
  //
  // It depends only on `pair`, so the grid above recomputed the same 128 values
  // 5220 times on the prefix shape. Measured with `rope_variants.cu`, that one
  // call was **80% of this kernel**: 0.935 -> 0.184 ms per call at `[522, ...]`.
  // The table is filled by `rope_frequency_kernel` below, on the device, from
  // the same `powf` the kernel used to call -- so the values are identical and
  // this changes no numbers. A host-side table would not be: `math.cuh`'s note
  // about device and host transcendentals differing applies here too.
  const float rate = frequency[pair];
  const float cosine = round_bf16(cosf(static_cast<float>(position) * rate));
  const float sine = round_bf16(sinf(static_cast<float>(position) * rate));

  // The packed projection and its bias share one layout -- q | k | v, each
  // segment `heads * head_dim` wide -- so one offset serves both.
  int segment;
  if (projection_head < q_heads) {
    segment = projection_head * head_dim;
  } else if (projection_head < q_heads + kv_heads) {
    segment = q_width + (projection_head - q_heads) * head_dim;
  } else {
    segment =
        q_width + kv_width + (projection_head - q_heads - kv_heads) * head_dim;
  }

  // The flat offsets are 64-bit even though every shape PI0.5 reaches fits in
  // 32: `tokens` is only bounded by the tensor a caller passes, and at the
  // action expert's width a token count of ~840k makes a 32-bit `source`
  // negative -- a read before `qkv` rather than a wrong answer. The sibling
  // kernels in this port (`add_position`, `embedding`, `geglu`,
  // `qkv_split_bias`) all widen the same way. The CUDA original does not.
  //
  // `kv_output_offset + token` below stays 32-bit on purpose: that sum is a
  // row index into the cache, so it is bounded by an allocation that exists --
  // passing 2^31 would need a 32 GB cache. The products above have no such
  // bound, which is the difference.
  const int64_t source = static_cast<int64_t>(token) * fused_width + segment;
  float first = __bfloat162float(qkv[source + pair]);
  float second = __bfloat162float(qkv[source + half_dim + pair]);
  if (bias != nullptr) {
    first += __bfloat162float(bias[segment + pair]);
    second += __bfloat162float(bias[segment + half_dim + pair]);
  }

  if (projection_head < q_heads) {
    // Q is per-token output: `[tokens, q_heads, head_dim]`, no cache involved.
    const int64_t destination =
        (static_cast<int64_t>(token) * q_heads + projection_head) * head_dim;
    rope_pair_bf16(round_bf16(first), round_bf16(second), cosine, sine,
                   q + destination, half_dim, pair);
    return;
  }

  const bool is_key = projection_head < q_heads + kv_heads;
  const int head = is_key ? projection_head - q_heads
                          : projection_head - q_heads - kv_heads;
  const int64_t destination =
      (static_cast<int64_t>(kv_output_offset + token) * kv_heads + head) * head_dim;
  __mt_bfloat16* target = is_key ? k : v;
  if (is_key) {
    rope_pair_bf16(round_bf16(first), round_bf16(second), cosine, sine,
                   target + destination, half_dim, pair);
  } else {
    // V is carried through unrotated: the bias add rounds once and nothing else
    // happens to it, exactly as `_write(t["v"], v_block)` does.
    target[destination + pair] = __float2bfloat16(first);
    target[destination + half_dim + pair] = __float2bfloat16(second);
  }
}
