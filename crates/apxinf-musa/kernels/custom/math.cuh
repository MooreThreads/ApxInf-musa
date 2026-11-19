#pragma once

#include <cstdint>
#include <musa_fp8.h>

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Shared device-side math helpers for custom MUSA operators.
//
// A port of `crates/apxinf-cuda/kernels/custom/math.cuh`, and the definitions
// have to stay bit-compatible with it: these are the functions the eager bridge
// also spelled out in Python (the torch bridge that recorded the reference; it
// is not part of this tree), and the recorded reference tensors were produced
// by that spelling. `gelu_tanh` here is the tanh approximation, not erf --
// `_gelu_tanh` in the bridge says so explicitly, and the two must not drift
// apart.

constexpr int kThreads = 256;

// The highest `gridDim.y` a launch may use. Kernels that put the row on
// `gridDim.y` loop `row += gridDim.y` when there are more rows than that, so
// the limit costs occupancy at worst rather than coverage.
constexpr int kMaxGridY = 65535;

// Fast transcendentals are a property of one generation, not of this port.
//
// `__musa_exp_f`, `__musa_tanh_f` and `__sincosf` are the MP31 (S5000/S4000)
// fast paths. On MP22 (M1000) the substitution that introduced them broke the
// PI0.5 and GR00T numerics (5b7d172); reverting the group restored the Orin
// gate. The vendored `mutlass::fast_tanh` already spells the rule as
// `#if __MUSA_ARCH__ >= 310`; keep it in one place so a shared call site cannot
// pick a generation-specific intrinsic by accident.
//
// `__MUSA_ARCH__` is defined only on the device pass (220 = mp_22, 310 = mp_31).
__device__ __forceinline__ float apxinf_exp_f(float value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
  return __musa_exp_f(value);
#else
  return expf(value);
#endif
}

__device__ __forceinline__ float apxinf_tanh_f(float value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
  return __musa_tanh_f(value);
#else
  return tanhf(value);
#endif
}

__device__ __forceinline__ void apxinf_sincosf(float value, float* sine, float* cosine) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
  __sincosf(value, sine, cosine);
#else
  *sine = sinf(value);
  *cosine = cosf(value);
#endif
}

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + apxinf_tanh_f(kAlpha * (value + 0.044715f * value * value * value)));
}

// The static-FP8 quantizer's per-value arithmetic, shared by the standalone
// kernel and every producer that writes E4M3 directly.
//
// Byte-for-byte contract with `quantize_bf16_e4m3_kernel`: the caller passes
// the value *after* the BF16 rounding the composed path would have applied,
// and this multiplies by `1 / scale`, clamps to the E4M3 range, and converts
// with round-to-nearest-even. NaN clamps to -448 by the clamp order, as the
// standalone kernel documents.
__device__ __forceinline__ uint8_t quantize_e4m3_value(float value,
                                                       float inverse_scale) {
  value *= inverse_scale;
  value = fminf(448.0f, fmaxf(-448.0f, value));
  return __musa_cvt_float_to_fp8(value, __MT_SATFINITE, __MT_E4M3);
}
