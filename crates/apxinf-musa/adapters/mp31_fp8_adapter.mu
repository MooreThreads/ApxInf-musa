// MP31 static-FP8 preparation and native fused epilogues.
#include "adapter_common.h"
#include "mp31_fp8/geglu.cuh"
#include "mp31_fp8/projection.cuh"
#include "mp31_fp8/projection_norm.cuh"
#include "../kernels/custom/rope.cuh"
#include "mp31_fp8/direct.cuh"
#include <cmath>
#include <map>
#include <new>
#include <tuple>

namespace {
thread_local std::map<std::tuple<int, int, float>, const float *> g_mp31_rope_frequency;

const float *mp31_rope_frequency(int half_dim, float theta, musaStream_t stream) {
  const int device = apxinf_adapters::stream_device(stream);
  const auto key = std::make_tuple(device, half_dim, theta);
  const auto found = g_mp31_rope_frequency.find(key);
  if (found != g_mp31_rope_frequency.end())
    return found->second;
  if (apxinf_adapters::capturing(stream))
    return nullptr;
  if (!apxinf_adapters::select_device(device))
    return nullptr;
  float *table = nullptr;
  if (musaMalloc(reinterpret_cast<void **>(&table), size_t(half_dim) * sizeof(float)) != musaSuccess)
    return nullptr;
  const int threads = 256;
  rope_frequency_kernel<<<(half_dim + threads - 1) / threads, threads, 0, stream>>>(
      table, half_dim, theta);
  if (musaGetLastError() != musaSuccess) {
    musaFree(table);
    return nullptr;
  }
  g_mp31_rope_frequency.emplace(key, table);
  return table;
}

struct Weight {
  MUtensorDescriptor descriptor;
  int k, n;
  uint8_t *direct_weight = nullptr;
  std::map<const void *, MUtensorDescriptor> activations;
};
musaError_t descriptor(MUtensorDescriptor *d, const void *p, int rows, int cols) {
  uint64_t dims[] = {uint64_t(cols), uint64_t(rows)}, strides[] = {uint64_t(cols)};
  return muTensorDescriptorEncode(d, MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8, 2, const_cast<void *>(p),
                                  dims, strides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE,
                                  0) == MUSA_SUCCESS
             ? musaSuccess
             : musaErrorInvalidValue;
}
} // namespace

// The action qkv weight is consumed only by the fused direct producer, so it
// skips the transposed TME copy and carries just the 16x32 register packing.
extern "C" musaError_t apxinf_musa_mp31_prepare_fp8(const void *in, void *packed, void *direct_packed,
                                                    int k, int n, void **handle,
                                                    musaStream_t stream) {
  const bool direct_only = k == 1024 && n == 2560;
  if (!in || !handle || (!packed && !direct_packed) ||
      direct_only != (direct_packed != nullptr) ||
      !((k == 1024 && (n == 2560 || n == 8192)) ||
        (n == 1024 && (k == 2048 || k == 4096))))
    return musaErrorInvalidValue;
  *handle = nullptr;
  auto *w = new (std::nothrow) Weight{};
  if (!w)
    return musaErrorMemoryAllocation;
  w->k = k;
  w->n = n;
  if (packed != nullptr) {
    auto status = descriptor(&w->descriptor, packed, n, k);
    if (status != musaSuccess) {
      delete w;
      return status;
    }
    apxinf_mp31::transpose_e4m3<<<dim3((n + 31) / 32, (k + 31) / 32), dim3(32, 8), 0, stream>>>(
        (const uint8_t *)in, (uint8_t *)packed, k, n);
    status = musaGetLastError();
    if (status != musaSuccess) {
      delete w;
      return status;
    }
  }
  if (direct_packed != nullptr) {
    w->direct_weight = static_cast<uint8_t *>(direct_packed);
    apxinf_mp31::pack_direct_weight<<<(n / 16) * (k / 32), 256, 0, stream>>>(
        (const uint8_t *)in, w->direct_weight, n, k);
    const auto status = musaGetLastError();
    if (status != musaSuccess) {
      delete w;
      return status;
    }
  }
  *handle = w;
  return musaSuccess;
}
extern "C" void apxinf_musa_mp31_release_fp8(void *handle) { delete static_cast<Weight *>(handle); }

extern "C" musaError_t apxinf_musa_mp31_fp8_geglu(const void *a, const void *handle, void *out,
                                                  int m, float alpha, float scale,
                                                  musaStream_t stream) {
  if (!a || !handle || !out || m != 10 || !std::isfinite(alpha) || !std::isfinite(scale) ||
      scale <= 0)
    return musaErrorInvalidValue;
  auto *w = static_cast<Weight *>(const_cast<void *>(handle));
  if (alpha == 0.f)
    return musaMemsetAsync(out, 0, size_t(m) * w->n / 2, stream);
  auto it = w->activations.find(a);
  if (it == w->activations.end()) {
    if (apxinf_adapters::capturing(stream))
      return musaErrorStreamCaptureUnsupported;
    MUtensorDescriptor ad{};
    auto status = descriptor(&ad, a, m, w->k);
    if (status != musaSuccess)
      return status;
    it = w->activations.emplace(a, ad).first;
  }
  apxinf_mp31::fp8_geglu<256, 1><<<w->n / 64, 256, 80 * 256, stream>>>(
      it->second, w->descriptor, (uint8_t *)out, m, w->n, w->k, alpha, 1.f / scale);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_mp31_fp8_qkv_rope(
    const void *a, const void *handle, const void *bias, void *q, void *k, void *v,
    int q_heads, int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, float alpha, musaStream_t stream) {
  if (!a || !handle || !q || !k || !v || q_heads != 8 || kv_heads != 1 || head_dim != 256 ||
      !std::isfinite(theta) || theta <= 0.f || position_offset < 0 || kv_output_offset < 0 ||
      !std::isfinite(alpha))
    return musaErrorInvalidValue;
  auto *w = static_cast<Weight *>(const_cast<void *>(handle));
  if (w->n != 2560 || w->k != 1024 || w->direct_weight == nullptr)
    return musaErrorInvalidValue;
  const float *frequency = mp31_rope_frequency(128, theta, stream);
  if (frequency == nullptr)
    return apxinf_adapters::capturing(stream) ? musaErrorStreamCaptureUnsupported
                                              : musaErrorMemoryAllocation;
  apxinf_mp31::direct_qkv_rope<16><<<80, 512, 0, stream>>>(
      (const uint8_t *)a, w->direct_weight, (const __mt_bfloat16 *)bias,
      (__mt_bfloat16 *)q, (__mt_bfloat16 *)k, (__mt_bfloat16 *)v, frequency,
      position_offset, kv_output_offset, alpha);
  return musaGetLastError();
}

extern "C" musaError_t apxinf_musa_mp31_fp8_projection_norm(
    const void *a, const void *handle, float *partial, const void *residual, const void *gate,
    const void *style, void *hidden, void *out, int m, int bf16_input, float inverse_input_scale,
    float alpha, float output_scale, float eps, musaStream_t stream) {
  if (!a || !handle || !partial || !residual || !gate || !style || !hidden || !out || m != 10 ||
      !std::isfinite(alpha) || !std::isfinite(output_scale) || output_scale <= 0.f ||
      !std::isfinite(1.f / output_scale) || !std::isfinite(eps) || eps < 0.f ||
      (bf16_input && (!std::isfinite(inverse_input_scale) || inverse_input_scale <= 0.f)))
    return musaErrorInvalidValue;
  auto *w = static_cast<Weight *>(const_cast<void *>(handle));
  if (w->n != 1024 || (w->k != 2048 && w->k != 4096))
    return musaErrorInvalidValue;
  MUtensorDescriptor ad{};
  if (!bf16_input) {
    auto it = w->activations.find(a);
    if (it == w->activations.end()) {
      if (apxinf_adapters::capturing(stream))
        return musaErrorStreamCaptureUnsupported;
      auto status = descriptor(&ad, a, m, w->k);
      if (status != musaSuccess)
        return status;
      it = w->activations.emplace(a, ad).first;
    }
    ad = it->second;
  }
  if (bf16_input)
    apxinf_mp31::partial_gemm<64, 256, 1, 8, true>
        <<<dim3(w->n / 64, 8), 256, 80 * 256, stream>>>(
            ad, w->descriptor, (const apxinf_mp31::Bf16 *)a, partial, m, w->n, w->k,
            inverse_input_scale);
  else if (w->k == 2048)
    apxinf_mp31::e4m3_projection_partial<1><<<dim3(w->n / 64, 8), 128, 80 * 256, stream>>>(
        ad, w->descriptor, partial, m, w->n);
  else
    apxinf_mp31::e4m3_projection_partial<2><<<dim3(w->n / 64, 8), 128, 80 * 512, stream>>>(
        ad, w->descriptor, partial, m, w->n);
  auto status = musaGetLastError();
  if (status != musaSuccess)
    return status;
  apxinf_mp31::partial_norm<8>
      <<<m, 256, 0, stream>>>(partial, (const __mt_bfloat16 *)residual,
                              (const __mt_bfloat16 *)gate, (const __mt_bfloat16 *)style,
                              (__mt_bfloat16 *)hidden, (uint8_t *)out, m, w->n, eps,
                              1.f / output_scale, alpha);
  return musaGetLastError();
}
