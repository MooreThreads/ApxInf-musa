#include "../kernels/mp31_attention/attention.cuh"
#include <musa_runtime.h>
#include <cmath>
#include <new>

namespace {
struct Descriptors {
  MUtensorDescriptor query, key, value;
};
static musaError_t encode_bf16(const void *data, int rows, int columns,
                               MUtensorDescriptor *out) {
  uint64_t dimensions[] = {uint64_t(columns), uint64_t(rows)};
  uint64_t strides[] = {uint64_t(columns * 2)};
  auto status = muTensorDescriptorEncode(
      out, MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16, 2, const_cast<void *>(data),
      dimensions, strides, MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE, 0);
  return status == MUSA_SUCCESS ? musaSuccess : musaErrorInvalidValue;
}
} // namespace
extern "C" musaError_t apxinf_musa_mp31_attention_prepare() {

  musaFuncAttributes attributes{};
  return musaFuncGetAttributes(&attributes,
                               apxinf_mp31_attention::attention_partials<64, 5>);

}
extern "C" musaError_t apxinf_musa_mp31_attention_descriptors(const void *query,
                                                              const void *key,
                                                              const void *value,
                                                              void **result) {
  if (!query || !key || !value || !result)
    return musaErrorInvalidValue;
  auto *descriptors = new (std::nothrow) Descriptors;
  if (!descriptors)
    return musaErrorMemoryAllocation;
  auto status = encode_bf16(query, 10, 8 * 256, &descriptors->query);
  if (status == musaSuccess)
    // The tensors contain 532 KV rows. Let TME zero-fill the final tile;
    // exposing 576 rows reads beyond the allocation and can propagate NaNs.
    status = encode_bf16(key, 532, 256, &descriptors->key);
  if (status == musaSuccess)
    status = encode_bf16(value, 532, 256, &descriptors->value);
  if (status != musaSuccess) {
    delete descriptors;
    return status;
  }
  *result = descriptors;
  return musaSuccess;
}
extern "C" void apxinf_musa_mp31_attention_release(void *descriptors) {
  delete static_cast<Descriptors *>(descriptors);
}
extern "C" musaError_t apxinf_musa_mp31_attention(const void *raw_descriptors,
                                                  float *partials, float *stats,
                                                  void *output, float inverse_output_scale,
                                                  musaStream_t stream) {

  if (!raw_descriptors || !partials || !stats || !output ||
      !std::isfinite(inverse_output_scale) || inverse_output_scale < 0.f)
    return musaErrorInvalidValue;
  auto const &descriptors = *static_cast<const Descriptors *>(raw_descriptors);
  // The first grid axis is the physical query head; the second is the KV split.
  // Five partitions are the validated default; the Rust workspace is sized for
  // the same split, so the two constants must stay equal.
  apxinf_mp31_attention::attention_partials<64, 5>
      <<<dim3(8, 5), 256, (16 * 256 + 2 * 64 * 512 + 16 * 64) * 2, stream>>>(
          descriptors.query, descriptors.key, descriptors.value,
          static_cast<mutlass::bfloat16_t *>(output), partials, stats, 10, 532);
  auto status = musaGetLastError();
  if (status != musaSuccess)
    return status;
  // A positive inverse scale writes the merged row as E4M3 through the next
  // projection's own input quantizer; zero keeps the BF16 contract.
  if (inverse_output_scale > 0.f)
    apxinf_mp31_attention::merge_partials<5, true><<<80, 256, 0, stream>>>(
        partials, stats, output, 10, inverse_output_scale);
  else
    apxinf_mp31_attention::merge_partials<5, false>
        <<<80, 256, 0, stream>>>(partials, stats, output, 10, 0.f);
  return musaGetLastError();

}
