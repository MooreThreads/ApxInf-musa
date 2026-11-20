struct alignas(16) Fp8SourcePack { __mt_bfloat16 values[8]; };
struct alignas(8) Fp8OutputPack { __mt_fp8_e4m3 values[8]; };

__global__ void quantize_bf16_e4m3_kernel(const __mt_bfloat16* input,
    __mt_fp8_e4m3* output, size_t count, float inverse_scale) {
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const size_t stride = size_t(gridDim.x) * blockDim.x * 8;
  for (size_t offset = start; offset < count; offset += stride) {
    if (offset + 8 <= count) {
      const Fp8SourcePack source = *reinterpret_cast<const Fp8SourcePack*>(input + offset);
      Fp8OutputPack destination;
      #pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        destination.values[lane] = __mt_fp8_e4m3(
            fminf(448.0f, fmaxf(-448.0f, float(source.values[lane]) * inverse_scale)));
      }
      *reinterpret_cast<Fp8OutputPack*>(output + offset) = destination;
    } else {
      for (size_t index = offset; index < count; ++index) {
        output[index] = __mt_fp8_e4m3(
            fminf(448.0f, fmaxf(-448.0f, float(input[index]) * inverse_scale)));
      }
    }
  }
}

// Concatenate two equally tall BF16 matrices along columns and quantize the
// result in the same pass. The consumer supplies the static per-tensor scale;
// no temporary BF16 concatenation buffer is materialized.
__global__ void concat_columns_quantize_bf16_e4m3_kernel(
    const __mt_bfloat16* first, const __mt_bfloat16* second,
    __mt_fp8_e4m3* output, int rows, int first_cols, int second_cols,
    float inverse_scale) {
  const int total_cols = first_cols + second_cols;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t output_base = static_cast<int64_t>(row) * total_cols;
    const int64_t first_base = static_cast<int64_t>(row) * first_cols;
    const int64_t second_base = static_cast<int64_t>(row) * second_cols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < total_cols;
         col += stride) {
      const float value = col < first_cols
          ? float(first[first_base + col])
          : float(second[second_base + col - first_cols]);
      output[output_base + col] = __mt_fp8_e4m3(
          fminf(448.0f, fmaxf(-448.0f, value * inverse_scale)));
    }
  }
}

// Concatenate two equally wide BF16 matrices along rows and quantize the
// result in the same pass. This is the layout produced by segmented vision
// attention before its output projection.
__global__ void concat_rows_quantize_bf16_e4m3_kernel(
    const __mt_bfloat16* first, const __mt_bfloat16* second,
    __mt_fp8_e4m3* output, int first_rows, int second_rows, int cols,
    float inverse_scale) {
  const int total_rows = first_rows + second_rows;
  const int stride = blockDim.x * gridDim.x;
  for (int row = blockIdx.y; row < total_rows; row += gridDim.y) {
    const __mt_bfloat16* source = row < first_rows
        ? first + static_cast<int64_t>(row) * cols
        : second + static_cast<int64_t>(row - first_rows) * cols;
    __mt_fp8_e4m3* destination = output + static_cast<int64_t>(row) * cols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols;
         col += stride) {
      destination[col] = __mt_fp8_e4m3(
          fminf(448.0f, fmaxf(-448.0f, float(source[col]) * inverse_scale)));
    }
  }
}
