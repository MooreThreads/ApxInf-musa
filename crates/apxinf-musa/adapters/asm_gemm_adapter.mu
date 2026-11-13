#include <musa.h>
#include <musa_runtime.h>

#include <climits>
#include <cstdint>
#include <map>
#include <utility>

#include "adapter_common.h"
#include "mubin_images.h"

namespace {

using namespace apxinf_adapters;

enum class DType : int {
  Bf16,
};

enum class Layout : int {
  Nt,
  Nn,
};

enum class Arch : int {
  Mp22,
};

enum class Schedule : int {
  Standard,
  WideMFast,
  NormalMFast,
  WideNFast,
};

// These names are the public strategy IDs. They are intentionally separate
// from the image array index, so regenerating image ordering does not change
// the Rust-facing meaning of a kernel choice.
enum class KernelId : int {
  Mp22Bf16Nt128_128x128,
  Mp22Bf16Nt128_128x64,
  Mp22Bf16Nt128_32x128,
  Mp22Bf16Nt256_256x128,
  Mp22Bf16Nt256_256x64,
  Mp22Bf16Nt512_256x256,
  Mp22Bf16Nn128_128x128,
  Mp22Bf16Nn128_128x64,
  Mp22Bf16Nn128_32x128,
  Mp22Bf16Nn256_256x128,
  Mp22Bf16Nn256_256x64,
  Mp22Bf16Nn512_256x256,
  Mp22Bf16Nn128_128x128WideMFast,
  Mp22Bf16Nn256_128x256WideMFast,
  Mp22Bf16Nn128_128x128NormalMFast,
  Mp22Bf16Nn256_256x128WideNFast,
};

struct KernelInfo {
  Arch arch;
  KernelId id;
  DType dtype;
  Layout layout;
  Schedule schedule;
  const unsigned char* image;
  unsigned long image_size;
  const char* symbol;
  int tile_m;
  int tile_n;
  int threads;
  int k_alignment;
};

// The table is the only place that maps a named strategy to an image and its
// launch metadata. The image symbols still come from the generated header.
const KernelInfo kKernelTable[] = {
    {Arch::Mp22, KernelId::Mp22Bf16Nt128_128x128, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt128_128x128, kMp22Bf16Nt128_128x128Size,
     "asm_bf16bf16gemm_nt_tc_128_128x128_epilogue", 128, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nt128_128x64, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt128_128x64, kMp22Bf16Nt128_128x64Size,
     "asm_bf16bf16gemm_nt_tc_128_128x64_epilogue", 128, 64, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nt128_32x128, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt128_32x128, kMp22Bf16Nt128_32x128Size,
     "asm_bf16bf16gemm_nt_tc_128_32x128_epilogue", 32, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nt256_256x128, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt256_256x128, kMp22Bf16Nt256_256x128Size,
     "asm_bf16bf16gemm_nt_tc_256_256x128_epilogue", 256, 128, 256, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nt256_256x64, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt256_256x64, kMp22Bf16Nt256_256x64Size,
     "asm_bf16bf16gemm_nt_tc_256_256x64_epilogue", 256, 64, 256, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nt512_256x256, DType::Bf16, Layout::Nt,
     Schedule::Standard, kMp22Bf16Nt512_256x256, kMp22Bf16Nt512_256x256Size,
     "asm_bf16bf16gemm_nt_tc_512_256x256_epilogue", 256, 256, 512, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn128_128x128, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn128_128x128, kMp22Bf16Nn128_128x128Size,
     "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue", 128, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn128_128x64, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn128_128x64, kMp22Bf16Nn128_128x64Size,
     "asm_bf16bf16gemm_nn_tc_128_128x64_epilogue", 128, 64, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn128_32x128, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn128_32x128, kMp22Bf16Nn128_32x128Size,
     "asm_bf16bf16gemm_nn_tc_128_32x128_epilogue", 32, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn256_256x128, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn256_256x128, kMp22Bf16Nn256_256x128Size,
     "asm_bf16bf16gemm_nn_tc_256_256x128_epilogue", 256, 128, 256, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn256_256x64, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn256_256x64, kMp22Bf16Nn256_256x64Size,
     "asm_bf16bf16gemm_nn_tc_256_256x64_epilogue", 256, 64, 256, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn512_256x256, DType::Bf16, Layout::Nn,
     Schedule::Standard, kMp22Bf16Nn512_256x256, kMp22Bf16Nn512_256x256Size,
     "asm_bf16bf16gemm_nn_tc_512_256x256_epilogue", 256, 256, 512, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn128_128x128WideMFast, DType::Bf16, Layout::Nn,
     Schedule::WideMFast, kMp22Bf16Nn128_128x128WideMFast, kMp22Bf16Nn128_128x128WideMFastSize,
     "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_wide_mfast", 128, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn256_128x256WideMFast, DType::Bf16, Layout::Nn,
     Schedule::WideMFast, kMp22Bf16Nn256_128x256WideMFast, kMp22Bf16Nn256_128x256WideMFastSize,
     "asm_bf16bf16gemm_nn_tc_256_128x256_epilogue_wide_mfast", 128, 256, 256, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn128_128x128NormalMFast, DType::Bf16, Layout::Nn,
     Schedule::NormalMFast, kMp22Bf16Nn128_128x128NormalMFast, kMp22Bf16Nn128_128x128NormalMFastSize,
     "asm_bf16bf16gemm_nn_tc_128_128x128_epilogue_normal_mfast", 128, 128, 128, 32},
    {Arch::Mp22, KernelId::Mp22Bf16Nn256_256x128WideNFast, DType::Bf16, Layout::Nn,
     Schedule::WideNFast, kMp22Bf16Nn256_256x128WideNFast, kMp22Bf16Nn256_256x128WideNFastSize,
     "asm_bf16bf16gemm_nn_tc_256_256x128_epilogue_wide_nfast", 256, 128, 256, 32},
};

thread_local std::map<std::pair<MUcontext, int>, MUfunction> functions;

const KernelInfo* find_kernel(Arch arch, DType dtype, Layout layout,
                              KernelId id) {
  for (const KernelInfo& info : kKernelTable) {
    if (info.arch == arch && info.id == id && info.dtype == dtype &&
        info.layout == layout) {
      return &info;
    }
  }
  return nullptr;
}

MUfunction load(const KernelInfo& info, KernelId id, void* stream) {
  MUcontext context = nullptr;
  if (muCtxGetCurrent(&context) != MUSA_SUCCESS || !context) {
    set_error("ASM GEMM: no current MUSA context");
    return nullptr;
  }
  const auto key = std::make_pair(context, static_cast<int>(id));
  auto found = functions.find(key);
  if (found != functions.end()) return found->second;
  if (capturing(stream)) {
    set_error("ASM GEMM must be prepared before graph capture");
    return nullptr;
  }
  MUmodule module = nullptr;
  if (muModuleLoadData(&module, info.image) != MUSA_SUCCESS) {
    set_error("ASM GEMM module load failed");
    return nullptr;
  }
  MUfunction function = nullptr;
  if (muModuleGetFunction(&function, module, info.symbol) != MUSA_SUCCESS) {
    muModuleUnload(module);
    set_error("ASM GEMM symbol lookup failed");
    return nullptr;
  }
  functions.emplace(key, function);
  return function;
}

__global__ void transpose_weight(const uint16_t* in, uint16_t* out, int k, int n) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < size_t(n)*k) out[i] = in[(i%k)*n+i/k];
}

int ceil_div(int value, int tile) { return (value + tile - 1) / tile; }

}  // namespace

extern "C" int apxinf_musa_asm_gemm_plan(
    const void* a, const void* b, void* out, int m, int n, int k, int arch,
    int dtype, int layout, int kernel_id, void* stream) {
  if (!a || !b || !out || m <= 0 || n <= 0 || k <= 0 ||
      int64_t(m) * k * 2 > INT_MAX || int64_t(n) * k * 2 > INT_MAX ||
      int64_t(m) * n * 2 > INT_MAX) {
    set_error("ASM GEMM plan: invalid shape or buffer exceeds int32 ABI");
    return -1;
  }
  const auto plan_arch = static_cast<Arch>(arch);
  const auto plan_dtype = static_cast<DType>(dtype);
  const auto plan_layout = static_cast<Layout>(layout);
  const auto plan_kernel = static_cast<KernelId>(kernel_id);
  const KernelInfo* info =
      find_kernel(plan_arch, plan_dtype, plan_layout, plan_kernel);
  if (!info) {
    set_error("ASM GEMM plan: unknown kernel plan");
    return -1;
  }
  if (k % info->k_alignment) {
    set_error("ASM GEMM plan: K does not satisfy kernel alignment");
    return -1;
  }
  if (!info->image_size) {
    set_error("ASM GEMM plan: kernel image is unavailable in this build");
    return -1;
  }
  if (!select_device(stream_device(stream))) {
    set_error("ASM GEMM plan: cannot select stream device");
    return -1;
  }
  MUfunction function = load(*info, plan_kernel, stream);
  if (!function) return -1;

  int out_bytes = m * n * 2;
  int a_bytes = m * k * 2;
  int b_bytes = n * k * 2;
  int zero = 0;
  int batch = 1;
  int stride_a = m * k;
  int stride_b = n * k;
  int stride_c = m * n;
  int ldb = plan_layout == Layout::Nt ? k : n;
  float alpha = 1.f, beta = 0.f, gamma = 0.f;
  void* nil = nullptr;
  void* args[] = {
      &out, &out_bytes, &zero, &a, &a_bytes, &zero, &b, &b_bytes, &zero,
      &nil, &zero, &zero, &nil, &zero, &zero, &nil, &zero, &zero,
      &nil, &zero, &zero, &m, &n, &k, &k, &ldb, &n, &alpha, &beta,
      &gamma, &batch, &stride_a, &stride_b, &stride_c};

  const int grid_m = ceil_div(m, info->tile_m);
  const int grid_n = ceil_div(n, info->tile_n);
  const bool m_fast = info->schedule == Schedule::WideMFast ||
                      info->schedule == Schedule::NormalMFast;
  auto status = muLaunchKernel(
      function, m_fast ? grid_m : grid_n, m_fast ? grid_n : grid_m, 1,
      info->threads, 1, 1, 0,
      reinterpret_cast<MUstream>(stream), args, nullptr);
  if (status != MUSA_SUCCESS) {
    set_error("ASM GEMM plan launch failed");
    return -1;
  }
  return 0;
}

extern "C" const char* apxinf_musa_asm_gemm_last_error() { return error_buffer(); }
extern "C" int apxinf_musa_asm_transpose(const void* in, void* out, int k, int n, void* stream) {
    if (!in || !out || k<=0 || n<=0 || int64_t(k)*n*2>INT_MAX || capturing(stream)) {
        set_error("ASM weight transpose: invalid shape or called during capture"); return -1;
    }
    if (!select_device(stream_device(stream))) {
        set_error("ASM weight transpose: cannot select stream device"); return -1;
    }
    transpose_weight<<<(size_t(n)*k+255)/256,256,0,static_cast<musaStream_t>(stream)>>>(
        static_cast<const uint16_t*>(in),static_cast<uint16_t*>(out),k,n);
    auto status=musaGetLastError();
    if (status!=musaSuccess) { set_error("ASM weight transpose: %s",musaGetErrorString(status)); return -1; }
    return 0;
}
