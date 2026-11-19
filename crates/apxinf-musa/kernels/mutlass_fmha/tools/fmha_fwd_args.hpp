// The C ABI of the three frozen mp_31 FMHA instantiations.
//
// One declaration per frozen kernel, because each is defined in its own
// translation unit: `doc/musa-backend.md` records that a recompiled TU can move
// an unrelated kernel's bits, and MATE's own build compiles one `.mu` per
// shared object. Keeping the split reproduces that arrangement exactly, which
// is what makes the replay against MATE's output a meaningful comparison.
//
// Both return null on success and a static message otherwise, so the caller
// reads the reason through the return value rather than a second call.
#pragma once

#include <cstdint>

/// The packed-QKV sibling of `apxinf_musa_mp31_fmha_fwd_dit_e4m3`: `q`, `k`
/// and `v` point at columns 0/1536/3072 of one 4608-wide BF16 row. The kernel's
/// row stride is `3 * heads * head_dim`, so the self-attention's fused QKV
/// projection feeds it without a split.
extern "C" const char* apxinf_musa_mp31_fmha_fwd_dit_e4m3_packed(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    float output_scale,
    void* stream);


/// GR00T vision attention: D=64, 16 heads, batch=2 x 256 tokens, non-causal,
/// E4M3 output with the vision output projection's activation scale folded in.
extern "C" const char* apxinf_musa_mp31_fmha_fwd_vision_e4m3(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    float output_scale,
    void* stream);

/// GR00T DiT attention: `TileM=64`, `HeadDimQK=64`, `HeadDimVO=48`, 32 heads,
/// D=48, non-causal, E4M3 output. `output_scale` is `1/activation_scale`; the
/// epilogue multiplies the FP32 accumulator by it before the E4M3 conversion.
extern "C" const char* apxinf_musa_mp31_fmha_fwd_dit_e4m3(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    float output_scale,
    void* stream);

extern "C" const char* apxinf_musa_mp31_fmha_fwd_tile128_packed(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    void* stream);

extern "C" const char* apxinf_musa_mp31_fmha_fwd_tile192_general(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    void* stream);

/// The unpacked decode: `TileM=16`, `IsPackGQA=false`, `IsSplit=false`.
///
/// MATE's packed decode puts all eight query heads in one TileM=128 tile, so a
/// `[10, 532]` call is one CTA. This instantiation keeps the same tensor-core
/// mainloop but gives each head its own block: the scheduler's head axis is
/// `params.h = 8`, so the grid is `(1, 8, 1)`.
extern "C" const char* apxinf_musa_mp31_fmha_fwd_tile16_unpacked(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    void* stream);
