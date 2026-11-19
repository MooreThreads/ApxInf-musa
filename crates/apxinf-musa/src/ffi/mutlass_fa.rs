//! Raw bindings for the vendored mp_31 D256 FMHA kernels' C ABI.
//!
//! The symbols are defined in `adapters/mutlass_fa_adapter.mu`, which owns the
//! metadata cache and the launch validation; the device code it drives is
//! `kernels/mutlass_fmha/fmha_fwd/*.mu`, vendored from MATE. As in `ffi::musa`,
//! only the symbols ApxInf calls are declared, by hand.
//!
//! Unlike `ffi::custom`, the definitions are absent from most builds: the
//! vendored tree is compiled only for mp_31 on a toolkit the render was
//! measured against (`build.rs` gates it, and `kernels::attention` gates the
//! call sites on the same `apxinf_musa_mutlass_fmha` cfg). An unreferenced
//! `extern` declaration costs nothing at link time, so the declarations here
//! are unconditional while the calls are not.
//!
//! This file is also one of the kernel-build-id inputs (`build.rs`'s
//! `KERNEL_INPUT_FILES`), so a signature change here re-keys the build even
//! when no `.mu` moved.

use std::ffi::c_void;

/// Which vendored instantiation of the D256 attention to run.
///
/// MATE's own rule picks between the packed tile 128 and the general tile 192
/// from the query length. `kernels::attention::mp31_fa_variant` applies that
/// rule plus two port overrides: 9-16 rows default to the unpacked tile 16, and
/// 17-256 rows use the unpacked tile 192 in place of the packed one. They are
/// variants rather than an automatic choice here for the same reason the MP22
/// assembly has three: which kernel ran has to be legible in a report, not
/// inferred from a shape.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutlassFaVariant {
    /// MATE 0.2.7 config `fmha_fwd_4386cd8f…`: `TileM` 128, packed GQA, two
    /// QK/PV consumer warpgroups. MATE's tile rule selects it for 9 to 16 query
    /// rows, which is PI0.5's ten-row action pass; this port defaults those rows
    /// to `UnpackedDecodeTile16` and keeps this variant behind
    /// `APXINF_MUSA_FA_UNPACKED=0`.
    PackedDecodeTile128 = 0,
    /// MATE 0.2.7 config `fmha_fwd_be3f5142…`: `TileM` 192, unpacked, three
    /// consumer warpgroups. MATE selects it for 257 or more query rows, which is
    /// PI0.5's prefix pass; this port also uses it for 17-256 rows in place of
    /// the packed tile. Every longer prefix lands on the same instantiation.
    GeneralTile192 = 1,
    /// MATE 0.2.7 config `fmha_fwd_c225fad4…`: `TileM` 16, unpacked, one
    /// consumer warpgroup. Selected by default for 9-16 query rows. Same
    /// arithmetic as the packed decode, but the scheduler's head axis is the
    /// eight query heads, so the grid is `(1, 8, 1)` instead of `(1, 1, 1)`.
    UnpackedDecodeTile16 = 2,
}

extern "C" {
    /// Packed BF16 non-causal attention over a dense BF16 KV.
    ///
    /// `q` and `out` are `[tokens, 8, 256]` contiguous BF16; `k` and `v` are
    /// `[key_tokens, 256]` contiguous BF16, of which all rows take part. The
    /// scale is `1/sqrt(256)` and is not a parameter.
    ///
    /// Returns 0 on success. On failure it returns non-zero and
    /// `apxinf_musa_mutlass_fa_d256_last_error` names the reason; the launch is
    /// asynchronous on `stream`.
    pub fn apxinf_musa_mutlass_fa_d256(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        key_tokens: i32,
        variant: i32,
        stream: *mut c_void,
    ) -> i32;

    /// The message behind the last non-zero return, as a static string.
    pub fn apxinf_musa_mutlass_fa_d256_last_error() -> *const std::os::raw::c_char;
}

/// Turn a status into the adapter's message, naming the operator.
pub fn check_mutlass_fa(status: i32, operation: &str) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: the adapter returns a pointer to a thread-local `char` array
        // that lives at least until the next call on this thread, and this
        // reads it before any call can happen.
        let detail = unsafe { crate::ffi::error_text(apxinf_musa_mutlass_fa_d256_last_error()) }
            .filter(|detail| !detail.is_empty())
            .unwrap_or_else(|| format!("the mutlass FA adapter returned {status}"));
        format!("MUSA {operation}: {detail}")
    })
}

extern "C" {
    /// GR00T DiT D48 attention with an E4M3 output.
    ///
    /// `q` is `[41, 32, 48]` BF16; `k`/`v` are `[key_tokens, 32, 48]` BF16;
    /// `out` is `[41, 32, 48]` E4M3. `output_scale` is `1/activation_scale` and
    /// is applied to the FP32 accumulator before the E4M3 conversion, which is
    /// what fuses the post-attention quantize.
    ///
    /// Returns 0 on success. On failure it returns non-zero and
    /// `apxinf_musa_mutlass_fa_gr00t_d48_last_error` names the reason.
    pub fn apxinf_musa_mutlass_fa_gr00t_d48_e4m3(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        key_tokens: i32,
        output_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// The message behind the last non-zero GR00T D48 return, as a static string.
    pub fn apxinf_musa_mutlass_fa_gr00t_d48_last_error() -> *const std::os::raw::c_char;
}

/// Turn the GR00T D48 adapter's status into its message, naming the operator.
pub fn check_gr00t_fa(status: i32, operation: &str) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: the adapter returns a pointer to a thread-local `char` array
        // that lives at least until the next call on this thread, and this
        // reads it before any call can happen.
        let detail =
            unsafe { crate::ffi::error_text(apxinf_musa_mutlass_fa_gr00t_d48_last_error()) }
                .filter(|detail| !detail.is_empty())
                .unwrap_or_else(|| format!("the GR00T D48 FA adapter returned {status}"));
        format!("MUSA {operation}: {detail}")
    })
}

extern "C" {
    /// The packed-QKV sibling of `apxinf_musa_mutlass_fa_gr00t_d48_e4m3`.
    ///
    /// `q`, `k` and `v` point at columns 0/1536/3072 of one 4608-wide BF16 row;
    /// the kernel's row stride is `3 * heads * head_dim`. Everything else is the
    /// same contract, including the E4M3 output and `output_scale`.
    pub fn apxinf_musa_mutlass_fa_gr00t_d48_e4m3_packed(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        key_tokens: i32,
        output_scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

extern "C" {
    /// GR00T vision attention with an E4M3 output.
    ///
    /// `q`/`k`/`v` are `[batch * tokens, 16, 64]` BF16 with `batch == 2` and
    /// `tokens == 256`; `out` is the same shape in E4M3. `output_scale` is
    /// `1/activation_scale` for the vision output projection.
    pub fn apxinf_musa_mutlass_fa_gr00t_vision_d64_e4m3(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        key_tokens: i32,
        output_scale: f32,
        stream: *mut c_void,
    ) -> i32;
}
