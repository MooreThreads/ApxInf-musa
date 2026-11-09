//! FP8 quantization operator contracts.
//!
//! **Not ported.** Every entry point here reports the gap rather than
//! computing something plausible: this file's operators define the FP8 data
//! model (per-tensor and per-row E4M3 quantization), and getting them subtly
//! wrong would produce numbers that look reasonable and are not. They are
//! outside the supported BF16 GR00T execution path. The GR00T loader rejects
//! unsupported precision variants before executing these operators.
//!
//! The signatures mirror `apxinf-cuda/src/kernels/quantization.rs` exactly, so
//! the model layer compiles unchanged and a request for an FP8 variant fails
//! naming the operator it needed.

use apxinf_core::{Result, Tensor};

use crate::context::MusaContext;
use crate::kernels::not_ported;

/// A per-row dynamically quantized FP8 tensor.
pub struct DynamicFp8Tensor {
    pub values: Tensor,
    pub scales: Tensor,
}

/// Quantize an FP16 tensor to E4M3 with a per-tensor scale.
pub fn quantize_f16_e4m3(_ctx: &MusaContext, _input: &Tensor, _scale: f32) -> Result<Tensor> {
    Err(not_ported("quantization::quantize_f16_e4m3"))
}

/// Quantize a BF16 tensor to E4M3 with a per-tensor scale.
pub fn quantize_bf16_e4m3(_ctx: &MusaContext, _input: &Tensor, _scale: f32) -> Result<Tensor> {
    Err(not_ported("quantization::quantize_bf16_e4m3"))
}

pub fn concat_rows_quantize_bf16_e4m3(
    _ctx: &MusaContext,
    _first: &Tensor,
    _second: &Tensor,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("quantization::concat_rows_quantize_bf16_e4m3"))
}

/// Quantize a BF16 tensor to E4M3 one row at a time.
pub fn quantize_rows_bf16_e4m3(_ctx: &MusaContext, _input: &Tensor) -> Result<DynamicFp8Tensor> {
    Err(not_ported("quantization::quantize_rows_bf16_e4m3"))
}

/// Row quantization into a padded output width.
pub fn quantize_rows_bf16_e4m3_padded(
    _ctx: &MusaContext,
    _input: &Tensor,
    _output_cols: usize,
) -> Result<DynamicFp8Tensor> {
    Err(not_ported("quantization::quantize_rows_bf16_e4m3_padded"))
}

/// Take the first `output_cols` columns of a BF16 tensor.
pub fn slice_columns_bf16(
    _ctx: &MusaContext,
    _input: &Tensor,
    _output_cols: usize,
) -> Result<Tensor> {
    Err(not_ported("quantization::slice_columns_bf16"))
}

/// Cast FP16 to BF16.
pub fn cast_f16_bf16(_ctx: &MusaContext, _input: &Tensor) -> Result<Tensor> {
    Err(not_ported("quantization::cast_f16_bf16"))
}
