//! Semantic BF16 operators, shaped after `apxinf-cuda-new`'s `ops` surface.
//!
//! PI0.5's device tree is written against these names, so the contract is
//! upstream's rather than this crate's: the same Args fields, the same
//! in-place output convention, the same semantics. What differs is the
//! implementation — each operator unpacks its Args and calls the native kernel
//! facade below, and an operator or variant this port cannot serve reports
//! itself by name instead of approximating.
//!
//! Only the names PI0.5's BF16, INT8 and static-FP8 paths import are mirrored
//! flat. The W8A8 route has its own module ([`w8a8`]) rather than a
//! [`GemmQuantization`] arm: its scale vectors are FP16 with a tile-ABI pad,
//! which the f32 contract cannot carry without a conversion on every GEMM.
//! Static FP8 is served through `GemmQuantization::Fp8UnitScale` and
//! `QuantizationSemantic::FixedScaleE4m3`; per-row FP8/NVFP4, GDN and the
//! packed-quantization family are not part of this surface.

mod attention;
mod cache;
mod gather;
mod gemm;
mod norm;
mod pointwise;
mod portable;
mod quantization;
mod rope;
#[path = "gemm/w8a8.rs"]
pub mod w8a8;

pub use attention::{attention, prepare_attention, AttentionArgs, AttentionMask, AttentionPolicy};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{gather, GatherArgs, GatherSemantic, PatchGeometry as GatherPatchGeometry};
pub use gemm::{
    gemm, gemm_ada_gate_residual_rms_norm, gemm_bias_gelu, gemm_bias_residual, gemm_fp8_qkv_rope,
    gemm_geglu, prepare_fp8_weight, GegluWeightLayout, GemmAdaGateResidualRmsNormArgs, GemmArgs,
    GemmBiasGeluArgs, GemmBiasResidualArgs, GemmGegluArgs, GemmPolicy, GemmQuantization,
};
pub use norm::{
    ada_gate_residual_rms_norm, adaptive_rms_norm, bias_residual, bias_residual_layer_norm,
    bias_residual_rms_norm, layer_norm, rms_norm, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs,
    BiasResidualArgs, BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, LayerNormArgs,
    RmsNormArgs,
};
pub use pointwise::{pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub(crate) use portable::{
    attention as portable_attention, embedding as portable_embedding, point as portable_point,
    rope as portable_rope,
};
pub use quantization::{quantization, QuantizationArgs, QuantizationSemantic};
pub use rope::{rope, RopeArgs, RopeSemantic};

pub use crate::workspace::{ExecutionSession, GraphWorkspace};

use apxinf_core::{DType, Error, Result, Tensor};

/// Run a fixed-shape forward pass that may prepare native executions.
pub fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::workspace::prepare_with_session(session, operation)
}

/// Run the same prepared forward pass without preparing anything new.
pub fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::workspace::with_session(session, operation)
}

/// Reject an output scale this backend cannot honour for a BF16 output.
///
/// BF16 outputs require scale 1.0. A scale other than 1.0 only makes sense for
/// a quantized output, and silently ignoring it would store unscaled values
/// that every consumer then reads as if they had been scaled. E4M3 outputs use
/// [`require_positive_output_scale`] instead.
fn require_unit_output_scale(operation: &str, output_scale: f32) -> Result<()> {
    if output_scale != 1.0 {
        return Err(Error::Other(format!(
            "MUSA {operation} writes BF16 only; output_scale {output_scale} needs a quantized output"
        )));
    }
    Ok(())
}

/// Reject an accumulation precision this backend's GEMM cannot produce.
///
/// The adapter's numerical contract is `MUBLAS_COMPUTE_32F`: f32 accumulators,
/// one rounding at the output. A narrower accumulator would round per
/// contraction step and move the numbers, so the request is refused rather than
/// answered with different arithmetic.
fn require_f32_accumulation(operation: &str, dtype: DType) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::Other(format!(
            "MUSA {operation} accumulates in F32; accumulation_dtype {dtype} is not supported"
        )));
    }
    Ok(())
}

/// Reject a request for a guarantee this backend cannot make.
fn require_deterministic(operation: &str, deterministic: bool) -> Result<()> {
    if deterministic {
        return Err(Error::Other(format!(
            "MUSA {operation} cannot guarantee bitwise determinism across its providers; \
             deterministic=true is refused rather than accepted without the guarantee"
        )));
    }
    Ok(())
}

/// Reject a workspace cap below what the operator must allocate for itself.
///
/// `required` counts only the intermediate the operator allocates on the
/// caller's behalf; vendor-library internal workspaces are outside this
/// backend's accounting and are documented as such.
fn require_workspace(operation: &str, required: usize, limit: usize) -> Result<()> {
    if required > limit {
        return Err(Error::Other(format!(
            "MUSA {operation} needs {required} bytes of its own workspace, above the caller's \
             workspace_limit {limit}"
        )));
    }
    Ok(())
}

/// Reject a scale an E4M3 output cannot honour.
///
/// The dequantization contract is `real = e4m3 * scale`, so the scale must be
/// finite and positive; zero or negative would make the reciprocal in the
/// quantizer meaningless.
fn require_positive_output_scale(operation: &str, output_scale: f32) -> Result<()> {
    if !output_scale.is_finite() || output_scale <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA {operation} needs a finite positive output_scale for an E4M3 output, got {output_scale}"
        )));
    }
    Ok(())
}

/// The output form a post-op was validated for.
///
/// Validating before the vendor GEMM means an illegal dtype or scale fails
/// without launching a full-size contraction, and leaves nothing half-recorded
/// in a captured traversal.
#[derive(Clone, Copy)]
enum ScaledOutput {
    Bf16,
    /// E4M3 with the validated dequantization scale.
    E4m3(f32),
}

fn require_scaled_output(out: &Tensor, scale: f32, operation: &str) -> Result<ScaledOutput> {
    match out.dtype() {
        DType::BF16 => {
            require_unit_output_scale(operation, scale)?;
            Ok(ScaledOutput::Bf16)
        }
        DType::F8E4M3 => {
            require_positive_output_scale(operation, scale)?;
            Ok(ScaledOutput::E4m3(scale))
        }
        dtype => Err(Error::Other(format!(
            "MUSA {operation} out expects BF16 or F8E4M3, got {dtype}"
        ))),
    }
}
