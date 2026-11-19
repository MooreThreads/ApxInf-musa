//! Normalization operator contracts.

use apxinf_core::{DType, Result, Shape, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, tensor_ptr};
use super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_output_dtype_tensor,
    require_output_matrix, require_positive_eps, require_vector_at_least_bf16,
    require_vector_bf16,
};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
use crate::kernels::not_ported;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;

/// BF16 RMS normalization over the last axis.
///
/// Which RMS kernel the caller requires.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RmsKernel {
    /// Width-directed: the register-cached lane kernel where it applies.
    Auto,
    /// The scalar two-pass kernel, selected by the accuracy harness.
    Scalar,
}

/// `input` is 2-D BF16 `[rows, cols]`, `weight` a BF16 `[cols]` vector.
pub fn rms_bf16(ctx: &MusaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    rms_bf16_with_kernel(ctx, input, weight, eps, RmsKernel::Auto)
}

/// The scalar two-pass RMSNorm, bypassing the width-directed lane kernel.
#[cfg_attr(not(apxinf_musa_available), allow(dead_code))]
pub(crate) fn rms_bf16_scalar(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    rms_bf16_with_kernel(ctx, input, weight, eps, RmsKernel::Scalar)
}

fn rms_bf16_with_kernel(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    kernel: RmsKernel,
) -> Result<Tensor> {
    require_positive_eps(eps, "rms_norm")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    rms_bf16_into_with_kernel(ctx, input, weight, &mut output, eps, kernel)?;
    Ok(output)
}

/// Write RMS normalization into a caller-provided `[rows, cols]` BF16 output.
pub fn rms_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    rms_bf16_into_with_kernel(ctx, input, weight, output, eps, RmsKernel::Auto)
}

fn rms_bf16_into_with_kernel(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
    kernel: RmsKernel,
) -> Result<()> {
    require_positive_eps(eps, "rms_norm")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_norm", "input")?;
    require_vector_bf16(ctx, weight, cols, "rms_norm", "weight")?;
    require_output_matrix(ctx, output, rows, cols, "rms_norm", "normalized")?;
    launch_rms(ctx, input, weight, output, rows, cols, eps, kernel)
}

#[cfg(apxinf_musa_available)]
fn launch_rms(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    kernel: RmsKernel,
) -> Result<()> {
    let rows = checked_i32("rms_norm", "rows", rows)?;
    let cols = checked_i32("rms_norm", "cols", cols)?;
    // SAFETY: the checks above established shapes, dtypes and one device; both
    // outputs are `[rows, cols]` BF16 allocations owned by the caller's
    // `Tensor`, which outlives the asynchronous launch on `ctx.stream()`.
    unsafe {
        let status = match kernel {
            RmsKernel::Auto => ffi::custom::apxinf_musa_rms_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            RmsKernel::Scalar => ffi::custom::apxinf_musa_rms_norm_bf16_scalar(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
        };
        check_native(status, "rms_norm")
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_rms(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _kernel: RmsKernel,
) -> Result<()> {
    Err(native_unavailable("norm::rms_bf16"))
}

/// BF16 layer normalization over the last axis.
///
/// Native: `kernels/custom/normalization.cuh`'s three-pass kernel, one block per
/// row, through `adapters/custom_kernels.mu`.
pub fn layer_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    layer_bf16_into(ctx, input, weight, bias, &mut output, eps)?;
    Ok(output)
}

/// Write layer normalization into a caller-provided `[rows, cols]` BF16 output.
pub fn layer_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    require_positive_eps(eps, "layer_norm")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_norm", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_norm", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "layer_norm", "normalized")?;
    launch_layer(ctx, input, weight, bias, output, rows, cols, eps)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel reproduces the bridge's arithmetic -- mean and variance over the
/// row in f32, then the affine transform, rounded once -- but not its summation
/// order: the bridge left the reduction to torch's `mean`, this one walks the
/// row per thread and combines across a warp/block tree. The two do not add the
/// same numbers in the same sequence, so this operator is gated rather than
/// expected to be bit-identical, as `norm.rms_bf16` is.
#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("layer_norm", "rows", rows)?;
    let cols = checked_i32("layer_norm", "cols", cols)?;
    // SAFETY: the checks above established shapes, dtypes and one device, and
    // all four buffers are owned by caller `Tensor`s that outlive the
    // asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_layer_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(bias)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "layer_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_layer(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::layer_bf16"))
}

/// Adaptive (adaLN) BF16 RMS normalization.
///
/// `style` is a single BF16 `[3 * cols]` vector holding scale, shift and gate
/// in that order.
pub fn adaptive_rms_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_rms_norm", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    adaptive_rms_bf16_into(ctx, input, style, &mut output, eps)?;
    Ok(output)
}

/// Write adaptive RMS normalization into a caller-provided output.
///
/// The kernel reads `style` as scale then shift, so the first `2 * cols`
/// entries are the contract; a longer vector is accepted because PI0.5's
/// modulation carries a third gate segment that this operator does not consume.
pub fn adaptive_rms_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    require_positive_eps(eps, "adaptive_rms_norm")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_rms_norm", "input")?;
    require_vector_at_least_bf16(ctx, style, 2 * cols, "adaptive_rms_norm", "style")?;
    require_output_matrix(ctx, output, rows, cols, "adaptive_rms_norm", "normalized")?;
    launch_adaptive_rms(ctx, input, style, output, rows, cols, eps)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel sums squares with a per-thread loop plus a two-level shuffle
/// tree, while a torch `mean` over the last axis does not.
/// The two do not accumulate in the same order, so this operator is gated
/// rather than expected to be bit-identical -- unlike the purely elementwise
/// ones. `hidden`-style outputs that skip the reduction do come out equal.
#[cfg(apxinf_musa_available)]
fn launch_adaptive_rms(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_rms_norm", "rows", rows)?;
    let cols = checked_i32("adaptive_rms_norm", "cols", cols)?;
    // SAFETY: as in `launch_rms` -- shapes, dtypes and one device are already
    // established, and the operands outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_ada_rms_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(style)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "adaptive_rms_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_adaptive_rms(
    _ctx: &MusaContext,
    _input: &Tensor,
    _style: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::adaptive_rms_bf16"))
}

/// Static-scale RMSNorm writing E4M3.
///
/// The register-cached fused producer serves the widths in
/// [`rms_e4m3_supported`]; any other width composes the BF16 producer with
/// [`super::quantization::quantize_fixed_e4m3`], which is byte-identical by
/// construction. Callers never have to know the predicate.
pub fn rms_quant_bf16_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    require_positive_eps(eps, "rms_quant_bf16_e4m3")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_quant_bf16_e4m3", "input")?;
    require_vector_bf16(ctx, weight, cols, "rms_quant_bf16_e4m3", "weight")?;
    let mut output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    if rms_e4m3_supported(cols) {
        rms_bf16_e4m3_into(ctx, input, weight, &mut output, eps, scale)?;
    } else {
        let mut scratch = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
        rms_bf16_into(ctx, input, weight, &mut scratch, eps)?;
        super::quantization::quantize_fixed_e4m3(ctx, &scratch, &mut output, scale)?;
    }
    Ok(output)
}

/// Static-scale LayerNorm writing E4M3; composes when the fused width is not
/// served, exactly like [`rms_quant_bf16_e4m3`].
///
/// The fused widths keep the register-cached one-pass statistics of the
/// existing block kernel, so their bytes can differ from the two-pass
/// composition by one ULP where a value sits on a rounding boundary.
pub fn layer_quant_bf16_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    require_positive_eps(eps, "layer_quant_bf16_e4m3")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_quant_bf16_e4m3", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_quant_bf16_e4m3", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_quant_bf16_e4m3", "bias")?;
    let mut output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    if layer_e4m3_supported(cols) {
        layer_bf16_e4m3_into(ctx, input, weight, bias, &mut output, eps, scale)?;
    } else {
        let mut scratch = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
        layer_bf16_into(ctx, input, weight, bias, &mut scratch, eps)?;
        super::quantization::quantize_fixed_e4m3(ctx, &scratch, &mut output, scale)?;
    }
    Ok(output)
}

// ── Static-E4M3 output, fused into the producer ─────────────────────────
//
// These are the model-path widths the register-cached vector/warp forms
// cover. A width outside a predicate is not refused by the operator: the
// `ops` facade composes the BF16 producer with `quantize_fixed_e4m3` instead,
// which is byte-identical by construction.

/// `rms_bf16_e4m3` serves `kThreads * 8` and `kThreads * 4` columns.
pub fn rms_e4m3_supported(cols: usize) -> bool {
    cols == 2048 || cols == 1024
}

/// RMSNorm writing E4M3 in one launch.
pub fn rms_bf16_e4m3_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
    scale: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "rms_norm_e4m3", "input")?;
    require_vector_bf16(ctx, weight, cols, "rms_norm_e4m3", "weight")?;
    require_output_dtype_tensor(
        ctx,
        output,
        &[rows, cols],
        DType::F8E4M3,
        "rms_norm_e4m3",
        "normalized",
    )?;
    if !rms_e4m3_supported(cols) {
        return Err(apxinf_core::Error::Other(format!(
            "MUSA rms_norm_e4m3: no fused kernel for width {cols}"
        )));
    }
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_rms_e4m3(ctx, input, weight, output, rows, cols, eps, inverse_scale)
}

#[cfg(apxinf_musa_available)]
fn launch_rms_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("rms_norm_e4m3", "rows", rows)?;
    let cols = checked_i32("rms_norm_e4m3", "cols", cols)?;
    // SAFETY: shapes, dtypes, one device and the fused width are established
    // above; the operands outlive the asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_rms_norm_bf16_e4m3(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "rms_norm_e4m3",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_rms_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("norm::rms_bf16_e4m3_into"))
}

/// `layer_bf16_e4m3` serves the GR00T/PI0.5 model widths.
pub fn layer_e4m3_supported(cols: usize) -> bool {
    matches!(cols, 1024 | 1152 | 1536 | 2048)
}

/// LayerNorm writing E4M3 in one launch.
pub fn layer_bf16_e4m3_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    eps: f32,
    scale: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm_e4m3", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_norm_e4m3", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_norm_e4m3", "bias")?;
    require_output_dtype_tensor(
        ctx,
        output,
        &[rows, cols],
        DType::F8E4M3,
        "layer_norm_e4m3",
        "normalized",
    )?;
    if !layer_e4m3_supported(cols) {
        return Err(apxinf_core::Error::Other(format!(
            "MUSA layer_norm_e4m3: no fused kernel for width {cols}"
        )));
    }
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_layer_e4m3(
        ctx,
        input,
        weight,
        bias,
        output,
        rows,
        cols,
        eps,
        inverse_scale,
    )
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_layer_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("layer_norm_e4m3", "rows", rows)?;
    let cols = checked_i32("layer_norm_e4m3", "cols", cols)?;
    // SAFETY: as in `launch_rms_e4m3`; all four buffers are caller `Tensor`s
    // that outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_layer_norm_bf16_e4m3(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(bias)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "layer_norm_e4m3",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_layer_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("norm::layer_bf16_e4m3_into"))
}

/// `layer_bf16_e4m3_dual` serves the GR00T DiT/vl widths.
pub fn layer_e4m3_dual_supported(cols: usize) -> bool {
    matches!(cols, 1024 | 1536 | 2048)
}

/// LayerNorm writing both BF16 and static E4M3 in one launch.
///
/// The E4M3 bytes match [`layer_bf16_e4m3_into`]; the BF16 store carries the
/// same rounded value the quantizer consumes. Widths outside
/// [`layer_e4m3_dual_supported`] are refused; callers that need a composition
/// use [`layer_quant_bf16_e4m3_dual`].
pub fn layer_bf16_e4m3_dual_into(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    normalized: &mut Tensor,
    quantized: &mut Tensor,
    eps: f32,
    scale: f32,
) -> Result<()> {
    require_positive_eps(eps, "layer_norm_e4m3_dual")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm_e4m3_dual", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_norm_e4m3_dual", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_norm_e4m3_dual", "bias")?;
    require_output_dtype_tensor(
        ctx,
        normalized,
        &[rows, cols],
        DType::BF16,
        "layer_norm_e4m3_dual",
        "normalized",
    )?;
    require_output_dtype_tensor(
        ctx,
        quantized,
        &[rows, cols],
        DType::F8E4M3,
        "layer_norm_e4m3_dual",
        "quantized",
    )?;
    if !layer_e4m3_dual_supported(cols) {
        return Err(apxinf_core::Error::Other(format!(
            "MUSA layer_norm_e4m3_dual: no fused kernel for width {cols}"
        )));
    }
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_layer_e4m3_dual(
        ctx,
        input,
        weight,
        bias,
        normalized,
        quantized,
        rows,
        cols,
        eps,
        inverse_scale,
    )
}

/// LayerNorm returning both BF16 and E4M3 outputs, composing the two passes
/// for a width the dual kernel does not serve.
pub fn layer_quant_bf16_e4m3_dual(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    require_positive_eps(eps, "layer_norm_e4m3_dual")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "layer_norm_e4m3_dual", "input")?;
    require_vector_bf16(ctx, weight, cols, "layer_norm_e4m3_dual", "weight")?;
    require_vector_bf16(ctx, bias, cols, "layer_norm_e4m3_dual", "bias")?;
    let mut normalized = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let mut quantized = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    if layer_e4m3_dual_supported(cols) {
        let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
        launch_layer_e4m3_dual(
            ctx,
            input,
            weight,
            bias,
            &normalized,
            &mut quantized,
            rows,
            cols,
            eps,
            inverse_scale,
        )?;
    } else {
        layer_bf16_into(ctx, input, weight, bias, &mut normalized, eps)?;
        super::quantization::quantize_fixed_e4m3(ctx, &normalized, &mut quantized, scale)?;
    }
    Ok((normalized, quantized))
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_layer_e4m3_dual(
    ctx: &MusaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    normalized: &Tensor,
    quantized: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("layer_norm_e4m3_dual", "rows", rows)?;
    let cols = checked_i32("layer_norm_e4m3_dual", "cols", cols)?;
    // SAFETY: as in `launch_layer_e4m3`; every buffer is a caller `Tensor`
    // that outlives the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_layer_norm_bf16_e4m3_dual(
                tensor_ptr(input)?,
                tensor_ptr(weight)?,
                tensor_ptr(bias)?,
                tensor_ptr(normalized)?,
                tensor_ptr(quantized)?,
                rows,
                cols,
                eps,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "layer_norm_e4m3_dual",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_layer_e4m3_dual(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _normalized: &Tensor,
    _quantized: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("norm::layer_bf16_e4m3_dual_into"))
}

/// `adaptive_rms_bf16_e4m3` serves `kThreads * 4` (1024) columns.
pub fn adaptive_rms_e4m3_supported(cols: usize) -> bool {
    cols == 1024
}

/// Adaptive RMSNorm writing E4M3 in one launch.
pub fn adaptive_rms_bf16_e4m3_into(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &mut Tensor,
    eps: f32,
    scale: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_rms_e4m3", "input")?;
    require_vector_at_least_bf16(ctx, style, 2 * cols, "adaptive_rms_e4m3", "style")?;
    require_output_dtype_tensor(
        ctx,
        output,
        &[rows, cols],
        DType::F8E4M3,
        "adaptive_rms_e4m3",
        "normalized",
    )?;
    if !adaptive_rms_e4m3_supported(cols) {
        return Err(apxinf_core::Error::Other(format!(
            "MUSA adaptive_rms_e4m3: no fused kernel for width {cols}"
        )));
    }
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_adaptive_rms_e4m3(ctx, input, style, output, rows, cols, eps, inverse_scale)
}

#[cfg(apxinf_musa_available)]
fn launch_adaptive_rms_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_rms_e4m3", "rows", rows)?;
    let cols = checked_i32("adaptive_rms_e4m3", "cols", cols)?;
    // SAFETY: as in `launch_rms_e4m3`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_ada_rms_norm_bf16_e4m3(
                tensor_ptr(input)?,
                tensor_ptr(style)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "adaptive_rms_e4m3",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_adaptive_rms_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _style: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("norm::adaptive_rms_bf16_e4m3_into"))
}

// ── FP8-output normalization (fused family not ported) ───────────────────
//
// The `*_quant_f16_e4m3` family writes E4M3 rather than BF16. The static-FP8
// route composes the BF16 normalization with an explicit quantizer instead of
// these fused entries. Signatures mirror `apxinf-cuda/src/kernels/norm.rs`;
// note those CUDA functions do **not** validate `eps`/`scale` finiteness
// (their BF16 siblings do), so a port should not add the check and then call
// itself faithful.

/// RMSNorm whose output is quantized to E4M3.
pub fn rms_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::rms_quant_f16_e4m3"))
}

/// LayerNorm whose output is quantized to E4M3.
pub fn layer_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::layer_quant_f16_e4m3"))
}

/// Adaptive RMSNorm whose output is quantized to E4M3.
pub fn adaptive_rms_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _style: &Tensor,
    _eps: f32,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("norm::adaptive_rms_quant_f16_e4m3"))
}

#[cfg(apxinf_musa_available)]
fn launch_adaptive_layer(
    ctx: &MusaContext,
    input: &Tensor,
    modulation: &Tensor,
    output: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_layer_norm", "rows", rows)?;
    let cols = checked_i32("adaptive_layer_norm", "cols", cols)?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_adaptive_layer_norm_bf16(
                tensor_ptr(input)?,
                tensor_ptr(modulation)?,
                tensor_ptr(output)?,
                rows,
                cols,
                eps,
                ctx.stream().handle(),
            ),
            "adaptive_layer_norm",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_adaptive_layer(
    _ctx: &MusaContext,
    _input: &Tensor,
    _modulation: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
) -> Result<()> {
    Err(native_unavailable("norm::adaptive_layer_bf16"))
}

/// Native device-resident adaptive LayerNorm.
pub fn adaptive_layer(
    ctx: &MusaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive LayerNorm", "input")?;
    require_vector_bf16(
        ctx,
        modulation,
        2 * cols,
        "adaptive LayerNorm",
        "modulation",
    )?;
    require_positive_eps(eps, "adaptive LayerNorm")?;
    let output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    #[cfg(apxinf_musa_available)]
    {
        launch_adaptive_layer(ctx, input, modulation, &output, rows, cols, eps)?;
        return Ok(output);
    }
    #[cfg(not(apxinf_musa_available))]
    Err(native_unavailable("norm::adaptive_layer_bf16"))
}

/// Adaptive LayerNorm writing BF16 and static E4M3 in one launch.
///
/// The arithmetic and BF16 rounding match [`adaptive_layer`]; the E4M3 bytes
/// match that result followed by
/// [`super::quantization::quantize_fixed_e4m3`].
pub fn adaptive_layer_bf16_e4m3_dual_into(
    ctx: &MusaContext,
    input: &Tensor,
    modulation: &Tensor,
    normalized: &mut Tensor,
    quantized: &mut Tensor,
    eps: f32,
    scale: f32,
) -> Result<()> {
    require_positive_eps(eps, "adaptive_layer_norm_e4m3")?;
    let (rows, cols) = require_matrix_bf16(ctx, input, "adaptive_layer_norm_e4m3", "input")?;
    require_vector_at_least_bf16(
        ctx,
        modulation,
        2 * cols,
        "adaptive_layer_norm_e4m3",
        "modulation",
    )?;
    require_output_matrix(
        ctx,
        normalized,
        rows,
        cols,
        "adaptive_layer_norm_e4m3",
        "normalized",
    )?;
    require_output_dtype_tensor(
        ctx,
        quantized,
        &[rows, cols],
        DType::F8E4M3,
        "adaptive_layer_norm_e4m3",
        "quantized",
    )?;
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_adaptive_layer_e4m3_dual(
        ctx,
        input,
        modulation,
        normalized,
        quantized,
        rows,
        cols,
        eps,
        inverse_scale,
    )
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_adaptive_layer_e4m3_dual(
    ctx: &MusaContext,
    input: &Tensor,
    modulation: &Tensor,
    normalized: &Tensor,
    quantized: &Tensor,
    rows: usize,
    cols: usize,
    eps: f32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("adaptive_layer_norm_e4m3", "rows", rows)?;
    let cols = checked_i32("adaptive_layer_norm_e4m3", "cols", cols)?;
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_adaptive_layer_norm_bf16_e4m3_dual(
                tensor_ptr(input)?,
                tensor_ptr(modulation)?,
                tensor_ptr(normalized)?,
                tensor_ptr(quantized)?,
                rows,
                cols,
                eps,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "adaptive_layer_norm_e4m3",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_adaptive_layer_e4m3_dual(
    _ctx: &MusaContext,
    _input: &Tensor,
    _modulation: &Tensor,
    _normalized: &Tensor,
    _quantized: &Tensor,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("norm::adaptive_layer_norm_bf16_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the normalization kernels.
    //!
    //! The replay harness covers `norm.rms_bf16` at its one recorded shape,
    //! `[522, 2048]` -- which takes the **vectorized** reduction, because
    //! `rows >= 16 && cols > 128 && cols % 4 == 0`. The block-reduction branch
    //! is therefore unreached by that evidence, and it is the one the small and
    //! ragged shapes use. These tests cover both, plus `ada_rms_norm`'s scale
    //! and shift.
    //!
    //! Needs a MUSA device, so they are `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape, Tensor};

    use crate::context::MusaContext;
    use crate::kernels::fused;
    use crate::kernels::norm::{
        adaptive_layer, adaptive_layer_bf16_e4m3_dual_into, adaptive_rms_bf16,
        adaptive_rms_bf16_e4m3_into, adaptive_rms_bf16_into, layer_bf16, layer_bf16_e4m3_into,
        layer_bf16_into, rms_bf16, rms_bf16_e4m3_into, rms_bf16_into,
    };
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values, zeros_as,
    };

    const EPS: f32 = 1e-6;

    fn check_rms(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x5eed_1234);
        let weight_values = values(cols, 0x0bad_c0de);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![cols]));

        let out = rms_bf16(ctx, &input, &weight, EPS).expect("native rms_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let weight_values: Vec<f32> = weight_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let sum: f32 = (0..cols).map(|c| input_values[base + c] * input_values[base + c]).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for col in 0..cols {
                let expected = round(input_values[base + col] * inverse_rms * weight_values[col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "rms [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, abs {error:.3e} > {tolerance:.3e}",
                    got[base + col]
                );
            }
        }
        println!("  rms_norm [{rows:>4} x {cols:>4}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn rms_norm_matches_a_reference_across_shapes() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // One block reduction now covers every shape: the single-warp
        // `rms_vector_square_mean_bf16` path was 89% of this kernel's runtime
        // and 6.4x slower than the same reduction over all threads, and the
        // two summed in different orders only because CUDA's dispatch did.
        // `[522, 2048]` is the recorded shape; the ragged ones pin the
        // widths where the old dispatch switched.
        for (rows, cols) in [(3, 5), (16, 130), (16, 256), (522, 2048), (7, 2047)] {
            check_rms(&ctx, rows, cols);
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn adaptive_rms_norm_applies_scale_and_shift() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check_adaptive_rms(&ctx, 7, 300);
        // `[10, 1024]` is the action expert's shape and a multiple of the
        // block, so it takes the register-cached branch; 300 above does not.
        check_adaptive_rms(&ctx, 10, 1024);
    }

    fn check_adaptive_rms(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x2468_ace0);
        let style_values = values(3 * cols, 0x1357_9bdf);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let style = upload(ctx, &style_values, Shape::new(vec![3 * cols]));

        let out = adaptive_rms_bf16(ctx, &input, &style, EPS).expect("native ada_rms_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let style_values: Vec<f32> = style_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let sum: f32 = (0..cols).map(|c| input_values[base + c] * input_values[base + c]).sum();
            let inverse_rms = 1.0 / (sum / cols as f32 + EPS).sqrt();
            for col in 0..cols {
                let normalized = input_values[base + col] * inverse_rms;
                let expected = round(normalized * (1.0 + style_values[col]) + style_values[cols + col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "ada_rms [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, abs {error:.3e}",
                    got[base + col]
                );
            }
        }
        println!("  ada_rms_norm [{rows} x {cols}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    /// `layer_norm` over a row, with a weight and a shift.
    ///
    /// The replay covers `[512, 1152]`; this adds a ragged shape small enough
    /// that most threads of a block find no column at all, which is where the
    /// three-pass structure (sum, then squared deviations, then the transform)
    /// has to keep its barriers straight.
    fn check_layer(ctx: &MusaContext, rows: usize, cols: usize) {
        let input_values = values(rows * cols, 0x2468_ace0);
        let weight_values = values(cols, 0x0bad_c0de);
        let bias_values = values(cols, 0x1357_9bdf);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let weight = upload(ctx, &weight_values, Shape::new(vec![cols]));
        let bias = upload(ctx, &bias_values, Shape::new(vec![cols]));

        let out = layer_bf16(ctx, &input, &weight, &bias, EPS).expect("native layer_norm");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let input_values: Vec<f32> = input_values.iter().map(|v| round(*v)).collect();
        let weight_values: Vec<f32> = weight_values.iter().map(|v| round(*v)).collect();
        let bias_values: Vec<f32> = bias_values.iter().map(|v| round(*v)).collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            let base = row * cols;
            let row_values = &input_values[base..base + cols];
            let mean = row_values.iter().sum::<f32>() / cols as f32;
            let variance =
                row_values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inverse_std = 1.0 / (variance + EPS).sqrt();
            for col in 0..cols {
                let expected =
                    round((row_values[col] - mean) * inverse_std * weight_values[col] + bias_values[col]);
                let error = (got[base + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "layer [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[base + col]
                );
            }
        }
        // `[512, 1152]` runs at a noticeable fraction of the gate, and it is
        // the reference's fault rather than the kernel's: the sum above is
        // sequential while the kernel splits the row across 256 threads and
        // combines a warp tree, so the mean and variance differ in the last f32
        // bits and occasionally that moves a bf16 rounding. The replay against
        // the *recorded* output -- torch's own reduction, not this one -- is
        // what bounds the kernel itself, and it lands at 7.6e-6 against a 9.7e-2
        // gate.
        println!("  layer_norm [{rows:>4} x {cols:>4}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn layer_norm_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        check_layer(&ctx, 3, 5);
        check_layer(&ctx, 16, 130);
        check_layer(&ctx, 512, 1152);
    }

    /// The fused E4M3 producers must be byte-identical to materializing the
    /// BF16 producer and running the standalone quantizer.
    fn quantized_bytes(ctx: &MusaContext, bf16: &Tensor, scale: f32) -> Vec<u8> {
        let mut quantized = zeros_as(ctx, bf16.shape().clone(), DType::F8E4M3);
        crate::kernels::quantization::quantize_fixed_e4m3(ctx, bf16, &mut quantized, scale)
            .expect("standalone quantizer");
        ctx.stream().synchronize().expect("synchronise");
        crate::transfers::to_cpu(&quantized)
            .expect("quantized back")
            .as_f8_e4m3()
            .expect("e4m3")
            .to_vec()
    }

    fn fused_bytes(ctx: &MusaContext, tensor: &Tensor) -> Vec<u8> {
        ctx.stream().synchronize().expect("synchronise");
        crate::transfers::to_cpu(tensor)
            .expect("fused back")
            .as_f8_e4m3()
            .expect("e4m3")
            .to_vec()
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn norm_e4m3_matches_the_composed_pair_byte_for_byte() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let scale = 0.0721f32;
        let eps = EPS;

        // RMSNorm [522, 2048]: the recorded shape, vector <8, 8>.
        let (rows, cols) = (522usize, 2048usize);
        let input = upload(
            &ctx,
            &values(rows * cols, 0x5eed_1234),
            Shape::new(vec![rows, cols]),
        );
        let weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
        let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        rms_bf16_into(&ctx, &input, &weight, &mut bf16, eps).expect("bf16 rms");
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        rms_bf16_e4m3_into(&ctx, &input, &weight, &mut fused_out, eps, scale)
            .expect("fused rms");
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "rms_norm [{rows} x {cols}] differs from the composed pair"
        );
        println!("  rms_norm e4m3 [{rows} x {cols}] byte-identical");

        // LayerNorm [512, 1152]: the vision width, warp <4, 9>.
        let (rows, cols) = (512usize, 1152usize);
        let input = upload(
            &ctx,
            &values(rows * cols, 0x2468_ace0),
            Shape::new(vec![rows, cols]),
        );
        let weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
        let bias = upload(&ctx, &values(cols, 0x1357_9bdf), Shape::new(vec![cols]));
        let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        layer_bf16_into(&ctx, &input, &weight, &bias, &mut bf16, eps).expect("bf16 layer");
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        layer_bf16_e4m3_into(&ctx, &input, &weight, &bias, &mut fused_out, eps, scale)
            .expect("fused layer");
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "layer_norm [{rows} x {cols}] differs from the composed pair"
        );
        println!("  layer_norm e4m3 [{rows} x {cols}] byte-identical");

        // The templated fused widths keep the register-cached one-pass
        // statistics of the 1152 sibling, while the composed pair uses the
        // generic two-pass producer. An element on an E4M3 rounding boundary
        // can therefore move by one ULP; everything else must match.
        for (rows, cols) in [(41usize, 1536usize), (512, 1024), (145, 2048)] {
            let input = upload(
                &ctx,
                &values(rows * cols, 0x5a17_c0de),
                Shape::new(vec![rows, cols]),
            );
            let weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
            let bias = upload(&ctx, &values(cols, 0x1357_9bdf), Shape::new(vec![cols]));
            let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
            layer_bf16_into(&ctx, &input, &weight, &bias, &mut bf16, eps).expect("bf16 layer");
            let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
            layer_bf16_e4m3_into(&ctx, &input, &weight, &bias, &mut fused_out, eps, scale)
                .expect("fused layer");
            let fused = fused_bytes(&ctx, &fused_out);
            let composed = quantized_bytes(&ctx, &bf16, scale);
            let differing = fused
                .iter()
                .zip(&composed)
                .filter(|(fused, composed)| fused != composed)
                .count();
            let max_delta = fused
                .iter()
                .zip(&composed)
                .map(|(fused, composed)| fused.abs_diff(*composed))
                .max()
                .unwrap_or(0);
            assert!(
                max_delta <= 1,
                "layer_norm [{rows} x {cols}] differs from the composed pair by {max_delta} ULP"
            );
            println!(
                "  layer_norm e4m3 [{rows} x {cols}] within one ULP ({differing} bytes differ)"
            );
        }

        // Adaptive LayerNorm [41, 1536]: BF16 and E4M3 from one launch.
        let (rows, cols) = (41usize, 1536usize);
        let input = upload(
            &ctx,
            &values(rows * cols, 0x2468_ace0),
            Shape::new(vec![rows, cols]),
        );
        let modulation = upload(
            &ctx,
            &values(2 * cols, 0x1357_9bdf),
            Shape::new(vec![2 * cols]),
        );
        let bf16 = adaptive_layer(&ctx, &input, &modulation, eps).expect("bf16 ada layer");
        let mut fused_bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        adaptive_layer_bf16_e4m3_dual_into(
            &ctx,
            &input,
            &modulation,
            &mut fused_bf16,
            &mut fused_out,
            eps,
            scale,
        )
        .expect("dual ada layer");
        assert_eq!(
            read_back(&fused_bf16),
            read_back(&bf16),
            "adaptive_layer_norm [{rows} x {cols}] BF16 result differs"
        );
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "adaptive_layer_norm [{rows} x {cols}] differs from the composed pair"
        );
        println!("  adaptive_layer_norm e4m3 [{rows} x {cols}] byte-identical");

        // Adaptive RMSNorm [10, 1024]: one block, vector <4>.
        let (rows, cols) = (10usize, 1024usize);
        let input = upload(
            &ctx,
            &values(rows * cols, 0x2468_ace0),
            Shape::new(vec![rows, cols]),
        );
        let style = upload(&ctx, &values(3 * cols, 0x1357_9bdf), Shape::new(vec![3 * cols]));
        let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        adaptive_rms_bf16_into(&ctx, &input, &style, &mut bf16, eps).expect("bf16 ada rms");
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        adaptive_rms_bf16_e4m3_into(&ctx, &input, &style, &mut fused_out, eps, scale)
            .expect("fused ada rms");
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "adaptive_rms_norm [{rows} x {cols}] differs from the composed pair"
        );
        println!("  adaptive_rms_norm e4m3 [{rows} x {cols}] byte-identical");

        // Bias + residual + RMSNorm [522, 2048].
        let (rows, cols) = (522usize, 2048usize);
        let projection = upload(
            &ctx,
            &values(rows * cols, 0x5eed_1234),
            Shape::new(vec![rows, cols]),
        );
        let residual = upload(
            &ctx,
            &values(rows * cols, 0xc2b2_ae35),
            Shape::new(vec![rows, cols]),
        );
        let weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
        let bias = upload(&ctx, &values(cols, 0x1357_9bdf), Shape::new(vec![cols]));
        let mut hidden = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        fused::bias_residual_rms_bf16_into(
            &ctx,
            &projection,
            Some(&bias),
            &residual,
            &weight,
            &mut hidden,
            &mut bf16,
            eps,
        )
        .expect("bf16 bias residual rms");
        let mut fused_hidden = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        fused::bias_residual_rms_e4m3_into(
            &ctx,
            &projection,
            Some(&bias),
            &residual,
            &weight,
            &mut fused_hidden,
            &mut fused_out,
            eps,
            scale,
        )
        .expect("fused bias residual rms");
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "bias_residual_rms [{rows} x {cols}] differs from the composed pair"
        );
        println!("  bias_residual_rms e4m3 [{rows} x {cols}] byte-identical");

        // Bias + residual + LayerNorm [512, 1152].
        let (rows, cols) = (512usize, 1152usize);
        let projection = upload(
            &ctx,
            &values(rows * cols, 0x2468_ace0),
            Shape::new(vec![rows, cols]),
        );
        let residual = upload(
            &ctx,
            &values(rows * cols, 0xc2b2_ae35),
            Shape::new(vec![rows, cols]),
        );
        let norm_weight = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
        let norm_bias = upload(&ctx, &values(cols, 0x1357_9bdf), Shape::new(vec![cols]));
        let projection_bias = upload(&ctx, &values(cols, 0x9e37_79b9), Shape::new(vec![cols]));
        let mut hidden = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        fused::bias_residual_layer_bf16_into(
            &ctx,
            &projection,
            Some(&projection_bias),
            &residual,
            &norm_weight,
            &norm_bias,
            &mut hidden,
            &mut bf16,
            eps,
        )
        .expect("bf16 bias residual layer");
        let mut fused_hidden = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
        let mut fused_out = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
        fused::bias_residual_layer_e4m3_into(
            &ctx,
            &projection,
            Some(&projection_bias),
            &residual,
            &norm_weight,
            &norm_bias,
            &mut fused_hidden,
            &mut fused_out,
            eps,
            scale,
        )
        .expect("fused bias residual layer");
        assert_eq!(
            fused_bytes(&ctx, &fused_out),
            quantized_bytes(&ctx, &bf16, scale),
            "bias_residual_layer [{rows} x {cols}] differs from the composed pair"
        );
        println!("  bias_residual_layer e4m3 [{rows} x {cols}] byte-identical");
    }
}
