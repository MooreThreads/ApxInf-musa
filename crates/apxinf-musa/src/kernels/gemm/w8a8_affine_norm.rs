//! Affine RMS/LayerNorm producers with the original BF16 rounding contract.
use super::*;

/// Normalize and quantize, optionally producing a bias+residual hidden stream.
/// `norm_bias = None` selects RMSNorm; `Some` selects LayerNorm. Unsupported
/// widths/alignment retain the ordinary BF16 producer and quantizer.
///
/// `APXINF_MUSA_BF16_RMS=scalar` forces the RMS fallback onto the scalar
/// two-pass kernel for accuracy runs; the plain path would otherwise take the
/// width-directed lane kernel for `[522, 2048]`.
#[allow(clippy::too_many_arguments)]
pub fn norm_quantize_w8a8_activation(
    ctx: &MusaContext,
    input: &Tensor,
    residual: Option<(&Tensor, Option<&Tensor>)>,
    weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f32,
    inv_c: Option<&Tensor>,
) -> Result<(Option<Tensor>, W8A8Activation)> {
    norm_quantize_with_scalar_rms(
        ctx,
        input,
        residual,
        weight,
        norm_bias,
        eps,
        inv_c,
        scalar_rms_requested(),
    )
}

fn scalar_rms_requested() -> bool {
    static REQUESTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUESTED.get_or_init(|| std::env::var("APXINF_MUSA_BF16_RMS").as_deref() == Ok("scalar"))
}

/// The producer behind [`norm_quantize_w8a8_activation`], with the scalar-RMS
/// choice explicit so tests do not depend on process-wide environment state.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(apxinf_musa_available), allow(unused_variables))]
pub(crate) fn norm_quantize_with_scalar_rms(
    ctx: &MusaContext,
    input: &Tensor,
    residual: Option<(&Tensor, Option<&Tensor>)>,
    weight: &Tensor,
    norm_bias: Option<&Tensor>,
    eps: f32,
    inv_c: Option<&Tensor>,
    force_scalar_rms: bool,
) -> Result<(Option<Tensor>, W8A8Activation)> {
    #[cfg(not(apxinf_musa_available))]
    {
        Err(native_unavailable("gemm::norm_quantize_w8a8_activation"))
    }
    #[cfg(apxinf_musa_available)]
    {
        use crate::kernels::contracts::require_vector_bf16;
        let op = "gemm::norm_quantize_w8a8_activation";
        let (rows, cols) = activation_matrix(ctx, input, op)?;
        require_vector_bf16(ctx, weight, cols, op, "norm weight")?;
        let ptr = |t: &Tensor| -> Result<*const c_void> {
            Ok(MusaBuffer::from_tensor(t).map_err(musa_error)?.ptr() as *const c_void)
        };
        let vector_ptr = |t: Option<&Tensor>, name| -> Result<*const c_void> {
            match t {
                Some(t) => {
                    require_vector_bf16(ctx, t, cols, op, name)?;
                    ptr(t)
                }
                None => Ok(std::ptr::null()),
            }
        };
        let source = ptr(input)?;
        let weights = ptr(weight)?;
        let bias = vector_ptr(norm_bias, "norm bias")?;
        let (res, projection_bias) = match residual {
            Some((r, b)) => {
                if activation_matrix(ctx, r, op)? != (rows, cols) {
                    return Err(Error::Other(format!(
                        "{op}: residual shape must match input"
                    )));
                }
                (ptr(r)?, vector_ptr(b, "projection bias")?)
            }
            None => (std::ptr::null(), std::ptr::null()),
        };
        let factors = equalization_factors_ptr(ctx, op, inv_c, cols)?;
        let alignment = if norm_bias.is_some() { 8 } else { 16 };
        let scalar_rms = force_scalar_rms && norm_bias.is_none() && residual.is_none();
        if cols != if norm_bias.is_some() { 1152 } else { 2048 }
            || [source, weights, bias, res, projection_bias]
                .iter()
                .any(|p| *p as usize % alignment != 0)
            || factors as usize % 16 != 0
            || scalar_rms
        {
            let (hidden, normalized) = match (residual, norm_bias) {
                (Some((r, b)), Some(nb)) => {
                    let out = crate::kernels::fused::bias_residual_layer_bf16(
                        ctx, input, b, r, weight, nb, eps,
                    )?;
                    (Some(out.hidden), out.normalized)
                }
                (Some((r, b)), None) => {
                    let out = crate::kernels::fused::bias_residual_rms_bf16(
                        ctx, input, b, r, weight, eps,
                    )?;
                    (Some(out.hidden), out.normalized)
                }
                (None, Some(nb)) => (
                    None,
                    crate::kernels::norm::layer_bf16(ctx, input, weight, nb, eps)?,
                ),
                (None, None) => (
                    None,
                    if scalar_rms {
                        crate::kernels::norm::rms_bf16_scalar(ctx, input, weight, eps)?
                    } else {
                        crate::kernels::norm::rms_bf16(ctx, input, weight, eps)?
                    },
                ),
            };
            return Ok((hidden, quantize_w8a8_activation(ctx, &normalized, inv_c)?));
        }
        let rows_i32 = checked_i32(op, "rows", rows)?;
        let cols_i32 = checked_i32(op, "cols", cols)?;
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Other(format!("{op}: size overflow")))?;
        let hidden = if residual.is_some() {
            let bytes = elements
                .checked_mul(2)
                .ok_or_else(|| Error::Other(format!("{op}: size overflow")))?;
            Some(
                crate::workspace::output_buffer(ctx, bytes)?
                    .as_tensor(Shape::new(vec![rows, cols]), DType::BF16)
                    .map_err(musa_error)?,
            )
        } else {
            None
        };
        let quantized = crate::workspace::output_buffer(ctx, elements)?;
        let row_scales = crate::workspace::output_buffer(ctx, padded_scale_bytes(rows, op)?)?;
        let hidden_ptr = match hidden.as_ref() {
            Some(t) => ptr(t)? as *mut c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: shapes, dtypes, devices and vector alignment are checked above.
        // Workspace owns disjoint outputs with full scale padding on this stream.
        unsafe {
            check_native(
                ffi::custom::apxinf_musa_norm_quantize_bf16_int8(
                    source,
                    res,
                    projection_bias,
                    weights,
                    bias,
                    hidden_ptr,
                    quantized.ptr(),
                    row_scales.ptr(),
                    rows_i32,
                    cols_i32,
                    eps,
                    factors,
                    ctx.stream().handle(),
                ),
                op,
            )?;
        }
        Ok((
            hidden,
            W8A8Activation {
                quantized,
                row_scales,
                rows,
                input_dim: cols,
            },
        ))
    }
}
