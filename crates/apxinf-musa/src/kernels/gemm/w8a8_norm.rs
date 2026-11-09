//! Adaptive RMSNorm producers for W8A8, with BF16 residual/activation rounding.
use super::*;

/// Adaptive RMSNorm followed by dynamic row quantization. Aligned 1024-column
/// inputs share the existing BF16 producer's reduction order in a single kernel.
pub fn adaptive_rms_quantize_w8a8_activation(
    ctx: &MusaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
    inv_c: Option<&Tensor>,
) -> Result<W8A8Activation> {
    Ok(norm_quantize(ctx, input, None, style, eps, inv_c)?.1)
}

/// Gated residual + Adaptive RMSNorm + dynamic quantization. Returns the BF16
/// residual stream and quantized normalized activation; the latter includes
/// optional equalization and must be consumed without quantizing a second time.
pub fn adaptive_gate_residual_rms_quantize_w8a8_activation(
    ctx: &MusaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    inv_c: Option<&Tensor>,
) -> Result<(Tensor, W8A8Activation)> {
    let (hidden, quantized) = norm_quantize(
        ctx,
        projection,
        Some((residual, gate_style)),
        norm_style,
        eps,
        inv_c,
    )?;
    Ok((hidden.expect("gated norm produces a residual"), quantized))
}

#[cfg_attr(not(apxinf_musa_available), allow(unused_variables))]
fn norm_quantize(
    ctx: &MusaContext,
    input: &Tensor,
    gated: Option<(&Tensor, &Tensor)>,
    style: &Tensor,
    eps: f32,
    inv_c: Option<&Tensor>,
) -> Result<(Option<Tensor>, W8A8Activation)> {
    #[cfg(not(apxinf_musa_available))]
    {
        Err(native_unavailable(
            "gemm::adaptive_rms_quantize_w8a8_activation",
        ))
    }
    #[cfg(apxinf_musa_available)]
    {
        use crate::kernels::contracts::{require_dtype_on, require_vector_bf16};
        let op = "gemm::adaptive_rms_quantize_w8a8_activation";
        let (rows, cols) = activation_matrix(ctx, input, op)?;
        let style_len = cols
            .checked_mul(3)
            .ok_or_else(|| Error::Other(format!("{op}: style size overflow")))?;
        require_vector_bf16(ctx, style, style_len, op, "norm style")?;
        let ptr = |t: &Tensor| -> Result<*const c_void> {
            Ok(MusaBuffer::from_tensor(t).map_err(musa_error)?.ptr() as *const c_void)
        };
        let source = ptr(input)?;
        let norm_style = ptr(style)?;
        let (residual, gate_style) = match gated {
            Some((residual, gate)) => {
                if activation_matrix(ctx, residual, op)? != (rows, cols) {
                    return Err(Error::Other(format!(
                        "{op}: residual shape must match input"
                    )));
                }
                require_vector_bf16(ctx, gate, style_len, op, "gate style")?;
                (ptr(residual)?, ptr(gate)?)
            }
            None => (std::ptr::null(), std::ptr::null()),
        };
        let factors = match inv_c {
            Some(t) => {
                require_dtype_on(ctx, t, DType::F32, op, "equalization")?;
                if t.shape().dims() != [cols] {
                    return Err(Error::Other(format!("{op}: equalization must be [{cols}]")));
                }
                ptr(t)?
            }
            None => std::ptr::null(),
        };
        if cols != 1024
            || [source, norm_style, residual, gate_style]
                .iter()
                .any(|p| *p as usize % 8 != 0)
            || factors as usize % 16 != 0
        {
            let (hidden, normalized) = match gated {
                Some((residual, gate)) => {
                    let outputs = crate::kernels::fused::adaptive_gate_residual_rms_bf16(
                        ctx, input, residual, gate, style, eps,
                    )?;
                    (Some(outputs.hidden), outputs.normalized)
                }
                None => (
                    None,
                    crate::kernels::norm::adaptive_rms_bf16(ctx, input, style, eps)?,
                ),
            };
            return Ok((hidden, quantize_w8a8_activation(ctx, &normalized, inv_c)?));
        }
        let rows_i32 = checked_i32(op, "rows", rows)?;
        let cols_i32 = checked_i32(op, "cols", cols)?;
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::Other(format!("{op}: size overflow")))?;
        let hidden = if gated.is_some() {
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
        // SAFETY: all sources have validated BF16 shapes/device; optional factors
        // are cols F32. Outputs are workspace-owned, disjoint and correctly sized,
        // including the 512 FP16 scale pad. Every vector address is aligned.
        unsafe {
            check_native(
                ffi::custom::apxinf_musa_ada_rms_quantize_bf16_int8(
                    source,
                    residual,
                    gate_style,
                    norm_style,
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
