//! Row-wise normalization contracts.

use apxinf_core::{DType, Result, Tensor};

use super::{require_scaled_output, ScaledOutput};
use crate::kernels::fused as fused_kernels;
use crate::kernels::norm as norm_kernels;
use crate::kernels::quantization;
use crate::MusaContext;

/// The last-axis width, for the fused-E4M3 width predicates.
///
/// The operators validate dimensionality themselves; a tensor that is not 2-D
/// falls through to the composed path, which rejects it with the operator's own
/// message rather than a width error from here.
fn last_dim(tensor: &Tensor) -> usize {
    tensor.shape().dims().last().copied().unwrap_or(0)
}

/// Inputs for RMS normalization over `input`'s last axis.
pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when the output is quantized. A BF16
    /// output requires 1.0; an E4M3 output requires a finite positive scale.
    pub output_scale: f32,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for ordinary layer normalization.
pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        bias: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for adaptive RMS normalization. `norm_style` is the scale-then-shift
/// pair, `[2 * cols]`; a longer vector is accepted because PI0.5's modulation
/// carries a third gate segment this operator does not consume.
pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        norm_style: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            norm_style,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for bias plus residual. Bias is optional by contract.
pub struct BiasResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> BiasResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            hidden,
        }
    }
}

pub struct BiasResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

pub struct BiasResidualLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        norm_bias: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            norm_bias,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for gated residual plus adaptive RMS normalization. `norm_style` is
/// `[2 * cols]`; `gate_style` is `[3 * cols]` and its gate is the third
/// segment.
pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual,
            norm_style,
            gate_style,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Run a normalization into `out`, through a BF16 scratch when `out` is E4M3.
///
/// The producers that still reach this helper compose; `ada_gate_residual_rms_norm`
/// has its own fused E4M3 render.
fn write_normalized(
    ctx: &MusaContext,
    out: &mut Tensor,
    scale: f32,
    operation: &str,
    write_bf16: impl FnOnce(&mut Tensor) -> Result<()>,
) -> Result<()> {
    match require_scaled_output(out, scale, operation)? {
        ScaledOutput::Bf16 => write_bf16(out),
        ScaledOutput::E4m3(scale) => {
            let mut scratch = ctx.allocate_output(out.shape().clone(), DType::BF16)?;
            write_bf16(&mut scratch)?;
            quantization::quantize_fixed_e4m3(ctx, &scratch, out, scale)
        }
    }
}

pub fn rms_norm(ctx: &MusaContext, args: RmsNormArgs<'_>) -> Result<()> {
    let RmsNormArgs {
        input,
        weight,
        normalized,
        eps,
        output_scale,
    } = args;
    if let ScaledOutput::E4m3(scale) = require_scaled_output(normalized, output_scale, "rms_norm")?
    {
        if norm_kernels::rms_e4m3_supported(last_dim(input)) {
            return norm_kernels::rms_bf16_e4m3_into(ctx, input, weight, normalized, eps, scale);
        }
    }
    write_normalized(ctx, normalized, output_scale, "rms_norm", |out| {
        norm_kernels::rms_bf16_into(ctx, input, weight, out, eps)
    })
}

pub fn layer_norm(ctx: &MusaContext, args: LayerNormArgs<'_>) -> Result<()> {
    let LayerNormArgs {
        input,
        weight,
        bias,
        normalized,
        eps,
        output_scale,
    } = args;
    if let ScaledOutput::E4m3(scale) =
        require_scaled_output(normalized, output_scale, "layer_norm")?
    {
        if norm_kernels::layer_e4m3_supported(last_dim(input)) {
            return norm_kernels::layer_bf16_e4m3_into(
                ctx, input, weight, bias, normalized, eps, scale,
            );
        }
    }
    write_normalized(ctx, normalized, output_scale, "layer_norm", |out| {
        norm_kernels::layer_bf16_into(ctx, input, weight, bias, out, eps)
    })
}

pub fn adaptive_rms_norm(ctx: &MusaContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    let AdaptiveRmsNormArgs {
        input,
        norm_style,
        normalized,
        eps,
        output_scale,
    } = args;
    if let ScaledOutput::E4m3(scale) =
        require_scaled_output(normalized, output_scale, "adaptive_rms_norm")?
    {
        if norm_kernels::adaptive_rms_e4m3_supported(last_dim(input)) {
            return norm_kernels::adaptive_rms_bf16_e4m3_into(
                ctx, input, norm_style, normalized, eps, scale,
            );
        }
    }
    write_normalized(ctx, normalized, output_scale, "adaptive_rms_norm", |out| {
        norm_kernels::adaptive_rms_bf16_into(ctx, input, norm_style, out, eps)
    })
}

pub fn bias_residual(ctx: &MusaContext, args: BiasResidualArgs<'_>) -> Result<()> {
    fused_kernels::bias_residual_bf16_into(ctx, args.input, args.bias, args.residual, args.hidden)
}

pub fn bias_residual_rms_norm(ctx: &MusaContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    let BiasResidualRmsNormArgs {
        input,
        bias,
        residual,
        weight,
        hidden,
        normalized,
        eps,
        output_scale,
    } = args;
    if let ScaledOutput::E4m3(scale) =
        require_scaled_output(normalized, output_scale, "bias_residual_rms_norm")?
    {
        if fused_kernels::bias_residual_rms_e4m3_supported(last_dim(input)) {
            return fused_kernels::bias_residual_rms_e4m3_into(
                ctx, input, bias, residual, weight, hidden, normalized, eps, scale,
            );
        }
    }
    write_normalized(
        ctx,
        normalized,
        output_scale,
        "bias_residual_rms_norm",
        |out| {
            fused_kernels::bias_residual_rms_bf16_into(
                ctx, input, bias, residual, weight, hidden, out, eps,
            )
        },
    )
}

pub fn bias_residual_layer_norm(
    ctx: &MusaContext,
    args: BiasResidualLayerNormArgs<'_>,
) -> Result<()> {
    let BiasResidualLayerNormArgs {
        input,
        bias,
        residual,
        weight,
        norm_bias,
        hidden,
        normalized,
        eps,
        output_scale,
    } = args;
    if let ScaledOutput::E4m3(scale) =
        require_scaled_output(normalized, output_scale, "bias_residual_layer_norm")?
    {
        if fused_kernels::bias_residual_layer_e4m3_supported(last_dim(input)) {
            return fused_kernels::bias_residual_layer_e4m3_into(
                ctx, input, bias, residual, weight, norm_bias, hidden, normalized, eps, scale,
            );
        }
    }
    write_normalized(
        ctx,
        normalized,
        output_scale,
        "bias_residual_layer_norm",
        |out| {
            fused_kernels::bias_residual_layer_bf16_into(
                ctx, input, bias, residual, weight, norm_bias, hidden, out, eps,
            )
        },
    )
}

pub fn ada_gate_residual_rms_norm(
    ctx: &MusaContext,
    args: AdaGateResidualRmsNormArgs<'_>,
) -> Result<()> {
    let AdaGateResidualRmsNormArgs {
        input,
        residual,
        norm_style,
        gate_style,
        hidden,
        normalized,
        eps,
        output_scale,
    } = args;
    match require_scaled_output(normalized, output_scale, "ada_gate_residual_rms_norm")? {
        ScaledOutput::Bf16 => fused_kernels::adaptive_gate_residual_rms_bf16_into(
            ctx, input, residual, gate_style, norm_style, hidden, normalized, eps,
        ),
        ScaledOutput::E4m3(scale) => fused_kernels::adaptive_gate_residual_rms_e4m3_into(
            ctx, input, residual, gate_style, norm_style, hidden, normalized, eps, scale,
        ),
    }
}
