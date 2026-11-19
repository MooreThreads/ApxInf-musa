//! GEMM contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::{
    require_deterministic, require_f32_accumulation, require_scaled_output,
    require_unit_output_scale, require_workspace, ScaledOutput,
};
use crate::kernels::activation;
use crate::kernels::gemm as gemm_kernels;
use crate::kernels::quantization;
use crate::kernels::{not_ported, TacticDispatch};
use crate::MusaContext;

pub use crate::kernels::gemm::prepare_fp8_weight;

/// What a model wants from GEMM, declared at load.
///
/// Every field is either enforced here or refused by name. `graph_safe` is the
/// exception in form rather than in effect: MUSA's routes are capture-safe
/// unconditionally, so `true` (the requirement) and `false` (a weaker request)
/// are both satisfied.
#[derive(Clone, Debug)]
pub struct GemmPolicy {
    /// MUSA GEMM accumulates in F32 — the adapter's `MUBLAS_COMPUTE_32F`
    /// contract — so any other value is refused.
    pub accumulation_dtype: DType,
    /// Bounds the bytes the operator allocates for itself: none for a plain
    /// GEMM, the gate buffer for [`gemm_geglu`]. Vendor-library internal
    /// workspaces are outside this backend's accounting.
    pub workspace_limit: usize,
    /// `false` requires an exact tactic record for this shape. A bucket
    /// extrapolation and the muBLAS default are refused; autotuning may measure
    /// the shape and write the record, which satisfies the constraint.
    pub allow_fallback: bool,
    /// Both values are accepted; see the type-level note.
    pub graph_safe: bool,
    /// Bitwise determinism is not guaranteed across the available providers, so
    /// `true` is refused rather than accepted without the guarantee.
    pub deterministic: bool,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
        }
    }
}

/// The constraints this backend carries into one GEMM call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmittedGemmPolicy {
    tactic: TacticDispatch,
}

impl GemmPolicy {
    /// Validate every field and keep only what execution must honour.
    ///
    /// `own_bytes` is the workspace this operator allocates for itself; checking
    /// it before the allocation means a refused call allocates none.
    fn admit(&self, operation: &str, own_bytes: usize) -> Result<AdmittedGemmPolicy> {
        require_f32_accumulation(operation, self.accumulation_dtype)?;
        require_deterministic(operation, self.deterministic)?;
        require_workspace(operation, own_bytes, self.workspace_limit)?;
        Ok(AdmittedGemmPolicy {
            tactic: if self.allow_fallback {
                TacticDispatch::AllowFallback
            } else {
                TacticDispatch::RequireExact
            },
        })
    }
}

/// Quantization contract for the operands supplied to one GEMM call.
///
/// [`Self::None`] and [`Self::Fp8UnitScale`] are served here. The W8A8 route is
/// reached through [`super::w8a8`] instead, whose scale vectors are FP16 with
/// the tile ABI's pad. The per-row [`Self::Fp8`] variant exists so the contract
/// matches upstream's; a caller that asks for it gets a named refusal instead
/// of a wrong answer.
#[derive(Clone, Copy)]
pub enum GemmQuantization<'a> {
    None,
    Fp8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    W8A8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    Fp8UnitScale,
    /// Quantize BF16 A to E4M3 using this scale before the unit-code GEMM.
    /// Alpha already includes activation/weight scales; no extra scale is applied.
    Fp8Bf16Activation {
        scale: f32,
    },
}

pub struct GemmArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    /// Canonical row-major output `[M, N]`. GEMM mutates this tensor.
    pub out: &'a mut Tensor,
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
}

impl<'a> GemmArgs<'a> {
    /// Create a GEMM with canonical contiguous row-major inputs:
    /// `a=[M,K]`, `b=[K,N]`, and `out=[M,N]`.
    pub fn new(a: &'a Tensor, b: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::None,
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
        }
    }
}

/// Try the fixed PI0.5 action qkv+RoPE producer. Returning `false` keeps the
/// caller on the existing GEMM followed by split/rope path for every other
/// shape or backend configuration.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_qkv_rope(
    ctx: &MusaContext,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    theta: f32,
    position_offset: usize,
    kv_output_offset: usize,
    alpha: f32,
) -> Result<bool> {
    gemm_kernels::mp31_fp8::try_qkv_rope(
        ctx,
        a,
        b,
        bias,
        q,
        k,
        v,
        theta,
        position_offset,
        kv_output_offset,
        alpha,
    )
}

/// How a gate/up projection is packed along N.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GegluWeightLayout {
    /// `[gate | up]`: the first half is the gate, the second the up.
    HalfSplit,
    /// `pair_width` gate columns followed by `pair_width` up columns,
    /// repeated. The composed producer pairs columns with this width.
    GateUpPairs { pair_width: usize },
}

pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub layout: GegluWeightLayout,
}

impl<'a> GemmGegluArgs<'a> {
    /// The half-split layout, which is what every existing caller uses.
    pub fn new(gemm: GemmArgs<'a>) -> Self {
        Self {
            gemm,
            layout: GegluWeightLayout::HalfSplit,
        }
    }
}

/// GEMM followed by bias and GELU. The output is BF16, or E4M3 when
/// `gemm.output_scale` carries the next layer's static scale.
pub struct GemmBiasGeluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

/// GEMM followed by bias and a residual add, into a BF16 output.
pub struct GemmBiasResidualArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
}

pub fn gemm(ctx: &MusaContext, args: GemmArgs<'_>) -> Result<()> {
    if let GemmQuantization::Fp8Bf16Activation { scale } = args.quantization {
        let bytes = args.a.shape().numel();
        args.policy.admit("gemm activation quantization", bytes)?;
        if !scale.is_finite() || scale <= 0.0 || !(1.0 / scale).is_finite() {
            return Err(Error::Other(
                "BF16 activation quantization needs a finite positive scale".into(),
            ));
        }
        let mut quantized = ctx.allocate_output(args.a.shape().clone(), DType::F8E4M3)?;
        quantization::quantize_fixed_e4m3(ctx, args.a, &mut quantized, scale)?;
        return gemm(
            ctx,
            GemmArgs {
                a: &quantized,
                quantization: GemmQuantization::Fp8UnitScale,
                ..args
            },
        );
    }
    match args.quantization {
        GemmQuantization::None => {
            require_plain(&args, "gemm")?;
            // A plain GEMM writes the caller's output and allocates nothing of its own.
            let admitted = args.policy.admit("gemm", 0)?;
            gemm_kernels::gemm_bf16_into_with(ctx, args.a, args.b, args.out, admitted.tactic)
        }
        GemmQuantization::Fp8UnitScale => {
            require_unit_output_scale("gemm", args.output_scale)?;
            let _ = args.policy.admit("gemm", 0)?;
            gemm_kernels::gemm_fp8_into(ctx, args.a, args.b, args.out, args.alpha, "gemm")
        }
        GemmQuantization::W8A8 { .. } => Err(not_ported("gemm with W8A8 operands")),
        GemmQuantization::Fp8Bf16Activation { .. } => {
            unreachable!("BF16 activation encoded before GEMM dispatch")
        }
        GemmQuantization::Fp8 { .. } => Err(not_ported("gemm with per-row FP8 operands")),
    }
}

/// `[M, K] @ [K, N]` followed by bias and GELU.
///
/// The static-FP8 route asks for an E4M3 output when the next GEMM consumes the
/// activation; `output_scale` is that next GEMM's calibrated scale. The composed
/// implementation (GEMM, bias+GELU, quantize) is deliberate: a fused kernel is
/// deferred until a profile shows the extra pass matters.
pub fn gemm_bias_gelu(ctx: &MusaContext, args: GemmBiasGeluArgs<'_>) -> Result<()> {
    let output = require_scaled_output(args.gemm.out, args.gemm.output_scale, "gemm_bias_gelu")?;
    let mut scratch = ctx.allocate_output(args.gemm.out.shape().clone(), DType::BF16)?;
    gemm_into_scratch(ctx, &args.gemm, &mut scratch)?;
    match output {
        ScaledOutput::Bf16 => {
            activation::bias_gelu_bf16_into(ctx, &scratch, Some(args.bias), args.gemm.out)
        }
        ScaledOutput::E4m3(scale) => {
            let cols = args.gemm.out.shape().dims().get(1).copied().unwrap_or(0);
            if activation::bias_gelu_e4m3_supported(cols) {
                activation::bias_gelu_bf16_e4m3_into(
                    ctx,
                    &scratch,
                    Some(args.bias),
                    args.gemm.out,
                    scale,
                )
            } else {
                let mut activated =
                    ctx.allocate_output(args.gemm.out.shape().clone(), DType::BF16)?;
                activation::bias_gelu_bf16_into(ctx, &scratch, Some(args.bias), &mut activated)?;
                quantization::quantize_fixed_e4m3(ctx, &activated, args.gemm.out, scale)
            }
        }
    }
}

/// `[M, K] @ [K, N]` followed by an optional bias and a residual add.
pub fn gemm_bias_residual(ctx: &MusaContext, args: GemmBiasResidualArgs<'_>) -> Result<()> {
    if args.gemm.out.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA gemm_bias_residual writes BF16 only, got {}",
            args.gemm.out.dtype()
        )));
    }
    require_unit_output_scale("gemm_bias_residual", args.gemm.output_scale)?;
    let mut scratch = ctx.allocate_output(args.gemm.out.shape().clone(), DType::BF16)?;
    gemm_into_scratch(ctx, &args.gemm, &mut scratch)?;
    super::norm::bias_residual(
        ctx,
        super::norm::BiasResidualArgs::new(&scratch, args.bias, args.residual, args.gemm.out),
    )
}

/// `[M, K] @ [K, 2N]` followed by gated GELU, writing `[M, N]`.
pub fn gemm_geglu(ctx: &MusaContext, args: GemmGegluArgs<'_>) -> Result<()> {
    if let GemmQuantization::Fp8Bf16Activation { scale } = args.gemm.quantization {
        let (m, n, _) = matrix_dims(args.gemm.a, args.gemm.b)?;
        let bytes = args.gemm.a.shape().numel() + m * n * 2;
        args.gemm.policy.admit("gemm_geglu", bytes)?;
        let mut quantized = ctx.allocate_output(args.gemm.a.shape().clone(), DType::F8E4M3)?;
        quantization::quantize_fixed_e4m3(ctx, args.gemm.a, &mut quantized, scale)?;
        let gemm = GemmArgs {
            a: &quantized,
            quantization: GemmQuantization::Fp8UnitScale,
            ..args.gemm
        };
        return gemm_geglu(ctx, GemmGegluArgs { gemm, ..args });
    }
    if matches!(args.layout, GegluWeightLayout::GateUpPairs { .. })
        && matches!(args.gemm.quantization, GemmQuantization::None)
    {
        return Err(Error::Other(
            "MUSA gemm_geglu pair layout requires the static-FP8 E4M3 output".into(),
        ));
    }
    match args.gemm.quantization {
        GemmQuantization::None => {
            require_plain(&args.gemm, "gemm_geglu")?;
            let [rows, _inner] = args.gemm.a.shape().dims() else {
                return Err(Error::Other(
                    "MUSA gemm_geglu activation must be a 2-D tensor".into(),
                ));
            };
            let [_, gate_cols] = args.gemm.b.shape().dims() else {
                return Err(Error::Other(
                    "MUSA gemm_geglu weight must be a 2-D tensor".into(),
                ));
            };
            // The gate is the one buffer this operator allocates for itself, so it is
            // what `workspace_limit` bounds. Computed before the allocation: a limit
            // that cannot cover it refuses the call with nothing allocated.
            let gate_bytes = rows
                .checked_mul(*gate_cols)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("MUSA gemm_geglu gate size overflow".into()))?;
            let admitted = args.gemm.policy.admit("gemm_geglu", gate_bytes)?;
            let mut gate = ctx.allocate_output(
                apxinf_core::Shape::new(vec![*rows, *gate_cols]),
                DType::BF16,
            )?;
            gemm_kernels::gemm_bf16_into_with(
                ctx,
                args.gemm.a,
                args.gemm.b,
                &mut gate,
                admitted.tactic,
            )?;
            activation::geglu_bf16_into(ctx, &gate, args.gemm.out)
        }
        GemmQuantization::Fp8UnitScale => {
            let output =
                require_scaled_output(args.gemm.out, args.gemm.output_scale, "gemm_geglu")?;
            let pair_width = match args.layout {
                GegluWeightLayout::HalfSplit => 0,
                GegluWeightLayout::GateUpPairs { pair_width } => pair_width,
            };
            let [rows, _inner] = args.gemm.a.shape().dims() else {
                return Err(Error::Other(
                    "MUSA gemm_geglu activation must be a 2-D tensor".into(),
                ));
            };
            let [_, gate_cols] = args.gemm.b.shape().dims() else {
                return Err(Error::Other(
                    "MUSA gemm_geglu weight must be a 2-D tensor".into(),
                ));
            };
            if gate_cols % 2 != 0 {
                return Err(Error::Other(
                    "MUSA gemm_geglu weight width must be even".into(),
                ));
            }
            let gate_bytes = rows
                .checked_mul(*gate_cols)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("MUSA gemm_geglu gate size overflow".into()))?;
            if let ScaledOutput::E4m3(scale) = output {
                args.gemm.policy.admit("gemm_geglu", 0)?;
                if matches!(args.gemm.quantization, GemmQuantization::Fp8UnitScale)
                    && gemm_kernels::mp31_fp8::try_geglu(
                        ctx,
                        args.gemm.a,
                        args.gemm.b,
                        args.gemm.out,
                        args.gemm.alpha,
                        scale,
                        pair_width,
                    )?
                {
                    return Ok(());
                }
            }
            let own_bytes = gate_bytes
                + if matches!(
                    args.gemm.quantization,
                    GemmQuantization::Fp8Bf16Activation { .. }
                ) {
                    args.gemm.a.shape().numel()
                } else {
                    0
                };
            let _ = args.gemm.policy.admit("gemm_geglu", own_bytes)?;
            let mut gate = ctx.allocate_output(
                apxinf_core::Shape::new(vec![*rows, *gate_cols]),
                DType::BF16,
            )?;
            gemm_kernels::gemm_fp8_into(
                ctx,
                args.gemm.a,
                args.gemm.b,
                &mut gate,
                args.gemm.alpha,
                "gemm_geglu",
            )?;
            match output {
                ScaledOutput::Bf16 => {
                    if pair_width != 0 {
                        return Err(Error::Other(
                            "MUSA gemm_geglu pair layout writes E4M3 only".into(),
                        ));
                    }
                    activation::geglu_bf16_into(ctx, &gate, args.gemm.out)
                }
                ScaledOutput::E4m3(scale) => {
                    activation::geglu_bf16_e4m3_into(ctx, &gate, args.gemm.out, scale, pair_width)
                }
            }
        }
        GemmQuantization::W8A8 { .. } => Err(not_ported("gemm_geglu with W8A8 operands")),
        GemmQuantization::Fp8Bf16Activation { .. } => {
            unreachable!("BF16 activation encoded before GEMM dispatch")
        }
        GemmQuantization::Fp8 { .. } => Err(not_ported("gemm_geglu with per-row FP8 operands")),
    }
}

/// Run the caller's GEMM into a fresh BF16 scratch, leaving `out` for the
/// post-GEMM operator.
fn gemm_into_scratch(ctx: &MusaContext, args: &GemmArgs<'_>, scratch: &mut Tensor) -> Result<()> {
    let mut scratch_args = GemmArgs::new(args.a, args.b, scratch);
    scratch_args.quantization = args.quantization;
    scratch_args.alpha = args.alpha;
    scratch_args.output_scale = 1.0;
    scratch_args.policy = args.policy.clone();
    gemm(ctx, scratch_args)
}

fn require_plain(args: &GemmArgs<'_>, operation: &str) -> Result<()> {
    if args.alpha != 1.0 {
        return Err(Error::Other(format!(
            "MUSA {operation} applies no alpha scaling; got alpha {}",
            args.alpha
        )));
    }
    require_unit_output_scale(operation, args.output_scale)
}

#[cfg(test)]
mod policy_tests {
    use super::*;

    fn policy() -> GemmPolicy {
        GemmPolicy::default()
    }

    #[test]
    fn the_default_policy_is_admitted() {
        let admitted = policy().admit("gemm", 0).expect("the default policy");
        assert_eq!(admitted.tactic, TacticDispatch::AllowFallback);
    }

    #[test]
    fn a_non_f32_accumulator_is_refused() {
        let error = GemmPolicy {
            accumulation_dtype: DType::F16,
            ..policy()
        }
        .admit("gemm", 0)
        .expect_err("F32 is the only accumulation contract");
        assert!(
            error.to_string().contains("accumulation_dtype f16"),
            "{error}"
        );
        assert!(error.to_string().contains("F32"), "{error}");
    }

    #[test]
    fn deterministic_is_refused() {
        let error = GemmPolicy {
            deterministic: true,
            ..policy()
        }
        .admit("gemm", 0)
        .expect_err("there is no bitwise-determinism contract to honour");
        assert!(error.to_string().contains("deterministic"), "{error}");
    }

    #[test]
    fn a_workspace_limit_below_the_gate_is_refused() {
        let error = GemmPolicy {
            workspace_limit: 1023,
            ..policy()
        }
        .admit("gemm_geglu", 4096)
        .expect_err("the gate does not fit");
        assert!(error.to_string().contains("4096"), "{error}");
        assert!(error.to_string().contains("1023"), "{error}");
        assert!(GemmPolicy {
            workspace_limit: 4096,
            ..policy()
        }
        .admit("gemm_geglu", 4096)
        .is_ok());
    }

    #[test]
    fn a_plain_gemm_needs_no_workspace() {
        assert!(GemmPolicy {
            workspace_limit: 0,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
    }

    #[test]
    fn allow_fallback_false_requires_an_exact_tactic() {
        let admitted = GemmPolicy {
            allow_fallback: false,
            ..policy()
        }
        .admit("gemm", 0)
        .expect("a valid policy");
        assert_eq!(admitted.tactic, TacticDispatch::RequireExact);
    }

    #[test]
    fn graph_safe_is_accepted_either_way() {
        assert!(GemmPolicy {
            graph_safe: true,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
        assert!(GemmPolicy {
            graph_safe: false,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
    }
}

/// FP8 projection followed by BF16 gate/residual and adaptive RMS normalization.
/// `gemm.out` is the final E4M3 output; `hidden` receives the rounded BF16 residual.
pub struct GemmAdaGateResidualRmsNormArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub residual: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub eps: f32,
}
impl<'a> GemmAdaGateResidualRmsNormArgs<'a> {
    pub fn new(
        gemm: GemmArgs<'a>,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            gemm,
            residual,
            gate_style,
            norm_style,
            hidden,
            eps,
        }
    }
}

pub fn gemm_ada_gate_residual_rms_norm(
    ctx: &MusaContext,
    args: GemmAdaGateResidualRmsNormArgs<'_>,
) -> Result<()> {
    let op = "gemm_ada_gate_residual_rms_norm";
    let (m, n, _k) = matrix_dims(args.gemm.a, args.gemm.b)?;
    let input_scale = match args.gemm.quantization {
        GemmQuantization::Fp8UnitScale => None,
        GemmQuantization::Fp8Bf16Activation { scale } => Some(scale),
        _ => {
            return Err(Error::Other(format!(
                "{op} only supports static FP8 projection"
            )))
        }
    };
    if !args.gemm.alpha.is_finite() || !args.eps.is_finite() || args.eps < 0.0 {
        return Err(Error::Other(format!(
            "{op} requires finite alpha and nonnegative finite eps"
        )));
    }
    let scale = args.gemm.output_scale;
    if !scale.is_finite()
        || scale <= 0.0
        || !(1.0 / scale).is_finite()
        || input_scale.is_some_and(|s| !s.is_finite() || s <= 0.0 || !(1.0 / s).is_finite())
    {
        return Err(Error::Other(format!(
            "{op} requires finite positive quantization scales"
        )));
    }
    gemm_kernels::mp31_fp8::validate_projection_norm(
        ctx,
        args.gemm.a,
        args.gemm.b,
        args.residual,
        args.gate_style,
        args.norm_style,
        args.hidden,
        args.gemm.out,
        input_scale,
    )?;
    let admitted = args.gemm.policy.admit(op, 0)?;
    if admitted.tactic == TacticDispatch::AllowFallback
        && gemm_kernels::mp31_fp8::try_projection_norm(
            ctx,
            args.gemm.a,
            args.gemm.b,
            args.residual,
            args.gate_style,
            args.norm_style,
            args.hidden,
            args.gemm.out,
            input_scale,
            args.gemm.alpha,
            scale,
            args.eps,
            args.gemm.policy.workspace_limit,
        )?
    {
        return Ok(());
    }
    let own = m
        .checked_mul(n)
        .and_then(|x| x.checked_mul(2))
        .and_then(|x| {
            x.checked_add(if input_scale.is_some() {
                args.gemm.a.shape().numel()
            } else {
                0
            })
        })
        .ok_or_else(|| Error::Other(format!("{op} scratch size overflow")))?;
    args.gemm.policy.admit(op, own)?;
    let mut projection = ctx.allocate_output(apxinf_core::Shape::new(vec![m, n]), DType::BF16)?;
    let normalized = args.gemm.out;
    gemm(
        ctx,
        GemmArgs {
            out: &mut projection,
            output_scale: 1.0,
            ..args.gemm
        },
    )?;
    let mut norm = super::AdaGateResidualRmsNormArgs::new(
        &projection,
        args.residual,
        args.norm_style,
        args.gate_style,
        args.hidden,
        normalized,
        args.eps,
    );
    norm.output_scale = scale;
    super::ada_gate_residual_rms_norm(ctx, norm)
}

#[cfg(all(test, apxinf_musa_available, apxinf_musa_mutlass_fmha))]
mod projection_norm_tests {
    use super::*;
    use crate::kernels::test_support::{upload, values};
    use crate::{workspace::ExecutionSession, MusaBuffer};
    use apxinf_core::{Graph, Shape};
    use std::sync::Arc;
    fn fp8(ctx: &MusaContext, rows: usize, cols: usize, seed: u32) -> Tensor {
        let mut rng = seed;
        let data = (0..rows * cols)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                (0x28 + ((rng >> 24) as u8 & 23)) | ((rng >> 16) as u8 & 128)
            })
            .collect::<Vec<_>>();
        crate::transfers::to_device(
            &Tensor::from_f8_e4m3(vec![rows, cols], &data).unwrap(),
            ctx.device_id(),
        )
        .unwrap()
    }
    fn bytes(t: &Tensor) -> Vec<u8> {
        let buffer = MusaBuffer::from_tensor(t).unwrap();
        let mut out = vec![0; buffer.len()];
        buffer.copy_to_host(&mut out).unwrap();
        out
    }
    fn decode(b: u8) -> f64 {
        let e = (b >> 3) & 15;
        let f = b & 7;
        let x = if e == 0 {
            f as f64 * 2f64.powi(-9)
        } else {
            (1. + f as f64 / 8.) * 2f64.powi(e as i32 - 7)
        };
        if b & 128 == 0 {
            x
        } else {
            -x
        }
    }
    #[test]
    #[ignore = "requires MP31; set APXINF_MUSA_BENCH_DEVICE"]
    fn projection_norm_alpha_scales_poison_graph_and_shapes() {
        let device = std::env::var("APXINF_MUSA_BENCH_DEVICE")
            .unwrap_or("5".into())
            .parse()
            .unwrap();
        let ctx = Arc::new(MusaContext::new(device).unwrap());
        if crate::MUSA_ARCH != "mp_31" {
            return;
        }
        let n = 1024;
        for (k, bf16_input) in [(2048, true), (4096, false)] {
            let w = fp8(&ctx, k, n, 13);
            prepare_fp8_weight(&ctx, &w).unwrap();
            for m in [10, 21, 10] {
                let a = if bf16_input {
                    upload(&ctx, &values(m * k, 83), Shape::new(vec![m, k]))
                } else {
                    fp8(&ctx, m, k, 83)
                };
                let residual = upload(&ctx, &values(m * n, 51), Shape::new(vec![m, n]));
                let gate = upload(&ctx, &values(3 * n, 23), Shape::new(vec![3 * n]));
                let style = upload(&ctx, &values(2 * n, 29), Shape::new(vec![2 * n]));
                for (alpha, scale) in [(0., 0.03125), (-0.037, 0.0197), (0.0019, 0.03125)] {
                    let input_scale = 0.0197;
                    let quant = if bf16_input {
                        GemmQuantization::Fp8Bf16Activation { scale: input_scale }
                    } else {
                        GemmQuantization::Fp8UnitScale
                    };
                    let mut projection = ctx
                        .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                        .unwrap();
                    let mut reference_hidden = ctx
                        .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                        .unwrap();
                    let mut reference = ctx
                        .allocate_output(Shape::new(vec![m, n]), DType::F8E4M3)
                        .unwrap();
                    let mut g = GemmArgs::new(&a, &w, &mut projection);
                    g.alpha = alpha;
                    g.quantization = quant;
                    gemm(&ctx, g).unwrap();
                    let mut norm = super::super::AdaGateResidualRmsNormArgs::new(
                        &projection,
                        &residual,
                        &style,
                        &gate,
                        &mut reference_hidden,
                        &mut reference,
                        1e-6,
                    );
                    norm.output_scale = scale;
                    super::super::ada_gate_residual_rms_norm(&ctx, norm).unwrap();
                    let norm_storage = MusaBuffer::alloc_on(&ctx, m * n + 128).unwrap();
                    norm_storage
                        .copy_from_host(&vec![127; m * n + 128])
                        .unwrap();
                    let hidden_storage = MusaBuffer::alloc_on(&ctx, m * n * 2 + 128).unwrap();
                    hidden_storage
                        .copy_from_host(&vec![127; m * n * 2 + 128])
                        .unwrap();
                    let mut normalized = norm_storage
                        .clone()
                        .into_tensor(Shape::new(vec![m, n]), DType::F8E4M3);
                    let mut hidden = hidden_storage
                        .clone()
                        .into_tensor(Shape::new(vec![m, n]), DType::BF16);
                    let call = |hidden: &mut Tensor, normalized: &mut Tensor| {
                        let mut g = GemmArgs::new(&a, &w, normalized);
                        g.alpha = alpha;
                        g.quantization = quant;
                        g.output_scale = scale;
                        gemm_ada_gate_residual_rms_norm(
                            &ctx,
                            GemmAdaGateResidualRmsNormArgs::new(
                                g, &residual, &style, &gate, hidden, 1e-6,
                            ),
                        )
                    };
                    call(&mut hidden, &mut normalized).unwrap();
                    ctx.synchronize().unwrap();
                    let actual = bytes(&normalized);
                    let expected = bytes(&reference);
                    let mut dot = 0.;
                    let mut aa = 0.;
                    let mut bb = 0.;
                    let mut differences = 0;
                    for (&x, &y) in actual[..m * n].iter().zip(&expected) {
                        assert_ne!(x & 127, 127, "unwritten/NaN output");
                        differences += usize::from(x != y);
                        let (x, y) = (decode(x), decode(y));
                        dot += x * y;
                        aa += x * x;
                        bb += y * y;
                    }
                    let cosine = dot / (aa * bb).sqrt();
                    // The shipped M=10 consumer keeps the FP32 intermediate
                    // (the measured default), so it intentionally differs from
                    // this composed BF16 reference. The measured value is
                    // 0.99988 for k=2048; the bound still rejects a broken
                    // alpha/scale path.
                    assert!(
                        cosine >= 0.9998,
                        "m={m} k={k} alpha={alpha} cosine={cosine} bytes={differences}"
                    );
                    assert!(actual[m * n..].iter().all(|b| *b == 127));
                    let hidden_bytes = bytes(&hidden);
                    assert!(hidden_bytes[m * n * 2..].iter().all(|b| *b == 127));
                    let got = hidden_bytes[..m * n * 2]
                        .chunks_exact(2)
                        .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
                        .collect::<Vec<_>>();
                    let expected_hidden = bytes(&reference_hidden)
                        .chunks_exact(2)
                        .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
                        .collect::<Vec<_>>();
                    assert!(
                        got.iter()
                            .take(m * n)
                            .zip(&expected_hidden)
                            .all(|(x, y)| (x - y).abs() <= 0.015625),
                        "hidden mismatch"
                    );
                    let session =
                        ExecutionSession::with_capacity(32 * 1024 * 1024, device).unwrap();
                    session
                        .prepare(|| call(&mut hidden, &mut normalized))
                        .unwrap();
                    let graph = crate::graph::capture(ctx.clone(), || {
                        session.run(|| call(&mut hidden, &mut normalized))
                    })
                    .unwrap();
                    for _ in 0..1000 {
                        graph.replay().unwrap();
                    }
                    ctx.synchronize().unwrap();
                    assert!(bytes(&normalized) == actual, "eager/replay mismatch");
                    assert!(bytes(&hidden) == hidden_bytes, "hidden replay mismatch");
                    eprintln!("m={m} k={k} alpha={alpha} scale={scale} cosine={cosine:.9} bytes={differences}");
                }
            }
        }
    }
}

fn matrix_dims(a: &Tensor, b: &Tensor) -> Result<(usize, usize, usize)> {
    let (a, b) = (a.shape().dims(), b.shape().dims());
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || a[0] == 0 || a[1] == 0 || b[1] == 0 {
        return Err(Error::Other(
            "GEMM requires nonempty compatible [M,K] and [K,N] matrices".into(),
        ));
    }
    Ok((a[0], b[1], a[1]))
}
