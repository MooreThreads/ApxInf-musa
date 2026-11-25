//! π0.5 FP8 MUSA transformer-layer computation.
//!
//! Copied from the CUDA `pi05` FP8 blocks and adapted to this port: runtime
//! intermediates are BF16 (the rest of the MUSA port), the attention epilogue
//! is BF16 followed by an explicit quantizer, and the precomputed patch route
//! is not used. Attention, RoPE/QKV splits and bias activation come from
//! `shared.rs`; this file keeps the linear layers and their explicit quantizers.

use crate::musa_pi05::backend::{ops, Context};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::musa_pi05::{
    Fp8StaticDeviceActionLayer, Fp8StaticDeviceLanguageLayer, Fp8StaticDeviceVisionBlock,
    GemmaVariantConfig,
};

use crate::musa_pi05::{Fp8StaticTransformerLayerScales, Fp8StaticVisionLayerScales};

use super::bf16_vector_prefix;
use super::shared::{
    bias_activation_bf16, mha_bf16, mqa_bf16, mqa_fp8, split_qkv_bias_bf16, split_qkv_rope_bf16,
    split_qkv_rope_into_cache_bf16,
};
use super::L3Policies;

pub struct Fp8StaticLanguageLayerOutput {
    pub hidden: Tensor,
    /// Prefix K/V are retained per layer for the paired action expert.
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Fp8StaticActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

fn output(ctx: &Context, shape: impl Into<Vec<usize>>, dtype: DType) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(shape.into()), dtype)
}

fn fixed_quantize(ctx: &Context, input: &Tensor, scale: f32) -> Result<Tensor> {
    let mut out = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args =
        ops::QuantizationArgs::new(ops::QuantizationSemantic::FixedScaleE4m3, input, &mut out);
    args.scale = scale;
    ops::quantization(ctx, args)?;
    Ok(out)
}

fn fp8_gemm(
    ctx: &Context,
    policies: &L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other("π0.5 FP8 GEMM shape mismatch".into()));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::BF16)?;
    let mut args = ops::GemmArgs::new(input, &weights.weight, &mut out);
    args.quantization = ops::GemmQuantization::Fp8UnitScale;
    args.alpha = input_scale * weights.weight_scale;
    args.policy = policies.gemm.clone();
    ops::gemm(ctx, args)?;
    Ok(out)
}

fn fp8_gemm_bias_gelu_quant(
    ctx: &Context,
    policies: &L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
    output_scale: f32,
) -> Result<Tensor> {
    let bias = weights
        .bias
        .as_ref()
        .ok_or_else(|| Error::Other("π0.5 FP8 fused GEMM+GELU requires a bias".into()))?;
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other("π0.5 FP8 GEMM+GELU shape mismatch".into()));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::F8E4M3)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out);
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.output_scale = output_scale;
    gemm.policy = policies.gemm.clone();
    ops::gemm_bias_gelu(ctx, ops::GemmBiasGeluArgs { gemm, bias })?;
    Ok(out)
}

fn fp8_gemm_bias_residual(
    ctx: &Context,
    policies: &L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
    residual: &Tensor,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 || dims[1] != weight_dims[0] {
        return Err(Error::Other(
            "π0.5 FP8 GEMM+bias+residual shape mismatch".into(),
        ));
    }
    let mut out = output(ctx, vec![dims[0], weight_dims[1]], DType::BF16)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out);
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.policy = policies.gemm.clone();
    ops::gemm_bias_residual(
        ctx,
        ops::GemmBiasResidualArgs {
            gemm,
            bias: weights.bias.as_ref(),
            residual,
        },
    )?;
    Ok(out)
}

fn fp8_gemm_geglu(
    ctx: &Context,
    policies: &L3Policies,
    input: &Tensor,
    input_scale: f32,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
    output_scale: f32,
    layout: ops::GegluWeightLayout,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    let weight_dims = weights.weight.shape().dims();
    if dims.len() != 2 || weight_dims.len() != 2 {
        return Err(Error::Other(
            "π0.5 FP8 GeGLU expects 2-D input and weight".into(),
        ));
    }
    if weight_dims[1] % 2 != 0 {
        return Err(Error::Other("π0.5 FP8 GeGLU width must be even".into()));
    }

    let mut out = output(ctx, vec![dims[0], weight_dims[1] / 2], DType::F8E4M3)?;
    let mut gemm = ops::GemmArgs::new(input, &weights.weight, &mut out);
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = input_scale * weights.weight_scale;
    gemm.output_scale = output_scale;
    gemm.policy = policies.gemm.clone();
    let mut args = ops::GemmGegluArgs::new(gemm);
    // The loader's packing and this layout must agree: the action layers are
    // interleaved, the prefix layers stay half-split for the composed
    // producer's wider vector form.
    args.layout = layout;
    ops::gemm_geglu(ctx, args)?;
    Ok(out)
}

fn rms_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::RmsNormArgs::new(input, weight, &mut normalized, eps);
    args.output_scale = scale;
    ops::rms_norm(ctx, args)?;
    Ok(normalized)
}

fn layer_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::LayerNormArgs::new(input, weight, bias, &mut normalized, eps);
    args.output_scale = scale;
    ops::layer_norm(ctx, args)?;
    Ok(normalized)
}

fn adaptive_rms_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 adaptive RMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let mut normalized = output(ctx, input.shape().dims().to_vec(), DType::F8E4M3)?;
    let mut args = ops::AdaptiveRmsNormArgs::new(input, &norm_style, &mut normalized, eps);
    args.output_scale = scale;
    ops::adaptive_rms_norm(ctx, args)?;
    Ok(normalized)
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_rms_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::BiasResidualRmsNormArgs::new(
        input,
        bias,
        residual,
        weight,
        &mut hidden,
        &mut normalized,
        eps,
    );
    args.output_scale = scale;
    ops::bias_residual_rms_norm(ctx, args)?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::BiasResidualLayerNormArgs::new(
        input,
        bias,
        residual,
        weight,
        norm_bias,
        &mut hidden,
        &mut normalized,
        eps,
    );
    args.output_scale = scale;
    ops::bias_residual_layer_norm(ctx, args)?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn fp8_projection_ada_norm(
    ctx: &Context,
    policies: &L3Policies,
    input: &Tensor,
    input_scale: f32,
    weight: &crate::musa_pi05::Fp8StaticLinearWeights,
    residual: &Tensor,
    norm_style: &Tensor,
    gate_style: &Tensor,
    eps: f32,
    output_scale: f32,
) -> Result<(Tensor, Tensor)> {
    if weight.bias.is_some() {
        return Err(Error::Other(
            "adaptive projection requires a bias-free weight".into(),
        ));
    }
    let norm_style = bf16_vector_prefix(norm_style, 2 * weight.weight.shape().dims()[1])?;
    let shape = residual.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), DType::BF16)?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut gemm = ops::GemmArgs::new(input, &weight.weight, &mut normalized);
    gemm.policy = policies.gemm.clone();
    gemm.alpha = input_scale * weight.weight_scale;
    gemm.output_scale = output_scale;
    gemm.quantization = if input.dtype() == DType::BF16 {
        ops::GemmQuantization::Fp8Bf16Activation { scale: input_scale }
    } else {
        ops::GemmQuantization::Fp8UnitScale
    };
    ops::gemm_ada_gate_residual_rms_norm(
        ctx,
        ops::GemmAdaGateResidualRmsNormArgs::new(
            gemm,
            residual,
            &norm_style,
            gate_style,
            &mut hidden,
            eps,
        ),
    )?;
    Ok((hidden, normalized))
}

#[allow(clippy::too_many_arguments)]
fn ada_gate_residual_rms_normalized_fp8(
    ctx: &Context,
    input: &Tensor,
    residual: &Tensor,
    norm_style: &Tensor,
    gate_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let cols = input
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 AdaRMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let shape = input.shape().dims().to_vec();
    let mut hidden = output(ctx, shape.clone(), input.dtype())?;
    let mut normalized = output(ctx, shape, DType::F8E4M3)?;
    let mut args = ops::AdaGateResidualRmsNormArgs::new(
        input,
        residual,
        &norm_style,
        gate_style,
        &mut hidden,
        &mut normalized,
        eps,
    );
    args.output_scale = scale;
    ops::ada_gate_residual_rms_norm(ctx, args)?;
    Ok((hidden, normalized))
}

pub fn language_layer_fp8_static(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceLanguageLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticLanguageLayerOutput> {
    language_layer_fp8_static_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        scales,
        input,
        compute_tail,
        position_offset,
        rms_eps,
        rope_theta,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn language_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceLanguageLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
    key_cache: Option<&Tensor>,
    value_cache: Option<&Tensor>,
) -> Result<Fp8StaticLanguageLayerOutput> {
    let normalized = rms_normalized_fp8(
        ctx,
        input,
        &weights.input_norm_scale,
        rms_eps,
        scales.attention_norm,
    )?;
    let qkv = fp8_gemm(
        ctx,
        policies,
        &normalized,
        scales.attention_norm,
        &weights.qkv,
    )?;
    let tokens = input.shape().dims()[0];
    // A prefix layer can write K/V straight into the padded action cache. The
    // old path produced `[tokens, head_dim]` and then paid one DtoD memcpy per
    // layer in `reserve_prefix`; the action path already writes into the cache
    // through this helper, so the two paths now share the same form.
    let (q, key, value) = match (key_cache, value_cache) {
        (Some(key_cache), Some(value_cache)) => {
            let q = split_qkv_rope_into_cache_bf16(
                ctx,
                &qkv,
                weights.qkv.bias.as_ref(),
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rope_theta,
                position_offset,
                0,
                key_cache,
                value_cache,
            )?;
            (q, key_cache.clone(), value_cache.clone())
        }
        _ => {
            let qkv = split_qkv_rope_bf16(
                ctx,
                &qkv,
                weights.qkv.bias.as_ref(),
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rope_theta,
                position_offset,
            )?;
            (qkv.q, qkv.k, qkv.v)
        }
    };
    // The composed path returns 3-D K/V; the cache path returns the padded 2-D
    // cache. Only the former needs the `[tokens, head_dim]` reshape.
    let key_out = if key.shape().dims().len() == 3 {
        key.reshape(vec![tokens, config.head_dim])?
    } else {
        key.clone()
    };
    let value_out = if value.shape().dims().len() == 3 {
        value.reshape(vec![tokens, config.head_dim])?
    } else {
        value.clone()
    };
    if !compute_tail {
        return Ok(Fp8StaticLanguageLayerOutput {
            hidden: input.clone(),
            key: key_out,
            value: value_out,
        });
    }
    // MUSA's attention is BF16-only, so the quantizer is always explicit.
    let attention = mqa_bf16(ctx, &policies.attention, &q, &key, &value, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let attention = fixed_quantize(ctx, &attention, scales.attention_output)?;
    let projected = fp8_gemm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
    )?;
    let (hidden, normalized) = bias_residual_rms_normalized_fp8(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
        scales.mlp_norm,
    )?;
    let activated = fp8_gemm_geglu(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.gate_up,
        scales.mlp_activation,
        ops::GegluWeightLayout::HalfSplit,
    )?;
    let hidden = fp8_gemm_bias_residual(
        ctx,
        policies,
        &activated,
        scales.mlp_activation,
        &weights.down,
        &hidden,
    )?;
    Ok(Fp8StaticLanguageLayerOutput {
        hidden,
        key: key_out,
        value: value_out,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_fp8_static(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceActionLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    next_norm_scale: f32,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticActionLayerOutput> {
    action_layer_fp8_static_with_policies(
        ctx,
        &L3Policies::default(),
        config,
        weights,
        scales,
        input,
        attention_normalized,
        attention_modulation,
        mlp_modulation,
        next_norm_modulation,
        next_norm_scale,
        prefix_k,
        prefix_v,
        position_offset,
        rms_eps,
        rope_theta,
    )
}

#[allow(clippy::too_many_arguments)]
fn action_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    config: GemmaVariantConfig,
    weights: &Fp8StaticDeviceActionLayer,
    scales: Fp8StaticTransformerLayerScales,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    next_norm_scale: f32,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Fp8StaticActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(normalized) => normalized.clone(),
        None => adaptive_rms_normalized_fp8(
            ctx,
            input,
            attention_modulation,
            rms_eps,
            scales.attention_norm,
        )?,
    };
    let q = {
        let fused_q = output(
            ctx,
            vec![input.shape().dims()[0], config.num_heads, config.head_dim],
            DType::BF16,
        )?;
        if ops::gemm_fp8_qkv_rope(
            ctx,
            &normalized,
            &weights.qkv.weight,
            weights.qkv.bias.as_ref(),
            &fused_q,
            prefix_k,
            prefix_v,
            rope_theta,
            position_offset,
            position_offset,
            scales.attention_norm * weights.qkv.weight_scale,
        )? {
            fused_q
        } else {
            let qkv = fp8_gemm(
                ctx,
                policies,
                &normalized,
                scales.attention_norm,
                &weights.qkv,
            )?;
            split_qkv_rope_into_cache_bf16(
                ctx,
                &qkv,
                weights.qkv.bias.as_ref(),
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                rope_theta,
                position_offset,
                position_offset,
                prefix_k,
                prefix_v,
            )?
        }
    };
    let key_tokens = position_offset + input.shape().dims()[0];
    // The prefix-cache attention views only the valid prefix rows, so the
    // reserved tail never reaches the softmax.
    let attention = mqa_fp8(
        ctx,
        &policies.attention,
        &q,
        prefix_k,
        prefix_v,
        key_tokens,
        scales.attention_output,
    )?;
    let attention = attention.reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let (hidden, normalized) = fp8_projection_ada_norm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
        input,
        mlp_modulation,
        attention_modulation,
        rms_eps,
        scales.mlp_norm,
    )?;
    let activated = fp8_gemm_geglu(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.gate_up,
        scales.mlp_activation,
        ops::GegluWeightLayout::GateUpPairs {
            pair_width: crate::musa_pi05::GEGLU_PAIR_WIDTH,
        },
    )?;
    let (hidden, normalized) = fp8_projection_ada_norm(
        ctx,
        policies,
        &activated,
        scales.mlp_activation,
        &weights.down,
        &hidden,
        next_norm_modulation,
        mlp_modulation,
        rms_eps,
        next_norm_scale,
    )?;
    Ok(Fp8StaticActionLayerOutput {
        hidden,
        next_normalized: normalized,
    })
}

pub fn vision_patch_embed_fp8_static(
    ctx: &Context,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let policies = L3Policies::default();
    let patches = fixed_quantize(ctx, patches, input_scale)?;
    vision_patch_embed_fp8_static_native_with_policies(
        ctx,
        &policies,
        weights,
        position_embedding,
        &patches,
        patches_per_view,
        input_scale,
    )
}

fn vision_patch_embed_fp8_static_native_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &crate::musa_pi05::Fp8StaticLinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
    input_scale: f32,
) -> Result<Tensor> {
    let projection = fp8_gemm(ctx, policies, patches, input_scale, weights)?;
    let mut out = output(ctx, projection.shape().dims().to_vec(), projection.dtype())?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::BiasPosition, &projection, &mut out);
    args.bias = weights.bias.as_ref();
    args.position = Some(position_embedding);
    args.tokens_per_view = patches_per_view;
    ops::gather(ctx, args)?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_fp8_static(
    ctx: &Context,
    weights: &Fp8StaticDeviceVisionBlock,
    scales: Fp8StaticVisionLayerScales,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    vision_layer_fp8_static_with_policies(
        ctx,
        &L3Policies::default(),
        weights,
        scales,
        input,
        patches_per_view,
        heads,
        head_dim,
        layer_norm_eps,
    )
}

#[allow(clippy::too_many_arguments)]
fn vision_layer_fp8_static_with_policies(
    ctx: &Context,
    policies: &L3Policies,
    weights: &Fp8StaticDeviceVisionBlock,
    scales: Fp8StaticVisionLayerScales,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    if patches_per_view == 0 {
        return Err(Error::Other(
            "PI0.5 FP8 vision layer requires patches_per_view > 0".into(),
        ));
    }
    let normalized = layer_normalized_fp8(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
        scales.attention_norm,
    )?;
    // MUSA has one dense attention contract and no packed-QKV epilogue, so the
    // generic rope split + attention path is the only one.
    let qkv = fp8_gemm(
        ctx,
        policies,
        &normalized,
        scales.attention_norm,
        &weights.qkv,
    )?;
    let tokens = input.shape().dims()[0];
    let qkv = split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = mha_bf16(
        ctx,
        &policies.attention,
        &qkv.q,
        &qkv.k,
        &qkv.v,
        patches_per_view,
    )?
    .reshape(vec![tokens, heads * head_dim])?;
    let attention = fixed_quantize(ctx, &attention, scales.attention_output)?;
    let projection = fp8_gemm(
        ctx,
        policies,
        &attention,
        scales.attention_output,
        &weights.output,
    )?;
    let (hidden, normalized) = bias_residual_layer_normalized_fp8(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
        scales.mlp_norm,
    )?;
    let activation = fp8_gemm_bias_gelu_quant(
        ctx,
        policies,
        &normalized,
        scales.mlp_norm,
        &weights.fc1,
        scales.mlp_activation,
    )?;
    fp8_gemm_bias_residual(
        ctx,
        policies,
        &activation,
        scales.mlp_activation,
        &weights.fc2,
        &hidden,
    )
}

pub(in crate::musa_pi05::model) mod backbone {
    use super::*;
    use crate::musa_pi05::backend::{Context, DeviceBuffer};
    use crate::musa_pi05::weights::*;
    use crate::musa_pi05::Pi05Config;
    use apxinf_core::{Error, Result, Tensor};
    use std::sync::Arc;
    pub struct Fp8StaticPrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Fp8StaticStepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub struct Fp8StaticBlocks {
        pub(in crate::musa_pi05::model) backend: Arc<Context>,
        pub(in crate::musa_pi05::model) config: Arc<Pi05Config>,
        pub(in crate::musa_pi05::model) weights: Arc<Fp8StaticWeights>,
        pub(in crate::musa_pi05::model) scales: Arc<Fp8StaticActivationScales>,
        pub(in crate::musa_pi05::model) policies: L3Policies,
    }
    impl Fp8StaticBlocks {
        pub(in crate::musa_pi05) fn new(
            backend: Arc<Context>,
            config: Arc<Pi05Config>,
            weights: Arc<Fp8StaticWeights>,
            scales: Arc<Fp8StaticActivationScales>,
            policies: L3Policies,
        ) -> Result<Self> {
            config.validate()?;
            scales.validate(&config)?;
            if weights.vision_layers.len() != config.vision_depth
                || weights.language_layers.len() != config.language.depth
                || weights.action_layers.len() != config.action_expert.depth
            {
                return Err(Error::Other("π0.5 device weight depth mismatch".into()));
            }
            Ok(Self {
                backend,
                config,
                weights,
                scales,
                policies,
            })
        }

        fn ctx(&self) -> &Context {
            &self.backend
        }

        // `native` is CUDA's "the patches are already calibrated E4M3"
        // contract. This port's `PrepareBlocks::preprocess` deliberately keeps
        // the runner's ingress BF16 (the graph input buffer, and the BF16/INT8
        // variants' contract), so the raw-RGB capture path hands this function
        // BF16 patches even when it calls `infer_native`. Quantizing here is
        // the one place that keeps both paths correct; a prequantized E4M3
        // gather would have to change the runner's patch dtype and is not worth
        // a second ingress format.
        pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
            if patches.dtype() != DType::BF16 {
                return Err(Error::DTypeMismatch {
                    expected: DType::BF16,
                    got: patches.dtype(),
                });
            }
            let patches = fixed_quantize(self.ctx(), patches, self.scales.vision_patch_input)?;
            self.encode_vision_fp8_patches(&patches)
        }

        pub fn embed_prefix(
            &self,
            vision_tokens: &Tensor,
            token_ids: &DeviceBuffer,
            token_count: usize,
        ) -> Result<Tensor> {
            if token_count == 0 || token_count > self.config.max_token_len {
                return Err(Error::Other(format!(
                    "π0.5 token count must be in 1..={}, got {token_count}",
                    self.config.max_token_len
                )));
            }
            let width = self.weights.token_embedding.shape().dims()[1];
            let mut language = output(self.ctx(), vec![token_count, width], DType::BF16)?;
            let mut args = ops::GatherArgs::new(
                ops::GatherSemantic::EmbeddingLookup,
                &self.weights.token_embedding,
                &mut language,
            );
            args.ids = Some(token_ids);
            ops::gather(self.ctx(), args)?;
            ops::concat_rows(self.ctx(), vision_tokens, &language)
        }

        pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Fp8StaticPrefixKvCache> {
            let mut hidden = prefix.clone();
            let mut keys = Vec::with_capacity(self.config.language.depth);
            let mut values = Vec::with_capacity(self.config.language.depth);
            let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
            for (index, (layer, scale)) in self
                .weights
                .language_layers
                .iter()
                .zip(&self.scales.language_layers)
                .enumerate()
            {
                let key_cache = output(
                    self.ctx(),
                    vec![cache_rows, self.config.language.head_dim],
                    DType::BF16,
                )?;
                let value_cache = output(
                    self.ctx(),
                    vec![cache_rows, self.config.language.head_dim],
                    DType::BF16,
                )?;
                let layer_output = language_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.language,
                    layer,
                    *scale,
                    &hidden,
                    index + 1 < self.config.language.depth,
                    0,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                    Some(&key_cache),
                    Some(&value_cache),
                )?;
                hidden = layer_output.hidden;
                keys.push(key_cache);
                values.push(value_cache);
            }
            Ok(Fp8StaticPrefixKvCache {
                keys,
                values,
                tokens: prefix.shape().dims()[0],
            })
        }

        fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
            let input = fixed_quantize(self.ctx(), time_embedding, self.scales.time_input)?;
            let hidden = fp8_gemm(
                self.ctx(),
                &self.policies,
                &input,
                self.scales.time_input,
                &self.weights.time_mlp_in,
            )?;
            let hidden = bias_activation_bf16(
                self.ctx(),
                &hidden,
                self.weights.time_mlp_in.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )?;
            let hidden = fixed_quantize(self.ctx(), &hidden, self.scales.time_hidden)?;
            let output = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.time_hidden,
                &self.weights.time_mlp_out,
            )?;
            bias_activation_bf16(
                self.ctx(),
                &output,
                self.weights.time_mlp_out.bias.as_ref(),
                ops::PointwiseActivation::Silu,
            )
        }

        fn modulation(
            &self,
            conditioning: &Tensor,
            weights: &crate::musa_pi05::Fp8StaticLinearWeights,
        ) -> Result<Tensor> {
            let projected = fp8_gemm(
                self.ctx(),
                &self.policies,
                conditioning,
                self.scales.conditioning,
                weights,
            )?;
            let modulation = match weights.bias.as_ref() {
                Some(bias) => bias_activation_bf16(
                    self.ctx(),
                    &projected,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => projected,
            };
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(
            &self,
            time_embedding: &Tensor,
        ) -> Result<Fp8StaticStepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let conditioning = fixed_quantize(self.ctx(), &conditioning, self.scales.conditioning)?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Fp8StaticStepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        fn denoise_step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Fp8StaticStepModulation,
            prefix: &Fp8StaticPrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 prefix KV/modulation depth mismatch".into(),
                ));
            }
            let state_fp8 = fixed_quantize(self.ctx(), state, self.scales.action_input)?;
            let hidden = fp8_gemm(
                self.ctx(),
                &self.policies,
                &state_fp8,
                self.scales.action_input,
                &self.weights.action_in,
            )?;
            let mut hidden = match self.weights.action_in.bias.as_ref() {
                Some(bias) => bias_activation_bf16(
                    self.ctx(),
                    &hidden,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => hidden,
            };

            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let (next_norm_modulation, next_norm_scale) =
                    if index + 1 < self.config.action_expert.depth {
                        (
                            &modulation.attention[index + 1],
                            self.scales.action_layers[index + 1].attention_norm,
                        )
                    } else {
                        (&modulation.final_norm, self.scales.action_final_norm)
                    };
                let output = action_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    self.config.action_expert,
                    layer,
                    self.scales.action_layers[index],
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention[index],
                    &modulation.mlp[index],
                    next_norm_modulation,
                    next_norm_scale,
                    &prefix.keys[index],
                    &prefix.values[index],
                    prefix.tokens,
                    self.config.rms_norm_eps,
                    self.config.rope_theta,
                )?;
                hidden = output.hidden;
                attention_normalized = Some(output.next_normalized);
            }
            let hidden = attention_normalized.ok_or_else(|| {
                Error::Other("π0.5 action expert must contain at least one layer".into())
            })?;
            let velocity = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.action_final_norm,
                &self.weights.action_out,
            )?;
            let velocity = match self.weights.action_out.bias.as_ref() {
                Some(bias) => bias_activation_bf16(
                    self.ctx(),
                    &velocity,
                    Some(bias),
                    ops::PointwiseActivation::None,
                )?,
                None => velocity,
            };
            let mut updated = output(self.ctx(), state.shape().dims().to_vec(), state.dtype())?;
            let mut args =
                ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut updated);
            args.secondary = Some(&velocity);
            args.dt = dt;
            ops::pointwise(self.ctx(), args)?;
            Ok(updated)
        }

        fn encode_vision_fp8_patches(&self, patches: &Tensor) -> Result<Tensor> {
            let mut hidden = vision_patch_embed_fp8_static_native_with_policies(
                self.ctx(),
                &self.policies,
                &self.weights.patch_embedding,
                &self.weights.position_embedding,
                patches,
                self.config.patches_per_view(),
                self.scales.vision_patch_input,
            )?;
            for (layer, scale) in self
                .weights
                .vision_layers
                .iter()
                .zip(&self.scales.vision_layers)
            {
                hidden = vision_layer_fp8_static_with_policies(
                    self.ctx(),
                    &self.policies,
                    layer,
                    *scale,
                    &hidden,
                    self.config.patches_per_view(),
                    self.config.vision_heads,
                    self.config.vision_head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            let hidden = layer_normalized_fp8(
                self.ctx(),
                &hidden,
                &self.weights.vision_post_norm.weight,
                &self.weights.vision_post_norm.bias,
                self.config.layer_norm_eps,
                self.scales.vision_post_norm,
            )?;
            let projected = fp8_gemm(
                self.ctx(),
                &self.policies,
                &hidden,
                self.scales.vision_post_norm,
                &self.weights.multimodal_projector,
            )?;
            match self.weights.multimodal_projector.bias.as_ref() {
                Some(bias) => bias_activation_bf16(
                    self.ctx(),
                    &projected,
                    Some(bias),
                    ops::PointwiseActivation::None,
                ),
                None => Ok(projected),
            }
        }
    }

    impl_blocks!(
        Fp8StaticBlocks,
        Fp8StaticPrefixKvCache,
        Fp8StaticStepModulation
    );
}

impl crate::musa_pi05::model::PrepareBlocks for backbone::Fp8StaticBlocks {
    fn backend(&self) -> &std::sync::Arc<crate::musa_pi05::backend::Context> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::musa_pi05::model::WorkspaceRequirements> {
        Ok(crate::musa_pi05::model::WorkspaceRequirements {
            bytes: self.config.graph_workspace_bytes_fp8_static(tokens)?,
        })
    }
    // The default `PrepareBlocks::preprocess` gathers BF16 patches; the FP8
    // patch-embedding entry quantizes them with `vision_patch_input`. Keeping
    // the ingress BF16 avoids changing the runner's patch/noise dtype contract;
    // a direct RGB-to-E4M3 gather is a later optimization if the profile asks.
}

#[cfg(test)]
mod tests {
    //! Numerical parity tests: with zero linear weights every residual path
    //! must return its input exactly. A dropped residual, a shape error or a
    //! wrong operator order moves a value; the synthetic "is it finite" smoke
    //! test cannot see any of those. Scale direction is not covered here: zero
    //! weights make the GEMM result zero regardless of `alpha`.

    use super::*;
    use crate::musa_pi05::backend::{self, Context};
    use crate::musa_pi05::{
        Fp8StaticDeviceLayerNorm, Fp8StaticDeviceVisionBlock, Fp8StaticLinearWeights,
        Fp8StaticTransformerLayerScales, Fp8StaticVisionLayerScales, LinearWeights,
    };
    use half::bf16;

    fn fp8_available(ctx: &Context) -> bool {
        apxinf_musa::FP8_ROUTE != "unsupported" && ctx.caps().arch_family.has_native_fp8()
    }

    fn zero_linear(input: usize, output: usize, ctx: &Context) -> Fp8StaticLinearWeights {
        Fp8StaticLinearWeights::from_host(
            &LinearWeights {
                weight: Tensor::from_f32(vec![input, output], &vec![0.0; input * output]).unwrap(),
                bias: None,
            },
            ctx,
        )
        .unwrap()
    }

    fn zero_linear_with_bias(input: usize, output: usize, ctx: &Context) -> Fp8StaticLinearWeights {
        Fp8StaticLinearWeights::from_host(
            &LinearWeights {
                weight: Tensor::from_f32(vec![input, output], &vec![0.0; input * output]).unwrap(),
                bias: Some(Tensor::from_f32(vec![output], &vec![0.0; output]).unwrap()),
            },
            ctx,
        )
        .unwrap()
    }

    fn bf16_vector(values: &[f32], ctx: &Context) -> Tensor {
        let host: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
        backend::to_device(ctx, &Tensor::from_bf16(vec![values.len()], &host).unwrap()).unwrap()
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fp8_static_zero_weight_language_layer_is_residual_identity() {
        let ctx = Context::new(0).unwrap();
        if !fp8_available(&ctx) {
            eprintln!(
                "skipping fp8_static_zero_weight_language_layer_is_residual_identity: no \
                 native FP8 route"
            );
            return;
        }
        let config = GemmaVariantConfig {
            width: 2048,
            depth: 1,
            mlp_dim: 64,
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
        };
        let norm = bf16_vector(&vec![1.0; config.width], &ctx);
        let weights = Fp8StaticDeviceLanguageLayer {
            input_norm_scale: norm.clone(),
            qkv: zero_linear(config.width, 2560, &ctx),
            output: zero_linear(config.width, config.width, &ctx),
            post_attention_norm_scale: norm,
            gate_up: zero_linear(config.width, 2 * config.mlp_dim, &ctx),
            down: zero_linear(config.mlp_dim, config.width, &ctx),
        };
        let source: Vec<bf16> = (0..9 * config.width)
            .map(|i| bf16::from_f32((i as f32 - 63.0) / 64.0))
            .collect();
        let input = backend::to_device(
            &ctx,
            &Tensor::from_bf16(vec![9, config.width], &source).unwrap(),
        )
        .unwrap();
        let output = language_layer_fp8_static(
            &ctx,
            config,
            &weights,
            Fp8StaticTransformerLayerScales {
                attention_norm: 0.01,
                attention_output: 0.01,
                mlp_norm: 0.01,
                mlp_activation: 0.01,
            },
            &input,
            true,
            0,
            1e-6,
            10_000.0,
        )
        .unwrap();
        let output = backend::to_cpu(&output.hidden).unwrap();
        assert_eq!(output.as_bf16().unwrap(), source.as_slice());
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fp8_static_zero_weight_vision_layer_is_residual_identity_across_views() {
        let ctx = Context::new(0).unwrap();
        if !fp8_available(&ctx) {
            eprintln!(
                "skipping fp8_static_zero_weight_vision_layer_is_residual_identity_across_views: \
                 no native FP8 route"
            );
            return;
        }
        let width = 72usize;
        let inner = 144usize;
        let heads = 1usize;
        let head_dim = 72usize;
        let affine = Fp8StaticDeviceLayerNorm {
            weight: bf16_vector(&vec![1.0; width], &ctx),
            bias: bf16_vector(&vec![0.0; width], &ctx),
        };
        let weights = Fp8StaticDeviceVisionBlock {
            norm1: Fp8StaticDeviceLayerNorm {
                weight: affine.weight.clone(),
                bias: affine.bias.clone(),
            },
            qkv: zero_linear_with_bias(width, 3 * width, &ctx),
            output: zero_linear_with_bias(width, width, &ctx),
            norm2: affine,
            fc1: zero_linear_with_bias(width, inner, &ctx),
            fc2: zero_linear_with_bias(inner, width, &ctx),
        };
        let source: Vec<bf16> = (0..8 * width)
            .map(|i| bf16::from_f32((i as f32 - 63.0) / 64.0))
            .collect();
        let input =
            backend::to_device(&ctx, &Tensor::from_bf16(vec![8, width], &source).unwrap()).unwrap();
        let output = vision_layer_fp8_static(
            &ctx,
            &weights,
            Fp8StaticVisionLayerScales {
                attention_norm: 0.01,
                attention_output: 0.01,
                mlp_norm: 0.01,
                mlp_activation: 0.01,
            },
            &input,
            4,
            heads,
            head_dim,
            1e-6,
        )
        .unwrap();
        let output = backend::to_cpu(&output).unwrap();
        assert_eq!(output.as_bf16().unwrap(), source.as_slice());
    }
}
