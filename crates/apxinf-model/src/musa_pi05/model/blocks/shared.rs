//! BF16 tensor producers shared by every PI0.5 precision.
//!
//! The precision lives in the linear layers. Normalization, element-wise
//! activation, attention and the RoPE/QKV splits operate on BF16 tensors in the
//! BF16, INT8 and static-FP8 routes alike, so they live here instead of in each
//! block file.

use crate::musa_pi05::backend::{
    ops::{
        ada_gate_residual_rms_norm, attention, gather, pointwise, rms_norm, rope,
        AdaGateResidualRmsNormArgs, AttentionArgs, AttentionPolicy, GatherArgs, GatherSemantic,
        PointwiseActivation, PointwiseArgs, PointwiseSemantic, RmsNormArgs, RopeArgs,
        RopeSemantic,
    },
    Context, DeviceBuffer,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::{bf16_vector_prefix, QkvTensors};

pub(super) fn rms_bf16(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    rms_norm(ctx, RmsNormArgs::new(input, weight, &mut output, eps))?;
    Ok(output)
}

pub(super) fn bias_activation_bf16(
    ctx: &Context,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: PointwiseActivation,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = bias;
    args.activation = activation;
    pointwise(ctx, args)?;
    Ok(output)
}

pub(super) fn euler_update_bf16(
    ctx: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(state.shape().clone(), DType::BF16)?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, state, &mut output);
    args.secondary = Some(velocity);
    args.dt = dt;
    pointwise(ctx, args)?;
    Ok(output)
}

pub(super) fn add_position_bf16(
    ctx: &Context,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut args = GatherArgs::new(GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(position);
    args.tokens_per_view = tokens_per_view;
    gather(ctx, args)?;
    Ok(output)
}

pub(super) fn mqa_bf16(
    ctx: &Context,
    policy: &AttentionPolicy,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q = query.shape().dims();
    let k = key.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || q.len() != 3
        || value.shape() != key.shape()
        || !matches!(k.len(), 2 | 3)
        || k[k.len() - 1] != q[2]
    {
        return Err(Error::Other("PI0.5 BF16 MQA shape/dtype mismatch".into()));
    }
    let kv_heads = if k.len() == 2 { 1 } else { k[1] };
    let available_tokens = k[0];
    if key_tokens == 0 || key_tokens > available_tokens || q[1] % kv_heads != 0 {
        return Err(Error::Other("PI0.5 BF16 MQA token/head mismatch".into()));
    }
    let kv_elements = key_tokens
        .checked_mul(kv_heads)
        .and_then(|count| count.checked_mul(q[2]))
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA size overflow".into()))?;
    let kv_bytes = kv_elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("PI0.5 BF16 MQA byte-size overflow".into()))?;
    let key = DeviceBuffer::from_tensor(key)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(Shape::new(vec![1, key_tokens, kv_heads, q[2]]), DType::BF16)
        })
        .map_err(Error::Other)?;
    let value = DeviceBuffer::from_tensor(value)
        .and_then(|buffer| buffer.view(0, kv_bytes))
        .and_then(|buffer| {
            buffer.as_tensor(Shape::new(vec![1, key_tokens, kv_heads, q[2]]), DType::BF16)
        })
        .map_err(Error::Other)?;
    let query4 = query.reshape(Shape::new(vec![1, q[0], q[1], q[2]]))?;
    let mut output = ctx.allocate_output(query4.shape().clone(), DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key, &value, &mut output);
    args.policy = policy.clone();
    attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

pub(super) fn mha_bf16(
    ctx: &Context,
    policy: &AttentionPolicy,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let shape = query.shape().dims();
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || value.dtype() != DType::BF16
        || shape.len() != 3
        || key.shape() != query.shape()
        || value.shape() != query.shape()
        || tokens_per_batch == 0
        || shape[0] % tokens_per_batch != 0
    {
        return Err(Error::Other("PI0.5 BF16 MHA shape/dtype mismatch".into()));
    }
    let batch = shape[0] / tokens_per_batch;
    let shape4 = Shape::new(vec![batch, tokens_per_batch, shape[1], shape[2]]);
    let query4 = query.reshape(shape4.clone())?;
    let key4 = key.reshape(shape4.clone())?;
    let value4 = value.reshape(shape4.clone())?;
    let mut output = ctx.allocate_output(shape4, DType::BF16)?;
    let mut args = AttentionArgs::new(&query4, &key4, &value4, &mut output);
    args.policy = policy.clone();
    attention(ctx, args)?;
    output.reshape(query.shape().clone())
}

pub(super) fn split_qkv_bias_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != 3 * heads * head_dim {
        return Err(Error::Other(
            "PI0.5 BF16 vision QKV shape/dtype mismatch".into(),
        ));
    }
    let output_shape = Shape::new(vec![shape[0], heads, head_dim]);
    let mut q = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(output_shape, DType::BF16)?;
    rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 1.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn split_qkv_rope_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(Shape::new(vec![shape[0], q_heads, head_dim]), DType::BF16)?;
    let kv_shape = Shape::new(vec![shape[0], kv_heads, head_dim]);
    let mut k = ctx.allocate_output(kv_shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(kv_shape, DType::BF16)?;
    rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn split_qkv_rope_into_cache_bf16(
    ctx: &Context,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    cache_offset: usize,
    key_cache: &Tensor,
    value_cache: &Tensor,
) -> Result<Tensor> {
    let shape = qkv.shape().dims();
    let expected_width = (q_heads + 2 * kv_heads)
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("PI0.5 BF16 QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16 || shape.len() != 2 || shape[1] != expected_width {
        return Err(Error::Other(
            "PI0.5 BF16 cached rotary QKV shape/dtype mismatch".into(),
        ));
    }
    let mut q = ctx.allocate_output(Shape::new(vec![shape[0], q_heads, head_dim]), DType::BF16)?;
    // At offset 0 the rope contract wants the exact `[rows, kv_heads, head_dim]`
    // view, while the cache itself is a padded 2-D `[total_rows, head_dim]`.
    // View the first `rows` rows and let the rope write through it; the caller's
    // full cache tensor shares the same buffer. The action path writes at a
    // non-zero offset and keeps its existing 2-D-cache form.
    let (mut key_target, mut value_target) = if cache_offset == 0 {
        let rows = shape[0];
        let bytes = rows
            .checked_mul(kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .and_then(|n| n.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("PI0.5 BF16 KV cache view overflow".into()))?;
        let view_shape = Shape::new(vec![rows, kv_heads, head_dim]);
        let key = DeviceBuffer::from_tensor(key_cache)
            .and_then(|buffer| buffer.view(0, bytes))
            .and_then(|buffer| buffer.as_tensor(view_shape.clone(), DType::BF16))
            .map_err(Error::Other)?;
        let value = DeviceBuffer::from_tensor(value_cache)
            .and_then(|buffer| buffer.view(0, bytes))
            .and_then(|buffer| buffer.as_tensor(view_shape, DType::BF16))
            .map_err(Error::Other)?;
        (key, value)
    } else {
        (key_cache.clone(), value_cache.clone())
    };
    rope(
        ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut key_target,
            v: &mut value_target,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: cache_offset,
        },
    )?;
    Ok(q)
}

pub(super) fn adaptive_gate_residual_rms_bf16(
    ctx: &Context,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<super::ResidualNormTensors> {
    let cols = projection
        .shape()
        .dims()
        .last()
        .copied()
        .ok_or_else(|| Error::Other("PI0.5 AdaRMS input has no columns".into()))?;
    let norm_style = bf16_vector_prefix(norm_style, 2 * cols)?;
    let mut hidden = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(projection.shape().clone(), DType::BF16)?;
    ada_gate_residual_rms_norm(
        ctx,
        AdaGateResidualRmsNormArgs::new(
            projection,
            residual,
            &norm_style,
            gate_style,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(super::ResidualNormTensors { hidden, normalized })
}
