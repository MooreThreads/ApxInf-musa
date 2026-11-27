//! Packed-QKV split and rotary position embedding contracts.

use apxinf_core::{Error, Result, Tensor};

use crate::kernels::attention as attention_kernels;
use crate::kernels::rope as rope_kernels;
use crate::{MusaBuffer, MusaContext};

/// Which packed-QKV split the arguments describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeSemantic {
    /// Grouped-query split with rotary embedding on Q and K.
    SplitQkvRope,
    /// Vision MHA split with an optional bias and no rotation.
    SplitQkvBias,
}

/// Splits a packed QKV projection into separate Q, K and V buffers.
///
/// `k` and `v` may point either at fresh per-call buffers (`kv_output_offset`
/// zero) or into a KV cache, in which case `kv_output_offset` is the row at
/// which this call's tokens are appended and the kernel writes in place.
pub struct RopeArgs<'a> {
    pub semantic: RopeSemantic,
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub position_offset: usize,
    pub kv_output_offset: usize,
}

pub fn rope(ctx: &MusaContext, args: RopeArgs<'_>) -> Result<()> {
    match args.semantic {
        RopeSemantic::SplitQkvRope => match args.kv_output_offset {
            0 => rope_kernels::split_qkv_apply_bf16_into(
                ctx,
                args.qkv,
                args.bias,
                args.q,
                args.k,
                args.v,
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.theta,
                args.position_offset,
            ),
            // The cache case: the kernel writes K/V at the offset and returns
            // Q alone, so the borrowed k/v view is the cache itself.
            _ => {
                if args.kv_heads != 1 {
                    return Err(Error::Other(format!(
                        "MUSA rope cache write requires kv_heads == 1, got {}",
                        args.kv_heads
                    )));
                }
                rope_kernels::apply_q_write_kv_bf16_into(
                    ctx,
                    args.qkv,
                    args.bias,
                    args.q,
                    args.q_heads,
                    args.kv_heads,
                    args.head_dim,
                    args.theta,
                    args.position_offset,
                    args.k,
                    args.v,
                    args.kv_output_offset,
                )
            }
        },
        RopeSemantic::SplitQkvBias => {
            if args.kv_output_offset != 0 {
                return Err(Error::Other(
                    "MUSA qkv split with a bias has no KV-cache form".into(),
                ));
            }
            attention_kernels::split_qkv_bias_bf16_into(
                ctx, args.qkv, args.bias, args.q, args.k, args.v, args.q_heads, args.head_dim,
            )
        }
    }
}

/// Apply split-half BF16 RoPE using prepared FP32 angles `[tokens, head_dim]`.
pub fn vision_axial_rope(ctx: &MusaContext, input: &Tensor, angles: &Tensor) -> Result<Tensor> {
    use apxinf_core::{DType, Device};
    let &[n, h, d] = input.shape().dims() else {
        return Err(Error::Other("expected rank-three vision RoPE input".into()));
    };
    if n == 0
        || h == 0
        || d == 0
        || d % 4 != 0
        || input.shape().numel() > i32::MAX as usize
        || input.dtype() != DType::BF16
        || angles.dtype() != DType::F32
        || angles.shape().dims() != [n, d]
        || input.device() != Device::Musa(ctx.device_id())
        || angles.device() != input.device()
    {
        return Err(Error::Other("invalid vision RoPE operands".into()));
    }
    #[cfg(apxinf_musa_available)]
    {
        let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
        let buffer = |t: &Tensor| MusaBuffer::from_tensor(t).map_err(Error::Other);
        let status = unsafe {
            crate::ffi::custom::apxinf_musa_mudnn_vision_rope(
                buffer(input)?.ptr(),
                buffer(&output)?.ptr(),
                buffer(angles)?.ptr(),
                n as i32,
                h as i32,
                d as i32,
                ctx.stream().handle(),
            )
        };
        crate::ffi::custom::check_mudnn(status, || "vision RoPE".into()).map_err(Error::Other)?;
        Ok(output)
    }
    #[cfg(not(apxinf_musa_available))]
    Err(Error::Other("MUSA vision RoPE unavailable".into()))
}
