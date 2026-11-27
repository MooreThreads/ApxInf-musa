//! Launch adapters for the bring-up portable MUSA operators.
//!
//! Validation and tensor ownership stay in Rust, while the ABI declarations
//! live in `ffi::custom`.

use crate::{MusaBuffer, MusaContext};
use apxinf_core::{Error, Result, Tensor};

fn buffer(tensor: &Tensor) -> Result<MusaBuffer> {
    MusaBuffer::from_tensor(tensor).map_err(Error::Other)
}

pub(crate) fn rope(
    ctx: &MusaContext,
    input: &Tensor,
    output: &Tensor,
    positions: &MusaBuffer,
    heads: usize,
    head_dim: usize,
    theta: f32,
    mode: i32,
    height_sections: usize,
    width_sections: usize,
) -> Result<()> {
    let elements = input.shape().numel() as i32;
    let input = buffer(input)?;
    let output = buffer(output)?;
    #[cfg(apxinf_musa_available)]
    unsafe {
        crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_rope(
            input.ptr(),
            output.ptr(),
            positions.ptr(),
            elements,
            heads as i32,
            head_dim as i32,
            theta,
            mode,
            height_sections as i32,
            width_sections as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Other)?;
    }
    #[cfg(not(apxinf_musa_available))]
    let _ = (
        ctx,
        input,
        output,
        positions,
        heads,
        head_dim,
        theta,
        mode,
        height_sections,
        width_sections,
    );
    #[cfg(not(apxinf_musa_available))]
    return Err(Error::Other("MUSA portable rope unavailable".into()));
    Ok(())
}

pub(crate) fn attention(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    query_len: usize,
    key_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<()> {
    let q = buffer(q)?;
    let k = buffer(k)?;
    let v = buffer(v)?;
    let output = buffer(output)?;
    #[cfg(apxinf_musa_available)]
    unsafe {
        crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_attention(
            q.ptr(),
            k.ptr(),
            v.ptr(),
            output.ptr(),
            query_len as i32,
            key_len as i32,
            heads as i32,
            kv_heads as i32,
            head_dim as i32,
            causal as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Other)?;
    }
    #[cfg(not(apxinf_musa_available))]
    let _ = (
        ctx, q, k, v, output, query_len, key_len, heads, kv_heads, head_dim, causal,
    );
    #[cfg(not(apxinf_musa_available))]
    return Err(Error::Other("MUSA portable attention unavailable".into()));
    Ok(())
}
