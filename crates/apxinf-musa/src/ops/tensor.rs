//! Model-neutral contiguous BF16 tensor operations backed by muDNN.
//!
//! These are deliberately small mathematical contracts: activation, broadcast
//! binary arithmetic, and last-axis normalization. Model-specific modulation,
//! residual and RoPE recipes remain in the model layer and compose these ops.

use std::ffi::c_void;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::error::musa_error;
use crate::ffi;
use crate::MusaContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    GeluTanh,
    Silu,
}

impl UnaryOp {
    const fn abi(self) -> i32 {
        match self {
            Self::GeluTanh => 0,
            Self::Silu => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Multiply,
    Maximum,
    Minimum,
}

impl BinaryOp {
    const fn abi(self) -> i32 {
        match self {
            Self::Add => 0,
            Self::Multiply => 1,
            Self::Maximum => 2,
            Self::Minimum => 3,
        }
    }
}

fn require_bf16_rank(
    ctx: &MusaContext,
    tensor: &Tensor,
    operation: &str,
    name: &str,
    max_rank: usize,
) -> Result<()> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects BF16, got {}",
            tensor.dtype()
        )));
    }
    let expected = Device::Musa(ctx.device_id());
    if tensor.device() != expected {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} is on {}, expected {expected}",
            tensor.device()
        )));
    }
    if tensor.shape().dims().is_empty() || tensor.shape().dims().len() > max_rank {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects rank 1..={max_rank}, got rank {}",
            tensor.shape().dims().len()
        )));
    }
    if tensor.shape().dims().contains(&0) {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} dimensions must be positive"
        )));
    }
    Ok(())
}

fn require_bf16(ctx: &MusaContext, tensor: &Tensor, operation: &str, name: &str) -> Result<()> {
    require_bf16_rank(ctx, tensor, operation, name, 5)
}

fn dims_i64(tensor: &Tensor, operation: &str, name: &str) -> Result<Vec<i64>> {
    tensor
        .shape()
        .dims()
        .iter()
        .copied()
        .map(|value| {
            i64::try_from(value).map_err(|_| {
                Error::Other(format!(
                    "MUSA {operation} {name} dimension {value} exceeds i64"
                ))
            })
        })
        .collect()
}

fn device_ptr(tensor: &Tensor) -> Result<*mut c_void> {
    Ok(MusaBuffer::from_tensor(tensor).map_err(musa_error)?.ptr() as *mut c_void)
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    if rank == 0 || rank > 5 {
        return Err(Error::Other(format!(
            "MUSA binary broadcast expects rank 1..=5, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    let mut output = vec![1; rank];
    for offset in 0..rank {
        let left_dim = left
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| left[index]);
        let right_dim = right
            .len()
            .checked_sub(offset + 1)
            .map_or(1, |index| right[index]);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return Err(Error::Other(format!(
                "MUSA binary operands are not broadcastable: {left:?} and {right:?}"
            )));
        }
        output[rank - offset - 1] = left_dim.max(right_dim);
    }
    Ok(output)
}

pub fn unary_bf16(ctx: &MusaContext, input: &Tensor, op: UnaryOp) -> Result<Tensor> {
    require_bf16(ctx, input, "unary", "input")?;
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let dims = dims_i64(input, "unary", "input")?;
        let rank = i32::try_from(dims.len())
            .map_err(|_| Error::Other("MUSA unary rank exceeds i32".into()))?;
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_unary_bf16(
                device_ptr(input)? as *const c_void,
                device_ptr(&output)?,
                dims.as_ptr(),
                rank,
                op.abi(),
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || format!("muDNN BF16 {op:?} {dims:?}"))
            .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, op);
        return Err(crate::kernels::native_unavailable("muDNN BF16 unary"));
    }
    Ok(output)
}

pub fn binary_bf16(
    ctx: &MusaContext,
    left: &Tensor,
    right: &Tensor,
    op: BinaryOp,
) -> Result<Tensor> {
    require_bf16(ctx, left, "binary", "left")?;
    require_bf16(ctx, right, "binary", "right")?;
    let output_shape = broadcast_shape(left.shape().dims(), right.shape().dims())?;
    let output = ctx.allocate_output(Shape::new(output_shape), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let left_dims = dims_i64(left, "binary", "left")?;
        let right_dims = dims_i64(right, "binary", "right")?;
        let output_dims = dims_i64(&output, "binary", "output")?;
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_binary_bf16(
                device_ptr(left)? as *const c_void,
                left_dims.as_ptr(),
                i32::try_from(left_dims.len())
                    .map_err(|_| Error::Other("MUSA binary left rank exceeds i32".into()))?,
                device_ptr(right)? as *const c_void,
                right_dims.as_ptr(),
                i32::try_from(right_dims.len())
                    .map_err(|_| Error::Other("MUSA binary right rank exceeds i32".into()))?,
                device_ptr(&output)?,
                output_dims.as_ptr(),
                i32::try_from(output_dims.len())
                    .map_err(|_| Error::Other("MUSA binary output rank exceeds i32".into()))?,
                op.abi(),
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 {op:?} {left_dims:?} and {right_dims:?}")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, left, right, op);
        return Err(crate::kernels::native_unavailable("muDNN BF16 binary"));
    }
    Ok(output)
}

/// Binary BF16 operation writing into caller-owned fixed storage. The output
/// shape must equal the broadcast result; unlike `binary_bf16`, this does not
/// consume a new prepared-arena slice.
pub fn binary_bf16_into(
    ctx: &MusaContext,
    left: &Tensor,
    right: &Tensor,
    output: &Tensor,
    op: BinaryOp,
) -> Result<()> {
    require_bf16(ctx, left, "binary", "left")?;
    require_bf16(ctx, right, "binary", "right")?;
    let expected = broadcast_shape(left.shape().dims(), right.shape().dims())?;
    if output.dtype() != DType::BF16 || output.shape().dims() != expected.as_slice() {
        return Err(Error::Other(format!(
            "binary output mismatch: expected {:?} BF16, got {:?} {}",
            expected,
            output.shape().dims(),
            output.dtype()
        )));
    }
    #[cfg(apxinf_musa_available)]
    {
        let left_dims = dims_i64(left, "binary", "left")?;
        let right_dims = dims_i64(right, "binary", "right")?;
        let output_dims = dims_i64(output, "binary", "output")?;
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_binary_bf16(
                device_ptr(left)? as *const c_void,
                left_dims.as_ptr(),
                i32::try_from(left_dims.len())
                    .map_err(|_| Error::Other("MUSA binary left rank exceeds i32".into()))?,
                device_ptr(right)? as *const c_void,
                right_dims.as_ptr(),
                i32::try_from(right_dims.len())
                    .map_err(|_| Error::Other("MUSA binary right rank exceeds i32".into()))?,
                device_ptr(output)?,
                output_dims.as_ptr(),
                i32::try_from(output_dims.len())
                    .map_err(|_| Error::Other("MUSA binary output rank exceeds i32".into()))?,
                op.abi(),
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 {op:?} into {:?}", output.shape().dims())
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, left, right, output, op);
        return Err(crate::kernels::native_unavailable("muDNN BF16 binary into"));
    }
    Ok(())
}

fn normalization_geometry(input: &Tensor, operation: &str) -> Result<(usize, usize)> {
    let dims = input.shape().dims();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other(format!("MUSA {operation} input must not be scalar")))?;
    let rows = dims[..dims.len() - 1]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| Error::Other(format!("MUSA {operation} row count overflow")))?;
    Ok((rows, cols))
}

/// Layer normalization over the last axis. Affine parameters must be supplied
/// together or both omitted, matching the Fast-WAM call sites and muDNN's
/// admitted inference contract.
pub fn layer_norm_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    require_bf16(ctx, input, "layer norm", "input")?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA layer norm epsilon must be finite and positive, got {eps}"
        )));
    }
    if weight.is_some() != bias.is_some() {
        return Err(Error::Other(
            "MUSA layer norm affine weight and bias must be both present or both absent".into(),
        ));
    }
    let (rows, cols) = normalization_geometry(input, "layer norm")?;
    for (name, tensor) in [("weight", weight), ("bias", bias)] {
        if let Some(tensor) = tensor {
            require_bf16(ctx, tensor, "layer norm", name)?;
            if tensor.shape().dims() != [cols] {
                return Err(Error::Other(format!(
                    "MUSA layer norm {name} must have shape [{cols}], got {:?}",
                    tensor.shape().dims()
                )));
            }
        }
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_layer_norm_bf16(
                device_ptr(input)? as *const c_void,
                weight
                    .map(device_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()) as *const c_void,
                bias.map(device_ptr)
                    .transpose()?
                    .unwrap_or(std::ptr::null_mut()) as *const c_void,
                device_ptr(&output)?,
                i32::try_from(rows)
                    .map_err(|_| Error::Other("MUSA layer norm rows exceed i32".into()))?,
                i32::try_from(cols)
                    .map_err(|_| Error::Other("MUSA layer norm cols exceed i32".into()))?,
                eps,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || format!("muDNN BF16 layer norm [{rows},{cols}]"))
            .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, weight, bias, eps);
        return Err(crate::kernels::native_unavailable("muDNN BF16 layer norm"));
    }
    Ok(output)
}

/// Unit-weight RMS normalization over the last axis. Fast-WAM applies its
/// learned Q/K weight in a separate BF16 multiply because the official eager
/// implementation rounds the normalized value to BF16 before multiplying.
pub fn rms_norm_bf16(ctx: &MusaContext, input: &Tensor, eps: f64) -> Result<Tensor> {
    require_bf16(ctx, input, "RMS norm", "input")?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA RMS norm epsilon must be finite and positive, got {eps}"
        )));
    }
    let (rows, cols) = normalization_geometry(input, "RMS norm")?;
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_rms_norm_bf16(
                device_ptr(input)? as *const c_void,
                device_ptr(&output)?,
                i32::try_from(rows)
                    .map_err(|_| Error::Other("MUSA RMS norm rows exceed i32".into()))?,
                i32::try_from(cols)
                    .map_err(|_| Error::Other("MUSA RMS norm cols exceed i32".into()))?,
                eps,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || format!("muDNN BF16 RMS norm [{rows},{cols}]"))
            .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, eps);
        return Err(crate::kernels::native_unavailable("muDNN BF16 RMS norm"));
    }
    Ok(output)
}

fn pairwise_rope_table_shape(input_dims: &[usize]) -> Result<[usize; 2]> {
    if input_dims.len() != 4 {
        return Err(Error::Other(format!(
            "MUSA pairwise RoPE input must have shape [B,H,T,D], got {input_dims:?}"
        )));
    }
    let head_dim = input_dims[3];
    if head_dim % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA pairwise RoPE head dimension must be even, got {head_dim}"
        )));
    }
    Ok([input_dims[2], head_dim / 2])
}

/// Apply adjacent-pair rotary position embedding to `[B, H, T, D]` BF16
/// values. `cos` and `sin` contain one F32 value per complex pair and have
/// shape `[T, D / 2]`.
///
/// The adapter casts each even/odd component to F32, performs both complex
/// products and their sum/difference in F32 through muDNN, then casts once to
/// BF16. This mirrors Fast-WAM's high-precision complex multiply without a
/// model-specific MUSA kernel.
pub fn pairwise_rope_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    require_bf16(ctx, input, "pairwise RoPE", "input")?;
    let dims = input.shape().dims();
    let expected_table = pairwise_rope_table_shape(dims)?;
    for (name, table) in [("cos", cos), ("sin", sin)] {
        if table.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "MUSA pairwise RoPE {name} expects F32, got {}",
                table.dtype()
            )));
        }
        let expected_device = Device::Musa(ctx.device_id());
        if table.device() != expected_device {
            return Err(Error::Other(format!(
                "MUSA pairwise RoPE {name} is on {}, expected {expected_device}",
                table.device()
            )));
        }
        if table.shape().dims() != expected_table {
            return Err(Error::Other(format!(
                "MUSA pairwise RoPE {name} must have shape {expected_table:?}, got {:?}",
                table.shape().dims()
            )));
        }
    }

    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let to_i32 = |name: &str, value: usize| {
            i32::try_from(value)
                .map_err(|_| Error::Other(format!("MUSA pairwise RoPE {name} exceeds i32")))
        };
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_pairwise_rope_bf16(
                device_ptr(input)? as *const c_void,
                device_ptr(cos)? as *const c_void,
                device_ptr(sin)? as *const c_void,
                device_ptr(&output)?,
                to_i32("batch", dims[0])?,
                to_i32("tokens", dims[2])?,
                to_i32("heads", dims[1])?,
                to_i32("head dimension", dims[3])?,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 pairwise RoPE {dims:?}")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, cos, sin);
        return Err(crate::kernels::native_unavailable(
            "muDNN BF16 pairwise RoPE",
        ));
    }
    Ok(output)
}

/// Materialize a BF16 axis permutation. Both input and output are contiguous;
/// `permutation[output_axis]` names the corresponding input axis.
pub fn permute_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    permutation: &[usize],
) -> Result<Tensor> {
    require_bf16_rank(ctx, input, "permute", "input", 8)?;
    let input_dims = input.shape().dims();
    if permutation.len() != input_dims.len() {
        return Err(Error::Other(format!(
            "MUSA BF16 permute needs one axis per input dimension and rank <= 8, got rank {} and permutation {permutation:?}",
            input_dims.len()
        )));
    }
    let output_dims = permuted_shape(input_dims, permutation)?;
    let output = ctx.allocate_output(Shape::new(output_dims), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let dims = dims_i64(input, "permute", "input")?;
        let axes = permutation
            .iter()
            .copied()
            .map(|axis| {
                i64::try_from(axis)
                    .map_err(|_| Error::Other("MUSA BF16 permute axis exceeds i64".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_permute_bf16(
                device_ptr(input)? as *const c_void,
                device_ptr(&output)?,
                dims.as_ptr(),
                axes.as_ptr(),
                i32::try_from(dims.len())
                    .map_err(|_| Error::Other("MUSA BF16 permute rank exceeds i32".into()))?,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 permute {dims:?} by {axes:?}")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    return Err(crate::kernels::native_unavailable("muDNN BF16 permute"));
    Ok(output)
}

fn permuted_shape(input_dims: &[usize], permutation: &[usize]) -> Result<Vec<usize>> {
    if permutation.len() != input_dims.len() {
        return Err(Error::Other(format!(
            "MUSA BF16 permute needs one axis per input dimension, got rank {} and permutation {permutation:?}",
            input_dims.len()
        )));
    }
    let mut seen = vec![false; input_dims.len()];
    let mut output_dims = Vec::with_capacity(input_dims.len());
    for &axis in permutation {
        if axis >= input_dims.len() || seen[axis] {
            return Err(Error::Other(format!(
                "MUSA BF16 permute axes must be a permutation of 0..{}, got {permutation:?}",
                input_dims.len()
            )));
        }
        seen[axis] = true;
        output_dims.push(input_dims[axis]);
    }
    Ok(output_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_broadcast_geometry_covers_fastwam_patterns() {
        assert_eq!(
            broadcast_shape(&[1, 294, 3072], &[3072]).unwrap(),
            vec![1, 294, 3072]
        );
        assert_eq!(
            broadcast_shape(&[1, 294, 3072], &[1, 294, 3072]).unwrap(),
            vec![1, 294, 3072]
        );
        assert_eq!(
            broadcast_shape(&[1, 294, 24, 128], &[294, 1, 64])
                .unwrap_err()
                .to_string()
                .contains("not broadcastable"),
            true
        );
    }

    #[test]
    fn binary_abi_values_are_stable() {
        assert_eq!(BinaryOp::Add.abi(), 0);
        assert_eq!(BinaryOp::Multiply.abi(), 1);
        assert_eq!(BinaryOp::Maximum.abi(), 2);
        assert_eq!(BinaryOp::Minimum.abi(), 3);
    }

    #[test]
    fn normalization_flattens_only_prefix_dimensions() {
        let tensor = Tensor::zeros(vec![1, 294, 24, 128], DType::BF16);
        assert_eq!(
            normalization_geometry(&tensor, "test").unwrap(),
            (7056, 128)
        );
    }

    #[test]
    fn pairwise_rope_derives_fastwam_table_geometry() {
        assert_eq!(
            pairwise_rope_table_shape(&[1, 98, 24, 128]).unwrap(),
            [98, 64]
        );
        assert!(pairwise_rope_table_shape(&[1, 98, 24, 127])
            .unwrap_err()
            .to_string()
            .contains("must be even"));
    }


    #[test]
    fn permute_geometry_rejects_duplicate_axes() {
        assert_eq!(
            permuted_shape(&[1, 3072, 3, 7, 14], &[0, 2, 3, 4, 1]).unwrap(),
            vec![1, 3, 7, 14, 3072]
        );
        assert!(permuted_shape(&[1, 3, 7, 14, 3072], &[0, 2, 3, 3, 1]).is_err());
    }
}
