//! KV-cache and row-manipulation contracts.

use std::ffi::c_void;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::error::musa_error;
use crate::ffi;
use crate::kernels::cache as cache_kernels;
use crate::kernels::elementwise;
use crate::MusaContext;

/// Allocate a persistent KV prefix and copy `prefix` into its first rows.
///
/// The tail deliberately keeps the arena's previous contents: `attention` reads
/// only the first `key_tokens` rows, which is exactly why the copy is allowed to
/// stop there.
pub fn reserve_prefix(ctx: &MusaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    cache_kernels::reserve_prefix_bf16(ctx, prefix, total_rows)
}

/// Concatenate two 2-D BF16 tensors along rows.
pub fn concat_rows(ctx: &MusaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let [first_rows, cols] = first.shape().dims() else {
        return Err(Error::Other(format!(
            "MUSA row concatenation first input must be a matrix, got {:?}",
            first.shape().dims()
        )));
    };
    let [second_rows, second_cols] = second.shape().dims() else {
        return Err(Error::Other(format!(
            "MUSA row concatenation second input must be a matrix, got {:?}",
            second.shape().dims()
        )));
    };
    let expected_device = Device::Musa(ctx.device_id());
    if first.dtype() != DType::BF16
        || second.dtype() != DType::BF16
        || first.device() != expected_device
        || second.device() != expected_device
        || cols != second_cols
    {
        return Err(Error::Other(format!(
            "MUSA row concatenation requires BF16 matrices on {expected_device} with equal widths; got {} {:?} and {} {:?}",
            first.dtype(), first.shape().dims(), second.dtype(), second.shape().dims()
        )));
    }
    let rows = first_rows
        .checked_add(*second_rows)
        .ok_or_else(|| Error::Other("MUSA row concatenation row count overflow".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, *cols]), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let ptr = |tensor: &Tensor| {
            Ok::<_, Error>(MusaBuffer::from_tensor(tensor).map_err(musa_error)?.ptr()
                as *mut c_void)
        };
        let checked = |name: &str, value: usize| {
            i32::try_from(value)
                .map_err(|_| Error::Other(format!("MUSA row concatenation {name} exceeds i32")))
        };
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_concat_rows_bf16(
                ptr(first)? as *const c_void,
                ptr(second)? as *const c_void,
                ptr(&output)?,
                checked("first rows", *first_rows)?,
                checked("second rows", *second_rows)?,
                checked("columns", *cols)?,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 row concatenation [{first_rows},{cols}] + [{second_rows},{cols}]")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    return Err(crate::kernels::native_unavailable(
        "muDNN BF16 row concatenation",
    ));
    Ok(output)
}

/// Return a zero-copy view over contiguous matrix rows.
pub fn contiguous_rows(
    ctx: &MusaContext,
    input: &Tensor,
    first_row: usize,
    row_count: usize,
) -> Result<Tensor> {
    elementwise::contiguous_rows(ctx, input, first_row, row_count)
}
