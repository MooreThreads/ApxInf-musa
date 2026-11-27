//! Model-neutral contiguous BF16 spatial operators backed by muDNN.
//!
//! Fast-WAM's Wan VAE is the first consumer, but the contracts deliberately
//! describe ordinary convolution and nearest-exact interpolation.
//! Layout conversion is not implicit: callers provide contiguous channel-first
//! NCW/NCHW/NCDHW tensors and receive the same layout.

use std::ffi::c_void;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::error::musa_error;
use crate::ffi;
use crate::MusaContext;

pub struct ConvolutionArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub padding: &'a [usize],
    pub stride: &'a [usize],
    pub dilation: &'a [usize],
    pub groups: usize,
}

fn require_bf16_contiguous(
    ctx: &MusaContext,
    tensor: &Tensor,
    operation: &str,
    name: &str,
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
    // `apxinf_core::Tensor` is contiguous by construction. Layout-bearing
    // backends expose a different storage type rather than strides here.
    Ok(())
}

fn positive_i32(values: &[usize], operation: &str, name: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(|value| {
            if value == 0 {
                return Err(Error::Other(format!(
                    "MUSA {operation} {name} values must be positive"
                )));
            }
            i32::try_from(value).map_err(|_| {
                Error::Other(format!("MUSA {operation} {name} value {value} exceeds i32"))
            })
        })
        .collect()
}

fn nonnegative_i32(values: &[usize], operation: &str, name: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Error::Other(format!("MUSA {operation} {name} value {value} exceeds i32"))
            })
        })
        .collect()
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

fn convolution_output_shape(args: &ConvolutionArgs<'_>) -> Result<Vec<usize>> {
    let input = args.input.shape().dims();
    let weight = args.weight.shape().dims();
    if !(input.len() == 4 || input.len() == 5) || weight.len() != input.len() {
        return Err(Error::Other(format!(
            "MUSA convolution expects rank-4 or rank-5 input/weight, got {} and {}",
            input.len(),
            weight.len()
        )));
    }
    let spatial = input.len() - 2;
    for (name, values) in [
        ("padding", args.padding),
        ("stride", args.stride),
        ("dilation", args.dilation),
    ] {
        if values.len() != spatial {
            return Err(Error::Other(format!(
                "MUSA convolution {name} needs {spatial} values, got {}",
                values.len()
            )));
        }
    }
    if args.groups == 0 || input[1] % args.groups != 0 || weight[0] % args.groups != 0 {
        return Err(Error::Other(
            "MUSA convolution groups must divide input and output channels".into(),
        ));
    }
    if weight[1]
        .checked_mul(args.groups)
        .ok_or_else(|| Error::Other("MUSA convolution channel count overflow".into()))?
        != input[1]
    {
        return Err(Error::Other(format!(
            "MUSA convolution weight input channels {} * groups {} do not match input channels {}",
            weight[1], args.groups, input[1]
        )));
    }
    if let Some(bias) = args.bias {
        if bias.shape().dims() != [weight[0]] {
            return Err(Error::Other(format!(
                "MUSA convolution bias must have shape [{}]",
                weight[0]
            )));
        }
    }
    let mut output = vec![input[0], weight[0]];
    for axis in 0..spatial {
        if args.stride[axis] == 0 || args.dilation[axis] == 0 || weight[axis + 2] == 0 {
            return Err(Error::Other(
                "MUSA convolution stride, dilation and kernel dimensions must be positive".into(),
            ));
        }
        let padded = input[axis + 2]
            .checked_add(
                args.padding[axis]
                    .checked_mul(2)
                    .ok_or_else(|| Error::Other("MUSA convolution padding overflow".into()))?,
            )
            .ok_or_else(|| Error::Other("MUSA convolution padded size overflow".into()))?;
        let extent = args.dilation[axis]
            .checked_mul(weight[axis + 2] - 1)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::Other("MUSA convolution kernel extent overflow".into()))?;
        if padded < extent {
            return Err(Error::Other(format!(
                "MUSA convolution kernel extent {extent} exceeds padded input {padded} on spatial axis {axis}"
            )));
        }
        output.push((padded - extent) / args.stride[axis] + 1);
    }
    Ok(output)
}

pub fn convolution_bf16(ctx: &MusaContext, args: ConvolutionArgs<'_>) -> Result<Tensor> {
    require_bf16_contiguous(ctx, args.input, "convolution", "input")?;
    require_bf16_contiguous(ctx, args.weight, "convolution", "weight")?;
    if let Some(bias) = args.bias {
        require_bf16_contiguous(ctx, bias, "convolution", "bias")?;
    }
    let output_shape = convolution_output_shape(&args)?;
    let output = ctx.allocate_output(Shape::new(output_shape), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let input_dims = dims_i64(args.input, "convolution", "input")?;
        let weight_dims = dims_i64(args.weight, "convolution", "weight")?;
        let output_dims = dims_i64(&output, "convolution", "output")?;
        let padding = nonnegative_i32(args.padding, "convolution", "padding")?;
        let stride = positive_i32(args.stride, "convolution", "stride")?;
        let dilation = positive_i32(args.dilation, "convolution", "dilation")?;
        let rank = i32::try_from(input_dims.len())
            .map_err(|_| Error::Other("MUSA convolution rank exceeds i32".into()))?;
        let groups = i32::try_from(args.groups)
            .map_err(|_| Error::Other("MUSA convolution groups exceed i32".into()))?;
        let bias = args
            .bias
            .map(device_ptr)
            .transpose()?
            .unwrap_or(std::ptr::null_mut());
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_convolution_bf16(
                device_ptr(args.input)? as *const c_void,
                device_ptr(args.weight)? as *const c_void,
                bias as *const c_void,
                device_ptr(&output)?,
                input_dims.as_ptr(),
                weight_dims.as_ptr(),
                output_dims.as_ptr(),
                rank,
                padding.as_ptr(),
                stride.as_ptr(),
                dilation.as_ptr(),
                groups,
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 convolution {input_dims:?} x {weight_dims:?}")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, args);
        return Err(crate::kernels::native_unavailable("muDNN BF16 convolution"));
    }
    Ok(output)
}

pub fn interpolate_nearest_exact_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    output_spatial: &[usize],
) -> Result<Tensor> {
    require_bf16_contiguous(ctx, input, "nearest-exact interpolate", "input")?;
    let input_shape = input.shape().dims();
    if !(3..=5).contains(&input_shape.len()) || output_spatial.len() != input_shape.len() - 2 {
        return Err(Error::Other(format!(
            "MUSA nearest-exact interpolate rank {} needs {} spatial output dimensions, got {}",
            input_shape.len(),
            input_shape.len().saturating_sub(2),
            output_spatial.len()
        )));
    }
    if output_spatial.contains(&0) {
        return Err(Error::Other(
            "MUSA nearest-exact interpolate output dimensions must be positive".into(),
        ));
    }
    let mut output_shape = input_shape[..2].to_vec();
    output_shape.extend_from_slice(output_spatial);
    let output = ctx.allocate_output(Shape::new(output_shape), DType::BF16)?;
    #[cfg(apxinf_musa_available)]
    {
        let input_dims = dims_i64(input, "nearest-exact interpolate", "input")?;
        let output_dims = dims_i64(&output, "nearest-exact interpolate", "output")?;
        let scales: Vec<f32> = input_shape[2..]
            .iter()
            .zip(output_spatial)
            .map(|(&source, &target)| source as f32 / target as f32)
            .collect();
        let rank = i32::try_from(input_dims.len())
            .map_err(|_| Error::Other("MUSA nearest-exact interpolate rank exceeds i32".into()))?;
        let status = unsafe {
            ffi::custom::apxinf_musa_mudnn_interpolate_nearest_exact_bf16(
                device_ptr(input)? as *const c_void,
                device_ptr(&output)?,
                input_dims.as_ptr(),
                output_dims.as_ptr(),
                rank,
                scales.as_ptr(),
                ctx.stream().handle(),
            )
        };
        ffi::custom::check_mudnn(status, || {
            format!("muDNN BF16 nearest-exact interpolate {input_dims:?} -> {output_dims:?}")
        })
        .map_err(musa_error)?;
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, output_spatial);
        return Err(crate::kernels::native_unavailable(
            "muDNN BF16 nearest-exact interpolate",
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_shape_covers_2d_and_3d() {
        let input_2d = Tensor::zeros(vec![1, 4, 9, 11], DType::BF16);
        let weight_2d = Tensor::zeros(vec![8, 2, 3, 5], DType::BF16);
        let args = ConvolutionArgs {
            input: &input_2d,
            weight: &weight_2d,
            bias: None,
            padding: &[1, 2],
            stride: &[2, 1],
            dilation: &[1, 1],
            groups: 2,
        };
        assert_eq!(convolution_output_shape(&args).unwrap(), vec![1, 8, 5, 11]);

        let input_3d = Tensor::zeros(vec![1, 4, 5, 9, 11], DType::BF16);
        let weight_3d = Tensor::zeros(vec![8, 4, 3, 3, 3], DType::BF16);
        let args = ConvolutionArgs {
            input: &input_3d,
            weight: &weight_3d,
            bias: None,
            padding: &[1, 1, 1],
            stride: &[1, 2, 2],
            dilation: &[1, 1, 1],
            groups: 1,
        };
        assert_eq!(
            convolution_output_shape(&args).unwrap(),
            vec![1, 8, 5, 5, 6]
        );
    }

    #[test]
    fn convolution_rejects_invalid_group_channels() {
        let input = Tensor::zeros(vec![1, 3, 4, 4], DType::BF16);
        let weight = Tensor::zeros(vec![4, 2, 3, 3], DType::BF16);
        let args = ConvolutionArgs {
            input: &input,
            weight: &weight,
            bias: None,
            padding: &[0, 0],
            stride: &[1, 1],
            dilation: &[1, 1],
            groups: 2,
        };
        assert!(convolution_output_shape(&args).is_err());
    }
}
