//! Static FP8 (E4M3) quantization contracts.
//!
//! The fp8_static route supplies one calibrated per-tensor scale, so this is a
//! single pass: multiply by `1 / scale`, clamp to the E4M3 finite range, and
//! convert to E4M3. There is no reduction and no scale vector, which is what
//! keeps the quantizer's cost close to its traffic; compare the W8A8 row
//! quantizer in [`super::gemm::w8a8`], which measures an amax per row and
//! writes an FP16 scale vector for the `s8mma` tiles.
//!
//! Both F16 and BF16 inputs are served because the two call sites differ by
//! design: weights are rounded to F16 before quantization (the CUDA static-FP8
//! preparation order, preserved here), while activations are BF16 like the rest
//! of the MUSA port.

use apxinf_core::{DType, Error, Result, Tensor};

#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i64, tensor_ptr};
use super::contracts::{require_device, require_output_dtype_tensor};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;

/// The kernel's multiplier, after checking that `scale` can be inverted.
///
/// A finite positive `scale` is not enough: a subnormal at or below
/// `f32::MAX.recip()` inverts to infinity, which the adapter would reject as a
/// generic device error. Fail here with the scale in the message instead.
pub(super) fn checked_inverse_scale(scale: f32) -> Result<f32> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA quantization fixed-scale E4M3 scale must be finite and positive, got {scale}"
        )));
    }
    let inverse = scale.recip();
    if !inverse.is_finite() {
        return Err(Error::Other(format!(
            "MUSA quantization fixed-scale E4M3 scale {scale} has no finite reciprocal"
        )));
    }
    Ok(inverse)
}

/// Quantize `input` into the caller-provided E4M3 `output` with `scale`.
///
/// `scale` is the dequantization multiplier: `real = e4m3_value * scale`. It
/// must be finite and positive with a finite reciprocal; the kernel receives
/// `1 / scale`.
pub fn quantize_fixed_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    output: &mut Tensor,
    scale: f32,
) -> Result<()> {
    quantize_with_inverse(ctx, input, output, checked_inverse_scale(scale)?)
}

/// [`quantize_fixed_e4m3`] for callers that already hold the kernel's multiplier.
///
/// The static-FP8 attention fallback receives `1 / output_scale` from the
/// model layer. Taking the reciprocal again would round twice and break the
/// byte identity with the composed BF16 + quantize path.
pub(super) fn quantize_fixed_e4m3_inverse(
    ctx: &MusaContext,
    input: &Tensor,
    output: &mut Tensor,
    inverse_scale: f32,
) -> Result<()> {
    if !inverse_scale.is_finite() || inverse_scale <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA quantization fixed-scale E4M3 inverse scale must be finite and positive, got \
             {inverse_scale}"
        )));
    }
    quantize_with_inverse(ctx, input, output, inverse_scale)
}

fn quantize_with_inverse(
    ctx: &MusaContext,
    input: &Tensor,
    output: &mut Tensor,
    inverse_scale: f32,
) -> Result<()> {
    let dtype = input.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err(Error::Other(format!(
            "MUSA quantization fixed-scale E4M3 input expects F16 or BF16, got {dtype}"
        )));
    }
    require_device(ctx, input, "quantization", "input")?;
    require_output_dtype_tensor(
        ctx,
        output,
        input.shape().dims(),
        DType::F8E4M3,
        "quantization",
        "out",
    )?;
    let count = input.numel();
    if count == 0 {
        return Err(Error::Other(
            "MUSA quantization fixed-scale E4M3 input is empty".into(),
        ));
    }
    launch(ctx, input, output, count, inverse_scale, dtype)
}

#[cfg(apxinf_musa_available)]
fn launch(
    ctx: &MusaContext,
    input: &Tensor,
    output: &Tensor,
    count: usize,
    inverse_scale: f32,
    dtype: DType,
) -> Result<()> {
    let count = checked_i64("quantization", "elements", count)?;
    // SAFETY: dtype, shape and device are established above; every pointer is
    // owned by a caller `Tensor` that outlives the asynchronous launch.
    unsafe {
        let status = match dtype {
            DType::BF16 => ffi::custom::apxinf_musa_quantize_bf16_e4m3(
                tensor_ptr(input)?,
                tensor_ptr(output)?,
                count,
                inverse_scale,
                ctx.stream().handle(),
            ),
            DType::F16 => ffi::custom::apxinf_musa_quantize_f16_e4m3(
                tensor_ptr(input)?,
                tensor_ptr(output)?,
                count,
                inverse_scale,
                ctx.stream().handle(),
            ),
            other => {
                return Err(Error::Other(format!(
                    "MUSA quantization fixed-scale E4M3 received {other}"
                )))
            }
        };
        check_native(status, "quantization::fixed_scale_e4m3")
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch(
    _ctx: &MusaContext,
    _input: &Tensor,
    _output: &Tensor,
    _count: usize,
    _inverse_scale: f32,
    _dtype: DType,
) -> Result<()> {
    Err(native_unavailable("quantization::fixed_scale_e4m3"))
}

#[cfg(test)]
mod host_tests {
    //! Host-side oracle for the E4M3 byte contract.
    //!
    //! This module is deliberately not gated on `apxinf_musa_available`: the
    //! encoder and the scale arithmetic are pure host code and should be
    //! checked in a toolkit-less build too.

    use super::checked_inverse_scale;

    pub(super) fn encode_e4m3(value: f32) -> u8 {
        if value.is_nan() {
            return 0x7f;
        }
        let sign = if value.is_sign_negative() { 0x80 } else { 0 };
        let magnitude = value.abs();
        if magnitude == 0.0 {
            return sign;
        }
        if !magnitude.is_finite() || magnitude >= 448.0 {
            return sign | 0x7e;
        }
        if magnitude < 2f32.powi(-6) {
            // `round_ties_even` can carry 7.5..8 up to the smallest normal,
            // whose byte is 0x08; E4M3 has no subnormal mantissa 8.
            let mantissa = (magnitude * 512.0).round_ties_even().min(8.0) as u8;
            return sign | mantissa;
        }
        let mut exponent = magnitude.log2().floor() as i32;
        let base = 2f32.powi(exponent);
        let mut mantissa = ((magnitude / base - 1.0) * 8.0).round_ties_even() as i32;
        if mantissa == 8 {
            mantissa = 0;
            exponent += 1;
        }
        let field = exponent + 7;
        sign | (((field << 3) | mantissa) as u8)
    }

    #[test]
    fn e4m3_rounds_the_subnormal_boundary_to_the_smallest_normal() {
        assert_eq!(encode_e4m3(7.499 / 512.0), 0x07);
        assert_eq!(encode_e4m3(7.5 / 512.0), 0x08); // exact tie, to even
        assert_eq!(encode_e4m3(0.015), 0x08);
        assert_eq!(encode_e4m3(0.0155), 0x08);
        assert_eq!(encode_e4m3(0.015624), 0x08);
        assert_eq!(encode_e4m3(-7.5 / 512.0), 0x88);
    }

    #[test]
    fn a_scale_without_a_finite_reciprocal_is_rejected() {
        assert_eq!(checked_inverse_scale(1.0).unwrap(), 1.0);
        assert!(checked_inverse_scale(f32::from_bits(1)).is_err());
        assert!(checked_inverse_scale(f32::MIN_POSITIVE).is_ok());
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the static-FP8 quantizer.
    //!
    //! The finite-value byte sequence is the contract: the host encoder below
    //! is the test oracle, and every special value that decides the E4M3 format
    //! is checked. NaN is excluded because the kernel's `fminf(448, fmaxf(-448,
    //! x))` clamp reduces it to -448 (`0xfe`), while the host encoder
    //! canonicalises it to `0x7f`; the production path only quantizes finite
    //! values. The device kernel's behaviour matches the CUDA reference's clamp
    //! order.
    //!
    //! Needs a MUSA device:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use super::host_tests::encode_e4m3;
    use super::*;
    use apxinf_core::Shape;
    use half::{bf16, f16};

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fixed_scale_e4m3_matches_the_host_encoder_for_finite_values() {
        let ctx = MusaContext::new(0).unwrap();
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            448.0,
            -448.0,
            449.0,
            0.5,
            0.25,
            1.0 / 512.0,
            1.0 / 1024.0,
            1.0 / 64.0,
            2.0 / 512.0,
            3.0 / 512.0,
            7.0 / 512.0,
            7.5 / 512.0,
            0.015,
            0.0155,
            8.0 / 512.0,
            15.0 / 512.0,
            16.0 / 512.0,
            255.0,
            256.0,
            447.0,
            448.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let host: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
        let input = crate::transfers::to_device(
            &Tensor::from_bf16(vec![values.len()], &host).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![values.len()]), DType::F8E4M3)
            .unwrap();
        quantize_fixed_e4m3(&ctx, &input, &mut output, 1.0).unwrap();
        ctx.synchronize().unwrap();
        let host_output = crate::transfers::to_cpu(&output)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()
            .to_vec();
        for (index, value) in values.iter().enumerate() {
            let rounded = host[index].to_f32();
            let expected = encode_e4m3(rounded);
            assert_eq!(
                host_output[index], expected,
                "value {value} ({rounded}) at {index}"
            );
        }
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fixed_scale_e4m3_applies_the_inverse_scale() {
        let ctx = MusaContext::new(0).unwrap();
        let input = crate::transfers::to_device(
            &Tensor::from_bf16(vec![2], &[bf16::from_f32(0.5), bf16::from_f32(-0.25)]).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![2]), DType::F8E4M3)
            .unwrap();
        // scale 0.5 means multiply by 2: 0.5 -> 1.0, -0.25 -> -0.5.
        quantize_fixed_e4m3(&ctx, &input, &mut output, 0.5).unwrap();
        ctx.synchronize().unwrap();
        let bytes = crate::transfers::to_cpu(&output)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()
            .to_vec();
        assert_eq!(bytes, vec![encode_e4m3(1.0), encode_e4m3(-0.5)]);
    }

    #[test]
    fn the_scale_must_be_finite_and_positive() {
        let ctx = match MusaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let input = crate::transfers::to_device(
            &Tensor::from_bf16(vec![1], &[bf16::from_f32(1.0)]).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![1]), DType::F8E4M3)
            .unwrap();
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(quantize_fixed_e4m3(&ctx, &input, &mut output, scale).is_err());
        }
    }

    /// The static-FP8 weight path packs to F16 before quantizing, so the F16
    /// arm of the kernel needs the same byte-level contract as BF16. F16 has
    /// a different mantissa width, so the BF16 case cannot stand in for it.
    #[test]
    #[ignore = "requires a MUSA device"]
    fn fixed_scale_e4m3_f16_matches_the_host_encoder_for_finite_values() {
        let ctx = MusaContext::new(0).unwrap();
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            448.0,
            -448.0,
            449.0,
            0.5,
            0.25,
            1.0 / 512.0,
            1.0 / 1024.0,
            1.0 / 64.0,
            2.0 / 512.0,
            3.0 / 512.0,
            7.0 / 512.0,
            7.5 / 512.0,
            0.015,
            0.0155,
            8.0 / 512.0,
            15.0 / 512.0,
            16.0 / 512.0,
            255.0,
            256.0,
            447.0,
            448.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let host: Vec<f16> = values.iter().map(|value| f16::from_f32(*value)).collect();
        let input = crate::transfers::to_device(
            &Tensor::from_f16(vec![values.len()], &host).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![values.len()]), DType::F8E4M3)
            .unwrap();
        quantize_fixed_e4m3(&ctx, &input, &mut output, 1.0).unwrap();
        ctx.synchronize().unwrap();
        let host_output = crate::transfers::to_cpu(&output)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()
            .to_vec();
        for (index, value) in values.iter().enumerate() {
            let rounded = host[index].to_f32();
            let expected = encode_e4m3(rounded);
            assert_eq!(
                host_output[index], expected,
                "value {value} ({rounded}) at {index}"
            );
        }
    }

    #[test]
    fn non_f16_or_bf16_input_is_rejected_before_the_device_check() {
        let ctx = match MusaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let input = Tensor::from_f32(vec![1], &[1.0]).unwrap();
        let mut output = Tensor::zeros(Shape::new(vec![1]), DType::F8E4M3);
        let error = quantize_fixed_e4m3(&ctx, &input, &mut output, 1.0)
            .expect_err("an F32 input must be refused");
        assert!(error.to_string().contains("expects F16 or BF16"), "{error}");
    }

    fn zero_length_device(ctx: &MusaContext, dtype: DType) -> Tensor {
        let base = ctx.allocate_output(Shape::new(vec![1]), dtype).unwrap();
        crate::buffer::MusaBuffer::from_tensor(&base)
            .unwrap()
            .view(0, 0)
            .unwrap()
            .as_tensor(Shape::new(vec![0]), dtype)
            .unwrap()
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn empty_input_and_wrong_output_dtype_are_rejected() {
        let ctx = MusaContext::new(0).unwrap();
        let empty = zero_length_device(&ctx, DType::BF16);
        let mut empty_out = zero_length_device(&ctx, DType::F8E4M3);
        let error = quantize_fixed_e4m3(&ctx, &empty, &mut empty_out, 1.0)
            .expect_err("an empty input must be refused");
        assert!(error.to_string().contains("is empty"), "{error}");

        let input = crate::transfers::to_device(
            &Tensor::from_bf16(vec![1], &[bf16::from_f32(1.0)]).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut wrong_out = ctx
            .allocate_output(Shape::new(vec![1]), DType::BF16)
            .unwrap();
        let error = quantize_fixed_e4m3(&ctx, &input, &mut wrong_out, 1.0)
            .expect_err("a BF16 output must be refused");
        assert!(error.to_string().contains("out expects f8_e4m3"), "{error}");
    }
}
