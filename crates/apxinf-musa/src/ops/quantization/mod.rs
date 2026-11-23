//! Quantization contracts.
//!
//! Shaped after `apxinf-cuda-new`'s `ops::quantization`: the model tree names a
//! semantic and supplies the operands, and the backend owns the kernel choice.
//! Only [`QuantizationSemantic::FixedScaleE4m3`] is served here, which is the
//! one the static-FP8 route calls; the other upstream semantics are absent
//! rather than stubbed, so a caller cannot import a name this port does not
//! implement.

use apxinf_core::{Result, Tensor};

use crate::kernels::quantization as kernels;
use crate::MusaContext;

/// Which quantization the arguments describe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantizationSemantic {
    /// `out = e4m3(input / scale)` with one static per-tensor scale.
    FixedScaleE4m3,
}

/// Inputs for one quantization call.
pub struct QuantizationArgs<'a> {
    pub semantic: QuantizationSemantic,
    pub input: &'a Tensor,
    /// Caller-provided output; its dtype selects the format.
    pub out: &'a mut Tensor,
    /// Reserved for the per-row/per-channel semantics; `FixedScaleE4m3` uses
    /// [`Self::scale`] and leaves this empty.
    pub scales: Option<&'a Tensor>,
    /// Dequantization multiplier: `real = e4m3_value * scale`.
    pub scale: f32,
}

impl<'a> QuantizationArgs<'a> {
    pub fn new(semantic: QuantizationSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input,
            out,
            scales: None,
            // Not a usable dequantization multiplier: an omitted scale fails
            // the quantizer instead of silently quantizing by 1.0.
            scale: 0.0,
        }
    }
}

pub fn quantization(ctx: &MusaContext, args: QuantizationArgs<'_>) -> Result<()> {
    match args.semantic {
        QuantizationSemantic::FixedScaleE4m3 => {
            if args.scales.is_some() {
                return Err(apxinf_core::Error::Other(
                    "MUSA quantization fixed-scale E4M3 takes no scale tensor".into(),
                ));
            }
            kernels::quantize_fixed_e4m3(ctx, args.input, args.out, args.scale)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{DType, Shape};

    #[test]
    fn an_unset_scale_is_the_invalid_sentinel() {
        let input = Tensor::zeros(Shape::new(vec![1]), DType::BF16);
        let mut out = Tensor::zeros(Shape::new(vec![1]), DType::F8E4M3);
        let args = QuantizationArgs::new(QuantizationSemantic::FixedScaleE4m3, &input, &mut out);
        assert_eq!(args.scale, 0.0);
    }
}
