//! fp8_static device weights: model aggregates and linear storage.

/// The gate/up interleave width used by the composed GeGLU producer.
///
/// The action layers pack gate/up in 32-column blocks and the producer pairs
/// column `c` with `c + 32`; the prefix layers keep the half-split
/// `[gate | up]` packing.
pub const GEGLU_PAIR_WIDTH: usize = 32;

mod linear {
    //! Device-ready static-FP8 linear weights.

    use apxinf_core::{DType, Error, Result, Tensor};

    use crate::musa_pi05::backend::{self, ops, Context};
    use crate::musa_pi05::weights::packing::concat_host_2d;
    use crate::musa_pi05::LinearWeights;

    use super::super::fp8_static_calibration::E4M3_MAX;

    #[derive(Debug)]
    pub struct Fp8StaticLinearWeights {
        /// `[input, output]` MUSA E4M3 matrix.
        pub weight: Tensor,
        pub weight_scale: f32,
        /// Bias stays BF16 and is fused into the consumer kernel.
        pub bias: Option<Tensor>,
    }

    impl Fp8StaticLinearWeights {
        pub fn from_host(linear: &LinearWeights, context: &Context) -> Result<Self> {
            Self::from_host_parts(&[linear], context)
        }

        /// Concatenate projections along their output dimension before applying
        /// one absmax quantization scale. This produces graph-ready QKV and
        /// gate/up matrices without runtime concatenation or mixed descales.
        pub fn from_host_parts(linears: &[&LinearWeights], context: &Context) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
            }
            let (weight_f16, weight_scale, bias) = pack_host_parts(linears)?;
            Self::finish(weight_f16, weight_scale, bias, context)
        }

        /// As [`from_host_parts`], but the gate/up columns are interleaved in
        /// `pair_width`-sized blocks so the composed producer can pair them
        /// without a per-element divide. The value set (and therefore the
        /// absmax scale) is unchanged.
        pub fn from_host_parts_interleaved(
            linears: &[&LinearWeights],
            pair_width: usize,
            context: &Context,
        ) -> Result<Self> {
            if linears.len() != 2 {
                return Err(Error::Other(
                    "interleaved gate/up packing expects exactly two projections".into(),
                ));
            }
            let (weight_f16, weight_scale, bias) = pack_host_parts(linears)?;
            let (weight_f16, bias) = interleave_gate_up(weight_f16, bias, pair_width)?;
            Self::finish(weight_f16, weight_scale, bias, context)
        }

        fn finish(
            weight_f16: Tensor,
            weight_scale: f32,
            bias: Option<Tensor>,
            context: &Context,
        ) -> Result<Self> {
            let weight_f16 = backend::to_device(context, &weight_f16)?;
            let mut weight = context.allocate_output(weight_f16.shape().clone(), DType::F8E4M3)?;
            let mut quantization = ops::QuantizationArgs::new(
                ops::QuantizationSemantic::FixedScaleE4m3,
                &weight_f16,
                &mut weight,
            );
            quantization.scale = weight_scale;
            ops::quantization(context, quantization)?;
            ops::prepare_fp8_weight(context, &weight)?;
            let bias = bias
                .as_ref()
                .map(|bias| backend::to_device(context, bias))
                .transpose()?;
            Ok(Self {
                weight,
                weight_scale,
                bias,
            })
        }
    }

    fn pack_host_parts(linears: &[&LinearWeights]) -> Result<(Tensor, f32, Option<Tensor>)> {
        let weight_host = concat_host_2d(&linears.iter().map(|x| &x.weight).collect::<Vec<_>>())?;
        let values = weight_host.to_f32_vec()?;
        let f16_max = half::f16::MAX.to_f32();
        if let Some(value) = values
            .iter()
            .find(|value| !value.is_finite() || value.abs() > f16_max)
        {
            return Err(Error::Other(format!(
                "FP8 weights must be finite and fit in F16, got {value}"
            )));
        }
        let amax = values
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        let scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
        let values = values
            .into_iter()
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
        let weight_f16 = Tensor::from_f16(weight_host.shape().dims().to_vec(), &values)?;
        let bias = if linears.iter().all(|x| x.bias.is_none()) {
            None
        } else if linears.iter().all(|x| x.bias.is_some()) {
            let biases = linears
                .iter()
                .map(|x| x.bias.as_ref().unwrap())
                .collect::<Vec<_>>();
            Some(concat_host_1d_bf16(&biases)?)
        } else {
            return Err(Error::Other(
                "cannot pack projections with a mixture of present and absent biases".into(),
            ));
        };
        Ok((weight_f16, scale, bias))
    }

    /// Permute `[gate | up]` into `[g0..g_p, u0..u_p, g_p.., ...]` along N.
    /// The value set is unchanged, so the packing absmax (and therefore the
    /// scale) is unchanged.
    fn interleave_gate_up(
        weight: Tensor,
        bias: Option<Tensor>,
        pair_width: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let dims = weight.shape().dims().to_vec();
        if dims.len() != 2 || dims[1] % 2 != 0 {
            return Err(Error::Other(
                "interleaved gate/up weight must be a [K, 2N] matrix".into(),
            ));
        }
        let (rows, cols) = (dims[0], dims[1]);
        let half = cols / 2;
        if pair_width == 0 || pair_width > half || half % pair_width != 0 {
            return Err(Error::Other(format!(
                "gate/up pair width {pair_width} must divide {half} and be non-zero"
            )));
        }
        let values = weight.to_f32_vec()?;
        let mut permuted = vec![0.0f32; values.len()];
        for row in 0..rows {
            let base = row * cols;
            for col in 0..half {
                let dst = 2 * pair_width * (col / pair_width) + (col % pair_width);
                permuted[base + dst] = values[base + col];
                permuted[base + dst + pair_width] = values[base + half + col];
            }
        }
        let values = permuted
            .into_iter()
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
        let weight = Tensor::from_f16(dims, &values)?;

        let bias = match bias {
            Some(bias) => {
                let values = bias.to_f32_vec()?;
                if values.len() != cols {
                    return Err(Error::Other(
                        "gate/up bias width does not match the packed weight".into(),
                    ));
                }
                let mut permuted = vec![0.0f32; cols];
                for col in 0..half {
                    let dst = 2 * pair_width * (col / pair_width) + (col % pair_width);
                    permuted[dst] = values[col];
                    permuted[dst + pair_width] = values[half + col];
                }
                let values = permuted
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                Some(Tensor::from_bf16(vec![cols], &values)?)
            }
            None => None,
        };
        Ok((weight, bias))
    }

    fn concat_host_1d_bf16(tensors: &[&Tensor]) -> Result<Tensor> {
        let mut output = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other("packed biases must be non-FP8 vectors".into()));
            }
            let values = tensor.to_f32_vec()?;
            if let Some(value) = values.iter().find(|value| !value.is_finite()) {
                return Err(Error::Other(format!(
                    "packed biases must be finite, got {value}"
                )));
            }
            output.extend(values.into_iter().map(half::bf16::from_f32));
        }
        Tensor::from_bf16(vec![output.len()], &output)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn linear(weight: &[f32], shape: [usize; 2], bias: Option<&[f32]>) -> LinearWeights {
            LinearWeights {
                weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
                bias: bias.map(|x| Tensor::from_f32(vec![x.len()], x).unwrap()),
            }
        }

        #[test]
        fn packs_qkv_before_quantization() {
            let q = linear(&[1., 2., 3., 4.], [2, 2], Some(&[1., 2.]));
            let k = linear(&[5., 6.], [2, 1], Some(&[3.]));
            let v = linear(&[7., 8.], [2, 1], Some(&[4.]));
            let (packed, scale, bias) = pack_host_parts(&[&q, &k, &v]).unwrap();
            assert_eq!(packed.shape().dims(), &[2, 4]);
            assert_eq!(packed.dtype(), DType::F16);
            assert_eq!(
                packed.to_f32_vec().unwrap(),
                vec![1., 2., 5., 7., 3., 4., 6., 8.]
            );
            assert_eq!(scale, 8.0 / E4M3_MAX);
            let bias = bias.unwrap();
            assert_eq!(bias.dtype(), DType::BF16);
            assert_eq!(bias.to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
        }

        #[test]
        fn interleaved_gate_up_pairs_gate_and_up_columns() {
            let gate = linear(&[1., 2., 3., 4.], [1, 4], None);
            let up = linear(&[5., 6., 7., 8.], [1, 4], None);
            let (packed, scale, _) = pack_host_parts(&[&gate, &up]).unwrap();
            let (packed, _) = interleave_gate_up(packed, None, 2).unwrap();
            // [g0,g1, u0,u1, g2,g3, u2,u3]: the producer pairs column `c`
            // with `c + pair_width`.
            assert_eq!(
                packed.to_f32_vec().unwrap(),
                vec![1., 2., 5., 6., 3., 4., 7., 8.]
            );
            // The permutation does not change the value set, so the packing
            // scale is untouched.
            assert_eq!(scale, 8.0 / E4M3_MAX);
        }

        #[test]
        fn non_finite_or_out_of_f16_range_weights_are_rejected() {
            let nan = linear(&[f32::NAN, 1.0], [1, 2], None);
            assert!(pack_host_parts(&[&nan]).is_err());
            let huge = linear(&[f32::MAX, 0.0], [1, 2], None);
            assert!(pack_host_parts(&[&huge]).is_err());
        }

        #[test]
        fn weight_scale_uses_values_before_fp16_rounding() {
            let source = linear(&[1.0004, -0.25], [1, 2], None);
            let (packed, scale, _) = pack_host_parts(&[&source]).unwrap();
            assert_eq!(scale, 1.0004 / E4M3_MAX);
            assert_eq!(packed.to_f32_vec().unwrap(), vec![1.0, -0.25]);
            let zero = linear(&[0.0, 0.0], [1, 2], None);
            assert_eq!(pack_host_parts(&[&zero]).unwrap().1, 1.0);
        }
    }
}
pub use linear::*;

use crate::musa_pi05::weights::bf16_to_device;
// Fully materialized static-FP8 π0.5 weights.

use crate::musa_pi05::backend::Context;
use apxinf_core::{Result, Tensor};

use crate::musa_pi05::{
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights, Pi05Weights,
    VisionBlockWeights,
};

#[derive(Debug)]
pub struct Fp8StaticDeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceVisionBlock {
    pub norm1: Fp8StaticDeviceLayerNorm,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub norm2: Fp8StaticDeviceLayerNorm,
    pub fc1: Fp8StaticLinearWeights,
    pub fc2: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Fp8StaticLinearWeights,
    pub down: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceActionLayer {
    pub input_modulation: Fp8StaticLinearWeights,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub post_attention_modulation: Fp8StaticLinearWeights,
    pub gate_up: Fp8StaticLinearWeights,
    pub down: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticWeights {
    pub patch_embedding: Fp8StaticLinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Fp8StaticDeviceVisionBlock>,
    pub vision_post_norm: Fp8StaticDeviceLayerNorm,
    pub multimodal_projector: Fp8StaticLinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Fp8StaticDeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Fp8StaticDeviceActionLayer>,
    pub action_final_modulation: Fp8StaticLinearWeights,
    pub action_in: Fp8StaticLinearWeights,
    pub action_out: Fp8StaticLinearWeights,
    pub time_mlp_in: Fp8StaticLinearWeights,
    pub time_mlp_out: Fp8StaticLinearWeights,
}

impl Fp8StaticWeights {
    pub fn from_host(weights: &Pi05Weights, backend: &Context) -> Result<Self> {
        crate::musa_pi05::backend::ops::prepare_attention(backend, 8, 1, 256)?;
        let vision_layers = weights
            .vision
            .blocks
            .iter()
            .map(|layer| Fp8StaticDeviceVisionBlock::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let language_layers = weights
            .language_layers
            .iter()
            .map(|layer| Fp8StaticDeviceLanguageLayer::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let action_layers = weights
            .action_layers
            .iter()
            .map(|layer| Fp8StaticDeviceActionLayer::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding: Fp8StaticLinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers,
            vision_post_norm: Fp8StaticDeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Fp8StaticLinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers,
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Fp8StaticLinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Fp8StaticLinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Fp8StaticLinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Fp8StaticLinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Fp8StaticDeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Fp8StaticDeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            norm1: Fp8StaticDeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.output, backend)?,
            norm2: Fp8StaticDeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Fp8StaticLinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Fp8StaticLinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Fp8StaticDeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            // The prefix layers keep the plain `[gate | up]` packing: their
            // composed producer's half-split packed4 kernel is much wider than
            // the pair-aware one, which only pays off for the action shape.
            gate_up: Fp8StaticLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp8StaticLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Fp8StaticDeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_modulation: modulation_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Fp8StaticLinearWeights::from_host_parts_interleaved(
                &[&weights.mlp.gate, &weights.mlp.up],
                GEGLU_PAIR_WIDTH,
                backend,
            )?,
            down: Fp8StaticLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &Context,
) -> Result<Fp8StaticLinearWeights> {
    Fp8StaticLinearWeights::from_host(&weights.modulation, backend)
}
