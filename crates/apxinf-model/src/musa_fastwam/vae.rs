//! Wan2.2 48-channel video VAE for the fixed Fast-WAM profile.
//!
//! This is a direct composition of the `VideoVAE38_` topology in upstream
//! `wan_video_vae.py`.  The official decoder streams one latent frame at a
//! time.  For the fixed three-frame profile, ordinary causal convolutions are
//! equivalent to processing the complete sequence; the two temporal resample
//! blocks retain the upstream special case explicitly: keep the first frame,
//! apply `time_conv` only to future frames, then interleave its two channel
//! groups on the time axis.

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_musa::ops::{self, BinaryOp, ConvolutionArgs, UnaryOp};
use std::cell::RefCell;

use super::{FastWamBackend, FastWamConfig, WanVaeWeights};

const VAE_CHANNELS: usize = 48;
const VAE_SPATIAL_TOKENS: usize = 14 * 28;
const NORMALIZE_EPS: f64 = 1.0e-12;

const VAE_MEAN: [f32; VAE_CHANNELS] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
];

const VAE_STD: [f32; VAE_CHANNELS] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

pub(crate) struct WanVae {
    backend: FastWamBackend,
    weights: WanVaeWeights,
    mean: Tensor,
    negative_mean: Tensor,
    inv_std: Tensor,
    std: Tensor,
    quarter: Tensor,
    negative_one: Tensor,
    positive_one: Tensor,
    spatial_attention_mask: Tensor,
    // Pad is implemented through row concatenation on this muDNN release.
    // Keep the zero-row operands in stable device storage so capture never
    // performs a CPU allocation plus H2D transfer.
    zero_rows_cache: RefCell<Vec<((usize, usize), Tensor)>>,
}

impl WanVae {
    pub(crate) fn from_host(
        backend: FastWamBackend,
        config: &FastWamConfig,
        weights: WanVaeWeights,
    ) -> Result<Self> {
        config.validate()?;
        let weights = weights.into_device(&backend)?;
        let channel_shape = [1, VAE_CHANNELS, 1, 1, 1];
        let mean = upload_bf16(&backend, &channel_shape, &VAE_MEAN)?;
        let negative_mean_values = VAE_MEAN.map(|value| -value);
        let negative_mean = upload_bf16(&backend, &channel_shape, &negative_mean_values)?;
        let inv_std_values = VAE_STD.map(|value| value.recip());
        let inv_std = upload_bf16(&backend, &channel_shape, &inv_std_values)?;
        let std = upload_bf16(&backend, &channel_shape, &VAE_STD)?;
        let quarter = upload_bf16(&backend, &[1], &[0.25])?;
        let negative_one = upload_bf16(&backend, &[1], &[-1.0])?;
        let positive_one = upload_bf16(&backend, &[1], &[1.0])?;
        let spatial_attention_mask = backend.to_device(&Tensor::from_i8(
            vec![VAE_SPATIAL_TOKENS, VAE_SPATIAL_TOKENS],
            &vec![1; VAE_SPATIAL_TOKENS * VAE_SPATIAL_TOKENS],
        )?)?;
        Ok(Self {
            backend,
            weights,
            mean,
            negative_mean,
            inv_std,
            std,
            quarter,
            negative_one,
            positive_one,
            spatial_attention_mask,
            zero_rows_cache: RefCell::new(Vec::new()),
        })
    }

    /// Encode the official merged two-camera image `[1,3,224,448]` into the
    /// first-frame latent `[1,48,1,14,28]`.
    pub(crate) fn encode_first_frame(&self, image: &Tensor) -> Result<Tensor> {
        if image.shape().dims() != [1, 3, 224, 448] {
            return Err(Error::Other(format!(
                "Wan VAE image must be [1,3,224,448], got {:?}",
                image.shape().dims()
            )));
        }
        let mut x = image.reshape(vec![1, 3, 1, 224, 448])?;
        x = self.patchify_2x2(&x)?;
        x = self.causal_conv3d(&x, "model.encoder.conv1", [1, 1, 1], [1, 1, 1])?;

        let encoder_dims = [(160, 160), (160, 320), (320, 640), (640, 640)];
        for (stage, &(input_channels, output_channels)) in encoder_dims.iter().enumerate() {
            let shortcut = self.avg_down3d(
                &x,
                output_channels,
                if stage == 1 || stage == 2 { 2 } else { 1 },
                if stage < 3 { 2 } else { 1 },
            )?;
            for block in 0..2 {
                x = self.residual_block(
                    &x,
                    &format!("model.encoder.downsamples.{stage}.downsamples.{block}"),
                )?;
            }
            if stage < 3 {
                x = self.downsample_spatial(
                    &x,
                    &format!("model.encoder.downsamples.{stage}.downsamples.2.resample.1"),
                )?;
                // The official encoder's first (and only) cache chunk stores
                // the temporal input and deliberately skips `time_conv`.
            }
            x = ops::binary_bf16(self.backend.context(), &x, &shortcut, BinaryOp::Add)?;
            debug_assert_eq!(x.shape().dims()[1], output_channels);
            debug_assert_eq!(shortcut.shape().dims()[1], output_channels);
            debug_assert!(input_channels <= output_channels);
        }

        x = self.residual_block(&x, "model.encoder.middle.0")?;
        x = self.attention_block(&x, "model.encoder.middle.1")?;
        x = self.residual_block(&x, "model.encoder.middle.2")?;
        x = self.channel_rms_norm(&x, "model.encoder.head.0.gamma")?;
        x = ops::unary_bf16(self.backend.context(), &x, UnaryOp::Silu)?;
        x = self.causal_conv3d(&x, "model.encoder.head.2", [1, 1, 1], [1, 1, 1])?;
        x = self.causal_conv3d(&x, "model.conv1", [0, 0, 0], [1, 1, 1])?;
        let mu = self.channel_slice(&x, 0, VAE_CHANNELS)?;
        let centered = ops::binary_bf16(
            self.backend.context(),
            &mu,
            &self.negative_mean,
            BinaryOp::Add,
        )?;
        ops::binary_bf16(
            self.backend.context(),
            &centered,
            &self.inv_std,
            BinaryOp::Multiply,
        )
    }

    /// Decode `[1,48,3,14,28]` latents into clamped video
    /// `[1,3,9,224,448]`.
    pub(crate) fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        if latents.shape().dims() != [1, 48, 3, 14, 28] {
            return Err(Error::Other(format!(
                "Wan VAE decode latents must be [1,48,3,14,28], got {:?}",
                latents.shape().dims()
            )));
        }
        let scaled = ops::binary_bf16(
            self.backend.context(),
            latents,
            &self.std,
            BinaryOp::Multiply,
        )?;
        let mut x = ops::binary_bf16(
            self.backend.context(),
            &scaled,
            &self.mean,
            BinaryOp::Add,
        )?;
        x = self.causal_conv3d(&x, "model.conv2", [0, 0, 0], [1, 1, 1])?;
        x = self.causal_conv3d(&x, "model.decoder.conv1", [1, 1, 1], [1, 1, 1])?;
        x = self.residual_block(&x, "model.decoder.middle.0")?;
        x = self.attention_block(&x, "model.decoder.middle.1")?;
        x = self.residual_block(&x, "model.decoder.middle.2")?;

        let decoder_dims = [(1024, 1024), (1024, 1024), (1024, 512), (512, 256)];
        for (stage, &(input_channels, output_channels)) in decoder_dims.iter().enumerate() {
            let stage_input = x.clone();
            for block in 0..3 {
                x = self.residual_block(
                    &x,
                    &format!("model.decoder.upsamples.{stage}.upsamples.{block}"),
                )?;
            }
            if stage < 3 {
                let temporal = stage < 2;
                if temporal {
                    x = self.temporal_upsample_main(
                        &x,
                        &format!("model.decoder.upsamples.{stage}.upsamples.3.time_conv"),
                    )?;
                }
                x = self.upsample_spatial(
                    &x,
                    &format!("model.decoder.upsamples.{stage}.upsamples.3.resample.1"),
                )?;
                let shortcut = self.dup_up3d(
                    &stage_input,
                    output_channels,
                    if temporal { 2 } else { 1 },
                    2,
                    temporal,
                )?;
                x = ops::binary_bf16(self.backend.context(), &x, &shortcut, BinaryOp::Add)?;
            }
            debug_assert_eq!(stage_input.shape().dims()[1], input_channels);
            debug_assert_eq!(x.shape().dims()[1], output_channels);
        }

        x = self.channel_rms_norm(&x, "model.decoder.head.0.gamma")?;
        x = ops::unary_bf16(self.backend.context(), &x, UnaryOp::Silu)?;
        x = self.causal_conv3d(&x, "model.decoder.head.2", [1, 1, 1], [1, 1, 1])?;
        x = self.unpatchify_2x2(&x)?;
        x = ops::binary_bf16(
            self.backend.context(),
            &x,
            &self.negative_one,
            BinaryOp::Maximum,
        )?;
        ops::binary_bf16(
            self.backend.context(),
            &x,
            &self.positive_one,
            BinaryOp::Minimum,
        )
    }

    fn residual_block(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let in_channels = input.shape().dims()[1];
        let first_weight = self.weight(&format!("{prefix}.residual.2.weight"))?;
        let out_channels = first_weight.shape().dims()[0];
        let shortcut = if in_channels == out_channels {
            input.clone()
        } else {
            self.causal_conv3d(input, &format!("{prefix}.shortcut"), [0, 0, 0], [1, 1, 1])?
        };
        let mut x = self.channel_rms_norm(input, &format!("{prefix}.residual.0.gamma"))?;
        x = ops::unary_bf16(self.backend.context(), &x, UnaryOp::Silu)?;
        x = self.causal_conv3d(&x, &format!("{prefix}.residual.2"), [1, 1, 1], [1, 1, 1])?;
        x = self.channel_rms_norm(&x, &format!("{prefix}.residual.3.gamma"))?;
        x = ops::unary_bf16(self.backend.context(), &x, UnaryOp::Silu)?;
        x = self.causal_conv3d(&x, &format!("{prefix}.residual.6"), [1, 1, 1], [1, 1, 1])?;
        ops::binary_bf16(self.backend.context(), &x, &shortcut, BinaryOp::Add)
    }

    fn attention_block(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE attention expects NCTHW".into()));
        };
        let (batch, channels, time, height, width) =
            (*batch, *channels, *time, *height, *width);
        if height * width != VAE_SPATIAL_TOKENS {
            return Err(Error::Other(format!(
                "Wan VAE attention expects {VAE_SPATIAL_TOKENS} spatial tokens, got {}",
                height * width
            )));
        }
        let mut x = ops::permute_bf16(self.backend.context(), input, &[0, 2, 1, 3, 4])?
            .reshape(vec![batch * time, channels, height, width])?;
        x = self.channel_rms_norm_4d(&x, &format!("{prefix}.norm.gamma"))?;
        x = self.conv2d(&x, &format!("{prefix}.to_qkv"), [0, 0], [1, 1])?;
        let q = self.channel_slice_4d(&x, 0, channels)?;
        let k = self.channel_slice_4d(&x, channels, channels)?;
        let v = self.channel_slice_4d(&x, 2 * channels, channels)?;
        let to_attention = |value: &Tensor| -> Result<Tensor> {
            Ok(ops::permute_bf16(self.backend.context(), value, &[0, 2, 3, 1])?
                .reshape(vec![batch * time, height * width, 1, channels])
                .and_then(|value| {
                    ops::permute_bf16(self.backend.context(), &value, &[0, 2, 1, 3])
                })?)
        };
        let q = to_attention(&q)?;
        let k = to_attention(&k)?;
        let v = to_attention(&v)?;
        x = ops::masked_math_attention_bf16(
            self.backend.context(),
            &q,
            &k,
            &v,
            &self.spatial_attention_mask,
        )?;
        x = ops::permute_bf16(self.backend.context(), &x, &[0, 2, 1, 3])?
            .reshape(vec![batch * time, height, width, channels])?;
        x = ops::permute_bf16(self.backend.context(), &x, &[0, 3, 1, 2])?;
        x = self.conv2d(&x, &format!("{prefix}.proj"), [0, 0], [1, 1])?;
        x = x.reshape(vec![batch, time, channels, height, width])?;
        x = ops::permute_bf16(self.backend.context(), &x, &[0, 2, 1, 3, 4])?;
        ops::binary_bf16(self.backend.context(), &x, input, BinaryOp::Add)
    }

    fn channel_rms_norm(&self, input: &Tensor, gamma_name: &str) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE channel RMS norm expects NCTHW".into()));
        };
        let mut x = ops::permute_bf16(self.backend.context(), input, &[0, 2, 3, 4, 1])?;
        x = ops::rms_norm_bf16(self.backend.context(), &x, NORMALIZE_EPS)?;
        let gamma = self.weight(gamma_name)?.reshape(vec![*channels])?;
        x = ops::binary_bf16(self.backend.context(), &x, &gamma, BinaryOp::Multiply)?;
        debug_assert_eq!(x.shape().dims(), [*batch, *time, *height, *width, *channels]);
        ops::permute_bf16(self.backend.context(), &x, &[0, 4, 1, 2, 3])
    }

    fn channel_rms_norm_4d(&self, input: &Tensor, gamma_name: &str) -> Result<Tensor> {
        let [batch, channels, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE image RMS norm expects NCHW".into()));
        };
        let mut x = ops::permute_bf16(self.backend.context(), input, &[0, 2, 3, 1])?;
        x = ops::rms_norm_bf16(self.backend.context(), &x, NORMALIZE_EPS)?;
        let gamma = self.weight(gamma_name)?.reshape(vec![*channels])?;
        x = ops::binary_bf16(self.backend.context(), &x, &gamma, BinaryOp::Multiply)?;
        debug_assert_eq!(x.shape().dims(), [*batch, *height, *width, *channels]);
        ops::permute_bf16(self.backend.context(), &x, &[0, 3, 1, 2])
    }

    fn causal_conv3d(
        &self,
        input: &Tensor,
        prefix: &str,
        padding: [usize; 3],
        stride: [usize; 3],
    ) -> Result<Tensor> {
        let padded = self.pad_ncthw(
            input,
            [2 * padding[0], 0],
            [padding[1], padding[1]],
            [padding[2], padding[2]],
        )?;
        ops::convolution_bf16(
            self.backend.context(),
            ConvolutionArgs {
                input: &padded,
                weight: self.weight(&format!("{prefix}.weight"))?,
                bias: Some(self.weight(&format!("{prefix}.bias"))?),
                padding: &[0, 0, 0],
                stride: &stride,
                dilation: &[1, 1, 1],
                groups: 1,
            },
        )
    }

    /// muDNN Pad on this MUSA release supports up to rank four.  Apply the
    /// NCTHW padding through rank-four views while preserving contiguous
    /// temporal/spatial order; this is mathematically the same constant pad
    /// and keeps the VAE free of custom kernels.
    fn pad_ncthw(
        &self,
        input: &Tensor,
        time: [usize; 2],
        height: [usize; 2],
        width: [usize; 2],
    ) -> Result<Tensor> {
        let [batch, channels, frames, rows, cols] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE pad expects NCTHW".into()));
        };
        let mut x = input.clone();
        if time != [0, 0] {
            let row_width = *batch * *channels * *rows * *cols;
            let rows_in = ops::permute_bf16(self.backend.context(), &x, &[2, 0, 1, 3, 4])?
                .reshape(vec![*frames, row_width])?;
            let before = self.zero_rows(time[0], row_width)?;
            let after = self.zero_rows(time[1], row_width)?;
            let rows_out = self.concat_rows(&before, &rows_in, &after)?;
            let out_frames = *frames + time[0] + time[1];
            x = rows_out
                .reshape(vec![out_frames, *batch, *channels, *rows, *cols])?;
            x = ops::permute_bf16(self.backend.context(), &x, &[1, 2, 0, 3, 4])?;
        }
        if height != [0, 0] || width != [0, 0] {
            let current_frames = x.shape().dims()[2];
            x = ops::permute_bf16(self.backend.context(), &x, &[0, 2, 1, 3, 4])?
                .reshape(vec![*batch * current_frames, *channels, *rows, *cols])?;
            x = self.pad_nchw(&x, height, width)?;
            let out_rows = *rows + height[0] + height[1];
            let out_cols = *cols + width[0] + width[1];
            x = x.reshape(vec![*batch, current_frames, *channels, out_rows, out_cols])?;
            x = ops::permute_bf16(self.backend.context(), &x, &[0, 2, 1, 3, 4])?;
        }
        Ok(x)
    }

    fn pad_nchw(&self, input: &Tensor, height: [usize; 2], width: [usize; 2]) -> Result<Tensor> {
        let [batch, channels, rows, cols] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE pad expects NCHW".into()));
        };
        let row_width = *batch * *channels * *cols;
        let rows_in = ops::permute_bf16(self.backend.context(), input, &[2, 0, 1, 3])?
            .reshape(vec![*rows, row_width])?;
        let rows_out = self.concat_rows(
            &self.zero_rows(height[0], row_width)?,
            &rows_in,
            &self.zero_rows(height[1], row_width)?,
        )?;
        let out_rows = *rows + height[0] + height[1];
        let x = rows_out
            .reshape(vec![out_rows, *batch, *channels, *cols])?;
        let x = ops::permute_bf16(self.backend.context(), &x, &[1, 2, 0, 3])?;
        let row_width = *batch * *channels * out_rows;
        let rows_in = ops::permute_bf16(self.backend.context(), &x, &[3, 0, 1, 2])?
            .reshape(vec![*cols, row_width])?;
        let rows_out = self.concat_rows(
            &self.zero_rows(width[0], row_width)?,
            &rows_in,
            &self.zero_rows(width[1], row_width)?,
        )?;
        let out_cols = *cols + width[0] + width[1];
        let x = rows_out
            .reshape(vec![out_cols, *batch, *channels, out_rows])?;
        ops::permute_bf16(self.backend.context(), &x, &[1, 2, 3, 0])
    }

    fn zero_rows(&self, rows: usize, width: usize) -> Result<Tensor> {
        if let Some((_, tensor)) = self
            .zero_rows_cache
            .borrow()
            .iter()
            .find(|((cached_rows, cached_width), _)| {
                *cached_rows == rows && *cached_width == width
            })
        {
            return Ok(tensor.clone());
        }
        let host = Tensor::zeros(vec![rows, width], DType::BF16);
        let device = self.backend.to_device(&host)?;
        self.zero_rows_cache
            .borrow_mut()
            .push(((rows, width), device.clone()));
        Ok(device)
    }

    fn concat_rows(&self, before: &Tensor, middle: &Tensor, after: &Tensor) -> Result<Tensor> {
        let joined = if before.shape().dims()[0] == 0 {
            middle.clone()
        } else if middle.shape().dims()[0] == 0 {
            before.clone()
        } else {
            ops::concat_rows(self.backend.context(), before, middle)?
        };
        if after.shape().dims()[0] == 0 {
            Ok(joined)
        } else if joined.shape().dims()[0] == 0 {
            Ok(after.clone())
        } else {
            ops::concat_rows(self.backend.context(), &joined, after)
        }
    }

    fn conv2d(
        &self,
        input: &Tensor,
        prefix: &str,
        padding: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Tensor> {
        ops::convolution_bf16(
            self.backend.context(),
            ConvolutionArgs {
                input,
                weight: self.weight(&format!("{prefix}.weight"))?,
                bias: Some(self.weight(&format!("{prefix}.bias"))?),
                padding: &padding,
                stride: &stride,
                dilation: &[1, 1],
                groups: 1,
            },
        )
    }

    fn downsample_spatial(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE downsample expects NCTHW".into()));
        };
        let frames = ops::permute_bf16(self.backend.context(), input, &[0, 2, 1, 3, 4])?
            .reshape(vec![batch * time, *channels, *height, *width])?;
        let frames = self.pad_nchw(&frames, [0, 1], [0, 1])?;
        let frames = self.conv2d(&frames, prefix, [0, 0], [2, 2])?;
        let output_height = frames.shape().dims()[2];
        let output_width = frames.shape().dims()[3];
        let frames = frames.reshape(vec![*batch, *time, *channels, output_height, output_width])?;
        ops::permute_bf16(self.backend.context(), &frames, &[0, 2, 1, 3, 4])
    }

    fn upsample_spatial(&self, input: &Tensor, prefix: &str) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE upsample expects NCTHW".into()));
        };
        let frames = ops::permute_bf16(self.backend.context(), input, &[0, 2, 1, 3, 4])?
            .reshape(vec![batch * time, *channels, *height, *width])?;
        let frames = ops::interpolate_nearest_exact_bf16(
            self.backend.context(),
            &frames,
            &[2 * height, 2 * width],
        )?;
        let frames = self.conv2d(&frames, prefix, [1, 1], [1, 1])?;
        let frames = frames.reshape(vec![*batch, *time, *channels, 2 * height, 2 * width])?;
        ops::permute_bf16(self.backend.context(), &frames, &[0, 2, 1, 3, 4])
    }

    fn temporal_upsample_main(&self, input: &Tensor, time_conv: &str) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE temporal upsample expects NCTHW".into()));
        };
        let first = self.temporal_slice(input, 0, 1)?;
        if *time == 1 {
            return Ok(first);
        }
        let future = self.temporal_slice(input, 1, time - 1)?;
        let future = self.causal_conv3d(&future, time_conv, [1, 0, 0], [1, 1, 1])?;
        let future = future.reshape(vec![*batch, 2, *channels, time - 1, *height, *width])?;
        let future = ops::permute_bf16(self.backend.context(), &future, &[0, 2, 3, 1, 4, 5])?
            .reshape(vec![*batch, *channels, 2 * (time - 1), *height, *width])?;
        self.concat_temporal(&first, &future)
    }

    fn avg_down3d(
        &self,
        input: &Tensor,
        output_channels: usize,
        factor_t: usize,
        factor_s: usize,
    ) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE AvgDown3D expects NCTHW".into()));
        };
        if height % factor_s != 0 || width % factor_s != 0 {
            return Err(Error::Other("Wan VAE AvgDown3D spatial geometry is not divisible".into()));
        }
        let pad_t = (factor_t - time % factor_t) % factor_t;
        let padded = if pad_t == 0 {
            input.clone()
        } else {
            self.pad_ncthw(input, [pad_t, 0], [0, 0], [0, 0])?
        };
        let padded_time = time + pad_t;
        let factor = factor_t * factor_s * factor_s;
        let grouped_channels = channels * factor;
        if grouped_channels % output_channels != 0 {
            return Err(Error::Other("Wan VAE AvgDown3D channel geometry is invalid".into()));
        }
        let group_size = grouped_channels / output_channels;
        let output_time = padded_time / factor_t;
        let output_height = height / factor_s;
        let output_width = width / factor_s;
        let x = padded.reshape(vec![
            *batch,
            *channels,
            output_time,
            factor_t,
            output_height,
            factor_s,
            output_width,
            factor_s,
        ])?;
        let x = ops::permute_bf16(self.backend.context(), &x, &[0, 1, 3, 5, 7, 2, 4, 6])?
            .reshape(vec![
                *batch,
                output_channels,
                group_size,
                output_time,
                output_height,
                output_width,
            ])?;
        if group_size == 1 {
            return x.reshape(vec![
                *batch,
                output_channels,
                output_time,
                output_height,
                output_width,
            ]);
        }
        if group_size != 4 {
            return Err(Error::Other(format!(
                "Wan VAE fixed profile expects AvgDown3D group size 1 or 4, got {group_size}"
            )));
        }
        let rows = ops::permute_bf16(self.backend.context(), &x, &[2, 0, 1, 3, 4, 5])?
            .reshape(vec![group_size, x.numel() / group_size])?;
        let mut sum = ops::contiguous_rows(self.backend.context(), &rows, 0, 1)?;
        for row in 1..group_size {
            let value = ops::contiguous_rows(self.backend.context(), &rows, row, 1)?;
            sum = ops::binary_bf16(self.backend.context(), &sum, &value, BinaryOp::Add)?;
        }
        let mean = ops::binary_bf16(
            self.backend.context(),
            &sum,
            &self.quarter,
            BinaryOp::Multiply,
        )?;
        mean.reshape(vec![
            *batch,
            output_channels,
            output_time,
            output_height,
            output_width,
        ])
    }

    fn dup_up3d(
        &self,
        input: &Tensor,
        output_channels: usize,
        factor_t: usize,
        factor_s: usize,
        drop_first_temporal: bool,
    ) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE DupUp3D expects NCTHW".into()));
        };
        let factor = factor_t * factor_s * factor_s;
        let total = output_channels
            .checked_mul(factor)
            .ok_or_else(|| Error::Other("Wan VAE DupUp3D channel overflow".into()))?;
        if total % channels != 0 {
            return Err(Error::Other("Wan VAE DupUp3D channel geometry is invalid".into()));
        }
        let repeats = total / channels;
        let repeated = self.repeat_interleave_channels(input, repeats)?;
        let repeated = repeated.reshape(vec![
            *batch,
            output_channels,
            factor_t,
            factor_s,
            factor_s,
            *time,
            *height,
            *width,
        ])?;
        let mut output = ops::permute_bf16(
            self.backend.context(),
            &repeated,
            &[0, 1, 5, 2, 6, 3, 7, 4],
        )?
        .reshape(vec![
            *batch,
            output_channels,
            time * factor_t,
            height * factor_s,
            width * factor_s,
        ])?;
        if drop_first_temporal && factor_t > 1 {
            output = self.temporal_slice(&output, factor_t - 1, output.shape().dims()[2] - factor_t + 1)?;
        }
        Ok(output)
    }

    fn repeat_interleave_channels(&self, input: &Tensor, repeats: usize) -> Result<Tensor> {
        if repeats == 1 {
            return Ok(input.clone());
        }
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE channel repeat expects NCTHW".into()));
        };
        let base = ops::permute_bf16(self.backend.context(), input, &[1, 0, 2, 3, 4])?
            .reshape(vec![*channels, batch * time * height * width])?;
        let mut blocks = base.clone();
        for _ in 1..repeats {
            blocks = ops::concat_rows(self.backend.context(), &blocks, &base)?;
        }
        let blocks = blocks.reshape(vec![repeats, *channels, *batch, *time, *height, *width])?;
        let blocks = ops::permute_bf16(self.backend.context(), &blocks, &[1, 0, 2, 3, 4, 5])?
            .reshape(vec![channels * repeats, *batch, *time, *height, *width])?;
        ops::permute_bf16(self.backend.context(), &blocks, &[1, 0, 2, 3, 4])
    }

    fn patchify_2x2(&self, input: &Tensor) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE patchify expects NCTHW".into()));
        };
        if height % 2 != 0 || width % 2 != 0 {
            return Err(Error::Other("Wan VAE patchify expects even spatial dimensions".into()));
        }
        let x = input.reshape(vec![*batch, *channels, *time, height / 2, 2, width / 2, 2])?;
        let x = ops::permute_bf16(self.backend.context(), &x, &[0, 1, 6, 4, 2, 3, 5])?;
        x.reshape(vec![*batch, channels * 4, *time, height / 2, width / 2])
    }

    fn unpatchify_2x2(&self, input: &Tensor) -> Result<Tensor> {
        let [batch, packed_channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE unpatchify expects NCTHW".into()));
        };
        if packed_channels % 4 != 0 {
            return Err(Error::Other("Wan VAE unpatchify channel count must be divisible by four".into()));
        }
        let channels = packed_channels / 4;
        let x = input.reshape(vec![*batch, channels, 2, 2, *time, *height, *width])?;
        let x = ops::permute_bf16(self.backend.context(), &x, &[0, 1, 4, 5, 3, 6, 2])?;
        x.reshape(vec![*batch, channels, *time, height * 2, width * 2])
    }

    fn temporal_slice(&self, input: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE temporal slice expects NCTHW".into()));
        };
        if count == 0 || start.checked_add(count).is_none_or(|end| end > *time) {
            return Err(Error::Other(format!(
                "Wan VAE temporal slice {start}..{} exceeds T={time}",
                start.saturating_add(count)
            )));
        }
        let rows = ops::permute_bf16(self.backend.context(), input, &[2, 0, 1, 3, 4])?
            .reshape(vec![*time, batch * channels * height * width])?;
        let rows = ops::contiguous_rows(self.backend.context(), &rows, start, count)?;
        let rows = rows.reshape(vec![count, *batch, *channels, *height, *width])?;
        ops::permute_bf16(self.backend.context(), &rows, &[1, 2, 0, 3, 4])
    }

    fn concat_temporal(&self, first: &Tensor, second: &Tensor) -> Result<Tensor> {
        let [batch, channels, first_time, height, width] = first.shape().dims() else {
            return Err(Error::Other("Wan VAE temporal concat expects NCTHW".into()));
        };
        let [second_batch, second_channels, second_time, second_height, second_width] =
            second.shape().dims()
        else {
            return Err(Error::Other("Wan VAE temporal concat expects NCTHW".into()));
        };
        if (batch, channels, height, width)
            != (second_batch, second_channels, second_height, second_width)
        {
            return Err(Error::Other("Wan VAE temporal concat geometry mismatch".into()));
        }
        let row_width = batch * channels * height * width;
        let first = ops::permute_bf16(self.backend.context(), first, &[2, 0, 1, 3, 4])?
            .reshape(vec![*first_time, row_width])?;
        let second = ops::permute_bf16(self.backend.context(), second, &[2, 0, 1, 3, 4])?
            .reshape(vec![*second_time, row_width])?;
        let rows = ops::concat_rows(self.backend.context(), &first, &second)?;
        let rows = rows.reshape(vec![first_time + second_time, *batch, *channels, *height, *width])?;
        ops::permute_bf16(self.backend.context(), &rows, &[1, 2, 0, 3, 4])
    }

    fn channel_slice(&self, input: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        let [batch, channels, time, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE channel slice expects NCTHW".into()));
        };
        if start + count > *channels {
            return Err(Error::Other("Wan VAE channel slice exceeds input".into()));
        }
        let rows = ops::permute_bf16(self.backend.context(), input, &[1, 0, 2, 3, 4])?
            .reshape(vec![*channels, batch * time * height * width])?;
        let rows = ops::contiguous_rows(self.backend.context(), &rows, start, count)?;
        let rows = rows.reshape(vec![count, *batch, *time, *height, *width])?;
        ops::permute_bf16(self.backend.context(), &rows, &[1, 0, 2, 3, 4])
    }

    fn channel_slice_4d(&self, input: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        let [batch, channels, height, width] = input.shape().dims() else {
            return Err(Error::Other("Wan VAE image channel slice expects NCHW".into()));
        };
        if start + count > *channels {
            return Err(Error::Other("Wan VAE image channel slice exceeds input".into()));
        }
        let rows = ops::permute_bf16(self.backend.context(), input, &[1, 0, 2, 3])?
            .reshape(vec![*channels, batch * height * width])?;
        let rows = ops::contiguous_rows(self.backend.context(), &rows, start, count)?;
        let rows = rows.reshape(vec![count, *batch, *height, *width])?;
        ops::permute_bf16(self.backend.context(), &rows, &[1, 0, 2, 3])
    }

    fn weight(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .tensors
            .get(name)
            .ok_or_else(|| Error::Other(format!("Wan VAE is missing {name}")))
    }
}

fn upload_bf16(backend: &FastWamBackend, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let values = values
        .iter()
        .copied()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_bf16(shape.to_vec(), &values)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vae_constants_match_release_profile() {
        assert_eq!(VAE_MEAN.len(), 48);
        assert_eq!(VAE_STD.len(), 48);
        assert!(VAE_STD.iter().all(|value| *value > 0.0));
        assert_eq!(VAE_SPATIAL_TOKENS, 392);
    }
}
