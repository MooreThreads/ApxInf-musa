//! Static-FP8 GR00T device weights.
//!
//! One calibrated per-tensor scale serves every linear: the weight scale is
//! derived here from the host BF16 magnitude, and the activation scale comes
//! from the shared calibration profile. Bias and every intermediate stay BF16
//! so residual, normalization and attention boundaries match the reference
//! graph exactly, like the CUDA static-FP8 executor.
//!
//! Fused E4M3 producers are used where the width matches their contract: the
//! norm hooks, `quantize_bias_gelu_reusable_input`, and the adaptive LayerNorm
//! dual-output kernel. Other widths compose the BF16 producer with
//! `quantize_fixed_e4m3`. Both paths round through BF16 before converting, so
//! any difference stays on the E4M3 grid (the fused norm kernels keep the
//! sibling's one-pass statistics, which can move a value on a rounding
//! boundary by one ULP).

use apxinf_core::{DType, Error, Result, Tensor};

use super::super::action_weights::Gr00tLinearWeights;
use super::super::backend::{kernels, RuntimeBackend};
use super::super::model::linear::DeviceLinearWeights;
use super::fp8_calibration::e4m3_weight_scale;

#[derive(Debug)]
pub(crate) struct Gr00tFp8LinearWeights {
    weight: Tensor,
    weight_scale: f32,
    activation_scale: f32,
    bias: Option<Tensor>,
}

impl Gr00tFp8LinearWeights {
    fn require_positive_scale(scale: f32) -> Result<()> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Other(format!(
                "GR00T FP8 activation scale must be finite and positive, got {scale}"
            )));
        }
        Ok(())
    }

    fn quantize_activation(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        backend.quantize_e4m3(input, self.activation_scale)
    }

    fn quantize_host_weight(weight: &Tensor, backend: &RuntimeBackend) -> Result<(Tensor, f32)> {
        let maximum = weight
            .to_f32_vec()?
            .into_iter()
            .fold(0.0f32, |value, next| value.max(next.abs()));
        let weight_scale = e4m3_weight_scale(maximum);
        let weight = backend.to_device(weight)?;
        let weight = backend.quantize_e4m3(&weight, weight_scale)?;
        Ok((weight, weight_scale))
    }

    pub(crate) fn from_host(
        weights: Gr00tLinearWeights,
        activation_scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        Self::require_positive_scale(activation_scale)?;
        let (weight, weight_scale) = Self::quantize_host_weight(&weights.weight, backend)?;
        Ok(Self {
            weight,
            weight_scale,
            activation_scale,
            bias: Some(backend.to_device(&weights.bias)?),
        })
    }

    pub(crate) fn matrix(
        weight: &Tensor,
        activation_scale: f32,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        Self::require_positive_scale(activation_scale)?;
        let (weight, weight_scale) = Self::quantize_host_weight(weight, backend)?;
        Ok(Self {
            weight,
            weight_scale,
            activation_scale,
            bias: None,
        })
    }
}

impl DeviceLinearWeights for Gr00tFp8LinearWeights {
    type ReusableInput = Tensor;

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let input = self.quantize_activation(input, backend)?;
        self.forward_quantized_tensor(&input, backend)
    }

    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn activation_scale(&self) -> Option<f32> {
        Some(self.activation_scale)
    }

    fn can_share_quantized_input_with(&self, other: &Self) -> bool {
        self.activation_scale == other.activation_scale
    }

    fn quantize_reusable_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(self.quantize_activation(input, backend)?))
    }

    fn forward_reusable_quantized(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        self.forward_quantized_tensor(input, backend)
    }

    /// Fuse the FFN's bias+GELU with the E4M3 quantizer.
    ///
    /// `self` is the down projection, so `self.activation_scale` is the scale
    /// its GEMM expects; the executor composes this on the first projection's
    /// BF16 output. `bias_gelu_bf16_e4m3_into` rounds to BF16 before the
    /// conversion, matching `bias_gelu_bf16` followed by
    /// `quantize_fixed_e4m3`. Widths outside `cols % 8` compose instead.
    fn quantize_bias_gelu_reusable_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        let dims = input.shape().dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        if kernels::activation::bias_gelu_e4m3_supported(dims[1]) {
            let mut output = backend
                .context()
                .allocate_output(input.shape().clone(), DType::F8E4M3)?;
            kernels::activation::bias_gelu_bf16_e4m3_into(
                backend.context(),
                input,
                Some(bias),
                &mut output,
                self.activation_scale,
            )?;
            Ok(Some(output))
        } else {
            let gated = kernels::activation::bias_gelu_bf16(backend.context(), input, Some(bias))?;
            Ok(Some(backend.quantize_e4m3(&gated, self.activation_scale)?))
        }
    }

    fn supports_fused_bias_gelu_quantization(&self) -> bool {
        true
    }

    /// The FEED-FORWARD residual path consumes the quantized normalized tensor
    /// returned by `layer_norm_quantized`, skipping the redundant quantize of
    /// the BF16 producer. The output bias and residual add stay fused through
    /// `bias_then_residual_bf16`.
    fn uses_quantized_output(&self) -> bool {
        true
    }

    /// Only the executor's concatenated QKV weight is queried. That weight is
    /// built from three linears that already share one activation scale, so
    /// the fused GEMM needs no per-projection quantization boundary.
    fn supports_fused_self_qkv(&self) -> bool {
        true
    }

    fn quantize_tensor_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(Some(self.quantize_activation(input, backend)?))
    }

    /// The executor owns the bias.
    ///
    /// Every caller (`linear`, `linear_quantized`, `linear_relu/silu/gelu`, and
    /// the feed-forward helpers in `model/executor.rs`) adds `weights.bias()`
    /// exactly once after this projection, matching the CUDA static-FP8
    /// executor. Adding it here as well would add it twice numerically and
    /// emit one extra device kernel per action-head linear.
    fn forward_quantized_tensor(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let alpha = self.activation_scale * self.weight_scale;
        backend.matmul_fp8(input, &self.weight, alpha)
    }

    fn supports_bias_residual_gemm(&self) -> bool {
        true
    }

    /// The fused self-QKV projection and its bias in one muDNN call.
    fn forward_reusable_quantized_bias(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let alpha = self.activation_scale * self.weight_scale;
        Ok(Some(backend.matmul_fp8_bias(
            input, &self.weight, bias, alpha,
        )?))
    }

    /// The FFN down projection, its bias and the residual add in one muDNN
    /// call. `input` is the E4M3 GeLU output the up projection already
    /// produced, so no extra quantize runs here.
    fn forward_reusable_quantized_bias_residual(
        &self,
        input: &Tensor,
        bias: &Tensor,
        residual: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let alpha = self.activation_scale * self.weight_scale;
        Ok(Some(backend.matmul_fp8_bias_residual(
            input, &self.weight, bias, residual, alpha,
        )?))
    }

    fn rms_norm_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(kernels::norm::rms_quant_bf16_e4m3(
            backend.context(),
            input,
            weight,
            eps,
            self.activation_scale,
        )?))
    }

    /// Fold `hidden = projection + residual` into the RMS norm + E4M3
    /// producer. Qwen3's projections carry no bias, so the fused kernel runs
    /// with `bias = None`: it writes `bf16(projection + residual)`, the same
    /// bytes `backend.add` produces, and normalizes that rounded row with the
    /// same block2 reduction as `rms_norm_bf16_e4m3_block2`.
    fn residual_rms_norm_quantized(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        norm_weight: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let dims = projection.shape().dims();
        let Some(&cols) = dims.get(1) else {
            return Ok(None);
        };
        if dims.len() != 2 || !kernels::fused::bias_residual_rms_e4m3_supported(cols) {
            return Ok(None);
        }
        let mut hidden = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        let mut normalized = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::F8E4M3)?;
        kernels::fused::bias_residual_rms_e4m3_into(
            backend.context(),
            projection,
            None,
            residual,
            norm_weight,
            &mut hidden,
            &mut normalized,
            eps,
            self.activation_scale,
        )?;
        Ok(Some((hidden, normalized)))
    }

    /// Fold the attention output bias, residual add, the DiT LayerNorm, and
    /// the E4M3 normalization this FFN input consumes into one launch.
    ///
    /// The bias is rounded to BF16 before the residual add, matching
    /// `bias_bf16` followed by the residual add kernel. The quantized output
    /// uses this (consumer) weight's calibrated scale.
    fn bias_residual_layer_norm_quantized(
        &self,
        projection: &Tensor,
        projection_bias: Option<&Tensor>,
        residual: &Tensor,
        norm_weight: &Tensor,
        norm_bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor, Self::ReusableInput)>> {
        let Some(bias) = projection_bias else {
            return Ok(None);
        };
        if projection.shape().dims() != [41, 1536]
            || residual.shape() != projection.shape()
            || bias.shape().dims() != [1536]
        {
            return Ok(None);
        }
        let fused = kernels::fused::bias_then_residual_layer_e4m3_dual(
            backend.context(),
            projection,
            bias,
            residual,
            norm_weight,
            norm_bias,
            eps,
            self.activation_scale,
        )?;
        Ok(Some((fused.hidden, fused.normalized, fused.quantized)))
    }

    /// The executor needs both the BF16 tensor for the graph and the E4M3
    /// operand for the projection. The dual-output producer computes both in
    /// one pass over the row, so dropping the half the FFN does not consume
    /// no longer costs a second LayerNorm.
    fn layer_norm_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let (normalized, quantized) = kernels::norm::layer_quant_bf16_e4m3_dual(
            backend.context(),
            input,
            weight,
            bias,
            eps,
            self.activation_scale,
        )?;
        Ok(Some((normalized, quantized)))
    }

    fn adaptive_layer_norm_quantized(
        &self,
        input: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let mut normalized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::BF16)?;
        let mut quantized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::F8E4M3)?;
        kernels::norm::adaptive_layer_bf16_e4m3_dual_into(
            backend.context(),
            input,
            modulation,
            &mut normalized,
            &mut quantized,
            eps,
            self.activation_scale,
        )?;
        Ok(Some((normalized, quantized)))
    }
}
