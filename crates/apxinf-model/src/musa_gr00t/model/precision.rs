//! Native-BF16 GR00T layer composition.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backbone::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use super::backend::RuntimeBackend;
use super::bf16_weights::Gr00tBf16LinearWeights;
use super::device_weights::DeviceLinearWeights;
use super::executor::{Gr00tExecutor, Gr00tPrecisionExecution};
use super::weights::fp8::Gr00tFp8LinearWeights;
use super::weights::fp8_calibration::Gr00tFp8Calibration;
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(crate) struct Gr00tBf16Execution;

impl Gr00tPrecisionExecution for Gr00tBf16Execution {
    type Dense = Gr00tBf16LinearWeights;
    type FeedForward = Gr00tBf16LinearWeights;
    type FusedQkv = Gr00tBf16LinearWeights;
    type Backbone = Gr00tBf16LinearWeights;

    const NAME: &'static str = "bf16";
    const SUPPORTS_CALIBRATION: bool = true;
    const USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM: bool = true;
    const USE_BIAS_RESIDUAL_LAYER_NORM: bool = true;
    const USE_CACHED_VISION_RESIDUAL_LAYER_NORM: bool = true;
    const USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM: bool = true;
    const USE_PRECOMPUTED_BF16_VISION_ROPE: bool = true;
    const USE_FUSED_QK_RMS_MROPE: bool = true;
    const USE_BATCHED_VISION_ATTENTION: bool = true;

    fn transfer_dense(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::Dense> {
        Gr00tBf16LinearWeights::from_host(weights, name, backend)
    }

    fn transfer_feed_forward(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward> {
        Gr00tBf16LinearWeights::from_host(weights, name, backend)
    }

    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        Ok(Some(Gr00tBf16LinearWeights::from_host(
            weights, name, backend,
        )?))
    }

    fn transfer_backbone_linears(
        &self,
        _text: &Qwen3VLTextWeights,
        _vision: &Qwen3VLVisionWeights,
        _backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        Ok(BTreeMap::new())
    }
}

pub(crate) type Gr00tBf16Executor = Gr00tExecutor<Gr00tBf16Execution>;

pub(crate) fn build(
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tBf16Executor> {
    Gr00tExecutor::from_backend(config, backbone, weights, Gr00tBf16Execution, backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_enables_batched_vision_attention() {
        assert!(Gr00tBf16Execution::USE_BATCHED_VISION_ATTENTION);
    }

    #[test]
    fn bf16_enables_qk_rms_mrope_fusion() {
        assert!(Gr00tBf16Execution::USE_FUSED_QK_RMS_MROPE);
    }

    #[test]
    fn bf16_enables_precomputed_vision_rope() {
        assert!(Gr00tBf16Execution::USE_PRECOMPUTED_BF16_VISION_ROPE);
    }

    #[test]
    fn bf16_enables_vision_mlp_residual_norm_fusion() {
        assert!(Gr00tBf16Execution::USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM);
    }

    #[test]
    fn bf16_explicitly_opts_into_packed8_bias_activation() {
        assert_eq!(Gr00tBf16Execution::USE_PACKED8_BF16_BIAS_ACTIVATION, false);
    }

    #[test]
    fn bf16_enables_bias_residual_adaptive_layer_norm() {
        assert!(Gr00tBf16Execution::USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM);
    }

    #[test]
    fn bf16_enables_bias_residual_layer_norm() {
        assert!(Gr00tBf16Execution::USE_BIAS_RESIDUAL_LAYER_NORM);
    }
}

/// Static-FP8 execution: calibrated per-tensor activation scales, E4M3
/// weights, BF16 biases and outputs.
///
/// Mirrors the CUDA static-FP8 executor's precision constants where they are
/// precision properties rather than Thor-specific tuning; the MUSA port has no
/// `sm`-based fusion policy, so the remaining fusions stay at their trait
/// defaults and the executor's composition path is used.
pub(crate) struct Gr00tFp8Execution {
    calibration: Gr00tFp8Calibration,
}

impl Gr00tFp8Execution {
    pub(crate) fn from_file(
        checkpoint: &Path,
        backbone: &Path,
        calibration: &Path,
        config: &Gr00tConfig,
        backbone_config: &Qwen3VLConfig,
    ) -> Result<Self> {
        let consumers = super::calibration::fp8_consumers(config, backbone_config);
        Ok(Self {
            calibration: Gr00tFp8Calibration::from_json_file(
                calibration,
                checkpoint,
                backbone,
                &consumers,
            )?,
        })
    }

    fn linear(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tFp8LinearWeights> {
        Gr00tFp8LinearWeights::from_host(weights, self.calibration.scale(name)?, backend)
    }

    fn matrix(
        &self,
        weight: &Tensor,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tFp8LinearWeights> {
        Gr00tFp8LinearWeights::matrix(weight, self.calibration.scale(name)?, backend)
    }
}

impl Gr00tPrecisionExecution for Gr00tFp8Execution {
    type Dense = Gr00tFp8LinearWeights;
    type FeedForward = Gr00tFp8LinearWeights;
    type FusedQkv = Gr00tFp8LinearWeights;
    type Backbone = Gr00tFp8LinearWeights;

    const NAME: &'static str = "fp8";
    const SUPPORTS_CALIBRATION: bool = false;
    const USE_PRECOMPUTED_BF16_VISION_ROPE: bool = true;
    const USE_BIAS_RESIDUAL_LAYER_NORM_QUANTIZED: bool = true;
    const USE_FUSED_QK_RMS_MROPE: bool = true;
    const USE_BATCHED_VISION_ATTENTION: bool = true;

    fn transfer_dense(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::Dense> {
        self.linear(weights, name, backend)
    }

    fn transfer_feed_forward(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward> {
        self.linear(weights, name, backend)
    }

    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        // The executor only concatenates equal-row Q/K/V and asks for the
        // fused linear here. Independent activation scales would erase the
        // calibrated boundaries, so a mismatch keeps the three-GEMM path.
        let prefix = name
            .strip_suffix(".fused_qkv")
            .ok_or_else(|| Error::Other(format!("GR00T fused QKV name {name:?} is malformed")))?;
        let query = self.calibration.scale(&format!("{prefix}.query"))?;
        let key = self.calibration.scale(&format!("{prefix}.key"))?;
        let value = self.calibration.scale(&format!("{prefix}.value"))?;
        if query != key || key != value {
            return Ok(None);
        }
        Ok(Some(Gr00tFp8LinearWeights::from_host(
            weights, query, backend,
        )?))
    }

    fn transfer_backbone_linears(
        &self,
        text: &Qwen3VLTextWeights,
        vision: &Qwen3VLVisionWeights,
        backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        let mut output = BTreeMap::new();
        for (index, layer) in text.layers.iter().enumerate() {
            let projections = [
                ("query", &layer.wq),
                ("key", &layer.wk),
                ("value", &layer.wv),
                ("output", &layer.wo),
                ("gate", &layer.w_gate),
                ("up", &layer.w_up),
                ("down", &layer.w_down),
            ];
            for (projection, weight) in projections {
                let name = format!("backbone.text.layers.{index}.{projection}");
                output.insert(name.clone(), self.matrix(weight, &name, backend)?);
            }
            // One wide QKV projection per text layer. This also makes
            // `shared_qkv` non-empty, which turns on the fused RMS-norm +
            // E4M3 producer in the executor, so a text layer emits one
            // projection and one normalization kernel instead of three and
            // two. Paired interleaved A/B on MUSA4 (2026-11-18): base
            // 16.8673/16.6839 ms against 16.6234/16.6530 ms with the fused
            // projection — the fused arms are the ones that clear 60 Hz.
            let fused_name = format!("backbone.text.layers.{index}.fused_qkv");
            let fused_weight =
                super::executor::concat_columns_bf16(&[&layer.wq, &layer.wk, &layer.wv])?;
            let fused_scale = self
                .calibration
                .scale(&format!("backbone.text.layers.{index}.query"))?;
            output.insert(
                fused_name,
                Gr00tFp8LinearWeights::matrix(&fused_weight, fused_scale, backend)?,
            );
        }
        let mut insert = |name: String, weight: &Tensor| -> Result<()> {
            output.insert(name.clone(), self.matrix(weight, &name, backend)?);
            Ok(())
        };
        insert(
            "backbone.vision.patch_embed".into(),
            &vision.patch_embed_weight,
        )?;
        for (index, block) in vision.blocks.iter().enumerate() {
            insert(format!("backbone.vision.blocks.{index}.qkv"), &block.qkv_w)?;
            insert(
                format!("backbone.vision.blocks.{index}.output"),
                &block.proj_w,
            )?;
            insert(format!("backbone.vision.blocks.{index}.fc1"), &block.fc1_w)?;
            insert(format!("backbone.vision.blocks.{index}.fc2"), &block.fc2_w)?;
        }
        insert("backbone.vision.merger.fc1".into(), &vision.merger.fc1_w)?;
        insert("backbone.vision.merger.fc2".into(), &vision.merger.fc2_w)?;
        for (index, merger) in vision.deepstack_mergers.iter().enumerate() {
            insert(
                format!("backbone.vision.deepstack.{index}.fc1"),
                &merger.fc1_w,
            )?;
            insert(
                format!("backbone.vision.deepstack.{index}.fc2"),
                &merger.fc2_w,
            )?;
        }
        if output.is_empty() {
            return Err(apxinf_core::Error::Other(
                "GR00T FP8 backbone plan is empty".into(),
            ));
        }
        Ok(output)
    }

    /// The attention epilogue already produced the o-projection's E4M3 input,
    /// so this skips the separate quantize and feeds the GEMM directly.
    fn attention_output_quantized(
        output: &Self::Dense,
        quantized: &Tensor,
        bias: Option<&Tensor>,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let projected = match bias {
            Some(bias) => output.forward_reusable_quantized_bias(quantized, bias, backend)?,
            None => Some(output.forward_reusable_quantized(quantized, backend)?),
        };
        Ok(projected)
    }
}

pub(crate) type Gr00tFp8Executor = Gr00tExecutor<Gr00tFp8Execution>;

pub(crate) fn build_fp8(
    checkpoint: &Path,
    backbone: &Path,
    calibration: &Path,
    config: Gr00tConfig,
    backbone_config: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tFp8Executor> {
    let execution =
        Gr00tFp8Execution::from_file(checkpoint, backbone, calibration, &config, &backbone_config)?;
    Gr00tExecutor::from_backend(config, backbone_config, weights, execution, backend)
}
