use super::super::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use apxinf_core::{Error, Result, Tensor};
use std::collections::HashMap;

/// Host-side checkpoint representation consumed by the native executor.
/// Device copies are created by the executor so this type does not depend on
/// the generic Backend trait.
pub struct NativeQwenWeights {
    pub config: Qwen3VLConfig,
    pub text: Qwen3VLTextWeights,
    pub vision: Qwen3VLVisionWeights,
    pub lm_head: Tensor,
}

impl NativeQwenWeights {
    pub fn from_map(config: Qwen3VLConfig, tensors: HashMap<String, Tensor>) -> Result<Self> {
        let text = Qwen3VLTextWeights::from_map(&config, tensors.clone())?;
        let vision = Qwen3VLVisionWeights::from_map(&config, tensors)?;
        let lm_head = match &text.lm_head {
            Some(t) => t.clone(),
            None => transpose_2d(&text.token_embedding)?,
        };
        Ok(Self {
            config,
            text,
            vision,
            lm_head,
        })
    }
}

impl NativeQwenWeights {
    pub fn to_device(self, runtime: &super::runtime::MusaQwenRuntime) -> Result<Self> {
        fn copy(runtime: &super::runtime::MusaQwenRuntime, t: &Tensor) -> Result<Tensor> {
            runtime.to_device(t)
        }
        let mut text = self.text;
        text.token_embedding = copy(runtime, &text.token_embedding)?;
        text.output_norm_weight = copy(runtime, &text.output_norm_weight)?;
        let lm_head = copy(runtime, &self.lm_head)?;
        if text.lm_head.is_some() {
            text.lm_head = Some(lm_head.clone());
        }
        for layer in &mut text.layers {
            layer.attn_norm_weight = copy(runtime, &layer.attn_norm_weight)?;
            layer.wq = copy(runtime, &layer.wq)?;
            layer.wk = copy(runtime, &layer.wk)?;
            layer.wv = copy(runtime, &layer.wv)?;
            layer.wo = copy(runtime, &layer.wo)?;
            layer.q_norm_weight = copy(runtime, &layer.q_norm_weight)?;
            layer.k_norm_weight = copy(runtime, &layer.k_norm_weight)?;
            layer.ffn_norm_weight = copy(runtime, &layer.ffn_norm_weight)?;
            layer.w_gate = copy(runtime, &layer.w_gate)?;
            layer.w_up = copy(runtime, &layer.w_up)?;
            layer.w_down = copy(runtime, &layer.w_down)?;
        }
        let mut vision = self.vision;
        vision.patch_embed_weight = copy(runtime, &vision.patch_embed_weight)?;
        vision.patch_embed_bias = copy(runtime, &vision.patch_embed_bias)?;
        vision.pos_embed = copy(runtime, &vision.pos_embed)?;
        for block in &mut vision.blocks {
            block.norm1_w = copy(runtime, &block.norm1_w)?;
            block.norm1_b = copy(runtime, &block.norm1_b)?;
            block.qkv_w = copy(runtime, &block.qkv_w)?;
            block.qkv_b = copy(runtime, &block.qkv_b)?;
            block.proj_w = copy(runtime, &block.proj_w)?;
            block.proj_b = copy(runtime, &block.proj_b)?;
            block.norm2_w = copy(runtime, &block.norm2_w)?;
            block.norm2_b = copy(runtime, &block.norm2_b)?;
            block.fc1_w = copy(runtime, &block.fc1_w)?;
            block.fc1_b = copy(runtime, &block.fc1_b)?;
            block.fc2_w = copy(runtime, &block.fc2_w)?;
            block.fc2_b = copy(runtime, &block.fc2_b)?;
        }
        for merger in std::iter::once(&mut vision.merger).chain(vision.deepstack_mergers.iter_mut())
        {
            merger.norm_w = copy(runtime, &merger.norm_w)?;
            merger.norm_b = copy(runtime, &merger.norm_b)?;
            merger.fc1_w = copy(runtime, &merger.fc1_w)?;
            merger.fc1_b = copy(runtime, &merger.fc1_b)?;
            merger.fc2_w = copy(runtime, &merger.fc2_w)?;
            merger.fc2_b = copy(runtime, &merger.fc2_b)?;
        }
        Ok(Self {
            config: self.config,
            text,
            vision,
            lm_head,
        })
    }
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "transpose_2d expected 2D tensor, got {}D",
            dims.len()
        )));
    }
    let [rows, cols] = [dims[0], dims[1]];
    match tensor.dtype() {
        apxinf_core::DType::F32 => {
            let data = tensor.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)
        }
        apxinf_core::DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)
        }
        dtype => Err(Error::Other(format!(
            "Qwen3-VL weight transpose does not support {dtype}"
        ))),
    }
}
