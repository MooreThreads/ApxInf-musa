//! Strict host-side checkpoint mapping for the converted Fast-WAM assets.
//!
//! The trusted offline converter is the only component that reads the
//! official PyTorch pickle. This loader accepts SafeTensors only, requires the
//! conversion metadata and consumes every tensor exactly once.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};

use super::{FastWamBackend, FastWamConfig};

const CONVERSION_SCHEMA: &str = "apxinf.fastwam.converted-checkpoint.v1";
const UPSTREAM_REVISION: &str = "7faa71108368fbb3b6885649f112af607427a2d4";
const MODEL_TENSOR_COUNT: usize = 1651;
const VAE_TENSOR_COUNT: usize = 196;

#[derive(Debug)]
pub(crate) struct LinearWeights {
    /// Matmul-ready `[input, output]` matrix.
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
}

#[derive(Debug)]
pub(crate) struct EmbeddingWeights {
    pub(crate) input: LinearWeights,
    pub(crate) output: LinearWeights,
}

#[derive(Debug)]
pub(crate) struct AttentionWeights {
    pub(crate) query: LinearWeights,
    pub(crate) key: LinearWeights,
    pub(crate) value: LinearWeights,
    pub(crate) output: LinearWeights,
    pub(crate) query_norm: Tensor,
    pub(crate) key_norm: Tensor,
}

#[derive(Debug)]
pub(crate) struct LayerNormWeights {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
}

#[derive(Debug)]
pub(crate) struct DiTBlockWeights {
    pub(crate) modulation: Tensor,
    pub(crate) self_attention: AttentionWeights,
    pub(crate) cross_attention: AttentionWeights,
    pub(crate) cross_attention_input_norm: LayerNormWeights,
    pub(crate) feed_forward_input: LinearWeights,
    pub(crate) feed_forward_output: LinearWeights,
}

#[derive(Debug)]
pub(crate) struct ExpertWeights {
    pub(crate) text_embedding: EmbeddingWeights,
    pub(crate) time_embedding: EmbeddingWeights,
    pub(crate) time_projection: LinearWeights,
    pub(crate) blocks: Vec<DiTBlockWeights>,
}

#[derive(Debug)]
pub(crate) struct VideoWeights {
    pub(crate) patch_embedding_weight: Tensor,
    pub(crate) patch_embedding_bias: Tensor,
    pub(crate) expert: ExpertWeights,
    pub(crate) head_modulation: Tensor,
    pub(crate) head: LinearWeights,
}

#[derive(Debug)]
pub(crate) struct ActionWeights {
    pub(crate) action_encoder: LinearWeights,
    pub(crate) expert: ExpertWeights,
    pub(crate) head: LinearWeights,
}

#[derive(Debug)]
pub(crate) struct FastWamWeights {
    pub(crate) video: VideoWeights,
    pub(crate) action: ActionWeights,
    pub(crate) proprio_encoder: LinearWeights,
}

/// Wan VAE is kept as a strict named map until its encoder/decoder blocks are
/// assembled. Names use the official converter's `model.` prefix.
#[derive(Debug)]
pub(crate) struct WanVaeWeights {
    pub(super) tensors: HashMap<String, Tensor>,
}

impl FastWamWeights {
    pub(crate) fn from_safetensors(config: &FastWamConfig, path: &Path) -> Result<Self> {
        config.validate()?;
        let (mut tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load converted Fast-WAM weights: {error}")))?;
        validate_metadata(&metadata, "Fast-WAM model")?;
        if tensors.len() != MODEL_TENSOR_COUNT {
            return Err(Error::Other(format!(
                "converted Fast-WAM model must contain {MODEL_TENSOR_COUNT} tensors, got {}",
                tensors.len()
            )));
        }

        let video_prefix = "mot.mixtures.video";
        let action_prefix = "mot.mixtures.action";
        let video = VideoWeights {
            patch_embedding_weight: take_bf16(
                &mut tensors,
                &format!("{video_prefix}.patch_embedding.weight"),
                &[
                    config.video_hidden_dim,
                    config.latent_channels,
                    config.patch_size[0],
                    config.patch_size[1],
                    config.patch_size[2],
                ],
            )?,
            patch_embedding_bias: take_bf16(
                &mut tensors,
                &format!("{video_prefix}.patch_embedding.bias"),
                &[config.video_hidden_dim],
            )?,
            expert: load_expert(
                &mut tensors,
                video_prefix,
                config.video_hidden_dim,
                config.video_ffn_dim,
                config,
            )?,
            head_modulation: take_bf16(
                &mut tensors,
                &format!("{video_prefix}.head.modulation"),
                &[1, 2, config.video_hidden_dim],
            )?,
            head: take_linear(
                &mut tensors,
                &format!("{video_prefix}.head.head"),
                config.video_hidden_dim,
                config.latent_channels * config.patch_size.iter().product::<usize>(),
            )?,
        };
        let action = ActionWeights {
            action_encoder: take_linear(
                &mut tensors,
                &format!("{action_prefix}.action_encoder"),
                config.action_dim,
                config.action_hidden_dim,
            )?,
            expert: load_expert(
                &mut tensors,
                action_prefix,
                config.action_hidden_dim,
                config.action_ffn_dim,
                config,
            )?,
            head: take_linear(
                &mut tensors,
                &format!("{action_prefix}.head"),
                config.action_hidden_dim,
                config.action_dim,
            )?,
        };
        let proprio_encoder = take_linear(
            &mut tensors,
            "proprio_encoder",
            config.proprio_dim,
            config.text_dim,
        )?;
        reject_unconsumed(tensors, "Fast-WAM model")?;
        Ok(Self {
            video,
            action,
            proprio_encoder,
        })
    }

    /// Upload the already validated canonical tensors once. The same concrete
    /// tree is retained so checkpoint names/shapes have one owner; after this
    /// call every tensor in the tree is MUSA-resident.
    pub(crate) fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            video: self.video.into_device(backend)?,
            action: self.action.into_device(backend)?,
            proprio_encoder: self.proprio_encoder.into_device(backend)?,
        })
    }
}

impl WanVaeWeights {
    pub(crate) fn from_safetensors(path: &Path) -> Result<Self> {
        let (tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load converted Wan VAE weights: {error}")))?;
        validate_metadata(&metadata, "Wan VAE")?;
        if tensors.len() != VAE_TENSOR_COUNT {
            return Err(Error::Other(format!(
                "converted Wan VAE must contain {VAE_TENSOR_COUNT} tensors, got {}",
                tensors.len()
            )));
        }
        for (name, tensor) in &tensors {
            if !name.starts_with("model.") || tensor.dtype() != DType::BF16 {
                return Err(Error::Other(format!(
                    "converted Wan VAE tensor {name} must use model.* naming and BF16, got {}",
                    tensor.dtype()
                )));
            }
        }
        expect_bf16_shape(
            tensors.get("model.encoder.conv1.weight").ok_or_else(|| {
                Error::Other("Wan VAE is missing model.encoder.conv1.weight".into())
            })?,
            &[160, 12, 3, 3, 3],
            "model.encoder.conv1.weight",
        )?;
        expect_bf16_shape(
            tensors
                .get("model.encoder.middle.1.to_qkv.weight")
                .ok_or_else(|| {
                    Error::Other("Wan VAE is missing model.encoder.middle.1.to_qkv.weight".into())
                })?,
            &[1920, 640, 1, 1],
            "model.encoder.middle.1.to_qkv.weight",
        )?;
        if !tensors
            .keys()
            .any(|name| name.starts_with("model.decoder."))
        {
            return Err(Error::Other(
                "converted Wan VAE contains no decoder tensors".into(),
            ));
        }
        Ok(Self { tensors })
    }

    pub(crate) fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        let tensors = self
            .tensors
            .into_iter()
            .map(|(name, tensor)| backend.to_device(&tensor).map(|tensor| (name, tensor)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { tensors })
    }
}

impl LinearWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            weight: backend.to_device(&self.weight)?,
            bias: backend.to_device(&self.bias)?,
        })
    }
}

impl EmbeddingWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            input: self.input.into_device(backend)?,
            output: self.output.into_device(backend)?,
        })
    }
}

impl AttentionWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            query: self.query.into_device(backend)?,
            key: self.key.into_device(backend)?,
            value: self.value.into_device(backend)?,
            output: self.output.into_device(backend)?,
            query_norm: backend.to_device(&self.query_norm)?,
            key_norm: backend.to_device(&self.key_norm)?,
        })
    }
}

impl LayerNormWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            weight: backend.to_device(&self.weight)?,
            bias: backend.to_device(&self.bias)?,
        })
    }
}

impl DiTBlockWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            modulation: backend.to_device(&self.modulation)?,
            self_attention: self.self_attention.into_device(backend)?,
            cross_attention: self.cross_attention.into_device(backend)?,
            cross_attention_input_norm: self
                .cross_attention_input_norm
                .into_device(backend)?,
            feed_forward_input: self.feed_forward_input.into_device(backend)?,
            feed_forward_output: self.feed_forward_output.into_device(backend)?,
        })
    }
}

impl ExpertWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            text_embedding: self.text_embedding.into_device(backend)?,
            time_embedding: self.time_embedding.into_device(backend)?,
            time_projection: self.time_projection.into_device(backend)?,
            blocks: self
                .blocks
                .into_iter()
                .map(|block| block.into_device(backend))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl VideoWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            patch_embedding_weight: backend.to_device(&self.patch_embedding_weight)?,
            patch_embedding_bias: backend.to_device(&self.patch_embedding_bias)?,
            expert: self.expert.into_device(backend)?,
            head_modulation: backend.to_device(&self.head_modulation)?,
            head: self.head.into_device(backend)?,
        })
    }
}

impl ActionWeights {
    fn into_device(self, backend: &FastWamBackend) -> Result<Self> {
        Ok(Self {
            action_encoder: self.action_encoder.into_device(backend)?,
            expert: self.expert.into_device(backend)?,
            head: self.head.into_device(backend)?,
        })
    }
}

fn load_expert(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    hidden: usize,
    feed_forward: usize,
    config: &FastWamConfig,
) -> Result<ExpertWeights> {
    let text_embedding = take_embedding(
        tensors,
        &format!("{prefix}.text_embedding"),
        config.text_dim,
        hidden,
    )?;
    let time_embedding = take_embedding(
        tensors,
        &format!("{prefix}.time_embedding"),
        config.freq_dim,
        hidden,
    )?;
    let time_projection = take_linear(
        tensors,
        &format!("{prefix}.time_projection.1"),
        hidden,
        6 * hidden,
    )?;
    let mut blocks = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let block = format!("{prefix}.blocks.{layer}");
        blocks.push(DiTBlockWeights {
            modulation: take_bf16(tensors, &format!("{block}.modulation"), &[1, 6, hidden])?,
            self_attention: take_attention(tensors, &format!("{block}.self_attn"), hidden, config)?,
            cross_attention: take_attention(
                tensors,
                &format!("{block}.cross_attn"),
                hidden,
                config,
            )?,
            cross_attention_input_norm: LayerNormWeights {
                weight: take_bf16(tensors, &format!("{block}.norm3.weight"), &[hidden])?,
                bias: take_bf16(tensors, &format!("{block}.norm3.bias"), &[hidden])?,
            },
            feed_forward_input: take_linear(
                tensors,
                &format!("{block}.ffn.0"),
                hidden,
                feed_forward,
            )?,
            feed_forward_output: take_linear(
                tensors,
                &format!("{block}.ffn.2"),
                feed_forward,
                hidden,
            )?,
        });
    }
    Ok(ExpertWeights {
        text_embedding,
        time_embedding,
        time_projection,
        blocks,
    })
}

fn take_embedding(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    input: usize,
    hidden: usize,
) -> Result<EmbeddingWeights> {
    Ok(EmbeddingWeights {
        input: take_linear(tensors, &format!("{prefix}.0"), input, hidden)?,
        output: take_linear(tensors, &format!("{prefix}.2"), hidden, hidden)?,
    })
}

fn take_attention(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    hidden: usize,
    config: &FastWamConfig,
) -> Result<AttentionWeights> {
    let attention_width = config.num_heads * config.head_dim;
    Ok(AttentionWeights {
        query: take_linear(tensors, &format!("{prefix}.q"), hidden, attention_width)?,
        key: take_linear(tensors, &format!("{prefix}.k"), hidden, attention_width)?,
        value: take_linear(tensors, &format!("{prefix}.v"), hidden, attention_width)?,
        output: take_linear(tensors, &format!("{prefix}.o"), attention_width, hidden)?,
        query_norm: take_bf16(
            tensors,
            &format!("{prefix}.norm_q.weight"),
            &[attention_width],
        )?,
        key_norm: take_bf16(
            tensors,
            &format!("{prefix}.norm_k.weight"),
            &[attention_width],
        )?,
    })
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<LinearWeights> {
    Ok(LinearWeights {
        weight: take_bf16(tensors, &format!("{prefix}.weight"), &[input, output])?,
        bias: take_bf16(tensors, &format!("{prefix}.bias"), &[output])?,
    })
}

fn take_bf16(tensors: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("converted Fast-WAM checkpoint is missing {name}")))?;
    expect_bf16_shape(&tensor, shape, name)?;
    Ok(tensor)
}

fn expect_bf16_shape(tensor: &Tensor, shape: &[usize], name: &str) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "converted Fast-WAM tensor {name} must be BF16 {shape:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn validate_metadata(metadata: &HashMap<String, String>, asset: &str) -> Result<()> {
    for (key, expected) in [
        ("format", CONVERSION_SCHEMA),
        ("upstream_revision", UPSTREAM_REVISION),
        ("linear_layout", "input_output"),
    ] {
        if metadata.get(key).map(String::as_str) != Some(expected) {
            return Err(Error::Other(format!(
                "converted {asset} metadata {key} must be {expected:?}, got {:?}",
                metadata.get(key)
            )));
        }
    }
    Ok(())
}

fn reject_unconsumed(tensors: HashMap<String, Tensor>, asset: &str) -> Result<()> {
    if tensors.is_empty() {
        return Ok(());
    }
    let mut names = tensors.into_keys().collect::<Vec<_>>();
    names.sort();
    Err(Error::Other(format!(
        "unconsumed converted {asset} tensors: {}",
        names.into_iter().take(16).collect::<Vec<_>>().join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_linear_layout_shape() {
        let tensor = Tensor::from_bf16(vec![7, 4], &[half::bf16::ZERO; 28]).unwrap();
        let mut tensors = HashMap::from([("layer.weight".into(), tensor)]);
        let error = take_bf16(&mut tensors, "layer.weight", &[4, 7]).unwrap_err();
        assert!(error.to_string().contains("[4, 7]"));
    }

    #[test]
    fn conversion_metadata_is_fail_closed() {
        let mut metadata = HashMap::new();
        assert!(validate_metadata(&metadata, "test").is_err());
        metadata.insert("format".into(), CONVERSION_SCHEMA.into());
        metadata.insert("upstream_revision".into(), UPSTREAM_REVISION.into());
        metadata.insert("linear_layout".into(), "output_input".into());
        assert!(validate_metadata(&metadata, "test").is_err());
        metadata.insert("linear_layout".into(), "input_output".into());
        validate_metadata(&metadata, "test").unwrap();
    }
}
