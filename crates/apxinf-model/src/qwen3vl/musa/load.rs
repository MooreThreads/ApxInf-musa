use super::weights::NativeQwenWeights;
use apxinf_core::{Error, Result};
use std::path::Path;

pub fn load_weights(model_dir: &Path) -> Result<NativeQwenWeights> {
    let config =
        super::super::Qwen3VLConfig::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(model_dir)
        .map_err(|e| Error::Other(format!("load {}: {e}", model_dir.display())))?;
    NativeQwenWeights::from_map(config, tensors)
}
