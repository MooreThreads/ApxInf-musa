//! The host-side source a streaming weight builder consumes.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{Error, Result, Tensor};
use apxinf_loader::safetensors::CheckpointReader;

/// One checkpoint, consumed by tensor name.
///
/// `Pi05Weights::from_map` and the streaming int8 builder both take tensors
/// through this seam, so extraction, folding, shape validation and the error
/// text are shared. The streaming builder validates each tensor when it takes
/// it, so with several bad tensors the first reported error can differ from
/// the map path's order.
pub(crate) trait WeightSource {
    /// Remove `name`; `Ok(None)` when absent or already taken.
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>>;

    fn take(&mut self, name: &str) -> Result<Tensor> {
        self.take_opt(name)?
            .ok_or_else(|| Error::Other(format!("missing PI0.5 weight `{name}`")))
    }

    fn take_any(&mut self, names: &[String]) -> Result<Tensor> {
        for name in names {
            if let Some(tensor) = self.take_opt(name)? {
                return Ok(tensor);
            }
        }
        Err(Error::Other(format!(
            "missing PI0.5 weight (accepted aliases: {})",
            names.join(", ")
        )))
    }
}

impl WeightSource for HashMap<String, Tensor> {
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        Ok(self.remove(name))
    }
}

/// A checkpoint read one tensor at a time from disk.
///
/// The reader holds headers only until a tensor is taken. Keys an export
/// wrapped in `model.` are resolved under the name the map path would leave,
/// so canonical, fully wrapped and mixed checkpoints all agree.
pub(crate) struct CheckpointSource {
    reader: CheckpointReader,
    /// Canonical tensor name -> the spelling the file uses, for keys the
    /// export wrapped in `model.`. Empty otherwise.
    aliases: HashMap<String, String>,
}

impl CheckpointSource {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let reader = CheckpointReader::open(path)
            .map_err(|error| Error::Other(format!("load PI0.5 SafeTensors: {error}")))?;
        let aliases = if super::host::needs_model_prefix_strip(reader.keys()) {
            model_prefix_aliases(reader.keys())
        } else {
            HashMap::new()
        };
        Ok(Self { reader, aliases })
    }
}

/// The map path's rule, applied to a header: a `model.`-wrapped key is
/// consulted under the name `normalize_lerobot_prefix` would leave.
fn model_prefix_aliases<'a>(keys: impl Iterator<Item = &'a str>) -> HashMap<String, String> {
    keys.filter_map(|key| {
        let canonical = super::host::canonical_lerobot_key(key);
        (canonical != key).then(|| (canonical.to_owned(), key.to_owned()))
    })
    .collect()
}

impl WeightSource for CheckpointSource {
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        let physical = match self.aliases.get(name) {
            Some(physical) => physical.as_str(),
            None => name,
        };
        self.reader.take(physical).map_err(|error| {
            // Name the tensor the builder asked for first: the file's spelling
            // carries `model.` only because of how the export was wrapped.
            Error::Other(format!("load PI0.5 SafeTensors: `{name}`: {error}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn model_prefixed_keys_are_rewritten_with_the_map_paths_rule() {
        let aliases = model_prefix_aliases(
            [
                "model.paligemma_with_expert.paligemma.lm_head.weight",
                "action_in_proj.weight",
                "model.action_in_proj.bias",
            ]
            .into_iter(),
        );
        assert_eq!(aliases.len(), 2);
        assert_eq!(
            aliases["paligemma_with_expert.paligemma.lm_head.weight"],
            "model.paligemma_with_expert.paligemma.lm_head.weight"
        );
        assert_eq!(aliases["action_in_proj.bias"], "model.action_in_proj.bias");
        assert!(!aliases.contains_key("action_in_proj.weight"));
    }

    #[test]
    fn a_mixed_prefix_checkpoint_resolves_each_key_by_its_own_spelling() {
        let dir = temp_checkpoint("mixed");
        let path = write_checkpoint(
            &dir.0,
            &[
                (
                    "model.paligemma_with_expert.paligemma.lm_head.weight",
                    &[1.0],
                ),
                ("action_in_proj.weight", &[2.0]),
                ("model.action_in_proj.bias", &[3.0]),
            ],
        );
        let mut source = CheckpointSource::open(&path).expect("open mixed checkpoint");
        for name in [
            "paligemma_with_expert.paligemma.lm_head.weight",
            "action_in_proj.weight",
            "action_in_proj.bias",
        ] {
            assert!(source.take_opt(name).expect("take").is_some(), "{name}");
        }
        assert!(source
            .take_opt("action_in_proj.bias")
            .expect("take twice")
            .is_none());
    }

    #[test]
    fn a_canonical_checkpoint_keeps_a_stray_wrapped_key_unreachable() {
        let dir = temp_checkpoint("canonical");
        let path = write_checkpoint(
            &dir.0,
            &[
                ("paligemma_with_expert.paligemma.lm_head.weight", &[1.0]),
                ("model.action_in_proj.weight", &[2.0]),
            ],
        );
        let mut source = CheckpointSource::open(&path).expect("open canonical checkpoint");
        assert!(source
            .take_opt("paligemma_with_expert.paligemma.lm_head.weight")
            .expect("take canonical")
            .is_some());
        assert!(
            source
                .take_opt("action_in_proj.weight")
                .expect("take stray wrapped")
                .is_none(),
            "a canonical export must not start resolving wrapped keys"
        );
    }

    #[test]
    fn a_streaming_read_error_names_the_canonical_tensor_and_the_load_context() {
        let dir = temp_checkpoint("truncated");
        let path = write_checkpoint(
            &dir.0,
            &[(
                "model.paligemma_with_expert.paligemma.lm_head.weight",
                &[1.0],
            )],
        );
        let mut source = CheckpointSource::open(&path).expect("open");
        let header_len = {
            let bytes = std::fs::read(&path).expect("read header");
            u64::from_le_bytes(bytes[..8].try_into().expect("header length"))
        };
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for truncation")
            .set_len(8 + header_len)
            .expect("truncate tensor data");
        let error = source
            .take_opt("paligemma_with_expert.paligemma.lm_head.weight")
            .expect_err("truncated tensor must fail");
        let message = error.to_string();
        assert!(message.contains("load PI0.5 SafeTensors: "), "{message}");
        assert!(
            message.contains("paligemma_with_expert.paligemma.lm_head.weight"),
            "{message}"
        );
    }

    struct TempCheckpoint(PathBuf);

    impl Drop for TempCheckpoint {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_checkpoint(case: &str) -> TempCheckpoint {
        let dir =
            std::env::temp_dir().join(format!("apxinf-pi05-source-{}-{case}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        TempCheckpoint(dir)
    }

    fn write_checkpoint(dir: &Path, tensors: &[(&str, &[f32])]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, values) in tensors {
            let start = data.len();
            for value in *values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [values.len()],
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
        let mut bytes = u64::try_from(header.len())
            .expect("header length fits u64")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, bytes).expect("write checkpoint");
        path
    }
}
