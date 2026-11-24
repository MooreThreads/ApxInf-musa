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
/// The reader holds headers only until a tensor is taken, and `model.`-prefixed
/// exports are looked up with the same prefix rule the map path applies.
pub(crate) struct CheckpointSource {
    reader: CheckpointReader,
    strip_model_prefix: bool,
}

impl CheckpointSource {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let reader = CheckpointReader::open(path)
            .map_err(|error| Error::Other(format!("load PI0.5 SafeTensors: {error}")))?;
        let strip_model_prefix = super::host::needs_model_prefix_strip(reader.keys());
        Ok(Self {
            reader,
            strip_model_prefix,
        })
    }

    fn key<'a>(&self, name: &'a str) -> std::borrow::Cow<'a, str> {
        if self.strip_model_prefix {
            std::borrow::Cow::Owned(format!("model.{name}"))
        } else {
            std::borrow::Cow::Borrowed(name)
        }
    }
}

impl WeightSource for CheckpointSource {
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        let key = self.key(name).into_owned();
        self.reader.take(&key).map_err(Error::Other)
    }
}
