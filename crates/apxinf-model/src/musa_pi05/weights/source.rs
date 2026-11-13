//! The host-side source a streaming weight builder consumes.

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};

/// One checkpoint, consumed by tensor name.
///
/// `Pi05Weights::from_map` and the streaming int8 builder both take tensors
/// through this seam, so extraction, folding and the error text are written
/// once and the two paths cannot drift.
pub(crate) trait WeightSource {
    /// Remove `name`; `Ok(None)` when absent or already taken.
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>>;
    /// Present in the checkpoint, taken or not.
    fn contains(&self, name: &str) -> bool;

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

    fn contains(&self, name: &str) -> bool {
        self.contains_key(name)
    }
}

/// A checkpoint map whose `model.` prefix has already been normalized away.
pub(crate) struct MapSource {
    tensors: HashMap<String, Tensor>,
}

impl MapSource {
    pub(crate) fn new(tensors: HashMap<String, Tensor>) -> Self {
        let mut tensors = tensors;
        super::host::normalize_lerobot_prefix(&mut tensors);
        Self { tensors }
    }

    /// Keys the loader did not consume; the real checkpoint leaves the unused
    /// action-expert `lm_head` (and any losing tied alias) behind.
    #[allow(dead_code)]
    pub(crate) fn remaining_keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

impl WeightSource for MapSource {
    fn take_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        Ok(self.tensors.remove(name))
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
