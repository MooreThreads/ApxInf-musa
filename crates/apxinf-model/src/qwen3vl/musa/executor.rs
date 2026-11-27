use super::NativeSequenceState;
use super::{load::load_weights, runtime::MusaQwenRuntime, weights::NativeQwenWeights};
use apxinf_core::{Error, Result, Tensor};
use apxinf_musa::ExecutionSession;
use std::path::Path;

pub struct Qwen3VlMusaExecutor {
    pub(crate) runtime: MusaQwenRuntime,
    pub(crate) weights: NativeQwenWeights,
    cache: Vec<Option<super::kv_cache::LayerCache>>,
    next_rope: u32,
    pub(crate) sequence: NativeSequenceState,
    session: Option<ExecutionSession>,
}

impl Qwen3VlMusaExecutor {
    pub fn load(device: usize, model_dir: &Path) -> Result<Self> {
        let runtime = MusaQwenRuntime::new(device)?;
        let weights = load_weights(model_dir)?.to_device(&runtime)?;
        Ok(Self {
            cache: (0..weights.text.layers.len()).map(|_| None).collect(),
            next_rope: 0,
            runtime,
            weights,
            sequence: NativeSequenceState::default(),
            session: None,
        })
    }
    /// Start a new sequence on the same stream, retaining per-layer KV capacity.
    pub fn reset(&mut self) -> Result<()> {
        for layer in self.cache.iter_mut().flatten() {
            layer.rewind();
        }
        self.next_rope = 0;
        self.sequence.reset();
        Ok(())
    }
    /// End the sequence and release KV allocations; model weights and workspace stay loaded.
    pub fn release_kv_cache(&mut self) -> Result<()> {
        self.runtime.synchronize()?;
        self.reset()?;
        self.cache.iter_mut().for_each(|layer| *layer = None);
        Ok(())
    }
    /// Encode preprocessed patches without consuming a language sequence position.
    pub fn encode_image(
        &self,
        pixels: &Tensor,
        grids: &[[u32; 3]],
    ) -> Result<super::NativeVisionOutput> {
        let uploaded;
        let pixels = if pixels.device() == apxinf_core::Device::Cpu {
            uploaded = self.runtime.to_device(pixels)?;
            &uploaded
        } else {
            pixels
        };
        super::vision::forward(&self.runtime, &self.weights, pixels, grids)
    }
    fn capacity(&self) -> usize {
        self.weights.config.text.max_position_embeddings.min(4096)
    }
    fn validate_ids(&self, ids: &[u32]) -> Result<()> {
        if ids.is_empty()
            || ids.len() > self.capacity().saturating_sub(self.sequence.position())
            || ids
                .iter()
                .any(|&t| t as usize >= self.weights.config.text.vocab_size)
        {
            return Err(Error::Other(
                "empty, out-of-vocabulary or over-capacity input".into(),
            ));
        }
        Ok(())
    }
    fn run(
        &mut self,
        ids: &[u32],
        positions: &[u32],
        next: u32,
        image: Option<(&[usize], &super::NativeVisionOutput)>,
    ) -> Result<Tensor> {
        let start = self.sequence.position();
        let cap = self.capacity();
        let session = if start == 0 {
            self.prefill_session(ids.len())?
        } else {
            None
        };
        let mut forward = || {
            super::language::forward(
                self.runtime.context(),
                &self.weights,
                &mut self.cache,
                start,
                cap,
                ids,
                positions,
                image,
            )
        };
        // Commit the sequence only after the result has independent storage.
        let ret = match session {
            Some(session) => session
                .prepare(forward)
                .and_then(|t| self.runtime.stable_copy(&t)),
            None => forward(),
        };
        match ret {
            Ok(t) => {
                super::validate_decode_transition(&mut self.sequence, ids.len(), start)
                    .map_err(Error::Other)?;
                self.next_rope = next;
                Ok(t)
            }
            Err(e) => {
                self.reset()?;
                Err(e)
            }
        }
    }
    fn prefill_session(&mut self, tokens: usize) -> Result<Option<ExecutionSession>> {
        let t = &self.weights.config.text;
        // The bump arena retains every layer's outputs through the traversal.
        // Budget BF16 projections/MLP, head-major attention scratch including
        // FP32 scores, and alignment. Very long prompts remain eager rather
        // than reserving quadratic temporary storage for every layer at once.
        let n = tokens as u128;
        let width = (t.hidden_size as u128).max(t.n_heads as u128 * t.head_dim as u128);
        let per_layer = n * (48 * width + 8 * t.intermediate_size as u128)
            + 8 * n * n * t.n_heads as u128
            + 16384;
        let mib = 1u128 << 20;
        let bytes = (per_layer * t.n_layers as u128 + 2 * t.vocab_size as u128 + 8 * mib)
            .div_ceil(mib)
            * mib;
        if bytes > 2 * 1024 * mib {
            return Ok(None);
        }
        let bytes = bytes as usize;
        if self
            .session
            .as_ref()
            .is_none_or(|s| s.workspace().capacity() < bytes)
        {
            self.session = Some(ExecutionSession::with_capacity(
                bytes,
                self.runtime.context().device_id(),
            )?);
        }
        Ok(self.session.clone())
    }
    /// Number of arena bytes consumed by the most recent session traversal.
    pub fn workspace_bytes_used(&self) -> usize {
        self.session.as_ref().map_or(0, |s| s.workspace().used())
    }
    pub fn prefill_text(&mut self, ids: &[u32]) -> Result<Tensor> {
        if self.sequence.position() != 0 {
            return Err(Error::Other("reset before prefill".into()));
        }
        self.validate_ids(ids)?;
        let pos = (0..ids.len() as u32)
            .flat_map(|i| [i; 3])
            .collect::<Vec<_>>();
        self.run(ids, &pos, ids.len() as u32, None)
    }
    pub fn prefill_image(
        &mut self,
        ids: &[u32],
        pixels: &Tensor,
        grids: &[[u32; 3]],
    ) -> Result<Tensor> {
        if self.sequence.position() != 0 {
            return Err(Error::Other("reset before prefill".into()));
        }
        self.validate_ids(ids)?;
        let &[[1, h, w]] = grids else {
            return Err(Error::Other("expected one still image".into()));
        };
        let m = self.weights.config.vision.spatial_merge_size as u32;
        if m == 0 || h == 0 || w == 0 || h % m != 0 || w % m != 0 {
            return Err(Error::Other("invalid image grid".into()));
        }
        let (h, w) = (h / m, w / m);
        let rows: Vec<_> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.weights.config.image_token_id).then_some(i))
            .collect();
        if rows.is_empty()
            || h.checked_mul(w).map(|n| n as usize) != Some(rows.len())
            || rows.windows(2).any(|r| r[1] != r[0] + 1)
        {
            return Err(Error::Other(
                "image pad rows must form one complete contiguous image".into(),
            ));
        }
        let first = rows[0] as u32;
        let after = rows[0] + rows.len();
        let mut pos = Vec::with_capacity(3 * ids.len());
        for i in 0..first {
            pos.extend([i; 3]);
        }
        for y in 0..h {
            for x in 0..w {
                pos.extend([first, first + y, first + x]);
            }
        }
        let mut next = first + h.max(w);
        for _ in after..ids.len() {
            pos.extend([next; 3]);
            next += 1;
        }
        let vis = self.encode_image(pixels, grids)?;
        self.run(ids, &pos, next, Some((&rows, &vis)))
    }
    pub fn decode_one(&mut self, id: u32) -> Result<Tensor> {
        if self.sequence.position() == 0 {
            return Err(Error::Other("prefill before decode".into()));
        }
        self.validate_ids(&[id])?;
        self.run(&[id], &[self.next_rope; 3], self.next_rope + 1, None)
    }
    pub fn greedy_token(&self, logits: &Tensor) -> Result<u32> {
        self.runtime.require_device(logits)?;
        if logits.shape().dims() != [1, self.weights.config.text.vocab_size] {
            return Err(Error::Other("expected one row of vocabulary logits".into()));
        }
        self.synchronize()?;
        let v = self.runtime.to_cpu(logits)?.to_f32_vec()?;
        if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Other("invalid logits".into()));
        }
        Ok((1..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b }) as u32)
    }
    pub fn synchronize(&self) -> Result<()> {
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires M1000 and a BF16 Qwen3-VL checkpoint"]
    fn reset_retains_kv_without_leaking_previous_request() -> Result<()> {
        let dir = std::env::var("APXINF_QWEN3VL_MODEL_DIR").unwrap();
        let mut model = Qwen3VlMusaExecutor::load(0, Path::new(&dir))?;
        let short = [32, 785, 13];
        let read = |t: Tensor| apxinf_musa::transfers::to_cpu(&t)?.to_f32_vec();
        let first = model.prefill_text(&short)?;
        let step = model.decode_one(17)?;
        model.synchronize()?;
        let expected = (read(first)?, read(step)?);
        model.reset()?;
        assert!(
            model.cache.iter().all(Option::is_some),
            "reset discarded KV storage"
        );
        model.prefill_text(&[42; 32])?;
        model.decode_one(99)?;
        model.reset()?;
        assert!(model.decode_one(17).is_err());
        let first = model.prefill_text(&short)?;
        let step = model.decode_one(17)?;
        model.synchronize()?;
        assert_eq!((read(first)?, read(step)?), expected);
        model.release_kv_cache()?;
        assert!(model.cache.iter().all(Option::is_none));
        assert!(model.decode_one(17).is_err());
        let first = model.prefill_text(&short)?;
        model.synchronize()?;
        assert_eq!(read(first)?, expected.0);
        Ok(())
    }
}
