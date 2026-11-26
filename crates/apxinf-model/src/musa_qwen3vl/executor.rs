use super::NativeSequenceState;
use super::{load::load_weights, runtime::MusaQwenRuntime, weights::NativeQwenWeights};
use apxinf_core::{Error, Result, Tensor};
use std::path::Path;

pub struct Qwen3VlMusaExecutor {
    pub(crate) runtime: MusaQwenRuntime,
    pub(crate) weights: NativeQwenWeights,
    cache: Vec<Option<super::kv_cache::LayerCache>>,
    next_rope: u32,
    pub(crate) sequence: NativeSequenceState,
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
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.cache.iter_mut().for_each(|l| *l = None);
        self.next_rope = 0;
        self.sequence.reset();
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
        let ret = super::language::forward(
            self.runtime.context(),
            &self.weights,
            &mut self.cache,
            start,
            cap,
            ids,
            positions,
            image,
        );
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
