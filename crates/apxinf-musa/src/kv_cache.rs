use crate::{MusaBuffer, MusaContext};
use apxinf_core::{DType, Error, KvCache, Result, Tensor};
use std::any::Any;

pub(crate) struct LayerCache {
    pub(crate) k: MusaBuffer,
    pub(crate) v: MusaBuffer,
}
pub(crate) struct MusaKvCache {
    ctx: std::sync::Arc<MusaContext>,
    pub(crate) layers: Vec<Option<LayerCache>>,
    n_kv_heads: usize,
    head_dim: usize,
    pub(crate) max_seq: usize,
    seq_len: usize,
}
impl MusaKvCache {
    pub(crate) fn new(
        ctx: &std::sync::Arc<MusaContext>,
        n: usize,
        kh: usize,
        d: usize,
        max: usize,
    ) -> Result<Self> {
        Ok(Self {
            ctx: ctx.clone(),
            layers: (0..n).map(|_| None).collect(),
            n_kv_heads: kh,
            head_dim: d,
            max_seq: max,
            seq_len: 0,
        })
    }
}
impl KvCache for MusaKvCache {
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor, n: usize) -> Result<()> {
        if layer >= self.layers.len() || n == 0 || n > self.max_seq.saturating_sub(self.seq_len) {
            return Err(Error::Other("MUSA KV append exceeds cache".into()));
        }
        if k.dtype() != DType::BF16
            || v.dtype() != DType::BF16
            || k.shape().dims() != [n, self.n_kv_heads, self.head_dim]
            || v.shape() != k.shape()
        {
            return Err(Error::Other("MUSA KV append shape/dtype mismatch".into()));
        }
        let bytes = self.max_seq * self.n_kv_heads * self.head_dim * 2;
        if self.layers[layer].is_none() {
            self.layers[layer] = Some(LayerCache {
                k: MusaBuffer::alloc_on(&self.ctx, bytes).map_err(Error::Other)?,
                v: MusaBuffer::alloc_on(&self.ctx, bytes).map_err(Error::Other)?,
            });
        }
        let cache = self.layers[layer].as_ref().unwrap();
        let kb = MusaBuffer::from_tensor(k).map_err(Error::Other)?;
        let vb = MusaBuffer::from_tensor(v).map_err(Error::Other)?;
        let off = self.seq_len * self.n_kv_heads * self.head_dim * 2;
        let kd = cache
            .k
            .view(off, n * self.n_kv_heads * self.head_dim * 2)
            .map_err(Error::Other)?;
        let vd = cache
            .v
            .view(off, n * self.n_kv_heads * self.head_dim * 2)
            .map_err(Error::Other)?;
        kd.copy_from_device_async(
            &kb,
            n * self.n_kv_heads * self.head_dim * 2,
            self.ctx.stream(),
        )
        .map_err(Error::Other)?;
        vd.copy_from_device_async(
            &vb,
            n * self.n_kv_heads * self.head_dim * 2,
            self.ctx.stream(),
        )
        .map_err(Error::Other)?;
        Ok(())
    }
    fn advance(&mut self, n: usize) {
        self.seq_len += n
    }
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn clear(&mut self) -> Result<()> {
        self.seq_len = 0;
        self.layers.iter_mut().for_each(|x| *x = None);
        Ok(())
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
