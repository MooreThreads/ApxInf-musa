use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_musa::{MusaBuffer, MusaContext};
/// Per-layer token-major storage. Only initialized rows can be passed to SDPA.
pub(super) struct LayerCache {
    k: MusaBuffer,
    v: MusaBuffer,
    written: usize,
}
impl LayerCache {
    /// Prior rows stay allocated but no longer belong to the new sequence.
    pub(super) fn rewind(&mut self) {
        self.written = 0;
    }
}

pub(super) fn append(
    slot: &mut Option<LayerCache>,
    ctx: &MusaContext,
    start: usize,
    cap: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let &[n, h, d] = k.shape().dims() else {
        return Err(Error::Other("invalid KV rank".into()));
    };
    let end = start
        .checked_add(n)
        .filter(|&e| e <= cap)
        .ok_or_else(|| Error::Other("KV capacity exceeded".into()))?;
    if n == 0
        || h == 0
        || d == 0
        || slot.as_ref().map_or(0, |l| l.written) != start
        || k.shape() != v.shape()
        || [k, v]
            .iter()
            .any(|t| t.dtype() != DType::BF16 || t.device() != Device::Musa(ctx.device_id()))
    {
        return Err(Error::Other("invalid KV append".into()));
    }
    let stride = h
        .checked_mul(d)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| Error::Other("KV stride overflow".into()))?;
    let size = cap
        .checked_mul(stride)
        .ok_or_else(|| Error::Other("KV size overflow".into()))?;
    if slot.is_none() {
        *slot = Some(LayerCache {
            k: MusaBuffer::alloc_on(ctx, size).map_err(Error::Other)?,
            v: MusaBuffer::alloc_on(ctx, size).map_err(Error::Other)?,
            written: 0,
        });
    }
    let layer = slot.as_mut().unwrap();
    let write = |dst: &MusaBuffer, src: &Tensor| -> Result<Tensor> {
        let src = MusaBuffer::from_tensor(src).map_err(Error::Other)?;
        dst.view(start * stride, n * stride)
            .map_err(Error::Other)?
            .copy_from_device_async(&src, n * stride, ctx.stream())
            .map_err(Error::Other)?;
        dst.view(0, end * stride)
            .and_then(|b| b.as_tensor(Shape::new(vec![end, h, d]), DType::BF16))
            .map_err(Error::Other)
    };
    let pair = (write(&layer.k, k)?, write(&layer.v, v)?);
    layer.written = end;
    Ok(pair)
}
