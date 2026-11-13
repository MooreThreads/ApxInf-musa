use crate::ops;
use crate::{MusaBuffer, MusaContext};
use apxinf_core::{
    Backend, DType, Device, Error, Graph, KvCache, NormalGenerator, Result, Shape, Tensor,
    TokenSampler, TokenSamplingSpec,
};
use std::any::Any;

pub struct MusaBackend {
    ctx: std::sync::Arc<MusaContext>,
}
unsafe impl Send for MusaBackend {}
unsafe impl Sync for MusaBackend {}
impl MusaBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            ctx: std::sync::Arc::new(MusaContext::new(device_id)?),
        })
    }
    pub fn context(&self) -> &MusaContext {
        &self.ctx
    }
    pub fn shared_context(&self) -> &std::sync::Arc<MusaContext> {
        &self.ctx
    }
    fn unsupported(op: &str) -> Error {
        Error::Other(format!("MUSA Qwen3-VL backend: {op} is not implemented"))
    }
    fn output(&self, input: &Tensor) -> Result<Tensor> {
        self.ctx
            .allocate_output(input.shape().clone(), input.dtype())
    }
    fn require_bf16(&self, t: &Tensor, op: &str) -> Result<()> {
        if t.device() != Device::Musa(self.ctx.device_id()) || t.dtype() != DType::BF16 {
            return Err(Self::unsupported(op));
        }
        Ok(())
    }
    fn point(&self, a: &Tensor, b: Option<&Tensor>, op: i32, factor: f32) -> Result<Tensor> {
        self.require_bf16(a, "non-BF16 pointwise")?;
        if let Some(b) = b {
            self.require_bf16(b, "non-BF16 pointwise")?;
            if b.shape() != a.shape() {
                return Err(Error::Other("pointwise shape mismatch".into()));
            }
        }
        let out = self.output(a)?;
        let aa = MusaBuffer::from_tensor(a).map_err(Error::Other)?;
        let oo = MusaBuffer::from_tensor(&out).map_err(Error::Other)?;
        let bb = b
            .map(|x| MusaBuffer::from_tensor(x).map_err(Error::Other))
            .transpose()?;
        #[cfg(apxinf_musa_available)]
        unsafe {
            crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_point(
                aa.ptr(),
                bb.as_ref().map_or(std::ptr::null_mut(), |x| x.ptr()),
                oo.ptr(),
                a.shape().numel() as i32,
                op,
                factor,
                self.ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
        }
        #[cfg(not(apxinf_musa_available))]
        return Err(Self::unsupported("portable pointwise"));
        Ok(out)
    }
    fn device_u32(&self, values: &[u32]) -> Result<MusaBuffer> {
        let b = MusaBuffer::alloc_on(&self.ctx, values.len() * 4).map_err(Error::Other)?;
        let bytes: &[u8] = bytemuck::cast_slice(values);
        b.copy_from_host(bytes).map_err(Error::Other)?;
        Ok(b)
    }
}

struct LayerCache {
    k: MusaBuffer,
    v: MusaBuffer,
}
struct MusaKvCache {
    ctx: std::sync::Arc<MusaContext>,
    layers: Vec<Option<LayerCache>>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    seq_len: usize,
}
impl MusaKvCache {
    fn new(
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

impl Backend for MusaBackend {
    fn rms_norm(&self, i: &Tensor, w: &Tensor, e: f32) -> Result<Tensor> {
        let mut o = self.output(i)?;
        ops::rms_norm(&self.ctx, ops::RmsNormArgs::new(i, w, &mut o, e))?;
        Ok(o)
    }
    fn silu(&self, i: &Tensor) -> Result<Tensor> {
        let mut o = self.output(i)?;
        ops::pointwise(
            &self.ctx,
            ops::PointwiseArgs {
                semantic: ops::PointwiseSemantic::BiasActivation,
                input: i,
                secondary: None,
                bias: None,
                out: &mut o,
                activation: ops::PointwiseActivation::Silu,
                dt: 0.0,
                output_scale: 1.0,
            },
        )?;
        Ok(o)
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.point(a, Some(b), 0, 0.0)
    }
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.point(a, Some(b), 1, 0.0)
    }
    fn scale(&self, i: &Tensor, f: f32) -> Result<Tensor> {
        self.point(i, None, 2, f)
    }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        self.require_bf16(a, "matmul")?;
        self.require_bf16(b, "matmul")?;
        let [m, k] = a.shape().dims() else {
            return Err(Error::Other("matmul A rank".into()));
        };
        let [bk, n] = b.shape().dims() else {
            return Err(Error::Other("matmul B rank".into()));
        };
        if k != bk {
            return Err(Error::Other("matmul shape mismatch".into()));
        };
        let mut o = self
            .ctx
            .allocate_output(Shape::new(vec![*m, *n]), DType::BF16)?;
        ops::gemm(&self.ctx, ops::GemmArgs::new(a, b, &mut o))?;
        Ok(o)
    }
    fn rope(&self, i: &Tensor, h: usize, d: usize, theta: f32, pos: u32) -> Result<Tensor> {
        self.rope_impl(i, h, d, theta, &[pos], 1, 0, 0)
    }
    fn rope_mrope(
        &self,
        i: &Tensor,
        h: usize,
        d: usize,
        theta: f32,
        sec: [usize; 3],
        p: &[u32],
    ) -> Result<Tensor> {
        self.rope_impl(i, h, d, theta, p, 3, sec[1], sec[2])
    }
    fn layer_norm(&self, i: &Tensor, w: &Tensor, b: &Tensor, e: f32) -> Result<Tensor> {
        let mut o = self.output(i)?;
        ops::layer_norm(&self.ctx, ops::LayerNormArgs::new(i, w, b, &mut o, e))?;
        Ok(o)
    }
    fn gelu_tanh(&self, i: &Tensor) -> Result<Tensor> {
        let mut o = self.output(i)?;
        ops::pointwise(
            &self.ctx,
            ops::PointwiseArgs {
                semantic: ops::PointwiseSemantic::BiasActivation,
                input: i,
                secondary: None,
                bias: None,
                out: &mut o,
                activation: ops::PointwiseActivation::Gelu,
                dt: 0.0,
                output_scale: 1.0,
            },
        )?;
        Ok(o)
    }
    fn add_bias(&self, i: &Tensor, b: &Tensor) -> Result<Tensor> {
        let mut o = self.output(i)?;
        ops::pointwise(
            &self.ctx,
            ops::PointwiseArgs {
                semantic: ops::PointwiseSemantic::BiasActivation,
                input: i,
                secondary: None,
                bias: Some(b),
                out: &mut o,
                activation: ops::PointwiseActivation::None,
                dt: 0.0,
                output_scale: 1.0,
            },
        )?;
        Ok(o)
    }
    fn rope_vision_2d(
        &self,
        i: &Tensor,
        h: usize,
        d: usize,
        theta: f32,
        p: &[u32],
    ) -> Result<Tensor> {
        self.rope_impl(i, h, d, theta, p, 2, 0, 0)
    }
    fn concat_2d(&self, ts: &[&Tensor]) -> Result<Tensor> {
        if ts.is_empty() {
            return Err(Error::Other("concat_2d empty".into()));
        };
        let rows = ts[0].shape().dims()[0];
        let cols = ts.iter().map(|t| t.shape().dims()[1]).sum();
        let mut out = self
            .ctx
            .allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
        let mut off = 0;
        for t in ts {
            let bytes = rows * t.shape().dims()[1] * 2;
            let src = MusaBuffer::from_tensor(t).map_err(Error::Other)?;
            let dst = MusaBuffer::from_tensor(&out)
                .map_err(Error::Other)?
                .view(off * rows * 2, bytes)
                .map_err(Error::Other)?;
            dst.copy_from_device_async(&src, bytes, self.ctx.stream())
                .map_err(Error::Other)?;
            off += t.shape().dims()[1];
        }
        Ok(out)
    }
    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq: usize,
        h: usize,
        d: usize,
    ) -> Result<Tensor> {
        let q3 = q.reshape(vec![seq, h, d])?;
        let k3 = k.reshape(vec![seq, h, d])?;
        let v3 = v.reshape(vec![seq, h, d])?;
        let mut o = self
            .ctx
            .allocate_output(Shape::new(vec![seq, h, d]), DType::BF16)?;
        self.attention_call(&q3, &k3, &v3, &mut o, seq, seq, h, h, d, false)?;
        o.reshape(vec![seq, h * d]).map_err(Into::into)
    }
    fn embedding(&self, t: &Tensor, ids: &[u32]) -> Result<Tensor> {
        self.require_bf16(t, "embedding")?;
        let [v, d] = t.shape().dims() else {
            return Err(Error::Other("embedding table rank".into()));
        };
        if ids.iter().any(|&x| x as usize >= *v) {
            return Err(Error::Other("embedding id out of bounds".into()));
        };
        let ib = self.device_u32(ids)?;
        let mut o = self
            .ctx
            .allocate_output(Shape::new(vec![ids.len(), *d]), DType::BF16)?;
        let tb = MusaBuffer::from_tensor(t).map_err(Error::Other)?;
        let ob = MusaBuffer::from_tensor(&o).map_err(Error::Other)?;
        #[cfg(apxinf_musa_available)]
        unsafe {
            crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_embedding(
                tb.ptr(),
                ib.ptr(),
                ob.ptr(),
                ids.len() as i32,
                *d as i32,
                self.ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
        }
        Ok(o)
    }
    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        kl: usize,
        _: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, l, h, kh, d, kl, true)
    }
    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        kl: usize,
        _: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, l, h, kh, d, kl, true)
    }
    fn create_kv_cache(&self, n: usize, kh: usize, d: usize, max: usize) -> Box<dyn KvCache> {
        Box::new(MusaKvCache::new(&self.ctx, n, kh, d, max).expect("MUSA KV cache allocation"))
    }
    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        l: usize,
        k: &Tensor,
        v: &Tensor,
        n: usize,
    ) -> Result<()> {
        kv.append(l, k, v, n)
    }
    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize()
    }
    fn begin_capture(&self) -> Result<()> {
        Err(Self::unsupported("graph capture"))
    }
    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Err(Self::unsupported("graph capture"))
    }
    fn device(&self) -> Device {
        Device::Musa(self.ctx.device_id())
    }
    fn to_device(&self, t: &Tensor) -> Result<Tensor> {
        crate::transfers::to_device(t, self.ctx.device_id())
    }
    fn to_cpu(&self, t: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cpu(t)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl MusaBackend {
    fn rope_impl(
        &self,
        i: &Tensor,
        h: usize,
        d: usize,
        theta: f32,
        p: &[u32],
        mode: i32,
        sh: usize,
        sw: usize,
    ) -> Result<Tensor> {
        self.require_bf16(i, "rope")?;
        let n = i.shape().numel();
        let pb = self.device_u32(p)?;
        let mut o = self.output(i)?;
        let a = MusaBuffer::from_tensor(i).map_err(Error::Other)?;
        let ob = MusaBuffer::from_tensor(&o).map_err(Error::Other)?;
        #[cfg(apxinf_musa_available)]
        unsafe {
            crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_rope(
                a.ptr(),
                ob.ptr(),
                pb.ptr(),
                n as i32,
                h as i32,
                d as i32,
                theta,
                mode,
                sh as i32,
                sw as i32,
                self.ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
        }
        Ok(o)
    }
    fn attention_call(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &mut Tensor,
        nq: usize,
        nk: usize,
        h: usize,
        kh: usize,
        d: usize,
        causal: bool,
    ) -> Result<()> {
        let qb = MusaBuffer::from_tensor(q).map_err(Error::Other)?;
        let kb = MusaBuffer::from_tensor(k).map_err(Error::Other)?;
        let vb = MusaBuffer::from_tensor(v).map_err(Error::Other)?;
        let ob = MusaBuffer::from_tensor(o).map_err(Error::Other)?;
        #[cfg(apxinf_musa_available)]
        unsafe {
            crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_portable_attention(
                qb.ptr(),
                kb.ptr(),
                vb.ptr(),
                ob.ptr(),
                nq as i32,
                nk as i32,
                h as i32,
                kh as i32,
                d as i32,
                causal as i32,
                self.ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
        }
        Ok(())
    }
    fn sdpa(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        kl: usize,
        causal: bool,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any()
            .downcast_ref::<MusaKvCache>()
            .ok_or_else(|| Error::Other("wrong MUSA KV cache".into()))?;
        let layer = cache
            .layers
            .get(l)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| Error::Other("KV layer not appended".into()))?;
        let mut o = self.ctx.allocate_output(q.shape().clone(), DType::BF16)?;
        let qn = q.shape().dims()[0];
        let kb = layer
            .k
            .as_tensor(Shape::new(vec![cache.max_seq * kh, d]), DType::BF16)
            .map_err(Error::Other)?;
        let vb = layer
            .v
            .as_tensor(Shape::new(vec![cache.max_seq * kh, d]), DType::BF16)
            .map_err(Error::Other)?;
        self.attention_call(q, &kb, &vb, &mut o, qn, kl, h, kh, d, causal)?;
        Ok(o)
    }
}
impl apxinf_core::SamplingBackend for MusaBackend {
    fn create_token_sampler(&self, _: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Err(MusaBackend::unsupported("token sampler"))
    }
    fn create_normal_generator(&self, o: Tensor) -> Result<Box<dyn NormalGenerator>> {
        crate::sampling::create_normal_generator(&self.ctx, o)
    }
}
