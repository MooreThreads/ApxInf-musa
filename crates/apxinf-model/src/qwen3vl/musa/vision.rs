use super::{runtime::MusaQwenRuntime, weights::NativeQwenWeights};
use apxinf_core::{DType, Error, Result, Tensor};

use super::super::vision_weights::Qwen3VLMerger;
pub struct NativeVisionOutput {
    pub primary: Tensor,
    pub deepstack: Vec<Tensor>,
}

pub fn forward(
    runtime: &MusaQwenRuntime,
    weights: &NativeQwenWeights,
    pixels: &Tensor,
    grids: &[[u32; 3]],
) -> Result<NativeVisionOutput> {
    runtime.require_device(pixels)?;
    let vc = &weights.config.vision;
    let hidden = vc.hidden_size;
    let heads = vc.num_heads;
    let dim = vc.head_dim();
    let merge = vc.spatial_merge_size;
    let eps = 1e-6;
    let &[[1, h, w]] = grids else {
        return Err(Error::Other(
            "native Qwen vision expects one still image".into(),
        ));
    };
    let (h, w) = (h as usize, w as usize);
    let n = h
        .checked_mul(w)
        .ok_or_else(|| Error::Other("vision grid overflow".into()))?;
    let [patch_width, _] = weights.vision.patch_embed_weight.shape().dims() else {
        return Err(Error::Other("invalid patch projection".into()));
    };
    if h == 0
        || w == 0
        || merge == 0
        || h % merge != 0
        || w % merge != 0
        || pixels.dtype() != DType::BF16
        || pixels.shape().dims() != [n, *patch_width]
    {
        return Err(Error::Other(
            "pixels and spatial merge grid do not match".into(),
        ));
    }
    let mut x = apxinf_musa::kernels::gemm::bf16(
        runtime.context(),
        pixels,
        &weights.vision.patch_embed_weight,
    )?;
    x = apxinf_musa::kernels::elementwise::bias_bf16(
        runtime.context(),
        &x,
        Some(&weights.vision.patch_embed_bias),
    )?;
    let pos = position_embeddings(runtime, &weights.vision.pos_embed, 1, h, w, merge, hidden)?;
    x = apxinf_musa::kernels::elementwise::add(runtime.context(), &x, &pos)?;
    let pos_ids = vision_pos_ids(1, h, w, merge);
    let pb = position_buffer(runtime, &pos_ids)?;
    let mut deep = Vec::new();
    for (i, b) in weights.vision.blocks.iter().enumerate() {
        let z = apxinf_musa::kernels::norm::layer_bf16(
            runtime.context(),
            &x,
            &b.norm1_w,
            &b.norm1_b,
            eps,
        )?;
        let qkv = apxinf_musa::kernels::gemm::bf16(runtime.context(), &z, &b.qkv_w)?;
        let qkv =
            apxinf_musa::kernels::elementwise::bias_bf16(runtime.context(), &qkv, Some(&b.qkv_b))?;
        let (q, k, v) = apxinf_musa::kernels::elementwise::split_qkv_uneven_bf16(
            runtime.context(),
            &qkv,
            hidden,
            hidden,
            hidden,
        )?;
        let q = q.reshape(vec![n, heads, dim])?;
        let k = k.reshape(vec![n, heads, dim])?;
        let v = v.reshape(vec![n, heads, dim])?;
        let q = apxinf_musa::ops::vision_axial_rope(runtime.context(), &q, &pb, 10000.)?;
        let k = apxinf_musa::ops::vision_axial_rope(runtime.context(), &k, &pb, 10000.)?;
        let attn = apxinf_musa::ops::vision_self_attention(runtime.context(), &q, &k, &v)?;
        let attn = apxinf_musa::kernels::gemm::bf16(runtime.context(), &attn, &b.proj_w)?;
        let attn = apxinf_musa::kernels::elementwise::bias_bf16(
            runtime.context(),
            &attn,
            Some(&b.proj_b),
        )?;
        x = apxinf_musa::kernels::elementwise::add(runtime.context(), &x, &attn)?;
        let z = apxinf_musa::kernels::norm::layer_bf16(
            runtime.context(),
            &x,
            &b.norm2_w,
            &b.norm2_b,
            eps,
        )?;
        let h1 = apxinf_musa::kernels::gemm::bf16(runtime.context(), &z, &b.fc1_w)?;
        let h1 = gelu_after_bias(runtime, &h1, &b.fc1_b)?;
        let h2 = apxinf_musa::kernels::gemm::bf16(runtime.context(), &h1, &b.fc2_w)?;
        let h2 =
            apxinf_musa::kernels::elementwise::bias_bf16(runtime.context(), &h2, Some(&b.fc2_b))?;
        x = apxinf_musa::kernels::elementwise::add(runtime.context(), &x, &h2)?;
        if weights.config.vision.deepstack_visual_indexes.contains(&i) {
            deep.push(x.clone());
        }
    }
    let primary = merge_primary(runtime, &weights.vision.merger, &x, n, hidden, merge, eps)?;
    let deepstack = weights
        .vision
        .deepstack_mergers
        .iter()
        .zip(deep.iter())
        .map(|(m, x)| merge_deep(runtime, m, x, n, hidden, merge, eps))
        .collect::<Result<Vec<_>>>()?;
    Ok(NativeVisionOutput { primary, deepstack })
}

fn merge_primary(
    r: &MusaQwenRuntime,
    m: &Qwen3VLMerger,
    x: &Tensor,
    n: usize,
    h: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let z = apxinf_musa::kernels::norm::layer_bf16(r.context(), x, &m.norm_w, &m.norm_b, eps)?;
    let z = reshape_merge(r, &z, n, h, merge)?;
    let z = apxinf_musa::kernels::gemm::bf16(r.context(), &z, &m.fc1_w)?;
    let z = gelu_after_bias(r, &z, &m.fc1_b)?;
    let z = apxinf_musa::kernels::gemm::bf16(r.context(), &z, &m.fc2_w)?;
    apxinf_musa::kernels::elementwise::bias_bf16(r.context(), &z, Some(&m.fc2_b))
}
fn merge_deep(
    r: &MusaQwenRuntime,
    m: &Qwen3VLMerger,
    x: &Tensor,
    n: usize,
    h: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let z = reshape_merge(r, x, n, h, merge)?;
    let z = apxinf_musa::kernels::norm::layer_bf16(r.context(), &z, &m.norm_w, &m.norm_b, eps)?;
    let z = apxinf_musa::kernels::gemm::bf16(r.context(), &z, &m.fc1_w)?;
    let z = gelu_after_bias(r, &z, &m.fc1_b)?;
    let z = apxinf_musa::kernels::gemm::bf16(r.context(), &z, &m.fc2_w)?;
    apxinf_musa::kernels::elementwise::bias_bf16(r.context(), &z, Some(&m.fc2_b))
}
fn reshape_merge(
    _r: &MusaQwenRuntime,
    x: &Tensor,
    n: usize,
    h: usize,
    merge: usize,
) -> Result<Tensor> {
    x.reshape(vec![n / (merge * merge), h * merge * merge])
}
fn vision_pos_ids(t: usize, h: usize, w: usize, m: usize) -> Vec<u32> {
    let mut o = Vec::new();
    for _ in 0..t {
        for mh in 0..h / m {
            for mw in 0..w / m {
                for ih in 0..m {
                    for iw in 0..m {
                        o.push((mh * m + ih) as u32);
                        o.push((mw * m + iw) as u32);
                    }
                }
            }
        }
    }
    o
}
fn position_buffer(r: &MusaQwenRuntime, p: &[u32]) -> Result<apxinf_musa::MusaBuffer> {
    let b = apxinf_musa::MusaBuffer::alloc_on(r.context(), p.len() * 4).map_err(Error::Other)?;
    b.copy_from_host(&p.iter().flat_map(|id| id.to_ne_bytes()).collect::<Vec<_>>())
        .map_err(Error::Other)?;
    Ok(b)
}
fn position_embeddings(
    r: &MusaQwenRuntime,
    table: &Tensor,
    _t: usize,
    h: usize,
    w: usize,
    m: usize,
    hidden: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    let side = (dims[0] as f64).sqrt() as usize;
    if side == 0 || dims != [side * side, hidden] {
        return Err(Error::Other("invalid vision position table".into()));
    }
    let vals = r.to_cpu(table)?.to_f32_vec()?;
    let ids = vision_pos_ids(1, h, w, m);
    let mut out = Vec::with_capacity(h * w * hidden);
    for pair in ids.chunks_exact(2) {
        let yf = pair[0] as f32 * (side - 1) as f32 / (h - 1).max(1) as f32;
        let xf = pair[1] as f32 * (side - 1) as f32 / (w - 1).max(1) as f32;
        let (y, x) = (yf as usize, xf as usize);
        let (dy, dx) = (yf - y as f32, xf - x as f32);
        let (y1, x1) = ((y + 1).min(side - 1), (x + 1).min(side - 1));
        for c in 0..hidden {
            let at = |y: usize, x: usize| vals[(y * side + x) * hidden + c];
            out.push(half::bf16::from_f32(
                (1. - dy) * (1. - dx) * at(y, x)
                    + (1. - dy) * dx * at(y, x1)
                    + dy * (1. - dx) * at(y1, x)
                    + dy * dx * at(y1, x1),
            ));
        }
    }
    r.to_device(&Tensor::from_bf16(vec![h * w, hidden], &out)?)
}

// Match the standalone Qwen graph: bias addition materializes a BF16 tensor before GELU.
fn gelu_after_bias(r: &MusaQwenRuntime, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let biased = apxinf_musa::kernels::elementwise::bias_bf16(r.context(), x, Some(bias))?;
    apxinf_musa::kernels::activation::bias_gelu_bf16(r.context(), &biased, None)
}
