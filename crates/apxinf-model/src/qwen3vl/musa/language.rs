use super::{
    kv_cache::{append, LayerCache},
    weights::NativeQwenWeights,
    NativeVisionOutput,
};
use apxinf_core::{Error, Result, Tensor};
use apxinf_musa::{
    kernels::{activation, elementwise, embedding, gemm, norm, rope},
    MusaBuffer, MusaContext,
};

pub(super) fn forward(
    ctx: &MusaContext,
    w: &NativeQwenWeights,
    cache: &mut [Option<LayerCache>],
    start: usize,
    cap: usize,
    ids: &[u32],
    positions: &[u32],
    image: Option<(&[usize], &NativeVisionOutput)>,
) -> Result<Tensor> {
    let tc = &w.config.text;
    let (n, h, kh, d) = (ids.len(), tc.n_heads, tc.n_kv_heads, tc.head_dim);
    let upload = |p: &[u32]| -> Result<MusaBuffer> {
        let bytes = p.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>();
        let b = MusaBuffer::alloc_on(ctx, bytes.len()).map_err(Error::Other)?;
        b.copy_from_host(&bytes).map_err(Error::Other)?;
        Ok(b)
    };
    let table = &w.text.token_embedding;
    if table.shape().dims() != [tc.vocab_size, tc.hidden_size]
        || ids.iter().any(|&id| id as usize >= tc.vocab_size)
        || positions.len() != 3 * n
    {
        return Err(Error::Other("invalid embedding input".into()));
    }
    let id_buf = upload(ids)?;
    let pos = upload(positions)?;
    let mut x = embedding::lookup(ctx, table, &id_buf, n)?;
    let rows = image
        .map(|(rows, _)| elementwise::prepare_row_indices(ctx, rows, n))
        .transpose()?;
    if let (Some(indices), Some((_, vision))) = (&rows, image) {
        x = elementwise::scatter_rows_bf16_prepared(ctx, &x, indices, &vision.primary, true)?;
    }
    for (i, l) in w.text.layers.iter().enumerate() {
        let z = norm::rms_bf16(ctx, &x, &l.attn_norm_weight, tc.rms_norm_eps)?;
        let project = |weight: &Tensor, scale: &Tensor, heads: usize| -> Result<Tensor> {
            let a = gemm::bf16(ctx, &z, weight)?.reshape(vec![n * heads, d])?;
            let a = norm::rms_bf16(ctx, &a, scale, tc.rms_norm_eps)?.reshape(vec![n, heads, d])?;
            rope::apply_mrope(ctx, &a, heads, d, tc.rope_theta, tc.mrope_section, &pos)
        };
        let q = project(&l.wq, &l.q_norm_weight, h)?;
        let k = project(&l.wk, &l.k_norm_weight, kh)?;
        let v = gemm::bf16(ctx, &z, &l.wv)?.reshape(vec![n, kh, d])?;
        let (k, v) = append(&mut cache[i], ctx, start, cap, &k, &v)?;
        // Full prefill has equal query and KV lengths; decode attends a longer prefix.
        let a = if start == 0 && n > 1 {
            apxinf_musa::kernels::attention::causal_gqa_prefill_bf16(ctx, &q, &k, &v)?
                .ok_or_else(|| Error::Other("MUSA causal GQA prefill unavailable".into()))?
        } else {
            apxinf_musa::kernels::attention::decode_gqa_bf16(ctx, &q, &k, &v)?
        }
        .reshape(vec![n, h * d])?;
        x = elementwise::add(ctx, &x, &gemm::bf16(ctx, &a, &l.wo)?)?;
        let z = norm::rms_bf16(ctx, &x, &l.ffn_norm_weight, tc.rms_norm_eps)?;
        let gate = activation::bias_silu_bf16(ctx, &gemm::bf16(ctx, &z, &l.w_gate)?, None)?;
        let up = gemm::bf16(ctx, &z, &l.w_up)?;
        let mlp = gemm::bf16(ctx, &elementwise::mul(ctx, &gate, &up)?, &l.w_down)?;
        x = elementwise::add(ctx, &x, &mlp)?;
        if let (Some(indices), Some((_, vision))) = (&rows, image) {
            if let Some(ds) = vision.deepstack.get(i) {
                x = elementwise::scatter_rows_bf16_prepared(ctx, &x, indices, ds, true)?;
            }
        }
    }
    let last = apxinf_musa::MusaBuffer::from_tensor(&x)
        .and_then(|b| b.view((n - 1) * tc.hidden_size * 2, tc.hidden_size * 2))
        .and_then(|b| b.as_tensor(vec![1, tc.hidden_size].into(), apxinf_core::DType::BF16))
        .map_err(Error::Other)?;
    let last = norm::rms_bf16(ctx, &last, &w.text.output_norm_weight, tc.rms_norm_eps)?;
    gemm::bf16(ctx, &last, &w.lm_head)
}
