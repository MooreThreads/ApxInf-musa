//! Prepared short-sequence MP31 attention; every other contract keeps its route.
use crate::MusaContext;
use apxinf_core::{Result, Tensor};

const PARTIAL_FLOATS: usize = 5 * 10 * 8 * 256;
const STAT_FLOATS: usize = 5 * 10 * 8;
const GUARD_FLOATS: usize = 64;
pub(crate) const WORKSPACE_BYTES: usize = (PARTIAL_FLOATS + STAT_FLOATS + 2 * GUARD_FLOATS) * 4;

#[derive(Default)]
pub(crate) struct State {
    #[cfg(apxinf_musa_mutlass_fmha)]
    workspace: Option<crate::buffer::MusaBuffer>,
    #[cfg(apxinf_musa_mutlass_fmha)]
    descriptors: std::collections::HashMap<[usize; 3], Descriptors>,
    #[cfg(apxinf_musa_mutlass_fmha)]
    recent: std::collections::VecDeque<[usize; 3]>,
}

#[cfg(apxinf_musa_mutlass_fmha)]
struct Descriptors(usize);
#[cfg(apxinf_musa_mutlass_fmha)]
impl Drop for Descriptors {
    fn drop(&mut self) {
        // Host POD only. Launches and graph nodes copy all three descriptors.
        unsafe { crate::ffi::custom::apxinf_musa_mp31_attention_release(self.0 as *mut _) };
    }
}

pub(crate) fn prepare(
    ctx: &MusaContext,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    #[cfg(apxinf_musa_mutlass_fmha)]
    {
        use crate::{
            buffer::MusaBuffer,
            ffi::{check_runtime, custom},
        };
        use apxinf_core::Error;
        if (heads, kv_heads, head_dim) != (8, 1, 256) {
            return Ok(());
        }
        let mut state = ctx
            .native_attention
            .lock()
            .map_err(|_| Error::Other("MP31 attention state poisoned".into()))?;
        if state.workspace.is_some() {
            return Ok(());
        }
        if crate::workspace::is_capturing() {
            return Err(Error::Other(
                "MP31 attention must be prepared before graph capture".into(),
            ));
        }
        unsafe { check_runtime(custom::apxinf_musa_mp31_attention_prepare()) }
            .map_err(Error::Other)?;
        let scratch =
            MusaBuffer::alloc_filled_on(ctx, WORKSPACE_BYTES, 255).map_err(Error::Other)?;
        state.workspace = Some(scratch);
        Ok(())
    }
    #[cfg(not(apxinf_musa_mutlass_fmha))]
    {
        let _ = (ctx, heads, kv_heads, head_dim);
        Ok(())
    }
}

/// `geometry` is the validated query tokens, heads, head dim, cache rows and valid KV rows.
/// The MQA caller has already checked dtype, contiguous storage and output coverage.
pub(crate) fn try_launch(
    ctx: &MusaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &mut Tensor,
    geometry: [usize; 5],
    workspace_limit: usize,
) -> Result<bool> {
    #[cfg(apxinf_musa_mutlass_fmha)]
    {
        use crate::ffi::{check_runtime, custom};
        use crate::kernels::contracts::tensor_ptr;
        use apxinf_core::Error;
        if geometry != [10, 8, 256, 532, 532] || workspace_limit < WORKSPACE_BYTES {
            return Ok(false);
        }
        let query = tensor_ptr(q)?;
        let key = tensor_ptr(k)?;
        let value = tensor_ptr(v)?;
        let output = tensor_ptr(out)?;
        let cache_key = [query as usize, key as usize, value as usize];
        if cache_key.iter().any(|ptr| ptr & 15 != 0) {
            return Ok(false);
        }
        let mut state = ctx
            .native_attention
            .lock()
            .map_err(|_| Error::Other("MP31 attention state poisoned".into()))?;
        if state.workspace.is_none() {
            return Ok(false);
        }
        if !state.descriptors.contains_key(&cache_key) {
            if crate::workspace::is_capturing() {
                return Err(Error::Other(
                    "MP31 attention descriptors were not prepared before capture".into(),
                ));
            }
            let mut handle = std::ptr::null_mut();
            unsafe {
                check_runtime(custom::apxinf_musa_mp31_attention_descriptors(
                    query,
                    key,
                    value,
                    &mut handle,
                ))
            }
            .map_err(Error::Other)?;
            if handle.is_null() {
                return Err(Error::Other(
                    "MP31 attention returned null descriptors".into(),
                ));
            }
            // Metadata does not retain activation tensors; stale addresses cannot be
            // launched without matching live input tensors. Bound long-running eager use.
            if state.descriptors.len() == 1024 {
                if let Some(old) = state.recent.pop_front() {
                    state.descriptors.remove(&old);
                }
            }
            state
                .descriptors
                .insert(cache_key, Descriptors(handle as usize));
        }
        state.recent.retain(|old| *old != cache_key);
        state.recent.push_back(cache_key);
        let handle = state.descriptors[&cache_key].0 as *const _;
        let base = state.workspace.as_ref().expect("prepared workspace").ptr() as *mut f32;
        let stats = unsafe { base.add(PARTIAL_FLOATS + GUARD_FLOATS) };
        // Hold the state lock until both dependent kernels have been enqueued on
        // the same context stream, so its one workspace can be safely reused.
        unsafe {
            check_runtime(custom::apxinf_musa_mp31_attention(
                handle,
                base,
                stats,
                output,
                ctx.stream().handle(),
            ))
        }
        .map_err(Error::Other)?;
        Ok(true)
    }
    #[cfg(not(apxinf_musa_mutlass_fmha))]
    {
        let _ = (ctx, q, k, v, out, geometry, workspace_limit);
        Ok(false)
    }
}

#[cfg(all(test, apxinf_musa_available, apxinf_musa_mutlass_fmha))]
mod tests {
    use super::*;
    use crate::{buffer::MusaBuffer, ops, workspace::ExecutionSession};
    use apxinf_core::{DType, Graph, Shape};
    use std::sync::Arc;

    fn tensor(ctx: &MusaContext, shape: &[usize], values: &[f32]) -> Tensor {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let raw = MusaBuffer::alloc_on(ctx, bytes.len()).unwrap();
        raw.copy_from_host(&bytes).unwrap();
        raw.into_tensor(Shape::new(shape.to_vec()), DType::BF16)
    }

    fn reference(q: &[f32], k: &[f32], v: &[f32], rows: usize) -> Vec<f32> {
        let mut out = vec![0.; 10 * 2048];
        for row in 0..10 {
            for head in 0..8 {
                let mut weights = vec![0f64; rows];
                let mut maximum = f64::NEG_INFINITY;
                for (n, score) in weights.iter_mut().enumerate() {
                    for d in 0..256 {
                        *score += q[(row * 8 + head) * 256 + d] as f64 * k[n * 256 + d] as f64;
                    }
                    *score /= 16.;
                    maximum = maximum.max(*score);
                }
                let mut sum = 0.;
                for score in &mut weights {
                    *score = (*score - maximum).exp();
                    sum += *score;
                }
                for d in 0..256 {
                    let mut value = 0.;
                    for (n, &weight) in weights.iter().enumerate() {
                        value += weight * v[n * 256 + d] as f64;
                    }
                    out[(row * 8 + head) * 256 + d] = (value / sum) as f32;
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "requires idle MP31"]
    fn mp31_attention_real_random_guards_replay_and_kv_switch() {
        let device = std::env::var("APXINF_MUSA_BENCH_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap();
        let ctx = Arc::new(MusaContext::new(device).unwrap());
        if !ctx.caps().arch_family.has_native_fp8() {
            return;
        }
        ops::prepare_attention(&ctx, 8, 1, 256).unwrap();
        let mut cases = Vec::new();
        for (index, rows) in [532, 543, 532].into_iter().enumerate() {
            let mut rng = 19u32 + index as u32;
            let mut values = |len, scale| {
                (0..len)
                    .map(|_| {
                        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((rng >> 24) as i32 - 128) as f32 * scale
                    })
                    .collect::<Vec<_>>()
            };
            let scale = if index == 2 { 0.125 } else { 0.015625 };
            let q = values(10 * 2048, scale);
            let k = values(rows * 256, scale);
            let v = values(rows * 256, 0.015625);
            cases.push((format!("random-{index}"), rows, q, k, v));
        }
        if let Ok(folder) = std::env::var("APXINF_MUSA_ATTENTION_REAL_INPUT_DIR") {
            let read = |name| {
                let data = std::fs::read(std::path::Path::new(&folder).join(name)).unwrap();
                data.chunks_exact(2)
                    .map(|bytes| {
                        f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16)
                    })
                    .collect::<Vec<_>>()
            };
            cases.push((
                "real-model-layer0-step0".into(),
                532,
                read("query.bf16"),
                read("key.bf16"),
                read("value.bf16"),
            ));
        }
        for (name, rows, query, key, value) in cases {
            assert_eq!(query.len(), 10 * 2048);
            assert_eq!(key.len(), rows * 256);
            assert_eq!(value.len(), rows * 256);
            let q = tensor(&ctx, &[1, 10, 8, 256], &query);
            let mut key_padded = key.clone();
            // Poison readable storage past the logical tensor. TME must not
            // expose these rows to the final KV tile, even with zero scores.
            key_padded.resize(576 * 256, f32::NAN);
            let k = tensor(&ctx, &[1, rows, 1, 256], &key_padded);
            let mut value_padded = value.clone();
            value_padded.resize(576 * 256, f32::NAN);
            let v = tensor(&ctx, &[1, rows, 1, 256], &value_padded);
            let raw = MusaBuffer::alloc_filled_on(&ctx, 10 * 2048 * 2 + 256, 127).unwrap();
            let mut out = raw.clone().into_tensor(q.shape().clone(), DType::BF16);
            let call =
                |out: &mut Tensor| ops::attention(&ctx, ops::AttentionArgs::new(&q, &k, &v, out));
            let before = ctx.native_attention.lock().unwrap().descriptors.len();
            call(&mut out).unwrap();
            ctx.synchronize().unwrap();
            let after = ctx.native_attention.lock().unwrap().descriptors.len();
            if rows != 532 {
                assert_eq!(before, after, "unvalidated KV length must retain MATE");
            } else {
                assert!(after > 0, "exact shape must exercise the MP31 route");
            }
            eprintln!("{name}: native descriptor cache entries {before}->{after}");
            let mut expected = vec![0; 10 * 2048 * 2 + 256];
            raw.copy_to_host(&mut expected).unwrap();
            let golden = reference(&query, &key, &value, rows);
            let mut dot = 0.;
            let mut aa = 0.;
            let mut bb = 0.;
            for (bytes, &y) in expected[..40960].chunks_exact(2).zip(&golden) {
                let x = f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16);
                assert!(x.is_finite() && x.abs() < 100., "unwritten/invalid output");
                dot += x as f64 * y as f64;
                aa += (x as f64).powi(2);
                bb += (y as f64).powi(2);
            }
            let cosine = dot / (aa * bb).sqrt();
            assert!(cosine >= 0.99999, "{name} cosine={cosine}");
            assert!(expected[40960..].iter().all(|&byte| byte == 127));
            assert!(!try_launch(
                &ctx,
                &q,
                &k,
                &v,
                &mut out,
                [10, 8, 256, rows, rows],
                WORKSPACE_BYTES - 1
            )
            .unwrap());
            let session = ExecutionSession::with_capacity(1024 * 1024, device).unwrap();
            session.prepare(|| call(&mut out)).unwrap();
            let graph =
                crate::graph::capture(ctx.clone(), || session.run(|| call(&mut out))).unwrap();
            let mut actual = vec![0; expected.len()];
            for replay in 0..1000 {
                raw.memset_async(127, expected.len(), ctx.stream()).unwrap();
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                raw.copy_to_host(&mut actual).unwrap();
                assert_eq!(expected, actual, "{name} replay {replay}");
            }
            let state = ctx.native_attention.lock().unwrap();
            let mut scratch = vec![0; WORKSPACE_BYTES];
            state
                .workspace
                .as_ref()
                .unwrap()
                .copy_to_host(&mut scratch)
                .unwrap();
            assert!(
                scratch[PARTIAL_FLOATS * 4..(PARTIAL_FLOATS + GUARD_FLOATS) * 4]
                    .iter()
                    .all(|&x| x == 255)
            );
            assert!(scratch[(PARTIAL_FLOATS + GUARD_FLOATS + STAT_FLOATS) * 4..]
                .iter()
                .all(|&x| x == 255));
            eprintln!("{name} kv={rows} cosine={cosine:.9} eager/graph/1000 replays bitwise identical; guards intact");
        }
    }
}
