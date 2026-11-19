//! Immutable MP31 FP8 weight preparation and native fused GEMM dispatch.
//! Canonical and packed allocations live with the context, preventing address reuse.
use crate::MusaContext;
use apxinf_core::{Result, Tensor};

#[derive(Default)]
pub(crate) struct PreparedWeights {
    entries: std::collections::HashMap<(usize, usize, usize), PreparedWeight>,
}
struct PreparedWeight {
    _source: Tensor,
    _packed: Option<Tensor>,
    _direct_packed: Option<Tensor>,
    descriptor: usize,
}
impl Drop for PreparedWeight {
    fn drop(&mut self) {
        #[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
        // SAFETY: preparation returned this uniquely owned host descriptor.
        unsafe {
            crate::ffi::custom::apxinf_musa_mp31_release_fp8(self.descriptor as *mut _)
        };
    }
}

fn enabled() -> bool {
    crate::MUSA_ARCH == "mp_31"
}

fn projection_enabled_for(k: usize) -> bool {
    crate::MUSA_ARCH == "mp_31" && matches!(k, 2048 | 4096)
}

/// Prepare an immutable canonical `[K,N]` E4M3 weight once, outside capture.
/// Unsupported targets/shapes retain the canonical vendor implementation.
/// The source must not be mutated after registration.
pub fn prepare_fp8_weight(ctx: &MusaContext, weight: &Tensor) -> Result<()> {
    let dims = weight.shape().dims();
    if !(enabled()
        && (dims == [1024, 8192]
            || dims == [1024, 2560]
            || (dims == [2048, 1024] && projection_enabled_for(2048))
            || (dims == [4096, 1024] && projection_enabled_for(4096))))
    {
        return Ok(());
    }

    prepare(ctx, weight)
}

#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
fn prepare(ctx: &MusaContext, weight: &Tensor) -> Result<()> {
    use super::super::contracts::{require_device, tensor_ptr};
    use crate::{buffer::MusaBuffer, error::musa_error, ffi};
    use apxinf_core::{DType, Shape};
    require_device(ctx, weight, "MP31 FP8 prepare", "weight")?;
    if weight.dtype() != DType::F8E4M3 {
        return Err(musa_error("MP31 preparation requires E4M3"));
    }
    if !ctx.caps().arch_family.has_native_fp8() {
        return Ok(());
    }
    if crate::workspace::is_capturing() {
        return Err(musa_error(
            "MP31 FP8 weights must be prepared before capture",
        ));
    }
    let source = tensor_ptr(weight)?;
    let mut weights = ctx
        .native_fp8_weights
        .lock()
        .map_err(|_| musa_error("MP31 weight lock poisoned"))?;
    let dims = weight.shape().dims();
    let (k, n) = (dims[0], dims[1]);
    if weights.entries.contains_key(&(source as usize, k, n)) {
        return Ok(());
    }
    // The action qkv weight is consumed only by the fused direct producer, so
    // it carries one 16x32 register packing instead of a transposed TME copy.
    let direct_only = k == 1024 && n == 2560;
    let packed = if direct_only {
        None
    } else {
        Some(
            MusaBuffer::alloc_on(ctx, k * n)
                .map_err(musa_error)?
                .into_tensor(Shape::new(vec![n, k]), DType::F8E4M3),
        )
    };
    let direct_packed = if direct_only {
        Some(
            MusaBuffer::alloc_on(ctx, k * n)
                .map_err(musa_error)?
                .into_tensor(Shape::new(vec![n, k]), DType::F8E4M3),
        )
    } else {
        None
    };
    let packed_ptr = packed
        .as_ref()
        .map(tensor_ptr)
        .transpose()?
        .unwrap_or(std::ptr::null_mut());
    let direct_ptr = direct_packed
        .as_ref()
        .map(tensor_ptr)
        .transpose()?
        .unwrap_or(std::ptr::null_mut());
    let mut descriptor = std::ptr::null_mut();
    // SAFETY: exact E4M3 sizes/devices checked above; output is uniquely owned.
    let status = unsafe {
        ffi::custom::apxinf_musa_mp31_prepare_fp8(
            source,
            packed_ptr,
            direct_ptr,
            k as i32,
            n as i32,
            &mut descriptor,
            ctx.stream().handle(),
        )
    };
    if status != ffi::musa::MUSA_SUCCESS {
        return Err(musa_error(ffi::runtime_error(status)));
    }
    weights.entries.insert(
        (source as usize, k, n),
        PreparedWeight {
            _source: weight.clone(),
            _packed: packed,
            _direct_packed: direct_packed,
            descriptor: descriptor as usize,
        },
    );
    Ok(())
}
#[cfg(not(all(apxinf_musa_available, apxinf_musa_mutlass_fmha)))]
fn prepare(_ctx: &MusaContext, _weight: &Tensor) -> Result<()> {
    Ok(())
}

/// Return false for a layout/shape without a proven native epilogue.
pub(crate) fn try_geglu(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    out: &Tensor,
    alpha: f32,
    scale: f32,
    pair_width: usize,
) -> Result<bool> {
    if !enabled() || pair_width != 32 || w.shape().dims() != [1024, 8192] {
        return Ok(false);
    }
    if a.shape().dims() != [10, 1024] {
        return Ok(false);
    }
    launch(ctx, a, w, out, alpha, scale)
}

/// Fuse the fixed action qkv projection with bias, rotary embedding, and the
/// direct K/V cache writes. This removes the intermediate `[10,2560]` BF16
/// tensor and the separate qkv-rope launch from the action graph.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_qkv_rope(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    theta: f32,
    position_offset: usize,
    kv_output_offset: usize,
    alpha: f32,
) -> Result<bool> {
    if !enabled()
        || a.shape().dims() != [10, 1024]
        || w.shape().dims() != [1024, 2560]
        || q.shape().dims() != [10, 8, 256]
        || w.dtype() != apxinf_core::DType::F8E4M3
        || a.dtype() != apxinf_core::DType::F8E4M3
        || q.dtype() != apxinf_core::DType::BF16
        || k.dtype() != apxinf_core::DType::BF16
        || v.dtype() != apxinf_core::DType::BF16
        || !theta.is_finite()
    {
        return Ok(false);
    }
    launch_qkv_rope(
        ctx,
        a,
        w,
        bias,
        q,
        k,
        v,
        theta,
        position_offset,
        kv_output_offset,
        alpha,
    )
}

#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
#[allow(clippy::too_many_arguments)]
fn launch_qkv_rope(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    theta: f32,
    position_offset: usize,
    kv_output_offset: usize,
    alpha: f32,
) -> Result<bool> {
    use super::super::contracts::{require_device, tensor_ptr};
    use crate::{error::musa_error, ffi};
    for (name, tensor) in [
        ("activation", a),
        ("weight", w),
        ("q", q),
        ("k", k),
        ("v", v),
    ] {
        require_device(ctx, tensor, "MP31 fused qkv-rope", name)?;
    }
    if let Some(bias) = bias {
        require_device(ctx, bias, "MP31 fused qkv-rope", "bias")?;
        if bias.dtype() != apxinf_core::DType::BF16 || bias.shape().dims() != [2560] {
            return Ok(false);
        }
    }
    let weights = ctx
        .native_fp8_weights
        .lock()
        .map_err(|_| musa_error("MP31 weight lock poisoned"))?;
    let Some(weight) = weights.entries.get(&(
        tensor_ptr(w)? as usize,
        w.shape().dims()[0],
        w.shape().dims()[1],
    )) else {
        return Ok(false);
    };
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: 10,
        n: 2560,
        k: 1024,
    })?;
    let status = unsafe {
        ffi::custom::apxinf_musa_mp31_fp8_qkv_rope(
            tensor_ptr(a)?,
            weight.descriptor as *const _,
            bias.map(|tensor| tensor_ptr(tensor).map(|ptr| ptr as *const _))
                .transpose()?
                .unwrap_or(std::ptr::null()),
            tensor_ptr(q)?,
            tensor_ptr(k)?,
            tensor_ptr(v)?,
            8,
            1,
            256,
            theta,
            position_offset as i32,
            kv_output_offset as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    if status != ffi::musa::MUSA_SUCCESS {
        return Err(musa_error(ffi::runtime_error(status)));
    }
    Ok(true)
}

#[cfg(not(all(apxinf_musa_available, apxinf_musa_mutlass_fmha)))]
#[allow(clippy::too_many_arguments)]
fn launch_qkv_rope(
    _ctx: &MusaContext,
    _a: &Tensor,
    _w: &Tensor,
    _bias: Option<&Tensor>,
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _theta: f32,
    _position_offset: usize,
    _kv_output_offset: usize,
    _alpha: f32,
) -> Result<bool> {
    Ok(false)
}
#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
fn launch(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    out: &Tensor,
    alpha: f32,
    scale: f32,
) -> Result<bool> {
    use super::super::contracts::{require_device, tensor_ptr};
    use crate::{error::musa_error, ffi};
    use apxinf_core::DType;
    for (name, tensor) in [("activation", a), ("weight", w), ("output", out)] {
        require_device(ctx, tensor, "MP31 GeGLU", name)?;
    }
    if a.dtype() != DType::F8E4M3
        || w.dtype() != DType::F8E4M3
        || out.dtype() != DType::F8E4M3
        || out.shape().dims() != [10, 4096]
        || !alpha.is_finite()
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(musa_error("MP31 GeGLU requires E4M3 operands/output, [10,4096] output and finite alpha/positive scale"));
    }
    let weights = ctx
        .native_fp8_weights
        .lock()
        .map_err(|_| musa_error("MP31 weight lock poisoned"))?;
    let Some(weight) = weights.entries.get(&(
        tensor_ptr(w)? as usize,
        w.shape().dims()[0],
        w.shape().dims()[1],
    )) else {
        return Ok(false);
    };
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: 10,
        n: 8192,
        k: 1024,
    })?;
    // SAFETY: registered immutable weight/descriptors and validated device buffers
    // remain live for the asynchronous launch and graph replays.
    let status = unsafe {
        ffi::custom::apxinf_musa_mp31_fp8_geglu(
            tensor_ptr(a)?,
            weight.descriptor as *const _,
            tensor_ptr(out)?,
            10,
            alpha,
            scale,
            ctx.stream().handle(),
        )
    };
    if status != ffi::musa::MUSA_SUCCESS {
        return Err(musa_error(ffi::runtime_error(status)));
    }
    Ok(true)
}

#[cfg(not(all(apxinf_musa_available, apxinf_musa_mutlass_fmha)))]
fn launch(
    _ctx: &MusaContext,
    _a: &Tensor,
    _w: &Tensor,
    _out: &Tensor,
    _alpha: f32,
    _scale: f32,
) -> Result<bool> {
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_projection_norm(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    style: &Tensor,
    hidden: &Tensor,
    out: &Tensor,
    input_scale: Option<f32>,
) -> Result<()> {
    use super::super::contracts::{
        require_dtype_on, require_output_dtype_tensor, require_vector_bf16,
    };
    use apxinf_core::DType;
    let op = "gemm_ada_gate_residual_rms_norm";
    let (m, n) = (a.shape().dims()[0], w.shape().dims()[1]);
    require_dtype_on(
        ctx,
        a,
        if input_scale.is_some() {
            DType::BF16
        } else {
            DType::F8E4M3
        },
        op,
        "activation",
    )?;
    require_dtype_on(ctx, w, DType::F8E4M3, op, "weight")?;
    require_output_dtype_tensor(ctx, residual, &[m, n], DType::BF16, op, "residual")?;
    require_output_dtype_tensor(ctx, hidden, &[m, n], DType::BF16, op, "hidden")?;
    require_output_dtype_tensor(ctx, out, &[m, n], DType::F8E4M3, op, "normalized")?;
    require_vector_bf16(ctx, gate, 3 * n, op, "gate style")?;
    require_vector_bf16(ctx, style, 2 * n, op, "norm style")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_projection_norm(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    style: &Tensor,
    hidden: &Tensor,
    out: &Tensor,
    input_scale: Option<f32>,
    alpha: f32,
    output_scale: f32,
    eps: f32,
    workspace_limit: usize,
) -> Result<bool> {
    if a.shape().dims().len() != 2
        || a.shape().dims()[0] != 10
        || w.shape().dims().len() != 2
        || w.shape().dims()[1] != 1024
        || !matches!(a.shape().dims()[1], 2048 | 4096)
        || workspace_limit < 8 * 10 * 1024 * 4
    {
        return Ok(false);
    }
    if !projection_enabled_for(a.shape().dims()[1]) {
        return Ok(false);
    }
    projection(
        ctx,
        a,
        w,
        residual,
        gate,
        style,
        hidden,
        out,
        input_scale,
        alpha,
        output_scale,
        eps,
    )
}
#[cfg(all(apxinf_musa_available, apxinf_musa_mutlass_fmha))]
#[allow(clippy::too_many_arguments)]
fn projection(
    ctx: &MusaContext,
    a: &Tensor,
    w: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    style: &Tensor,
    hidden: &Tensor,
    out: &Tensor,
    input_scale: Option<f32>,
    alpha: f32,
    output_scale: f32,
    eps: f32,
) -> Result<bool> {
    use super::super::contracts::tensor_ptr;
    use crate::{error::musa_error, ffi};
    use apxinf_core::{DType, Shape};
    let weights = ctx
        .native_fp8_weights
        .lock()
        .map_err(|_| musa_error("MP31 weight lock poisoned"))?;
    let Some(weight) = weights.entries.get(&(
        tensor_ptr(w)? as usize,
        w.shape().dims()[0],
        w.shape().dims()[1],
    )) else {
        return Ok(false);
    };
    let k = a.shape().dims()[1];
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm { m: 10, n: 1024, k })?;
    let partial = ctx.allocate_output(Shape::new(vec![8, 10, 1024]), DType::F32)?;
    // SAFETY: the operator validates device, shape, dtype, scale and vector lengths;
    // the prepared immutable weight and graph arena remain live through replay.
    let status = unsafe {
        ffi::custom::apxinf_musa_mp31_fp8_projection_norm(
            tensor_ptr(a)?,
            weight.descriptor as *const _,
            tensor_ptr(&partial)?.cast(),
            tensor_ptr(residual)?,
            tensor_ptr(gate)?,
            tensor_ptr(style)?,
            tensor_ptr(hidden)?,
            tensor_ptr(out)?,
            10,
            i32::from(input_scale.is_some()),
            input_scale.map_or(1.0, |s| 1.0 / s),
            alpha,
            output_scale,
            eps,
            ctx.stream().handle(),
        )
    };
    if status != ffi::musa::MUSA_SUCCESS {
        return Err(musa_error(ffi::runtime_error(status)));
    }
    Ok(true)
}
#[cfg(not(all(apxinf_musa_available, apxinf_musa_mutlass_fmha)))]
#[allow(clippy::too_many_arguments)]
fn projection(
    _ctx: &MusaContext,
    _a: &Tensor,
    _w: &Tensor,
    _residual: &Tensor,
    _gate: &Tensor,
    _style: &Tensor,
    _hidden: &Tensor,
    _out: &Tensor,
    _input_scale: Option<f32>,
    _alpha: f32,
    _output_scale: f32,
    _eps: f32,
) -> Result<bool> {
    Ok(false)
}

#[cfg(all(test, apxinf_musa_available, apxinf_musa_mutlass_fmha))]
mod tests {
    use super::*;
    use crate::{ops, MusaBuffer};
    use apxinf_core::{DType, Graph, Shape};
    use std::sync::Arc;
    fn bytes(t: &Tensor) -> Vec<u8> {
        let b = MusaBuffer::from_tensor(t).unwrap();
        let mut out = vec![0; b.len()];
        b.copy_to_host(&mut out).unwrap();
        out
    }
    fn e4m3(ctx: &MusaContext, shape: Vec<usize>, seed: u32) -> Tensor {
        let mut s = seed;
        let data = (0..shape.iter().product())
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                0x28 + ((s >> 24) as u8 & 23) | ((s >> 16) as u8 & 0x80)
            })
            .collect::<Vec<_>>();
        crate::transfers::to_device(
            &Tensor::from_f8_e4m3(shape, &data).unwrap(),
            ctx.device_id(),
        )
        .unwrap()
    }
    #[test]
    #[ignore = "requires MP31; set APXINF_MUSA_BENCH_DEVICE"]
    fn prepared_geglu_alpha_coverage_graph_and_shape_switch() {
        let device = std::env::var("APXINF_MUSA_BENCH_DEVICE")
            .unwrap_or("0".into())
            .parse()
            .unwrap();
        let ctx = Arc::new(MusaContext::new(device).unwrap());
        if crate::MUSA_ARCH != "mp_31" || !ctx.caps().arch_family.has_native_fp8() {
            return;
        }
        let w = e4m3(&ctx, vec![1024, 8192], 13);
        prepare_fp8_weight(&ctx, &w).unwrap();
        prepare_fp8_weight(&ctx, &w).unwrap();
        let source = bytes(&w);
        {
            let weights = ctx.native_fp8_weights.lock().unwrap();
            assert_eq!(weights.entries.len(), 1);
            let packed = bytes(
                weights
                    .entries
                    .values()
                    .next()
                    .unwrap()
                    ._packed
                    .as_ref()
                    .unwrap(),
            );
            for k in 0..1024 {
                for n in 0..8192 {
                    assert_eq!(source[k * 8192 + n], packed[n * 1024 + k]);
                }
            }
        }
        for t in [10, 21, 10] {
            let a = e4m3(&ctx, vec![t, 1024], 83);
            for (alpha, scale) in [
                (0.0, 0.03125),
                (-0.37, 0.03125),
                (0.0019, 0.03125),
                (0.0, 0.0197),
                (-0.37, 0.0197),
                (0.0019, 0.0197),
            ] {
                let raw = MusaBuffer::alloc_on(&ctx, t * 4096 + 128).unwrap();
                raw.copy_from_host(&vec![0x7f; t * 4096 + 128]).unwrap();
                let mut out = raw
                    .clone()
                    .into_tensor(Shape::new(vec![t, 4096]), DType::F8E4M3);
                let mut gate = ctx
                    .allocate_output(Shape::new(vec![t, 8192]), DType::BF16)
                    .unwrap();
                let mut reference = ctx
                    .allocate_output(Shape::new(vec![t, 4096]), DType::F8E4M3)
                    .unwrap();
                super::super::fp8::gemm_fp8_into(&ctx, &a, &w, &mut gate, alpha, "GeGLU reference")
                    .unwrap();
                crate::kernels::activation::geglu_bf16_e4m3_into(
                    &ctx,
                    &gate,
                    &mut reference,
                    scale,
                    32,
                )
                .unwrap();
                let call = |out: &mut Tensor| {
                    let mut gemm = ops::GemmArgs::new(&a, &w, out);
                    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
                    gemm.alpha = alpha;
                    gemm.output_scale = scale;
                    let mut args = ops::GemmGegluArgs::new(gemm);
                    args.layout = ops::GegluWeightLayout::GateUpPairs { pair_width: 32 };
                    ops::gemm_geglu(&ctx, args)
                };
                call(&mut out).unwrap();
                ctx.synchronize().unwrap();
                let actual = bytes(&out);
                let reference = bytes(&reference);
                let decode = |b: u8| -> f64 {
                    let e = (b >> 3) & 15;
                    let f = b & 7;
                    let value = if e == 0 {
                        (f as f64) * 2f64.powi(-9)
                    } else {
                        (1.0 + f as f64 / 8.0) * 2f64.powi(e as i32 - 7)
                    };
                    if b & 128 == 0 {
                        value
                    } else {
                        -value
                    }
                };
                let mut dot = 0.;
                let mut aa = 0.;
                let mut bb = 0.;
                let mut mismatches = 0;
                for (&x, &y) in actual[..t * 4096].iter().zip(&reference) {
                    assert_ne!(x & 127, 127, "unwritten/NaN output");
                    mismatches += usize::from(x != y);
                    let x = decode(x);
                    let y = decode(y);
                    dot += x * y;
                    aa += x * x;
                    bb += y * y;
                }
                if alpha == 0. {
                    assert_eq!(aa, 0.);
                } else {
                    let cosine = dot / (aa * bb).sqrt();
                    assert!(
                        cosine >= 0.99999,
                        "t={t} alpha={alpha} scale={scale} cosine={cosine} mismatches={mismatches}"
                    );
                    eprintln!(
                        "t={t} alpha={alpha} scale={scale} cosine={cosine:.9} byte_differences={mismatches}"
                    );
                }
                assert!(
                    actual[t * 4096..].iter().all(|b| *b == 0x7f),
                    "output boundary overwritten"
                );
                let session =
                    crate::workspace::ExecutionSession::with_capacity(32 * 1024 * 1024, device)
                        .unwrap();
                session.prepare(|| call(&mut out)).unwrap();
                let graph =
                    crate::graph::capture(ctx.clone(), || session.run(|| call(&mut out))).unwrap();
                for replay in 0..1000 {
                    graph.replay().unwrap();
                    ctx.synchronize().unwrap();
                    if replay % 100 == 0 || replay == 999 {
                        assert!(bytes(&out) == actual, "replay differs from eager");
                    }
                }
            }
        }
    }
}
