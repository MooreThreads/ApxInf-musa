//! The TileLang operator dispatch table and its launcher.
//!
//! [`generated`] — built by `build.rs` from the kernels/tilelang inventory —
//! lists every TileLang operator linked into this build's kernel static
//! library (GEMM entries today): exact-shape matching, a per-operator status
//! and unique entry symbols (`kTl<N>_*`, renamed by gen.py on the generated
//! sources). With the `tilelang` feature off the table is empty, [`serves`]
//! always answers `None` and every GEMM keeps the native route — the same
//! three-state semantics the build applies (doc/tilelang-kernels.md).
//!
//! Runtime escape hatch: `APXINF_TILELANG=off` disables every TileLang
//! operator for the run without rebuilding; the default and any other value
//! are auto. The variable is read once through a `OnceLock`, exactly as the
//! neighboring workspace knobs do, and named as a literal so the
//! backend_parity scanner's two-way check holds.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::super::contracts::tensor_ptr;
use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::context::MusaContext;

/// The part entry: `call(a, b, partials, stream)`, matching the extern "C"
/// signature of the generated sources.
pub(crate) type PartFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> i32;
/// The one-time entry: raises the device function's dynamic shared memory
/// limit and similar attributes.
pub(crate) type InitFn = unsafe extern "C" fn() -> i32;
/// The reduce entry: `call(partials, a_scale, b_scale, out, stream)`.
pub(crate) type ReduceFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void) -> i32;

/// An operator's adoption level; only `Enabled` entries dispatch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TilelangStatus {
    Enabled,
    Experimental,
    Off,
}

/// One linked TileLang GEMM (fields mirror the generated table; the readiness
/// flags make each stage's one-time init lazily executed, which is why the
/// static table carries interior mutability).
pub(crate) struct TilelangGemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub status: TilelangStatus,
    /// Bytes of split-K INT32 partials: `splits * m * n * 4`, from the
    /// inventory.
    pub scratch: usize,
    pub part: PartFn,
    pub part_init: InitFn,
    pub reduce: ReduceFn,
    pub reduce_init: InitFn,
    part_ready: AtomicBool,
    reduce_ready: AtomicBool,
}

// Flat include: the generated file refers to this module's types as
// `super::tilelang::…`, so it must sit at the tilelang module's top level,
// not in a nested submodule.
type Bf16UnaryFn = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32;

/// BF16 uses a bit-preserving, one-time weight permutation and optional FP32 reduction.
pub(crate) struct TilelangBf16Gemm {
    m: usize,
    n: usize,
    k: usize,
    status: TilelangStatus,
    scratch: usize,
    part: PartFn,
    part_init: InitFn,
    pack: Bf16UnaryFn,
    pack_init: InitFn,
    reduce: Option<Bf16UnaryFn>,
    reduce_init: Option<InitFn>,
    part_ready: AtomicBool,
    pack_ready: AtomicBool,
    reduce_ready: AtomicBool,
}

include!(concat!(env!("OUT_DIR"), "/tilelang_kernels.rs"));

/// Whether `value` explicitly disables the route (only "off", case
/// insensitive).
fn is_off(value: Option<&str>) -> bool {
    value.is_some_and(|value| value.eq_ignore_ascii_case("off"))
}

/// The runtime policy, read once per process like the neighboring workspace
/// knobs — no allocation or environment lock on the dispatch path.
fn policy_off() -> bool {
    static POLICY: OnceLock<bool> = OnceLock::new();
    *POLICY.get_or_init(|| is_off(std::env::var("APXINF_TILELANG").ok().as_deref()))
}

/// Whether an Enabled entry serves this exact shape.
pub(crate) fn serves(m: usize, n: usize, k: usize) -> Option<&'static TilelangGemm> {
    if policy_off() {
        return None;
    }
    TILELANG_GEMMS.iter().find(|entry| {
        entry.m == m && entry.n == n && entry.k == k && entry.status == TilelangStatus::Enabled
    })
}

/// Whether an Enabled entry serves `(n, k)` at any decode batch: weights are
/// prepacked before any batch is known. The model side's KN prepack keys on
/// this; the policy matches [`serves`].
pub(super) fn serves_gemm(n: usize, k: usize) -> bool {
    if policy_off() {
        return false;
    }
    TILELANG_GEMMS
        .iter()
        .any(|entry| entry.n == n && entry.k == k && entry.status == TilelangStatus::Enabled)
}

/// Runs one TileLang GEMM (part + reduce).
///
/// Records the same graph-shape ledger entry the native route would; the
/// split-K partials come from the workspace arena. Each stage's one-time init
/// (dynamic shared memory attributes) runs under its readiness flag, which is
/// published only after the init succeeded: a concurrent first call either
/// runs the idempotent init itself or observes the flag after success, and a
/// failed init stays unpublished so the next call retries it.
pub(super) fn launch(
    ctx: &MusaContext,
    entry: &TilelangGemm,
    a: *mut c_void,
    b: *mut c_void,
    a_scale: *mut c_void,
    b_scale: *mut c_void,
    out: *mut c_void,
) -> Result<()> {
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: entry.m,
        n: entry.n,
        k: entry.k,
    })?;
    let scratch = crate::workspace::output_buffer(ctx, entry.scratch)?;
    let stream = ctx.stream().handle();
    // SAFETY: the symbols resolved at link time (kernel static library); the
    // pointers come from w8a8's dispatch block, validated against the plan,
    // and the scratch is an arena slice this function just allocated. The
    // calls run asynchronously on ctx's stream and every buffer outlives
    // them.
    unsafe {
        if !entry.part_ready.load(Ordering::Acquire) {
            if (entry.part_init)() != 0 {
                return Err(Error::Other(
                    "MUSA gemm::w8a8 TileLang part init failed".into(),
                ));
            }
            entry.part_ready.store(true, Ordering::Release);
        }
        if (entry.part)(a, b, scratch.ptr(), stream) != 0 {
            return Err(Error::Other("MUSA gemm::w8a8 TileLang part failed".into()));
        }
        if !entry.reduce_ready.load(Ordering::Acquire) {
            if (entry.reduce_init)() != 0 {
                return Err(Error::Other(
                    "MUSA gemm::w8a8 TileLang reduce init failed".into(),
                ));
            }
            entry.reduce_ready.store(true, Ordering::Release);
        }
        if (entry.reduce)(scratch.ptr(), a_scale, b_scale, out, stream) != 0 {
            return Err(Error::Other(
                "MUSA gemm::w8a8 TileLang reduce failed".into(),
            ));
        }
    }
    Ok(())
}

fn serves_bf16(m: usize, n: usize, k: usize) -> Option<&'static TilelangBf16Gemm> {
    if policy_off() {
        return None;
    }
    TILELANG_BF16_GEMMS
        .iter()
        .find(|op| (op.m, op.n, op.k) == (m, n, k) && op.status == TilelangStatus::Enabled)
}

fn initialize(init: InitFn, ready: &AtomicBool) -> Result<()> {
    if !ready.load(Ordering::Acquire) {
        if unsafe { init() } != 0 {
            return Err(Error::Other("BF16 TileLang init failed".into()));
        }
        ready.store(true, Ordering::Release);
    }
    Ok(())
}

/// Called by the existing immutable BF16 weight preparation entry points.
/// The assembly registration retains the source, preventing pointer reuse.
pub(super) fn prepare_bf16_weight(ctx: &MusaContext, weight: &Tensor) -> Result<()> {
    let dims = weight.shape().dims();
    if dims.len() != 2 {
        return Ok(());
    }
    let (k, n) = (dims[0], dims[1]);
    let Some(op) = serves_bf16(10, n, k) else {
        return Ok(());
    };
    if crate::workspace::is_capturing() {
        return Err(Error::Other(
            "prepare BF16 TileLang weights before graph capture".into(),
        ));
    }
    let mut weights = ctx
        .asm_weights
        .lock()
        .map_err(|_| Error::Other("BF16 prepared-weight lock poisoned".into()))?;
    let source = tensor_ptr(weight)?;
    let key = (source as usize, k, n);
    if weights.tilelang.contains_key(&key) {
        return Ok(());
    }
    initialize(op.pack_init, &op.pack_ready)?;
    let packed = crate::MusaBuffer::alloc_on(ctx, k * n * 2)
        .map_err(crate::error::musa_error)?
        .into_tensor(Shape::new(vec![k * n]), DType::BF16);
    if unsafe { (op.pack)(source, tensor_ptr(&packed)?, ctx.stream().handle()) } != 0 {
        return Err(Error::Other(
            "BF16 TileLang weight preparation failed".into(),
        ));
    }
    weights.tilelang.insert(key, packed);
    Ok(())
}

/// Exact-shape BF16 decode dispatch; unsupported or disabled entries stay native.
pub(super) fn launch_bf16(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    let Some(op) = serves_bf16(m, n, k) else {
        return Ok(false);
    };
    let key = (tensor_ptr(weight)? as usize, k, n);
    let packed = ctx
        .asm_weights
        .lock()
        .map_err(|_| Error::Other("BF16 prepared-weight lock poisoned".into()))?
        .tilelang
        .get(&key)
        .cloned()
        .ok_or_else(|| {
            Error::Other("BF16 TileLang weight was not prepared at model load".into())
        })?;
    initialize(op.part_init, &op.part_ready)?;
    let scratch = if op.reduce.is_some() {
        Some(crate::workspace::output_buffer(ctx, op.scratch)?)
    } else {
        None
    };
    let destination = if let Some(buffer) = &scratch {
        buffer.ptr()
    } else {
        tensor_ptr(output)?
    };
    let stream = ctx.stream().handle();
    if unsafe {
        (op.part)(
            tensor_ptr(activation)?,
            tensor_ptr(&packed)?,
            destination,
            stream,
        )
    } != 0
    {
        return Err(Error::Other("BF16 TileLang GEMM failed".into()));
    }
    if let (Some(reduce), Some(init)) = (op.reduce, op.reduce_init) {
        initialize(init, &op.reduce_ready)?;
        if unsafe { reduce(destination, tensor_ptr(output)?, stream) } != 0 {
            return Err(Error::Other("BF16 TileLang reduction failed".into()));
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_explicit_off_disables() {
        assert!(!is_off(None));
        assert!(!is_off(Some("auto")));
        assert!(!is_off(Some("")));
        assert!(!is_off(Some("1")));
        assert!(is_off(Some("off")));
        assert!(is_off(Some("OFF")));
        assert!(is_off(Some("Off")));
    }

    #[test]
    fn unknown_shapes_never_serve() {
        // Holds with an empty table (feature off) and a populated one
        // (feature on): shapes outside the whitelist must stay native.
        assert!(serves(1, 2, 3).is_none());
        assert!(serves(10, 1024, 8192).is_none());
    }

    #[test]
    fn linked_entries_carry_positive_scratch() {
        // Integrity of the generated table itself: every linked entry must
        // declare positive scratch (`splits * m * n * 4`), else launch would
        // ask the workspace for zero bytes.
        for entry in TILELANG_GEMMS.iter() {
            assert!(
                entry.scratch > 0,
                "entry {}x{}x{} has no scratch",
                entry.m,
                entry.n,
                entry.k
            );
        }
    }

    /// Numerical regression: every linked entry must reproduce a CPU
    /// reference bit for bit. Integer GEMM accumulates exactly in INT32, so
    /// the split order cannot matter; the scales are dyadic so fp16 carries
    /// them exactly, and the kernel's fp32 chain (acc * a_scale, then *
    /// b_scale, then one bf16 round) is reproducible on the host.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn linked_entries_match_a_cpu_reference_bit_for_bit() {
        use half::{bf16, f16};

        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for entry in TILELANG_GEMMS.iter() {
            let (m, n, k) = (entry.m, entry.n, entry.k);

            // Deterministic operands: xorshift ints in [-100, 100) and
            // dyadic fp16 scales (small_int / 2^10 carries exactly).
            let mut state = 0x2545_F491u32;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            };
            let a: Vec<i8> = (0..m * k).map(|_| (next() % 201) as i32 as i8).collect();
            let b: Vec<i8> = (0..k * n).map(|_| (next() % 201) as i32 as i8).collect();
            let a_scale: Vec<f32> = (0..m).map(|i| (64.0 + (i % 64) as f32) / 1024.0).collect();
            let b_scale: Vec<f32> = (0..n).map(|j| (48.0 + (j % 48) as f32) / 1024.0).collect();

            // The same upload pattern the w8a8 device tests use: raw
            // allocation plus a host copy; scale buffers carry the 512-entry
            // zero pad production packs (the kernel reads only [0, len)).
            let upload_i8 = |ctx: &MusaContext, values: &[i8]| {
                let bytes: Vec<u8> = values.iter().map(|v| *v as u8).collect();
                let buffer = crate::MusaBuffer::alloc_on(ctx, bytes.len()).expect("alloc i8");
                buffer.copy_from_host(&bytes).expect("upload i8");
                buffer
            };
            let upload_f16 = |ctx: &MusaContext, values: &[f32]| {
                let mut bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                    .collect();
                bytes.resize(bytes.len() + 512 * 2, 0);
                let buffer = crate::MusaBuffer::alloc_on(ctx, bytes.len()).expect("alloc f16");
                buffer.copy_from_host(&bytes).expect("upload f16");
                buffer
            };

            let a_buf = upload_i8(&ctx, &a);
            let b_buf = upload_i8(&ctx, &b);
            let a_scale_buf = upload_f16(&ctx, &a_scale);
            let b_scale_buf = upload_f16(&ctx, &b_scale);
            let out = crate::MusaBuffer::alloc_on(&ctx, m * n * 2).expect("alloc output");

            // The session contract: a shape must be seen by a prepare
            // traversal before it runs, so the launch lives inside one —
            // which also executes it, warming the kernels like production.
            let arena = crate::workspace::GraphWorkspace::new(8 << 20, 0).expect("workspace");
            let session = crate::workspace::ExecutionSession::new(arena);
            session
                .prepare(|| {
                    launch(
                        &ctx,
                        entry,
                        a_buf.ptr(),
                        b_buf.ptr(),
                        a_scale_buf.ptr(),
                        b_scale_buf.ptr(),
                        out.ptr(),
                    )
                })
                .unwrap();
            ctx.synchronize().unwrap();

            let mut got = vec![0u8; m * n * 2];
            out.copy_to_host(&mut got).unwrap();
            let got_bits: Vec<u16> = got
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            for idx in 0..m * n {
                let (row, col) = (idx / n, idx % n);
                let acc: i64 = (0..k)
                    .map(|j| a[row * k + j] as i64 * b[j * n + col] as i64)
                    .sum();
                // The kernel's chain: (acc as f32 * a_scale) * b_scale, then
                // one round to bf16 — left-associative here matches it.
                let scaled = (acc as i32) as f32 * a_scale[row] * b_scale[col];
                let want = bf16::from_f32(scaled).to_bits();
                assert_eq!(
                    got_bits[idx], want,
                    "entry {}x{}x{} at row={row} col={col}",
                    entry.m, entry.n, entry.k
                );
            }
        }
    }
}
