//! Helpers shared by the device-gated operator tests.
//!
//! Compiled only under `cfg(test)`. These exist because every native operator
//! test needs the same four things — encode an f32 to bf16 bits, read it back,
//! put bytes on the device, and take them off — and copying them per module
//! would mean four places to fix when the encoding changes.
//!
//! Encoding is round-to-nearest-even, and the decoders used by the references
//! below consume the same bits, so a test measures the operator and not the
//! conversion. The alternative — `apxinf_core::bf16` — would work for building
//! a CPU tensor, but the device path here goes through raw bytes either way.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;

/// Round-to-nearest-even to bf16, returned as the 16 bits.
pub fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

pub fn from_bf16_bits(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Deterministic values in [-1, 1), so a failing shape can be re-run.
pub fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xffff) as f32 / 32768.0 - 1.0
        })
        .collect()
}

pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
        .collect()
}

/// Put `values` on the device as a BF16 tensor of `shape`.
pub fn upload(ctx: &MusaContext, values: &[f32], shape: Shape) -> Tensor {
    let bytes = bf16_bytes(values);
    assert_eq!(bytes.len(), shape.numel() * 2);
    let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate operand");
    buffer.copy_from_host(&bytes).expect("upload operand");
    buffer.as_tensor(shape, DType::BF16).expect("view as bf16")
}

/// A same-shaped tensor of another dtype, for the tests that check a wrong
/// operand is *refused* rather than reinterpreted.
///
/// Zeroed on purpose: the call under test must not reach a kernel, so a test
/// that starts passing because something ran is not a test that passed.
pub fn zeros_as(ctx: &MusaContext, shape: Shape, dtype: DType) -> Tensor {
    let bytes = vec![0u8; shape.numel() * dtype.size_in_bytes()];
    let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate operand");
    buffer.copy_from_host(&bytes).expect("upload operand");
    buffer
        .as_tensor(shape, dtype)
        .expect("view as the requested dtype")
}

pub fn read_back(tensor: &Tensor) -> Vec<f32> {
    let buffer = MusaBuffer::from_tensor(tensor).expect("device tensor");
    let mut bytes = vec![0u8; tensor.size_in_bytes()];
    buffer.copy_to_host(&mut bytes).expect("read back");
    bytes
        .chunks_exact(2)
        .map(|pair| from_bf16_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// The gate every operator test here uses: the largest absolute error must stay
/// under 1% of the output's own largest magnitude.
///
/// Absolute rather than relative, because these outputs are sums of many terms
/// that cancel — a relative gate reports the f32 accumulation floor as if it
/// were kernel error on the elements that land near zero. Same criterion as
/// the replay harness uses, and the muBLAS heuristic probe before it.
pub const GATE_FRACTION: f32 = 1e-2;

pub fn gate_for(output: &[f32]) -> f32 {
    output
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        * GATE_FRACTION
}

/// Run `operation` against an explicit arena, the way a captured traversal
/// does. The session resets the workspace before the closure and installs it
/// for the duration of the call.
pub fn with_workspace<T>(
    workspace: crate::workspace::GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    crate::workspace::ExecutionSession::new(workspace).run(operation)
}

/// Capture a graph and keep the value the traversal produced.
///
/// `graph::capture` records a `Result<()>` traversal; the operator tests need
/// the produced output alongside the graph to compare replays against.
pub fn capture_with_output<T>(
    context: std::sync::Arc<MusaContext>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(crate::graph::CapturedGraph, T)> {
    let output = std::cell::RefCell::new(None);
    let graph = crate::graph::capture(context, || {
        *output.borrow_mut() = Some(operation()?);
        Ok(())
    })?;
    Ok((
        graph,
        output
            .into_inner()
            .expect("a successful capture always produced a value"),
    ))
}

/// Tight numeric gates for the shared-kernel parity tests.
///
/// `GATE_FRACTION` above is an absolute 1%-of-peak bound: right for a
/// reduction-order difference in a fused kernel, and too loose to see a
/// fast-transcendental substitution. The 5b7d172 sites (`__musa_exp_f`,
/// `__musa_tanh_f`, `__sincosf`) passed the 1% gate while PI0.5 produced NaN
/// and GR00T fell to 0.9611 cosine, because a broken transcendental moves the
/// value by far less than 1% of the row maximum but by whole bf16 ulps.
///
/// These helpers compare against a host reference that reproduces the kernel's
/// own rounding chain, so a change of implementation shows up in bf16 ulp
/// distance. The bounds are calibrated on mp_22 (see the measured values the
/// callers print); a future generation whose libm differs more may need a
/// larger `TIGHT_MAX_ULP`, recorded here with the measurement that justified it.
pub const TIGHT_MIN_COSINE: f64 = 0.99999;
pub const TIGHT_MAX_REL_L2: f64 = 1e-3;
pub const TIGHT_MAX_ULP: u32 = 2;

pub fn cosine(got: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(got.len(), expected.len(), "cosine length mismatch");
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    for (got, expected) in got.iter().zip(expected) {
        let (got, expected) = (*got as f64, *expected as f64);
        dot += got * expected;
        got_norm += got * got;
        expected_norm += expected * expected;
    }
    let denominator = (got_norm * expected_norm).sqrt();
    if denominator == 0.0 {
        return if got_norm == expected_norm { 1.0 } else { 0.0 };
    }
    dot / denominator
}

pub fn relative_l2(got: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(got.len(), expected.len(), "relative L2 length mismatch");
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    for (got, expected) in got.iter().zip(expected) {
        let delta = *got as f64 - *expected as f64;
        difference += delta * delta;
        norm += (*expected as f64) * (*expected as f64);
    }
    difference.sqrt() / norm.sqrt().max(f64::MIN_POSITIVE)
}

/// Map a bf16 bit pattern onto an ordered i32 so `+x` and `-x` are adjacent
/// around zero and the distance is the number of representable bf16 values.
fn bf16_key(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7fff);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn max_bf16_ulp(got: &[f32], expected: &[f32]) -> u32 {
    assert_eq!(got.len(), expected.len(), "ulp length mismatch");
    got.iter()
        .zip(expected)
        .map(|(got, expected)| {
            (bf16_key(to_bf16_bits(*got)) - bf16_key(to_bf16_bits(*expected))).unsigned_abs()
        })
        .max()
        .unwrap_or(0)
}

pub fn assert_all_finite(what: &str, values: &[f32]) {
    if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        panic!("{what}: non-finite value {value} at index {index}");
    }
}

/// Assert a host-reference comparison at bf16 resolution, not at 1% of peak.
///
/// Prints the three measured numbers so a failing run says which bound moved
/// and by how much.
pub fn assert_tight(what: &str, got: &[f32], expected: &[f32]) {
    assert_all_finite(what, got);
    let cosine = cosine(got, expected);
    let relative_l2 = relative_l2(got, expected);
    let ulp = max_bf16_ulp(got, expected);
    println!("  {what}: cosine {cosine:.12}  rel_l2 {relative_l2:.3e}  max_ulp {ulp}");
    assert!(
        cosine >= TIGHT_MIN_COSINE,
        "{what}: cosine {cosine:.12} < {TIGHT_MIN_COSINE}"
    );
    assert!(
        relative_l2 <= TIGHT_MAX_REL_L2,
        "{what}: relative L2 {relative_l2:.3e} > {TIGHT_MAX_REL_L2}"
    );
    assert!(
        ulp <= TIGHT_MAX_ULP,
        "{what}: bf16 ulp distance {ulp} > {TIGHT_MAX_ULP}"
    );
}
