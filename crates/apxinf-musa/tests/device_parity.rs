//! Tight device parity for the shared kernels behind the 5b7d172 regression.
//!
//! The kernel-level tests in `src/kernels/{activation,attention}.rs` compare
//! against a host reference at 1%-of-peak, which is the right bound for a
//! reduction-order difference and too loose for a changed transcendental. These
//! exercise the same arithmetic through the typed `ops` surface at bf16
//! resolution: finite, cosine, relative L2 and bf16 ulp distance.
//!
//! The RoPE path is covered by `kernels::rope`'s tests, whose host reference
//! reproduces the full rounding chain and now carries the same tight assertion.
//!
//! Needs a MUSA device, so every case is `#[ignore]`d:
//! `cargo test -p apxinf-musa --test device_parity -- --ignored --test-threads=1 --nocapture`.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_musa::{ops, transfers, MusaContext};
use half::bf16;

// Keep in step with `test_support::{TIGHT_MIN_COSINE, TIGHT_MAX_REL_L2,
// TIGHT_MAX_ULP}`; an integration test cannot import the lib's `cfg(test)` module.
const MIN_COSINE: f64 = 0.99999;
const MAX_REL_L2: f64 = 1e-3;
const MAX_ULP: u32 = 2;

/// Deterministic values in [-1, 1), the same LCG every kernel test uses.
fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xffff) as f32 / 32768.0 - 1.0
        })
        .collect()
}

fn rounded(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn upload(ctx: &MusaContext, dims: &[usize], values: &[f32]) -> Tensor {
    let values: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
    let host = Tensor::from_bf16(dims.to_vec(), &values).expect("host tensor");
    transfers::to_device(&host, ctx.device_id()).expect("upload tensor")
}

fn output(ctx: &MusaContext, dims: &[usize]) -> Tensor {
    ctx.allocate_output(Shape::new(dims.to_vec()), DType::BF16)
        .expect("allocate output")
}

fn read(ctx: &MusaContext, tensor: &Tensor) -> Vec<f32> {
    ctx.synchronize().expect("synchronise");
    transfers::to_cpu(tensor)
        .expect("read back")
        .as_bf16()
        .expect("bf16 values")
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

fn cosine(got: &[f32], expected: &[f32]) -> f64 {
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

fn relative_l2(got: &[f32], expected: &[f32]) -> f64 {
    let mut difference = 0.0f64;
    let mut norm = 0.0f64;
    for (got, expected) in got.iter().zip(expected) {
        let delta = *got as f64 - *expected as f64;
        difference += delta * delta;
        norm += (*expected as f64) * (*expected as f64);
    }
    difference.sqrt() / norm.sqrt().max(f64::MIN_POSITIVE)
}

fn bf16_key(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7fff);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn max_bf16_ulp(got: &[f32], expected: &[f32]) -> u32 {
    got.iter()
        .zip(expected)
        .map(|(got, expected)| {
            (bf16_key(bf16::from_f32(*got).to_bits())
                - bf16_key(bf16::from_f32(*expected).to_bits()))
            .unsigned_abs()
        })
        .max()
        .unwrap_or(0)
}

/// The bound every case here uses; a generation whose libm differs more would
/// need a larger `MAX_ULP`, recorded with the measurement that justified it.
fn assert_tight(what: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    if let Some((index, value)) = got.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        panic!("{what}: non-finite value {value} at index {index}");
    }
    let cosine = cosine(got, expected);
    let relative_l2 = relative_l2(got, expected);
    let ulp = max_bf16_ulp(got, expected);
    println!("  {what}: cosine {cosine:.12}  rel_l2 {relative_l2:.3e}  max_ulp {ulp}");
    assert!(cosine >= MIN_COSINE, "{what}: cosine {cosine:.12} < {MIN_COSINE}");
    assert!(
        relative_l2 <= MAX_REL_L2,
        "{what}: relative L2 {relative_l2:.3e} > {MAX_REL_L2}"
    );
    assert!(ulp <= MAX_ULP, "{what}: bf16 ulp distance {ulp} > {MAX_ULP}");
}

#[test]
#[ignore = "needs a MUSA device"]
fn pointwise_silu_matches_a_tight_reference() {
    let ctx = MusaContext::new(0).expect("MUSA device 0");
    for (rows, cols) in [(7usize, 320usize), (3, 5)] {
        let input_values = values(rows * cols, 0x5eed_1234);
        let input = upload(&ctx, &[rows, cols], &input_values);
        let mut out = output(&ctx, &[rows, cols]);
        let mut args =
            ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, &input, &mut out);
        args.activation = ops::PointwiseActivation::Silu;
        ops::pointwise(&ctx, args).expect("ops silu");
        let got = read(&ctx, &out);
        let expected: Vec<f32> = input_values
            .iter()
            .map(|value| {
                let value = rounded(*value);
                rounded(value / (1.0 + (-value).exp()))
            })
            .collect();
        assert_tight(&format!("ops silu [{rows}x{cols}]"), &got, &expected);
    }
}

#[test]
#[ignore = "needs a MUSA device"]
fn pointwise_geglu_matches_a_tight_reference() {
    let ctx = MusaContext::new(0).expect("MUSA device 0");
    let alpha = 0.7978845608028654f32;
    let gelu = |value: f32| {
        let value = rounded(value);
        0.5 * value * (1.0 + (alpha * (value + 0.044715 * value * value * value)).tanh())
    };
    for (rows, cols) in [(7usize, 320usize), (3, 10)] {
        let input_values = values(rows * 2 * cols, 0x0bad_c0de);
        let input = upload(&ctx, &[rows, 2 * cols], &input_values);
        let mut out = output(&ctx, &[rows, cols]);
        ops::pointwise(
            &ctx,
            ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, &input, &mut out),
        )
        .expect("ops geglu");
        let got = read(&ctx, &out);
        let mut expected = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let gate = input_values[row * 2 * cols + col];
                let up = rounded(input_values[row * 2 * cols + cols + col]);
                expected.push(rounded(gelu(gate) * up));
            }
        }
        assert_tight(&format!("ops geglu [{rows}x{cols}]"), &got, &expected);
    }
}

#[test]
#[ignore = "needs a MUSA device"]
fn attention_matches_a_host_reference() {
    let ctx = MusaContext::new(0).expect("MUSA device 0");
    let (tokens, heads, head_dim) = (8usize, 4usize, 72usize);
    let width = tokens * heads * head_dim;
    let q_values = values(width, 0x5eed_1234);
    let k_values = values(width, 0x0bad_c0de);
    let v_values = values(width, 0x1357_9bdf);
    let q = upload(&ctx, &[1, tokens, heads, head_dim], &q_values);
    let k = upload(&ctx, &[1, tokens, heads, head_dim], &k_values);
    let v = upload(&ctx, &[1, tokens, heads, head_dim], &v_values);
    let mut out = output(&ctx, &[1, tokens, heads, head_dim]);
    ops::attention(&ctx, ops::AttentionArgs::new(&q, &k, &v, &mut out)).expect("ops attention");
    let got = read(&ctx, &out);

    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut expected = vec![0.0f32; got.len()];
    for token in 0..tokens {
        for head in 0..heads {
            let query_base = (token * heads + head) * head_dim;
            let mut scores = vec![0.0f64; tokens];
            for key_token in 0..tokens {
                let key_base = (key_token * heads + head) * head_dim;
                let mut dot = 0.0f64;
                for dim in 0..head_dim {
                    dot += f64::from(rounded(q_values[query_base + dim]))
                        * f64::from(rounded(k_values[key_base + dim]));
                }
                scores[key_token] = dot * scale;
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for dim in 0..head_dim {
                let mut accumulated = 0.0f64;
                for key_token in 0..tokens {
                    let value_base = (key_token * heads + head) * head_dim;
                    accumulated += exps[key_token] / sum
                        * f64::from(rounded(v_values[value_base + dim]));
                }
                expected[query_base + dim] = rounded(accumulated as f32);
            }
        }
    }
    // The MHA path rounds probabilities to BF16 on some routes; the measured
    // distance on both mp_22 and mp_31 is 0 ulp, so the default bound holds.
    assert_tight("ops attention", &got, &expected);
}
