//! Native FP8 (E4M3) GEMM.
//!
//! `[M, K] @ [K, N] -> [M, N]` with E4M3 operands, f32 accumulation, one f32
//! `alpha` applied in the epilogue and one rounding to BF16. `mp_31` builds
//! publish the muDNN provider; every other generation publishes `unsupported`.
//!
//! Layout: row-major `[M, K] @ [K, N] -> [M, N]`. `mudnn_adapter.mu` sets the
//! descriptors with `SetNdInfo({m, k})`, `{k, n}` and `{m, n}` and
//! `SetTranspose(false, false)`, so `weight` is the model's canonical `[K, N]`
//! row-major tensor and nothing is transposed in memory.
//!
//! `mudnn_adapter.mu` caches one descriptor per `(m, n, k)` and rewrites
//! `alpha` on every call. The descriptor is created on first use outside
//! capture, which is why the prepare traversal has to warm every shape first.

use apxinf_core::{DType, Error, Result, Tensor};

use super::super::contracts::require_device;
#[cfg(apxinf_musa_available)]
use super::super::contracts::{checked_i32, tensor_ptr};
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
use crate::MusaContext;

pub(crate) fn require_contraction(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    operation: &str,
) -> Result<(usize, usize, usize)> {
    require_device(ctx, activation, operation, "activation")?;
    require_device(ctx, weight, operation, "weight")?;
    require_device(ctx, output, operation, "out")?;
    let (rows, inner) = matrix_shape(activation, operation, "activation")?;
    let (weight_inner, cols) = matrix_shape(weight, operation, "weight")?;
    if inner != weight_inner {
        return Err(Error::Other(format!(
            "MUSA {operation} contraction mismatch: [{rows}x{inner}] @ [{weight_inner}x{cols}]"
        )));
    }
    if output.shape().dims() != [rows, cols] {
        return Err(Error::Other(format!(
            "MUSA {operation} out expects [{rows}, {cols}], got {:?}",
            output.shape().dims()
        )));
    }
    if output.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA {operation} out expects BF16, got {}",
            output.dtype()
        )));
    }
    Ok((rows, inner, cols))
}

fn matrix_shape(tensor: &Tensor, operation: &str, name: &str) -> Result<(usize, usize)> {
    if tensor.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects F8E4M3, got {}",
            tensor.dtype()
        )));
    }
    match tensor.shape().dims() {
        [rows, cols] if *rows > 0 && *cols > 0 => Ok((*rows, *cols)),
        dims => Err(Error::Other(format!(
            "MUSA {operation} {name} expects a non-empty 2-D tensor, got {dims:?}"
        ))),
    }
}

/// Check that this build and device have the native FP8 route.
///
/// `Error::Other` rather than `UnsupportedOp`: the message needs the running
/// device name and the published route, which `UnsupportedOp(&'static str)`
/// cannot carry, and no caller dispatches on the variant.
pub(crate) fn require_supported(ctx: &MusaContext, operation: &str) -> Result<()> {
    if crate::FP8_ROUTE != "mudnn" || !ctx.caps().arch_family.has_native_fp8() {
        return Err(Error::Other(format!(
            "MUSA {operation}: native FP8 GEMM is unavailable on {} (build route {}, requires mp_31)",
            ctx.caps().device_name,
            crate::FP8_ROUTE
        )));
    }
    Ok(())
}

pub fn gemm_fp8_into(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    alpha: f32,
    operation: &str,
) -> Result<()> {
    require_supported(ctx, operation)?;
    if !alpha.is_finite() {
        return Err(Error::Other(format!(
            "MUSA {operation} alpha must be finite, got {alpha}"
        )));
    }
    let (rows, inner, cols) = require_contraction(ctx, activation, weight, output, operation)?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: rows,
        n: cols,
        k: inner,
    })?;
    launch(ctx, activation, weight, output, rows, inner, cols, alpha)
}

fn require_bias(ctx: &MusaContext, bias: &Tensor, cols: usize, operation: &str) -> Result<()> {
    require_device(ctx, bias, operation, "bias")?;
    if bias.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA {operation} bias expects BF16, got {}",
            bias.dtype()
        )));
    }
    if bias.shape().dims() != [cols] {
        return Err(Error::Other(format!(
            "MUSA {operation} bias expects [{cols}], got {:?}",
            bias.shape().dims()
        )));
    }
    Ok(())
}

fn require_residual(
    ctx: &MusaContext,
    residual: &Tensor,
    rows: usize,
    cols: usize,
    operation: &str,
) -> Result<()> {
    require_device(ctx, residual, operation, "residual")?;
    if residual.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA {operation} residual expects BF16, got {}",
            residual.dtype()
        )));
    }
    if residual.shape().dims() != [rows, cols] {
        return Err(Error::Other(format!(
            "MUSA {operation} residual expects [{rows}, {cols}], got {:?}",
            residual.shape().dims()
        )));
    }
    Ok(())
}

/// `[M, K] @ [K, N] + bias -> [M, N]` through muDNN's bias epilogue.
///
/// `bias` is BF16 `[N]`; the output stays row-major BF16. The fused call does
/// the bias add in f32 inside the GEMM epilogue, so it replaces a separate
/// `bias_bf16` launch and its read/write of the `[M, N]` output.
pub fn gemm_fp8_bias_into(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    bias: &Tensor,
    alpha: f32,
    operation: &str,
) -> Result<()> {
    require_supported(ctx, operation)?;
    if !alpha.is_finite() {
        return Err(Error::Other(format!(
            "MUSA {operation} alpha must be finite, got {alpha}"
        )));
    }
    let (rows, inner, cols) = require_contraction(ctx, activation, weight, output, operation)?;
    require_bias(ctx, bias, cols, operation)?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: rows,
        n: cols,
        k: inner,
    })?;
    launch_bias(ctx, activation, weight, output, bias, rows, inner, cols, alpha)
}

/// `[M, K] @ [K, N] + residual + bias -> [M, N]` through muDNN's bias and
/// residual epilogue. `residual` is BF16 `[M, N]`.
pub fn gemm_fp8_bias_residual_into(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    bias: &Tensor,
    residual: &Tensor,
    alpha: f32,
    operation: &str,
) -> Result<()> {
    require_supported(ctx, operation)?;
    if !alpha.is_finite() {
        return Err(Error::Other(format!(
            "MUSA {operation} alpha must be finite, got {alpha}"
        )));
    }
    let (rows, inner, cols) = require_contraction(ctx, activation, weight, output, operation)?;
    require_bias(ctx, bias, cols, operation)?;
    require_residual(ctx, residual, rows, cols, operation)?;
    crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
        m: rows,
        n: cols,
        k: inner,
    })?;
    launch_bias_residual(
        ctx, activation, weight, output, bias, residual, rows, inner, cols, alpha,
    )
}

#[cfg(apxinf_musa_available)]
fn launch(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    rows: usize,
    inner: usize,
    cols: usize,
    alpha: f32,
) -> Result<()> {
    let m = checked_i32("gemm_fp8", "rows", rows)?;
    let n = checked_i32("gemm_fp8", "cols", cols)?;
    let k = checked_i32("gemm_fp8", "inner", inner)?;
    // SAFETY: the contraction, dtype and device checks above establish that
    // every pointer is a live E4M3 or BF16 buffer on this context's device;
    // the adapter keys its descriptor by shape and the launch is asynchronous.
    unsafe {
        ffi::custom::check_mudnn(
            ffi::custom::apxinf_musa_mudnn_fp8_gemm(
                tensor_ptr(activation)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                m,
                n,
                k,
                alpha,
                ctx.stream().handle(),
            ),
            || format!("muDNN FP8 GEMM {rows}x{inner}x{cols}"),
        )
        .map_err(musa_error)
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _inner: usize,
    _cols: usize,
    _alpha: f32,
) -> Result<()> {
    Err(crate::kernels::native_unavailable("gemm_fp8"))
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_bias(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    bias: &Tensor,
    rows: usize,
    inner: usize,
    cols: usize,
    alpha: f32,
) -> Result<()> {
    let m = checked_i32("gemm_fp8_bias", "rows", rows)?;
    let n = checked_i32("gemm_fp8_bias", "cols", cols)?;
    let k = checked_i32("gemm_fp8_bias", "inner", inner)?;
    // SAFETY: the contraction, dtype and device checks above establish that
    // every pointer is a live E4M3 or BF16 buffer on this context's device;
    // the adapter keys its descriptor by shape and the launch is asynchronous.
    unsafe {
        ffi::custom::check_mudnn(
            ffi::custom::apxinf_musa_mudnn_fp8_gemm_bias(
                tensor_ptr(activation)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                tensor_ptr(bias)?,
                m,
                n,
                k,
                alpha,
                ctx.stream().handle(),
            ),
            || format!("muDNN FP8 GEMM+bias {rows}x{inner}x{cols}"),
        )
        .map_err(musa_error)
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_bias(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _bias: &Tensor,
    _rows: usize,
    _inner: usize,
    _cols: usize,
    _alpha: f32,
) -> Result<()> {
    Err(crate::kernels::native_unavailable("gemm_fp8_bias"))
}

#[cfg(apxinf_musa_available)]
#[allow(clippy::too_many_arguments)]
fn launch_bias_residual(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
    rows: usize,
    inner: usize,
    cols: usize,
    alpha: f32,
) -> Result<()> {
    let m = checked_i32("gemm_fp8_bias_residual", "rows", rows)?;
    let n = checked_i32("gemm_fp8_bias_residual", "cols", cols)?;
    let k = checked_i32("gemm_fp8_bias_residual", "inner", inner)?;
    // SAFETY: as above; the residual is BF16 `[rows, cols]` and the bias BF16
    // `[cols]`, both on this context's device.
    unsafe {
        ffi::custom::check_mudnn(
            ffi::custom::apxinf_musa_mudnn_fp8_gemm_bias_residual(
                tensor_ptr(activation)?,
                tensor_ptr(weight)?,
                tensor_ptr(output)?,
                tensor_ptr(bias)?,
                tensor_ptr(residual)?,
                m,
                n,
                k,
                alpha,
                ctx.stream().handle(),
            ),
            || format!("muDNN FP8 GEMM+bias+residual {rows}x{inner}x{cols}"),
        )
        .map_err(musa_error)
    }
}

#[cfg(not(apxinf_musa_available))]
#[allow(clippy::too_many_arguments)]
fn launch_bias_residual(
    _ctx: &MusaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _output: &Tensor,
    _bias: &Tensor,
    _residual: &Tensor,
    _rows: usize,
    _inner: usize,
    _cols: usize,
    _alpha: f32,
) -> Result<()> {
    Err(crate::kernels::native_unavailable(
        "gemm_fp8_bias_residual",
    ))
}

/// Skip a device test when this build/device has no native FP8 route.
///
/// `#[ignore]` means "needs hardware", not "this board has the route"; M1000
/// (mp_22) has no E4M3 MMA, so a test that unwrapped the vendor call would
/// panic there instead of reporting a skip.
#[cfg(all(test, apxinf_musa_available))]
fn skip_without_native_fp8(ctx: &MusaContext, test: &str) -> bool {
    match require_supported(ctx, test) {
        Ok(()) => false,
        Err(reason) => {
            eprintln!("skipping {test}: {reason}");
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{Shape, Tensor};

    fn fp8(shape: [usize; 2]) -> Tensor {
        Tensor::from_f8_e4m3(shape.to_vec(), &vec![0u8; shape[0] * shape[1]]).unwrap()
    }

    #[test]
    fn cpu_tensors_are_rejected() {
        let ctx = match MusaContext::new(0) {
            Ok(ctx) => ctx,
            Err(_) => return,
        };
        let activation = fp8([4, 8]);
        let weight = fp8([8, 16]);
        let mut output = Tensor::zeros(Shape::new(vec![4, 16]), DType::BF16);
        // The shapes agree, so this asserts the device contract: host tensors
        // never reach a vendor call, whether or not this build has a route.
        let error = gemm_fp8_into(&ctx, &activation, &weight, &mut output, 1.0, "test");
        assert!(error.is_err());
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod device_tests {
    //! Device tests for the vendor FP8 GEMM.
    //!
    //! The reference decodes the exact E4M3 bytes and accumulates in f64, then
    //! applies the same `alpha`; the output is BF16, so the comparison is a
    //! cosine rather than byte equality. The shapes include `k = 9`, which is
    //! not a tensor-core multiple, because the vendor path must accept the
    //! model's odd contractions rather than silently pad them.

    use super::*;
    use apxinf_core::Shape;

    fn decode_e4m3(byte: u8) -> f64 {
        let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = (byte >> 3) & 0x0f;
        let mantissa = byte & 0x07;
        if exponent == 0x0f {
            if mantissa == 0x07 {
                return f64::NAN;
            }
            return sign * (1.0 + mantissa as f64 / 8.0) * 2f64.powi(8);
        }
        if exponent == 0 {
            return sign * (mantissa as f64) * 2f64.powi(-9);
        }
        sign * (1.0 + mantissa as f64 / 8.0) * 2f64.powi(exponent as i32 - 7)
    }

    fn next_random(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    fn finite_byte(state: &mut u32) -> u8 {
        let mut byte = (next_random(state) >> 24) as u8;
        if byte & 0x7f == 0x7f {
            byte = (byte & 0x80) | 0x7e;
        }
        byte
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fp8_gemm_matches_a_double_reference() {
        let ctx = MusaContext::new(0).unwrap();
        if skip_without_native_fp8(&ctx, "fp8_gemm_matches_a_double_reference") {
            return;
        }
        for (m, n, k) in [(4usize, 8usize, 16usize), (10, 32, 64), (33, 17, 9)] {
            let mut state = 0x51ed270bu32 ^ (m * 131 + n * 17 + k) as u32;
            let a_host: Vec<u8> = (0..m * k).map(|_| finite_byte(&mut state)).collect();
            let b_host: Vec<u8> = (0..k * n).map(|_| finite_byte(&mut state)).collect();
            let a = crate::transfers::to_device(
                &Tensor::from_f8_e4m3(vec![m, k], &a_host).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let b = crate::transfers::to_device(
                &Tensor::from_f8_e4m3(vec![k, n], &b_host).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let mut out = ctx
                .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            let alpha = 0.5f32;
            gemm_fp8_into(&ctx, &a, &b, &mut out, alpha, "test").unwrap();
            ctx.synchronize().unwrap();
            let got = crate::transfers::to_cpu(&out)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let mut dot = 0.0;
            let mut got_norm = 0.0;
            let mut want_norm = 0.0;
            for row in 0..m {
                for col in 0..n {
                    let want = alpha as f64
                        * (0..k)
                            .map(|inner| {
                                decode_e4m3(a_host[row * k + inner])
                                    * decode_e4m3(b_host[inner * n + col])
                            })
                            .sum::<f64>();
                    let value = got[row * n + col] as f64;
                    dot += value * want;
                    got_norm += value * value;
                    want_norm += want * want;
                }
            }
            let cosine = dot / (got_norm.sqrt() * want_norm.sqrt());
            assert!(
                cosine >= 0.9999,
                "{m}x{n}x{k}: cosine {cosine} is below the FP8 GEMM gate"
            );
        }
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn malformed_contractions_are_rejected_before_the_vendor_call() {
        let ctx = MusaContext::new(0).unwrap();
        if skip_without_native_fp8(
            &ctx,
            "malformed_contractions_are_rejected_before_the_vendor_call",
        ) {
            return;
        }
        let device =
            |tensor: Tensor| crate::transfers::to_device(&tensor, ctx.device_id()).unwrap();
        let activation = device(Tensor::from_f8_e4m3(vec![4, 8], &[0x38u8; 32]).unwrap());
        let weight = device(Tensor::from_f8_e4m3(vec![8, 16], &[0x38u8; 128]).unwrap());
        let mut output = ctx
            .allocate_output(Shape::new(vec![4, 16]), DType::BF16)
            .unwrap();

        let wrong_inner = device(Tensor::from_f8_e4m3(vec![9, 16], &[0x38u8; 144]).unwrap());
        let error = gemm_fp8_into(&ctx, &activation, &wrong_inner, &mut output, 1.0, "test")
            .expect_err("a mismatched contraction must be refused");
        assert!(error.to_string().contains("contraction"), "{error}");

        let bf16_activation = device(Tensor::zeros(Shape::new(vec![4, 8]), DType::BF16));
        let error = gemm_fp8_into(&ctx, &bf16_activation, &weight, &mut output, 1.0, "test")
            .expect_err("a BF16 activation must be refused by the FP8 entry point");
        assert!(error.to_string().contains("expects F8E4M3"), "{error}");

        let mut wrong_out = ctx
            .allocate_output(Shape::new(vec![4, 16]), DType::F8E4M3)
            .unwrap();
        let error = gemm_fp8_into(&ctx, &activation, &weight, &mut wrong_out, 1.0, "test")
            .expect_err("an E4M3 output must be refused");
        assert!(error.to_string().contains("out expects BF16"), "{error}");
    }

    /// The four action-expert shapes and the aligned prefix shapes, timed at the
    /// `gemm_fp8_into` entry point. Warm L2 is one weight buffer used back to
    /// back; cold L2 round-robins enough copies that the working set exceeds the
    /// 60 MiB L2, which is what a layer-step sees when it re-reads weights a
    /// step later. Run with
    /// `APXINF_MUSA_BENCH_DEVICE=4 cargo test -p apxinf-musa --release --lib -- --ignored --nocapture bench_fp8_gemm_shapes`.
    /// `APXINF_MUSA_BENCH_COLD=1` selects the cold variant.
    #[test]
    #[ignore = "requires a MUSA device; set APXINF_MUSA_BENCH_DEVICE"]
    fn bench_fp8_gemm_shapes() {
        use std::time::Instant;

        let device = std::env::var("APXINF_MUSA_BENCH_DEVICE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let cold = std::env::var("APXINF_MUSA_BENCH_COLD").is_ok_and(|value| value == "1");
        let ctx = MusaContext::new(device).expect("MUSA device");
        if skip_without_native_fp8(&ctx, "bench_fp8_gemm_shapes") {
            return;
        }
        let iterations = 200usize;
        // (m, n, k, label). The prefix shapes are here because the same kernel
        // covers them on the decode fast path; the odd k=588/n=4304 shapes are
        // refused by alignment and stay on muDNN.
        let shapes = [
            (10usize, 2560usize, 1024usize, "action qkv"),
            (10, 1024, 2048, "action o"),
            (10, 8192, 1024, "action gate_up"),
            (10, 1024, 4096, "action down"),
            (522, 2560, 2048, "prefix qkv"),
            (522, 2048, 2048, "prefix o"),
            (522, 16384, 2048, "prefix mlp up"),
            (522, 2048, 16384, "prefix mlp down"),
            // GR00T N1.7 (S5000 static FP8): the 4-step DiT M=41 family, the
            // M=145 text layers, and the M=512 vision tower.
            (41, 4608, 1536, "gr00t dit qkv"),
            (41, 1536, 1536, "gr00t dit o"),
            (41, 1536, 2048, "gr00t dit cross q"),
            (41, 6144, 1536, "gr00t dit up"),
            (41, 1536, 6144, "gr00t dit down"),
            (128, 1536, 2048, "gr00t dit cross kv"),
            (145, 2048, 2048, "gr00t text q/o"),
            (145, 1024, 2048, "gr00t text k/v"),
            (145, 6144, 2048, "gr00t text gate/up"),
            (145, 2048, 6144, "gr00t text down"),
            (512, 3072, 1024, "gr00t vision qkv"),
            (512, 1024, 1024, "gr00t vision o"),
            (512, 4096, 1024, "gr00t vision fc1"),
            (512, 1024, 4096, "gr00t vision fc2"),
            (128, 4096, 4096, "gr00t vision merger fc1"),
            (128, 2048, 4096, "gr00t vision merger fc2"),
        ];
        for (m, n, k, label) in shapes {
            let activation = crate::transfers::to_device(
                &Tensor::from_f8_e4m3(vec![m, k], &vec![0x38u8; m * k]).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let weight_bytes = k * n;
            // 64 MiB of weights so that the round-robin working set cannot be
            // served from the 60 MiB L2. A warm run keeps one copy.
            let copies = if cold {
                (64usize << 20).div_ceil(weight_bytes).max(2)
            } else {
                1
            };
            let weights: Vec<Tensor> = (0..copies)
                .map(|_| {
                    crate::transfers::to_device(
                        &Tensor::from_f8_e4m3(vec![k, n], &vec![0x38u8; weight_bytes]).unwrap(),
                        ctx.device_id(),
                    )
                    .unwrap()
                })
                .collect();
            let mut out = ctx
                .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            for _ in 0..5 {
                gemm_fp8_into(&ctx, &activation, &weights[0], &mut out, 0.5, "bench").unwrap();
            }
            ctx.synchronize().unwrap();
            let start = Instant::now();
            for step in 0..iterations {
                let weight = &weights[step % copies];
                gemm_fp8_into(&ctx, &activation, weight, &mut out, 0.5, "bench").unwrap();
            }
            ctx.synchronize().unwrap();
            let per_call = start.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;
            let gbs = weight_bytes as f64 / (per_call * 1.0e-6) / 1.0e9;
            let tflops = 2.0 * (m * n * k) as f64 / (per_call * 1.0e-6) / 1.0e12;
            println!(
                "  fp8 gemm {label}: m={m} n={n} k={k} copies={copies} \
                 {per_call:.2} us/call, {gbs:.1} GB/s weights, {tflops:.1} TFLOPS"
            );
        }
    }

    /// A/B the muDNN bias/residual epilogue against the composed
    /// `gemm_fp8_into` + `bias_then_residual_bf16` on the DiT shapes.
    #[test]
    #[ignore = "requires a MUSA device; set APXINF_MUSA_BENCH_DEVICE"]
    fn bench_fp8_bias_epilogue_shapes() {
        use half::bf16;
        use std::hint::black_box;
        use std::time::Instant;

        let device = std::env::var("APXINF_MUSA_BENCH_DEVICE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let ctx = MusaContext::new(device).expect("MUSA device");
        if skip_without_native_fp8(&ctx, "bench_fp8_bias_epilogue_shapes") {
            return;
        }
        let iterations = 200usize;
        let shapes = [
            (41usize, 1536usize, 1536usize, "dit o"),
            (41, 4608, 1536, "dit qkv"),
            (41, 6144, 1536, "dit up"),
            (41, 1536, 6144, "dit down"),
        ];
        for (m, n, k, label) in shapes {
            let activation = crate::transfers::to_device(
                &Tensor::from_f8_e4m3(vec![m, k], &vec![0x38u8; m * k]).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let weight = crate::transfers::to_device(
                &Tensor::from_f8_e4m3(vec![k, n], &vec![0x38u8; k * n]).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let bias = crate::transfers::to_device(
                &Tensor::from_bf16(vec![n], &vec![bf16::from_f32(0.25); n]).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let residual = crate::transfers::to_device(
                &Tensor::from_bf16(vec![m, n], &vec![bf16::from_f32(0.5); m * n]).unwrap(),
                ctx.device_id(),
            )
            .unwrap();
            let mut plain = ctx
                .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            let mut fused = ctx
                .allocate_output(Shape::new(vec![m, n]), DType::BF16)
                .unwrap();

            gemm_fp8_into(&ctx, &activation, &weight, &mut plain, 0.5, "bench").unwrap();
            let composed = crate::kernels::fused::bias_then_residual_bf16(
                &ctx,
                &plain,
                Some(&bias),
                &residual,
            )
            .unwrap();
            gemm_fp8_bias_residual_into(
                &ctx, &activation, &weight, &mut fused, &bias, &residual, 0.5, "bench",
            )
            .unwrap();
            ctx.synchronize().unwrap();
            let composed_values = crate::transfers::to_cpu(&composed).unwrap();
            let fused_values = crate::transfers::to_cpu(&fused).unwrap();
            let max_abs = composed_values
                .as_bf16()
                .unwrap()
                .iter()
                .zip(fused_values.as_bf16().unwrap())
                .map(|(left, right)| (f32::from(*left) - f32::from(*right)).abs())
                .fold(0.0f32, f32::max);

            let start = Instant::now();
            for _ in 0..iterations {
                gemm_fp8_into(&ctx, &activation, &weight, &mut plain, 0.5, "bench").unwrap();
                black_box(
                    crate::kernels::fused::bias_then_residual_bf16(
                        &ctx,
                        &plain,
                        Some(&bias),
                        &residual,
                    )
                    .unwrap(),
                );
            }
            ctx.synchronize().unwrap();
            let composed_us = start.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;

            let start = Instant::now();
            for _ in 0..iterations {
                gemm_fp8_bias_residual_into(
                    &ctx, &activation, &weight, &mut fused, &bias, &residual, 0.5, "bench",
                )
                .unwrap();
            }
            ctx.synchronize().unwrap();
            let fused_us = start.elapsed().as_secs_f64() * 1.0e6 / iterations as f64;
            println!(
                "  fp8 bias+residual {label}: m={m} n={n} k={k} \
                 composed {composed_us:.2} us, fused {fused_us:.2} us, max_abs {max_abs:.6}"
            );
        }
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod session_tests {
    use super::*;
    use crate::workspace::{ExecutionSession, OperationShape};
    use apxinf_core::Shape;

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fp8_gemm_is_recorded_by_the_prepare_traversal() {
        let ctx = MusaContext::new(0).unwrap();
        if skip_without_native_fp8(&ctx, "fp8_gemm_is_recorded_by_the_prepare_traversal") {
            return;
        }
        let (m, n, k) = (4usize, 8usize, 16usize);
        let a = crate::transfers::to_device(
            &Tensor::from_f8_e4m3(vec![m, k], &vec![0x38u8; m * k]).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let b = crate::transfers::to_device(
            &Tensor::from_f8_e4m3(vec![k, n], &vec![0x38u8; k * n]).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let mut out = ctx
            .allocate_output(Shape::new(vec![m, n]), DType::BF16)
            .unwrap();
        let session = ExecutionSession::with_capacity(1 << 20, ctx.device_id()).unwrap();
        crate::ops::prepare_with_session(&session, || {
            gemm_fp8_into(&ctx, &a, &b, &mut out, 1.0, "test")
        })
        .unwrap();
        assert!(
            session
                .prepared_shapes()
                .contains(&OperationShape::Gemm { m, n, k }),
            "prepare traversal did not record the FP8 GEMM: {:?}",
            session.prepared_shapes()
        );
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod capture_tests {
    //! muDNN caches one descriptor per `(m, n, k)` and rewrites `alpha` on
    //! every call. PI0.5's language layers share a shape but not an
    //! `alpha = input_scale * weight_scale`, so capture must record the value
    //! per call rather than leave the last one in the descriptor.

    use super::*;
    use apxinf_core::{Graph, Shape};
    use std::sync::Arc;

    fn device_fp8(ctx: &MusaContext, shape: [usize; 2], byte: u8) -> Tensor {
        crate::transfers::to_device(
            &Tensor::from_f8_e4m3(shape.to_vec(), &vec![byte; shape[0] * shape[1]]).unwrap(),
            ctx.device_id(),
        )
        .unwrap()
    }

    fn bits(tensor: &Tensor) -> Vec<u32> {
        crate::transfers::to_cpu(tensor)
            .unwrap()
            .to_f32_vec()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    #[ignore = "requires a MUSA device"]
    fn fp8_gemm_capture_replays_each_layers_alpha() {
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        if crate::FP8_ROUTE != "mudnn" {
            eprintln!("skipping fp8_gemm_capture_replays_each_layers_alpha: route is not mudnn");
            return;
        }
        if skip_without_native_fp8(&ctx, "fp8_gemm_capture_replays_each_layers_alpha") {
            return;
        }
        let (m, n, k) = (4usize, 8usize, 16usize);
        let activation = device_fp8(&ctx, [m, k], 0x38);
        let weight = device_fp8(&ctx, [k, n], 0x38);
        let mut first = ctx
            .allocate_output(Shape::new(vec![m, n]), DType::BF16)
            .unwrap();
        let mut second = ctx
            .allocate_output(Shape::new(vec![m, n]), DType::BF16)
            .unwrap();

        // Warm up the shape and record what each alpha should produce.
        gemm_fp8_into(&ctx, &activation, &weight, &mut first, 0.5, "test").unwrap();
        gemm_fp8_into(&ctx, &activation, &weight, &mut second, 2.0, "test").unwrap();
        ctx.synchronize().unwrap();
        let expected_first = bits(&first);
        let expected_second = bits(&second);

        // Overwrite both outputs so a replay that recorded nothing cannot pass
        // by observing the warm-up values.
        gemm_fp8_into(&ctx, &activation, &weight, &mut first, 4.0, "test").unwrap();
        gemm_fp8_into(&ctx, &activation, &weight, &mut second, 4.0, "test").unwrap();
        ctx.synchronize().unwrap();

        let graph = crate::graph::capture(Arc::clone(&ctx), || {
            gemm_fp8_into(ctx.as_ref(), &activation, &weight, &mut first, 0.5, "test")?;
            gemm_fp8_into(ctx.as_ref(), &activation, &weight, &mut second, 2.0, "test")
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();

        assert_eq!(bits(&first), expected_first);
        assert_eq!(bits(&second), expected_second);
    }
}
