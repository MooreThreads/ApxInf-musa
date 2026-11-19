//! Activation operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{
    bf16_output, matrix_tensor, require_matrix_bf16, require_optional_vector_bf16,
    output_tensor, require_dtype_on, require_output_dtype_tensor, require_output_matrix,
};
#[cfg(apxinf_musa_available)]
use super::contracts::{checked_i32, optional_tensor_ptr, tensor_ptr};
use crate::context::MusaContext;
#[cfg(apxinf_musa_available)]
use crate::error::musa_error;
#[cfg(apxinf_musa_available)]
use crate::ffi;
#[cfg(apxinf_musa_available)]
use crate::kernels::check_native;
#[cfg(not(apxinf_musa_available))]
use crate::kernels::native_unavailable;
use crate::kernels::not_ported;

/// Activation codes shared with the CUDA backend's kernel ABI: the same
/// integer selects the same nonlinearity, so a tactic or a diagnosable name
/// never has to be renumbered across backends.
pub(super) const ACTIVATION_NONE: i32 = 0;
pub(super) const ACTIVATION_GELU: i32 = 1;
pub(super) const ACTIVATION_SILU: i32 = 2;
pub(super) const ACTIVATION_RELU: i32 = 3;

/// Bias plus the activation selected by `activation`.
///
/// Native: one elementwise kernel on the device, `kernels/custom/activation.cuh`
/// through `adapters/custom_kernels.mu`. All three of `bias_gelu_bf16`,
/// `bias_silu_bf16` and `elementwise::bias_bf16` land here, because they are one
/// CUDA entry point three ways — the bridge recorded them as three `kind`
/// branches of a single signature for the same reason.
pub(super) fn bias_activation(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    activation: i32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias activation", "input")?;
    let mut output = matrix_tensor(ctx, rows, cols, bf16_output(ctx, rows, cols)?);
    bias_activation_into(ctx, input, value, &mut output, activation)?;
    Ok(output)
}

pub(super) fn bias_activation_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
    activation: i32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias activation", "input")?;
    require_optional_vector_bf16(ctx, value, cols, "bias activation", "bias")?;
    require_output_matrix(ctx, output, rows, cols, "bias activation", "out")?;
    launch(ctx, input, value, output, rows, cols, activation)
}

/// The launch, split by whether this build has a MUSA toolkit.
///
/// # Numerics
/// The kernel computes in f32 and rounds once on the way out, which is what the
/// eager bridge did — it upcast both operands, added and activated in f32, and
/// let the write convert back. GELU is the tanh approximation rather than erf,
/// matching both `_gelu_tanh` in the bridge and `gelu_tanh` in
/// `kernels/custom/math.cuh`; the three spellings have to move together or the
/// recorded reference tensors stop describing this kernel.
///
/// # Stream
/// `ctx.stream()`, as everywhere else in this backend. That is ordered against
/// the torch operators around it — see the streaming note in the eager bridge
/// that recorded the reference (it is not part of this tree) — so no
/// synchronisation is needed here.
#[cfg(apxinf_musa_available)]
fn launch(
    ctx: &MusaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    output: &Tensor,
    rows: usize,
    cols: usize,
    activation: i32,
) -> Result<()> {
    let rows = checked_i32("bias_activation", "rows", rows)?;
    let cols = checked_i32("bias_activation", "cols", cols)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: `require_matrix_bf16` and `require_vector_bf16` have established
    // that the input and the optional bias are contiguous BF16 tensors on this
    // context's device, and `matrix_tensor` that the output is a `[rows, cols]`
    // BF16 allocation there. `tensor_ptr` borrows storage the caller's `Tensor`s
    // keep alive across the call, and the null bias is the kernel's documented
    // "no bias" case. The launch is asynchronous on `ctx.stream()`.
    unsafe {
        ffi::check_runtime(ffi::custom::apxinf_musa_bias_activation_bf16(
            tensor_ptr(input)?,
            bias_pointer,
            tensor_ptr(output)?,
            rows,
            cols,
            activation,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    Ok(())
}

#[cfg(not(apxinf_musa_available))]
fn launch(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _activation: i32,
) -> Result<()> {
    Err(native_unavailable("activation::bias_activation_bf16"))
}

/// Bias plus GELU.
pub fn bias_gelu_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    bias_activation(ctx, input, value, ACTIVATION_GELU)
}

/// Write bias plus GELU into a caller-provided output.
pub fn bias_gelu_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    bias_activation_into(ctx, input, value, output, ACTIVATION_GELU)
}

/// Whether the fused E4M3 bias+GELU epilogue serves this width.
///
/// Mirrors the adapter's `cols % 8 == 0` vector check. Every device allocation
/// from `MusaContext::allocate_output` is 256-byte aligned, so the pointer
/// alignment the adapter also checks cannot fail on a model-path call.
pub fn bias_gelu_e4m3_supported(cols: usize) -> bool {
    cols % 8 == 0
}

/// Bias plus GELU writing E4M3, as one launch.
///
/// The kernel rounds to BF16 before applying `scale`'s reciprocal, so the bytes
/// match `bias_gelu_bf16_into` followed by `quantize_fixed_e4m3`.
pub fn bias_gelu_bf16_e4m3_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
    scale: f32,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias gelu", "input")?;
    require_optional_vector_bf16(ctx, value, cols, "bias gelu", "bias")?;
    require_output_dtype_tensor(
        ctx,
        output,
        &[rows, cols],
        DType::F8E4M3,
        "bias gelu",
        "out",
    )?;
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    if !bias_gelu_e4m3_supported(cols) {
        return Err(Error::Other(format!(
            "MUSA bias_gelu_e4m3: no fused kernel for width {cols}"
        )));
    }
    launch_e4m3(
        ctx,
        input,
        value,
        output,
        rows,
        cols,
        ACTIVATION_GELU,
        inverse_scale,
    )
}

/// Bias plus GELU writing E4M3, as one launch, returning a new tensor.
///
/// The consumer (the following FC2 projection) then takes the E4M3 tensor
/// directly, so the vision block drops both the BF16 round trip of the
/// `[512,4096]` FC1 output and the separate quantize launch.
pub fn bias_gelu_bf16_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "bias gelu", "input")?;
    let mut output = output_tensor(
        ctx,
        input.shape().clone(),
        DType::F8E4M3,
        crate::workspace::output_buffer(ctx, rows * cols)?,
    );
    bias_gelu_bf16_e4m3_into(ctx, input, value, &mut output, scale)?;
    Ok(output)
}

/// The fused E4M3 launch, split by whether this build has a MUSA toolkit.
#[cfg(apxinf_musa_available)]
fn launch_e4m3(
    ctx: &MusaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    output: &Tensor,
    rows: usize,
    cols: usize,
    activation: i32,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("bias_activation_e4m3", "rows", rows)?;
    let cols = checked_i32("bias_activation_e4m3", "cols", cols)?;
    let bias_pointer = optional_tensor_ptr(bias)?;
    // SAFETY: the checks above established a BF16 `[rows, cols]` input, an
    // optional BF16 bias of `cols` and an E4M3 `[rows, cols]` output, all on
    // this context's device and alive across the asynchronous launch.
    unsafe {
        ffi::check_runtime(ffi::custom::apxinf_musa_bias_activation_bf16_e4m3(
            tensor_ptr(input)?,
            bias_pointer,
            tensor_ptr(output)?,
            rows,
            cols,
            activation,
            inverse_scale,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    Ok(())
}

#[cfg(not(apxinf_musa_available))]
fn launch_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
    _output: &Tensor,
    _rows: usize,
    _cols: usize,
    _activation: i32,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("activation::bias_gelu_bf16_e4m3_into"))
}

/// Bias plus SiLU.
pub fn bias_silu_bf16(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    bias_activation(ctx, input, value, ACTIVATION_SILU)
}

/// Write bias plus SiLU into a caller-provided output.
pub fn bias_silu_bf16_into(
    ctx: &MusaContext,
    input: &Tensor,
    value: Option<&Tensor>,
    output: &mut Tensor,
) -> Result<()> {
    bias_activation_into(ctx, input, value, output, ACTIVATION_SILU)
}

/// Bias plus ReLU.
pub fn bias_relu_bf16(ctx: &MusaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, value, ACTIVATION_RELU)
}

/// Native device-resident SiLU gate for Qwen's gated MLP.
pub fn silu_mul_bf16(ctx: &MusaContext, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    require_dtype_on(ctx, gate, DType::BF16, "silu multiply", "gate")?;
    require_dtype_on(ctx, up, DType::BF16, "silu multiply", "up")?;
    if gate.shape() != up.shape() || gate.numel() == 0 {
        return Err(Error::Other(
            "MUSA SiLU multiply requires equal non-empty input shapes".into(),
        ));
    }
    let output = output_tensor(
        ctx,
        gate.shape().clone(),
        DType::BF16,
        crate::workspace::output_buffer(ctx, gate.size_in_bytes())?,
    );
    #[cfg(apxinf_musa_available)]
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_silu_mul_bf16(
                tensor_ptr(gate)?,
                tensor_ptr(up)?,
                tensor_ptr(&output)?,
                i64::try_from(gate.numel()).map_err(|_| {
                    Error::Other("MUSA SiLU multiply element count overflow".into())
                })?,
                ctx.stream().handle(),
            ),
            "silu_mul_bf16",
        )?;
    }
    #[cfg(not(apxinf_musa_available))]
    return Err(native_unavailable("activation::silu_mul_bf16"));
    Ok(output)
}

pub fn silu_bf16(ctx: &MusaContext, input: &Tensor) -> Result<Tensor> {
    require_dtype_on(ctx, input, DType::BF16, "silu", "input")?;
    if input.numel() == 0 {
        return Err(Error::Other("MUSA SiLU requires a non-empty input".into()));
    }
    let output = output_tensor(
        ctx,
        input.shape().clone(),
        DType::BF16,
        crate::workspace::output_buffer(ctx, input.size_in_bytes())?,
    );
    #[cfg(apxinf_musa_available)]
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_silu_bf16(
                tensor_ptr(input)?,
                tensor_ptr(&output)?,
                i64::try_from(input.numel())
                    .map_err(|_| Error::Other("MUSA SiLU element count overflow".into()))?,
                ctx.stream().handle(),
            ),
            "silu_bf16",
        )?;
    }
    #[cfg(not(apxinf_musa_available))]
    return Err(native_unavailable("activation::silu_bf16"));
    Ok(output)
}

/// Gated GELU: splits `[rows, 2 * half]` into gate and up halves.
///
/// Native: `kernels/custom/activation.cuh`'s grid-stride kernel, through
/// `adapters/custom_kernels.mu`. PI0.5 never calls this on its own -- its gate/up
/// projection is fused into the GEMM -- but the fused operator is served by
/// calling this one, so it is on the model's path through
/// [`crate::kernels::gemm::bf16_geglu_fused`].
pub fn geglu_bf16(ctx: &MusaContext, gate_up: &Tensor) -> Result<Tensor> {
    let (rows, cols) = require_matrix_bf16(ctx, gate_up, "geglu", "gate_up")?;
    if cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu expects an even inner dimension, got {cols}"
        )));
    }
    let half = cols / 2;
    let mut output = matrix_tensor(ctx, rows, half, bf16_output(ctx, rows, half)?);
    geglu_bf16_into(ctx, gate_up, &mut output)?;
    Ok(output)
}

/// Write gated GELU into a caller-provided `[rows, cols / 2]` output.
pub fn geglu_bf16_into(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, gate_up, "geglu", "gate_up")?;
    if cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu expects an even inner dimension, got {cols}"
        )));
    }
    let half = cols / 2;
    require_output_matrix(ctx, output, rows, half, "geglu", "out")?;
    launch_geglu(ctx, gate_up, output, rows, half)
}

/// The launch, split by whether this build has a MUSA toolkit.
#[cfg(apxinf_musa_available)]
fn launch_geglu(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &Tensor,
    rows: usize,
    inner: usize,
) -> Result<()> {
    let rows = checked_i32("geglu", "rows", rows)?;
    let inner = checked_i32("geglu", "inner", inner)?;
    // SAFETY: `require_matrix_bf16` established `gate_up` as `[rows, 2 * inner]`
    // BF16 on this device, `matrix_tensor` that the output is `[rows, inner]`
    // there, and both outlive the asynchronous launch on `ctx.stream()`.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_geglu_bf16(
                tensor_ptr(gate_up)?,
                tensor_ptr(output)?,
                rows,
                inner,
                ctx.stream().handle(),
            ),
            "geglu",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_geglu(
    _ctx: &MusaContext,
    _gate_up: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _inner: usize,
) -> Result<()> {
    Err(native_unavailable("activation::geglu_bf16"))
}

/// Gated GELU written directly as static E4M3.
///
/// Same arithmetic as [`geglu_bf16_into`] followed by
/// [`crate::kernels::quantization::quantize_fixed_e4m3`]. The fused kernel
/// rounds the product to BF16 before converting, which is the composed pair's
/// contract, so enabling the fusion cannot move the model's bytes.
pub fn geglu_bf16_e4m3_into(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &mut Tensor,
    scale: f32,
    pair_width: usize,
) -> Result<()> {
    let (rows, cols) = require_matrix_bf16(ctx, gate_up, "geglu", "gate_up")?;
    if cols % 2 != 0 {
        return Err(Error::Other(format!(
            "MUSA geglu expects an even inner dimension, got {cols}"
        )));
    }
    let half = cols / 2;
    if pair_width != 0
        && (pair_width % 4 != 0 || pair_width > half || half % pair_width != 0)
    {
        return Err(Error::Other(format!(
            "MUSA geglu pair width {pair_width} must divide {half}, be a multiple of 4, and not exceed it"
        )));
    }
    require_output_dtype_tensor(
        ctx,
        output,
        &[rows, half],
        DType::F8E4M3,
        "geglu",
        "out",
    )?;
    let inverse_scale = super::quantization::checked_inverse_scale(scale)?;
    launch_geglu_e4m3(
        ctx,
        gate_up,
        output,
        rows,
        half,
        pair_width,
        inverse_scale,
    )
}

#[cfg(apxinf_musa_available)]
fn launch_geglu_e4m3(
    ctx: &MusaContext,
    gate_up: &Tensor,
    output: &Tensor,
    rows: usize,
    inner: usize,
    pair_width: usize,
    inverse_scale: f32,
) -> Result<()> {
    let rows = checked_i32("geglu", "rows", rows)?;
    let inner = checked_i32("geglu", "inner", inner)?;
    let pair_width = checked_i32("geglu", "pair_width", pair_width)?;
    // SAFETY: the shapes, dtype and device are established by the checks
    // above, and both tensors outlive the asynchronous launch.
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_geglu_bf16_e4m3(
                tensor_ptr(gate_up)?,
                tensor_ptr(output)?,
                rows,
                inner,
                pair_width,
                inverse_scale,
                ctx.stream().handle(),
            ),
            "geglu",
        )
    }
}

#[cfg(not(apxinf_musa_available))]
fn launch_geglu_e4m3(
    _ctx: &MusaContext,
    _gate_up: &Tensor,
    _output: &Tensor,
    _rows: usize,
    _inner: usize,
    _pair_width: usize,
    _inverse_scale: f32,
) -> Result<()> {
    Err(native_unavailable("activation::geglu_bf16_e4m3"))
}

// ── FP16 activation (not ported) ─────────────────────────────────────────

pub fn bias_gelu_bf16_packed8(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: &Tensor,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_gelu_bf16_packed8"))
}

pub fn bias_gelu_quant_bf16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: &Tensor,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_gelu_quant_bf16_e4m3"))
}

/// `silu(gate) * up`, written as static E4M3.
///
/// The BF16 product is rounded first and then converted with the same
/// fixed-scale arithmetic [`super::quantization::quantize_fixed_e4m3`] uses,
/// in one launch. The composed pair (`silu_mul_bf16` + `quantize_fixed_e4m3`)
/// measured ~11.9 us/call for the text widths and paid a `[145,6144]` BF16
/// round trip per layer.
pub fn silu_mul_quant_bf16_e4m3(
    ctx: &MusaContext,
    gate: &Tensor,
    up: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    require_dtype_on(ctx, gate, DType::BF16, "silu multiply", "gate")?;
    require_dtype_on(ctx, up, DType::BF16, "silu multiply", "up")?;
    if gate.shape() != up.shape() || gate.numel() == 0 {
        return Err(Error::Other(
            "MUSA SiLU multiply requires equal non-empty input shapes".into(),
        ));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!(
            "MUSA SiLU multiply E4M3 scale must be finite and positive, got {scale}"
        )));
    }
    let output = output_tensor(
        ctx,
        gate.shape().clone(),
        DType::F8E4M3,
        crate::workspace::output_buffer(ctx, gate.numel())?,
    );
    #[cfg(apxinf_musa_available)]
    unsafe {
        check_native(
            ffi::custom::apxinf_musa_silu_mul_bf16_e4m3(
                tensor_ptr(gate)?,
                tensor_ptr(up)?,
                tensor_ptr(&output)?,
                i64::try_from(gate.numel()).map_err(|_| {
                    Error::Other("MUSA SiLU multiply element count overflow".into())
                })?,
                1.0f32 / scale,
                ctx.stream().handle(),
            ),
            "silu_mul_quant_bf16_e4m3",
        )?;
    }
    #[cfg(not(apxinf_musa_available))]
    return Err(native_unavailable("activation::silu_mul_quant_bf16_e4m3"));
    Ok(output)
}

/// Bias plus SiLU in FP16.
pub fn bias_silu_f16(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_silu_f16"))
}

/// Bias plus SiLU, output quantized to E4M3.
pub fn bias_silu_quant_f16_e4m3(
    _ctx: &MusaContext,
    _input: &Tensor,
    _bias: Option<&Tensor>,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("activation::bias_silu_quant_f16_e4m3"))
}

/// Gated GELU whose output is quantized to E4M3.
pub fn geglu_quant_f16_e4m3(
    _ctx: &MusaContext,
    _gate_up: &Tensor,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("activation::geglu_quant_f16_e4m3"))
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    //! Device tests for the bias/activation kernel.
    //!
    //! The replay harness already compares this
    //! operator against recorded reference tensors on six real shapes covering
    //! all three activation codes, and it is the stronger evidence. What it
    //! cannot reach is the **bias-less** path: every recorded call carried a
    //! bias, because that is what the model does, so `bias == nullptr` — a real
    //! branch in the kernel, reached by `elementwise::bias_bf16(ctx, x, None)`
    //! — has no reference to be compared against. That is what this checks.
    //!
    //! Needs a MUSA device, so it is `#[ignore]`d:
    //! `cargo test -p apxinf-musa --lib -- --ignored --nocapture`.

    use apxinf_core::{DType, Shape};

    use crate::context::MusaContext;
    use crate::kernels::activation::{
        bias_gelu_bf16, bias_gelu_bf16_e4m3_into, bias_gelu_bf16_into, bias_relu_bf16,
        bias_silu_bf16, geglu_bf16, geglu_bf16_e4m3_into, geglu_bf16_into,
    };
    use crate::kernels::elementwise::bias_bf16;
    use crate::kernels::test_support::{
        from_bf16_bits, gate_for, read_back, to_bf16_bits, upload, values, zeros_as,
    };

    /// The kernel's own spellings, in f32, so the reference differs only where
    /// `tanhf`/`expf` differ between the host libm and the device.
    fn reference(kind: i32, input: &[f32], bias: Option<f32>) -> Vec<f32> {
        input
            .iter()
            .map(|value| {
                let mut x = *value;
                if let Some(bias) = bias {
                    x += bias;
                }
                match kind {
                    1 => {
                        let alpha = 0.7978845608028654f32;
                        0.5 * x * (1.0 + (alpha * (x + 0.044715 * x * x * x)).tanh())
                    }
                    2 => x / (1.0 + (-x).exp()),
                    3 => x.max(0.0),
                    _ => x,
                }
            })
            .map(|v| from_bf16_bits(to_bf16_bits(v)))
            .collect()
    }

    fn check(ctx: &MusaContext, rows: usize, cols: usize, kind: i32) {
        let input_values = values(rows * cols, 0x5eed_1234);
        let bias_values = values(cols, 0x0bad_c0de);
        let input = upload(ctx, &input_values, Shape::new(vec![rows, cols]));
        let bias = upload(ctx, &bias_values, Shape::new(vec![cols]));

        let out = match kind {
            0 => bias_bf16(ctx, &input, Some(&bias)),
            1 => bias_gelu_bf16(ctx, &input, Some(&bias)),
            2 => bias_silu_bf16(ctx, &input, Some(&bias)),
            3 => bias_relu_bf16(ctx, &input, Some(&bias)),
            _ => panic!("unknown activation code {kind}"),
        }
        .expect("native activation");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        // The reference consumes the bf16-rounded inputs the kernel sees.
        let rounded_input: Vec<f32> = input_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();
        let rounded_bias: Vec<f32> = bias_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();

        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for col in 0..cols {
                let index = row * cols + col;
                // The bias goes in *before* the activation, which is what the
                // kernel does; getting this order wrong is a mistake worth
                // having caught here rather than in a diff downstream.
                let expected =
                    reference(kind, &rounded_input[index..index + 1], Some(rounded_bias[col]))[0];
                let error = (got[index] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "kind {kind} [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[index]
                );
            }
        }
        println!("  kind {kind}  [{rows} x {cols}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_activation_matches_a_reference_for_every_kind() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for kind in [0, 1, 2, 3] {
            check(&ctx, 3, 5, kind);
        }
        // A row count above one and a column count that is not a multiple of
        // the block size, so the grid-stride loop takes more than one step.
        check(&ctx, 257, 300, 1);
        // A width that is a multiple of eight takes the vectorized branch,
        // which is the one the vision MLP's `[512, 4304]` GELU runs on.
        for kind in [0, 1, 2, 3] {
            check(&ctx, 3, 320, kind);
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_activation_without_bias_matches_every_kind() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        for cols in [5, 320] {
            let input_values = values(3 * cols, 0x5eed_1234);
            let input = upload(&ctx, &input_values, Shape::new(vec![3, cols]));
            let rounded: Vec<f32> = input_values
                .iter()
                .map(|value| from_bf16_bits(to_bf16_bits(*value)))
                .collect();
            for kind in [0, 1, 2, 3] {
                let output = super::bias_activation(&ctx, &input, None, kind)
                    .expect("native activation without bias");
                ctx.stream().synchronize().expect("synchronise");
                let actual = read_back(&output);
                let expected = reference(kind, &rounded, None);
                let tolerance = gate_for(&actual);
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "kind {kind}, cols {cols}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_activation_without_a_bias_copies_the_input() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let input_values = values(4 * 32, 0x1357_9bdf);
        let input = upload(&ctx, &input_values, Shape::new(vec![4, 32]));

        let out = bias_bf16(&ctx, &input, None).expect("bespoke bias-less activation");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        // Code 0 with a null bias is the identity on bf16 values, which makes
        // this the one case in the whole operator where bit equality is the
        // right assertion rather than a gate.
        let expected: Vec<f32> = input_values
            .iter()
            .map(|v| from_bf16_bits(to_bf16_bits(*v)))
            .collect();
        assert_eq!(got, expected, "bias-less activation must copy the input");
        println!("  bias-less identity over {} elements: bit-identical", got.len());
    }

    /// `geglu_bf16`: `gelu_tanh(gate) * up`, gate from the first half.
    ///
    /// The replay covers the two shapes the split gate/up projection produces
    /// (`[522, 32768]` and `[10, 8192]`) and both come out **bit-identical**,
    /// which is stronger evidence than this. What it does not reach is a shape
    /// where the row/column split of a flat index does not line up with the
    /// block, or an odd inner width, which the facade refuses.
    fn check_geglu(ctx: &MusaContext, rows: usize, cols: usize) {
        let inner = cols / 2;
        let gate_up_values = values(rows * cols, 0x5eed_1234);
        let gate_up = upload(ctx, &gate_up_values, Shape::new(vec![rows, cols]));

        let out = geglu_bf16(ctx, &gate_up).expect("native geglu");
        ctx.stream().synchronize().expect("synchronise");
        let got = read_back(&out);

        let round = |v: f32| from_bf16_bits(to_bf16_bits(v));
        let alpha = 0.7978845608028654f32;
        let tolerance = gate_for(&got);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for col in 0..inner {
                let index = row * cols + col;
                let gate = round(gate_up_values[index]);
                let up = round(gate_up_values[index + inner]);
                let gelu = 0.5 * gate * (1.0 + (alpha * (gate + 0.044715 * gate * gate * gate)).tanh());
                let expected = round(gelu * up);
                let error = (got[row * inner + col] - expected).abs();
                worst = worst.max(error);
                assert!(
                    error <= tolerance,
                    "geglu [{rows}x{cols}] at [{row}][{col}]: {} vs {expected}, \
                     abs {error:.3e} > {tolerance:.3e}",
                    got[row * inner + col]
                );
            }
        }
        println!("  geglu [{rows:>4} x {cols:>5}]  worst abs {worst:.3e} (gate {tolerance:.3e})");
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_matches_a_reference() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        // An inner width of 5 is not a multiple of the block size and neither
        // is 151, so both take a ragged last grid-stride step.
        check_geglu(&ctx, 3, 10);
        check_geglu(&ctx, 7, 302);
        check_geglu(&ctx, 522, 32768);
    }

    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_refuses_an_odd_inner_dimension() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let gate_up = upload(&ctx, &values(3 * 5, 0x1357_9bdf), Shape::new(vec![3, 5]));
        let error = geglu_bf16(&ctx, &gate_up).expect_err("an odd width has no gate/up split");
        assert!(error.to_string().contains("even inner dimension"), "{error}");
        println!("  geglu refused: {error}");
    }

    /// The fused E4M3 producer must be the composed pair's bytes, not merely
    /// close: the FP8 calibration and every gold cosine were recorded against
    /// the quantizer reading a materialized BF16 GeGLU.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_e4m3_matches_the_composed_pair_byte_for_byte() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let scale = 0.0721f32;
        for (rows, inner) in [(3usize, 10usize), (10, 4096), (522, 32768)] {
            let gate_up = upload(
                &ctx,
                &values(rows * 2 * inner, 0x51ed_270b),
                Shape::new(vec![rows, 2 * inner]),
            );
            let mut activated = zeros_as(&ctx, Shape::new(vec![rows, inner]), DType::BF16);
            geglu_bf16_into(&ctx, &gate_up, &mut activated).expect("bf16 geglu");
            let mut composed = zeros_as(&ctx, Shape::new(vec![rows, inner]), DType::F8E4M3);
            crate::kernels::quantization::quantize_fixed_e4m3(
                &ctx,
                &activated,
                &mut composed,
                scale,
            )
            .expect("standalone quantizer");

            let mut fused = zeros_as(&ctx, Shape::new(vec![rows, inner]), DType::F8E4M3);
            geglu_bf16_e4m3_into(&ctx, &gate_up, &mut fused, scale, 0).expect("fused geglu");
            ctx.stream().synchronize().expect("synchronise");

            let expected = crate::transfers::to_cpu(&composed)
                .expect("composed back")
                .as_f8_e4m3()
                .expect("composed e4m3")
                .to_vec();
            let got = crate::transfers::to_cpu(&fused)
                .expect("fused back")
                .as_f8_e4m3()
                .expect("fused e4m3")
                .to_vec();
            assert_eq!(
                got, expected,
                "geglu e4m3 [{rows} x {inner}] differs from the composed pair"
            );
            println!("  geglu e4m3 [{rows} x {inner}] byte-identical");
        }
    }

    /// The pair-interleaved weight layout must be a permutation, not a
    /// different computation: `pair_width = 64` over a permuted gate/up buffer
    /// has to produce the half-split path's bytes.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn geglu_e4m3_pairs_match_the_half_split_byte_for_byte() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let scale = 0.0721f32;
        let pair_width = 64usize;
        for (rows, inner) in [(10usize, 4096usize), (3, 512)] {
            let half = values(rows * 2 * inner, 0x0bad_c0de);
            let half_up = upload(&ctx, &half, Shape::new(vec![rows, 2 * inner]));

            // Permute [gate | up] into [g0..g63, u0..u63, g64..u64, ...].
            let mut interleaved = vec![0.0f32; rows * 2 * inner];
            for row in 0..rows {
                let base = row * 2 * inner;
                for col in 0..inner {
                    let block = col / pair_width;
                    let within = col % pair_width;
                    let dst = 2 * pair_width * block + within;
                    interleaved[base + dst] = half[base + col];
                    interleaved[base + dst + pair_width] = half[base + inner + col];
                }
            }
            let pairs = upload(&ctx, &interleaved, Shape::new(vec![rows, 2 * inner]));

            let mut expected = zeros_as(&ctx, Shape::new(vec![rows, inner]), DType::F8E4M3);
            geglu_bf16_e4m3_into(&ctx, &half_up, &mut expected, scale, 0).expect("half split");
            let mut got = zeros_as(&ctx, Shape::new(vec![rows, inner]), DType::F8E4M3);
            geglu_bf16_e4m3_into(&ctx, &pairs, &mut got, scale, pair_width)
                .expect("pair interleaved");
            ctx.stream().synchronize().expect("synchronise");

            let expected = crate::transfers::to_cpu(&expected)
                .expect("half split back")
                .as_f8_e4m3()
                .expect("half split e4m3")
                .to_vec();
            let got = crate::transfers::to_cpu(&got)
                .expect("pairs back")
                .as_f8_e4m3()
                .expect("pairs e4m3")
                .to_vec();
            assert_eq!(
                got, expected,
                "pair-aware geglu [{rows} x {inner}] differs from the half split"
            );
            println!("  geglu pairs [{rows} x {inner}] byte-identical");
        }
    }

    /// The fused E4M3 bias+GELU must be byte-identical to materializing the
    /// BF16 activation and running the standalone quantizer.
    #[test]
    #[ignore = "needs a MUSA device"]
    fn bias_gelu_e4m3_matches_the_composed_pair_byte_for_byte() {
        let ctx = MusaContext::new(0).expect("MUSA device 0");
        let scale = 0.0721f32;
        for (rows, cols) in [(8usize, 16usize), (512, 4304)] {
            let input = upload(
                &ctx,
                &values(rows * cols, 0x5eed_1234),
                Shape::new(vec![rows, cols]),
            );
            let bias = upload(&ctx, &values(cols, 0x0bad_c0de), Shape::new(vec![cols]));
            let mut bf16 = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::BF16);
            bias_gelu_bf16_into(&ctx, &input, Some(&bias), &mut bf16).expect("bf16 gelu");
            let mut composed = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
            crate::kernels::quantization::quantize_fixed_e4m3(
                &ctx,
                &bf16,
                &mut composed,
                scale,
            )
            .expect("standalone quantizer");
            let mut fused = zeros_as(&ctx, Shape::new(vec![rows, cols]), DType::F8E4M3);
            bias_gelu_bf16_e4m3_into(&ctx, &input, Some(&bias), &mut fused, scale)
                .expect("fused bias gelu");
            ctx.stream().synchronize().expect("synchronise");
            let expected = crate::transfers::to_cpu(&composed)
                .expect("composed back")
                .as_f8_e4m3()
                .expect("composed e4m3")
                .to_vec();
            let got = crate::transfers::to_cpu(&fused)
                .expect("fused back")
                .as_f8_e4m3()
                .expect("fused e4m3")
                .to_vec();
            assert_eq!(
                got, expected,
                "bias_gelu e4m3 [{rows} x {cols}] differs from the composed pair"
            );
            println!("  bias_gelu e4m3 [{rows} x {cols}] byte-identical");
        }
    }
}
