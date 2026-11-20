//! Safe MUSA surface used by the backend-native GR00T port.
//!
//! Model code sees concrete resources and model-neutral operations here. Raw
//! FFI and provider-specific entry points remain private to `apxinf-musa`.

use std::any::Any;
use std::sync::Arc;

use apxinf_core::{DType, Device, Error, Graph, Result, Shape, Tensor};

pub(crate) use apxinf_musa::MusaBuffer as DeviceBuffer;
pub(crate) use apxinf_musa::MusaContext as Context;
pub(crate) use apxinf_musa::kernels::gemm::{decode_scope, prepare_bf16_nn_weight};

pub(crate) mod transfers {
    pub(crate) use apxinf_musa::transfers::{
        copy_cpu_to_device_pinned, to_cpu, to_device, PinnedBuffer,
    };
}

pub(crate) mod tuning {
    pub(crate) use apxinf_musa::kernels::gemm::configure_tuning;
    pub(crate) use apxinf_musa::tuning::{TuningDb, TuningMode, TuningPaths};
}

pub(crate) use tuning::TuningMode;

/// Compatibility facade for the execution engine while prepared execution is
/// migrated to the repository-wide session API.
pub(crate) mod kernels {
    pub(crate) use apxinf_musa::kernels::*;

    use apxinf_core::{Error, Result};

    pub(crate) struct GraphWorkspace {
        session: apxinf_musa::ExecutionSession,
    }

    impl GraphWorkspace {
        pub(crate) fn new(capacity_bytes: usize, device: usize) -> Result<Self> {
            Ok(Self {
                session: apxinf_musa::ExecutionSession::with_capacity(capacity_bytes, device)?,
            })
        }
    }

    pub(crate) fn prepare_with_workspace<T>(
        workspace: &GraphWorkspace,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        workspace.session.prepare(operation)
    }

    pub(crate) fn with_workspace<T>(
        workspace: &GraphWorkspace,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        workspace.session.run(operation)
    }

    pub(crate) fn with_workspace_eager<T>(
        workspace: &GraphWorkspace,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        workspace
            .session
            .prepare(operation)
            .map_err(|error| Error::Other(format!("MUSA prepared execution failed: {error}")))
    }
}

/// Concrete model-local backend used by the self-contained executor.
pub(crate) struct RuntimeBackend {
    context: Arc<Context>,
}

pub(crate) type GpuBackend = RuntimeBackend;

impl RuntimeBackend {
    pub(crate) fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            context: Arc::new(Context::new(device_id)?),
        })
    }

    pub(crate) fn context(&self) -> &Context {
        &self.context
    }

    pub(crate) fn device_id(&self) -> usize {
        self.context.device_id()
    }

    pub(crate) fn device(&self) -> Device {
        Device::Musa(self.device_id())
    }

    pub(crate) fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_device(tensor, self.device_id())
    }

    pub(crate) fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cpu(tensor)
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.context.synchronize()
    }

    pub(crate) fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let _scope = decode_scope();
        kernels::gemm::bf16(self.context(), a, b)
    }

    pub(crate) fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        kernels::norm::layer_bf16(self.context(), input, weight, bias, eps)
    }

    pub(crate) fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        kernels::activation::bias_gelu_bf16(self.context(), input, None)
    }

    pub(crate) fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        kernels::elementwise::bias_bf16(self.context(), input, Some(bias))
    }

    pub(crate) fn silu(&self, input: &Tensor) -> Result<Tensor> {
        kernels::activation::silu_bf16(self.context(), input)
    }

    pub(crate) fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add(self.context(), a, b)
    }

    /// Quantize a BF16 activation to E4M3 with a fixed dequantization scale.
    ///
    /// The static-FP8 executor owns the scale (from the calibration profile);
    /// the kernel receives its reciprocal.
    pub(crate) fn quantize_e4m3(&self, input: &Tensor, scale: f32) -> Result<Tensor> {
        let mut output = self
            .context
            .allocate_output(input.shape().clone(), DType::F8E4M3)?;
        kernels::quantization::quantize_fixed_e4m3(&self.context, input, &mut output, scale)?;
        Ok(output)
    }

    /// E4M3 x E4M3 contraction with a per-tensor dequantization factor,
    /// accumulating in F32 and writing BF16.
    pub(crate) fn matmul_fp8(
        &self,
        activation: &Tensor,
        weight: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        let a = activation.shape().dims();
        let b = weight.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(Error::Other(format!(
                "GR00T FP8 GEMM shape mismatch: {a:?} @ {b:?}"
            )));
        }
        let mut output = self
            .context
            .allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
        kernels::gemm::gemm_fp8_into(
            &self.context,
            activation,
            weight,
            &mut output,
            alpha,
            "gr00t fp8 linear",
        )?;
        Ok(output)
    }

    /// `activation @ weight + bias` in one muDNN call, for the static-FP8
    /// linears whose bias is BF16.
    pub(crate) fn matmul_fp8_bias(
        &self,
        activation: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        let a = activation.shape().dims();
        let b = weight.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(Error::Other(format!(
                "GR00T FP8 bias GEMM shape mismatch: {a:?} @ {b:?}"
            )));
        }
        let mut output = self
            .context
            .allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
        kernels::gemm::gemm_fp8_bias_into(
            &self.context,
            activation,
            weight,
            &mut output,
            bias,
            alpha,
            "gr00t fp8 linear bias",
        )?;
        Ok(output)
    }

    /// `activation @ weight + residual + bias` in one muDNN call, for the
    /// static-FP8 routes whose residual and bias are both BF16.
    pub(crate) fn matmul_fp8_bias_residual(
        &self,
        activation: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        residual: &Tensor,
        alpha: f32,
    ) -> Result<Tensor> {
        let a = activation.shape().dims();
        let b = weight.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(Error::Other(format!(
                "GR00T FP8 bias+residual GEMM shape mismatch: {a:?} @ {b:?}"
            )));
        }
        if residual.shape().dims() != [a[0], b[1]] {
            return Err(Error::Other(format!(
                "GR00T FP8 bias+residual residual expects [{}, {}], got {:?}",
                a[0],
                b[1],
                residual.shape().dims()
            )));
        }
        let mut output = self
            .context
            .allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
        kernels::gemm::gemm_fp8_bias_residual_into(
            &self.context,
            activation,
            weight,
            &mut output,
            bias,
            residual,
            alpha,
            "gr00t fp8 linear bias residual",
        )?;
        Ok(output)
    }

    pub(crate) fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::mul(self.context(), a, b)
    }

    pub(crate) fn rope_vision_2d(
        &self,
        input: &Tensor,
        heads: usize,
        head_dim: usize,
        theta: f32,
        positions: &[u32],
    ) -> Result<Tensor> {
        let bytes: Vec<u8> = positions
            .iter()
            .flat_map(|position| position.to_ne_bytes())
            .collect();
        let position_buffer = DeviceBuffer::alloc(bytes.len(), self.device_id())
            .map_err(|error| Error::Other(format!("MUSA position allocation: {error}")))?;
        position_buffer
            .copy_from_host(&bytes)
            .map_err(|error| Error::Other(format!("MUSA position upload: {error}")))?;
        kernels::rope::apply_vision_2d(
            self.context(),
            input,
            heads,
            head_dim,
            theta,
            &position_buffer,
        )
    }

    pub(crate) fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _seq_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        kernels::attention::noncausal(self.context(), q, k, v, heads, head_dim)?
            .reshape(vec![q.shape().dims()[0], heads * head_dim])
    }

    pub(crate) fn capture_graph<T>(
        &self,
        operation: impl FnMut() -> Result<T>,
    ) -> Result<(Box<dyn Graph>, T)> {
        let (graph, output) = apxinf_musa::capture_isolated(Arc::clone(&self.context), operation)?;
        Ok((Box::new(graph), output))
    }

    pub(crate) fn as_any(&self) -> &dyn Any {
        self
    }
}

pub(crate) fn create(device: Device) -> Result<Arc<RuntimeBackend>> {
    match device {
        Device::Musa(device_id) => Ok(Arc::new(RuntimeBackend::new(device_id)?)),
        device => Err(Error::UnsupportedDevice(device)),
    }
}

pub(crate) fn without_autotune<T>(operation: impl FnOnce() -> T) -> T {
    apxinf_musa::tuning::without_autotune(operation)
}

pub(super) fn try_packed8_bias_activation(
    _backend: &RuntimeBackend,
    _input: &Tensor,
    _bias: &Tensor,
    _activation: i32,
) -> Result<Option<Tensor>> {
    Ok(None)
}
