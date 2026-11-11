//! Safe MUSA surface used by the backend-native GR00T port.
//!
//! Model code sees concrete resources and model-neutral operations here. Raw
//! FFI and provider-specific entry points remain private to `apxinf-musa`.

use std::any::Any;
use std::sync::Arc;

use apxinf_core::{Device, Error, Graph, Result, Tensor};

pub(crate) use apxinf_musa::MusaBuffer as DeviceBuffer;
pub(crate) use apxinf_musa::MusaContext as Context;
pub(crate) use apxinf_musa::{CapturedGraph, OperationShape};

pub(crate) mod transfers {
    pub(crate) use apxinf_musa::transfers::{copy_cpu_to_device, to_cpu, to_device};
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
