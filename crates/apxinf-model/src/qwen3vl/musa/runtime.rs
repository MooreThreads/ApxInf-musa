use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// MUSA resources owned by the native Qwen3-VL executor.
pub struct MusaQwenRuntime {
    context: Arc<apxinf_musa::MusaContext>,
}

impl MusaQwenRuntime {
    pub fn new(device: usize) -> Result<Self> {
        Ok(Self {
            context: Arc::new(apxinf_musa::MusaContext::new(device)?),
        })
    }
    pub fn context(&self) -> &apxinf_musa::MusaContext {
        &self.context
    }
    pub fn device(&self) -> Device {
        Device::Musa(self.context.device_id())
    }
    pub fn synchronize(&self) -> Result<()> {
        self.context.synchronize()
    }
    pub fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        apxinf_musa::transfers::to_device(tensor, self.context.device_id())
    }
    pub fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        apxinf_musa::transfers::to_cpu(tensor)
    }
    pub fn output(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        self.context.allocate_output(shape, dtype)
    }
    /// The next prefill may overwrite arena storage while the caller still owns these logits.
    pub(super) fn stable_copy(&self, tensor: &Tensor) -> Result<Tensor> {
        self.require_device(tensor)?;
        let source = apxinf_musa::MusaBuffer::from_tensor(tensor).map_err(Error::Other)?;
        let bytes = source.len();
        let destination =
            apxinf_musa::MusaBuffer::alloc_on(&self.context, bytes).map_err(Error::Other)?;
        destination
            .copy_from_device_async(&source, bytes, self.context.stream())
            .map_err(Error::Other)?;
        destination
            .as_tensor(tensor.shape().clone(), tensor.dtype())
            .map_err(Error::Other)
    }
    pub fn require_device(&self, tensor: &Tensor) -> Result<()> {
        if tensor.device() != self.device() {
            return Err(Error::DeviceMismatch {
                expected: self.device(),
                got: tensor.device(),
            });
        }
        Ok(())
    }
}
