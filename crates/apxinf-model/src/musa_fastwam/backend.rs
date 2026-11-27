//! Safe MUSA seam for the backend-native Fast-WAM port.
//!
//! Family code composes these model-neutral operations. Raw FFI and provider
//! descriptors remain private to `apxinf-musa`.

use std::sync::Arc;

use apxinf_core::{DType, Result, Shape, Tensor};
use apxinf_musa::ops::{self, BinaryOp, GemmArgs};

#[derive(Clone)]
pub(crate) struct FastWamBackend {
    context: Arc<apxinf_musa::MusaContext>,
}

impl FastWamBackend {
    pub(crate) fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            context: Arc::new(apxinf_musa::MusaContext::new(device_id)?),
        })
    }

    pub(crate) fn context(&self) -> &apxinf_musa::MusaContext {
        &self.context
    }

    pub(crate) fn context_arc(&self) -> Arc<apxinf_musa::MusaContext> {
        Arc::clone(&self.context)
    }

    pub(crate) fn device_id(&self) -> usize {
        self.context.device_id()
    }

    pub(crate) fn synchronize(&self) -> Result<()> {
        self.context.synchronize()
    }

    pub(crate) fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        apxinf_musa::transfers::to_device(tensor, self.device_id())
    }

    /// Apply a checkpoint linear whose stored matrix is canonical `[K, N]`.
    /// Activations are matrices `[M, K]`; the returned bias add is BF16 and
    /// follows NumPy last-axis broadcasting.
    pub(crate) fn linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor> {
        let [rows, inner] = input.shape().dims() else {
            return Err(apxinf_core::Error::Other(format!(
                "Fast-WAM linear input must be [M,K], got {:?}",
                input.shape().dims()
            )));
        };
        let [weight_inner, output] = weight.shape().dims() else {
            return Err(apxinf_core::Error::Other(format!(
                "Fast-WAM linear weight must be [K,N], got {:?}",
                weight.shape().dims()
            )));
        };
        if inner != weight_inner || bias.shape().dims() != [*output] {
            return Err(apxinf_core::Error::Other(format!(
                "Fast-WAM linear geometry mismatch: input {:?}, weight {:?}, bias {:?}",
                input.shape().dims(),
                weight.shape().dims(),
                bias.shape().dims()
            )));
        }
        let mut projection = self
            .context
            .allocate_output(Shape::new(vec![*rows, *output]), DType::BF16)?;
        ops::gemm(
            &self.context,
            GemmArgs::new(input, weight, &mut projection),
        )?;
        ops::binary_bf16(&self.context, &projection, bias, BinaryOp::Add)
    }
}
