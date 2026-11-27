//! Synchronous device-side greedy selection. Only the scalar result crosses to the host.
use crate::MusaContext;
use apxinf_core::{DType, Device, Error, Result, Tensor};

pub fn greedy_argmax(ctx: &MusaContext, logits: &Tensor) -> Result<u32> {
    let [1, width] = *logits.shape().dims() else {
        return Err(Error::Other("expected one row of logits".into()));
    };
    let n = i32::try_from(width).map_err(|_| Error::Other("vocabulary too large".into()))?;
    if n == 0 || logits.device() != Device::Musa(ctx.device_id()) {
        return Err(Error::Other("empty or wrong-device logits".into()));
    }
    let kind = match logits.dtype() {
        DType::BF16 => 0,
        DType::F16 => 1,
        DType::F32 => 2,
        DType::I32 => 3,
        _ => return Err(Error::Other("unsupported logits dtype".into())),
    };
    #[cfg(apxinf_musa_available)]
    {
        let input = crate::MusaBuffer::from_tensor(logits).map_err(Error::Other)?;
        if input.len() < width * logits.dtype().size_in_bytes() {
            return Err(Error::Other("logits storage is too small".into()));
        }
        let mut id = -1;
        let status = unsafe {
            crate::ffi::custom::apxinf_musa_mudnn_greedy(
                input.ptr(),
                n,
                kind,
                &mut id,
                ctx.stream().handle(),
            )
        };
        crate::ffi::custom::check_mudnn(status, || "greedy argmax".into()).map_err(Error::Other)?;
        if !(0..n).contains(&id) {
            return Err(Error::Other("invalid logits".into()));
        }
        Ok(id as u32)
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = kind;
        Err(Error::Other("MUSA greedy argmax unavailable".into()))
    }
}
