//! Safe tensor transfer operations and private copy primitives.
//!
//! Same surface as the CUDA backend's, with the names the seam uses: `device`,
//! not `cuda`, so model code written against `crate::accelerator::gpu` reads
//! the same whichever backend is compiled in.
//!
//! **Paired with `apxinf-cuda/src/transfers.rs`.** The two are separate files, not
//! two names for one: this is a port, and a port is allowed to be better in
//! places — the copies differ on most of their lines today, and that is the
//! intended state. What they are not allowed to be is *silently* different,
//! which is how `APXINF_MUSA_ALLOC_CACHE=0` came to mean the opposite of
//! `APXINF_CUDA_ALLOC_CACHE=0`. So a fix here is usually a fix there, and the
//! question before finishing one is whether the other side has the same defect.
//! `tests/backend_parity.rs` locks down the half of this a test can see — the
//! environment knobs — and says why it is only that half.

use crate::error::musa_error;
use crate::ffi;
use crate::MusaBuffer;
use apxinf_core::{Device, Error, Result, Storage, Tensor};

/// Transfer a CPU tensor to a MUSA device.
pub fn to_device(tensor: &Tensor, device_id: usize) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other("tensor is already on GPU".into()));
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let buffer = MusaBuffer::alloc(bytes.len(), device_id).map_err(musa_error)?;
    buffer.copy_from_host(bytes).map_err(musa_error)?;
    Ok(buffer.into_tensor(tensor.shape().clone(), tensor.dtype()))
}

/// Copy a CPU tensor into shape-identical, stable-address device storage.
pub fn copy_cpu_to_device(source: &Tensor, destination: &Tensor) -> Result<()> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    let device_id = match destination.device() {
        Device::Musa(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other(format!(
            "fixed MUSA input mismatch: source {:?} {}, destination {:?} {}",
            source.shape().dims(),
            source.dtype(),
            destination.shape().dims(),
            destination.dtype()
        )));
    }
    let source = source
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let destination = destination
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected MUSA storage".into()))?;
    copy_host_to_device(device_id, source, destination.ptr).map_err(musa_error)
}

/// A page-locked host staging buffer for the fixed-input upload path.
///
/// `musaMemcpy` from ordinary host memory makes the driver stage every byte
/// through a bounce buffer. A pinned source can be DMA'd asynchronously on the
/// context stream, so the prepared GR00T graph fills its four stable-address
/// inputs through one of these and enqueues the uploads before the replay on
/// the same stream.
pub struct PinnedBuffer {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

// SAFETY: the pointer owns a host allocation whose lifetime moves with the
// struct, and the prepared-plan invariant keeps a single thread filling it.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    /// Page-lock `bytes` of host memory.
    pub fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            ffi::check_runtime(ffi::musa::musaMallocHost(&mut ptr, bytes)).map_err(|error| {
                Error::Other(format!(
                    "MUSA pinned host allocation of {bytes} bytes failed: {error}"
                ))
            })?;
        }
        Ok(Self { ptr, bytes })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Copy `source` into the staging buffer and enqueue one asynchronous
    /// host-to-device copy into `destination` on `stream`.
    ///
    /// The destination must belong to the device that owns `stream`. The
    /// caller must not overwrite the staging bytes again until the copy has
    /// completed on the stream; the prepared-input contract synchronizes
    /// between calls.
    pub fn stage_and_upload(
        &self,
        source: &[u8],
        destination: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> std::result::Result<(), String> {
        if source.len() > self.bytes {
            return Err(format!(
                "source is {} bytes, pinned staging buffer is {}",
                source.len(),
                self.bytes
            ));
        }
        if source.is_empty() {
            return Ok(());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), self.ptr.cast::<u8>(), source.len());
            ffi::check_runtime(ffi::musa::musaMemcpyAsync(
                destination,
                self.ptr,
                source.len(),
                ffi::musa::MUSA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ))
        }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::musa::musaFreeHost(self.ptr);
            }
        }
    }
}

/// Copy a CPU tensor into shape-identical device storage through a pinned
/// staging buffer, asynchronously on `stream`.
///
/// The device pointer addresses must have been fixed before capture, which is
/// the same requirement as [`copy_cpu_to_device`]. No synchronization happens
/// here: the caller enqueues everything that reads the destination on
/// `stream` after this call.
pub fn copy_cpu_to_device_pinned(
    source: &Tensor,
    destination: &Tensor,
    staging: &PinnedBuffer,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    let _device_id = match destination.device() {
        Device::Musa(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other(format!(
            "fixed MUSA input mismatch: source {:?} {}, destination {:?} {}",
            source.shape().dims(),
            source.dtype(),
            destination.shape().dims(),
            destination.dtype()
        )));
    }
    let source = source
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let destination = destination
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected MUSA storage".into()))?;
    staging
        .stage_and_upload(source, destination.ptr as *mut std::ffi::c_void, stream)
        .map_err(musa_error)
}

/// Copy a fixed-shape device tensor into another fixed-shape device tensor on
/// the backend stream. This is used by prepared denoise loops to move the
/// result into a persistent ping-pong state buffer before rewinding scratch.
pub fn copy_device_to_device(
    source: &Tensor,
    destination: &Tensor,
    stream: &crate::MusaStream,
) -> Result<()> {
    if source.device() != destination.device()
        || source.shape() != destination.shape()
        || source.dtype() != destination.dtype()
    {
        return Err(Error::Other(format!(
            "fixed MUSA device copy mismatch: source {:?} {}, destination {:?} {}",
            source.shape().dims(),
            source.dtype(),
            destination.shape().dims(),
            destination.dtype()
        )));
    }
    let source = MusaBuffer::from_tensor(source).map_err(musa_error)?;
    let destination = MusaBuffer::from_tensor(destination).map_err(musa_error)?;
    destination
        .copy_from_device_async(&source, source.len(), stream)
        .map_err(musa_error)
}

/// Transfer a MUSA tensor back to CPU.
pub fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    let handle = match tensor.storage() {
        Storage::Gpu { handle, .. } => handle,
        _ => return Err(Error::Other("tensor is not on GPU".into())),
    };
    let device_id = match tensor.device() {
        Device::Musa(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    let bytes = copy_device_to_host(device_id, handle.ptr, handle.len).map_err(musa_error)?;
    Tensor::from_raw(tensor.shape().clone(), tensor.dtype(), Device::Cpu, bytes)
}

pub(crate) fn copy_host_to_device(
    device_id: usize,
    source: &[u8],
    destination: usize,
) -> std::result::Result<(), String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("MUSA device id {device_id} does not fit in i32"))?;
    unsafe {
        ffi::check_runtime(ffi::musa::musaSetDevice(device))?;
        ffi::check_runtime(ffi::musa::musaMemcpy(
            destination as *mut std::ffi::c_void,
            source.as_ptr().cast(),
            source.len(),
            ffi::musa::MUSA_MEMCPY_HOST_TO_DEVICE,
        ))
    }
}

pub(crate) fn copy_device_to_host(
    device_id: usize,
    source: usize,
    len: usize,
) -> std::result::Result<Vec<u8>, String> {
    let mut destination = vec![0u8; len];
    let device = i32::try_from(device_id)
        .map_err(|_| format!("MUSA device id {device_id} does not fit in i32"))?;
    unsafe {
        ffi::check_runtime(ffi::musa::musaSetDevice(device))?;
        ffi::check_runtime(ffi::musa::musaDeviceSynchronize())?;
        ffi::check_runtime(ffi::musa::musaMemcpy(
            destination.as_mut_ptr().cast(),
            source as *const std::ffi::c_void,
            len,
            ffi::musa::MUSA_MEMCPY_DEVICE_TO_HOST,
        ))?;
    }
    Ok(destination)
}
