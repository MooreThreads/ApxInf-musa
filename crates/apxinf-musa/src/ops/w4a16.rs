//! W4A16 linear: vectorized G16 decode to BF16 scratch followed by MuBLAS BF16 GEMM.
//! Language MLP and Vision use the same original G16 symmetric quantization.

use crate::{MusaBuffer, MusaContext};
use apxinf_core::{DType, Error, Result, Tensor};

pub struct W4A16WeightView<'a> {
    pub packed: &'a MusaBuffer,
    pub scales: &'a Tensor,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
}

pub fn linear(
    ctx: &MusaContext,
    a: &Tensor,
    w: W4A16WeightView<'_>,
    out: &mut Tensor,
) -> Result<()> {
    let dims = a.shape().dims();
    let group = w.group_size;
    if group == 0 {
        return Err(Error::Other("W4 group size must be non-zero".into()));
    }
    let kp =
        w.k.checked_add(group - 1)
            .ok_or_else(|| Error::Other("W4 K overflow".into()))?
            / group
            * group;
    if dims.len() != 2
        || dims[1] != w.k
        || a.dtype() != DType::BF16
        || out.dtype() != DType::BF16
        || out.shape().dims() != [dims[0], w.n]
    {
        return Err(Error::Other("W4A16 shape/dtype mismatch".into()));
    }
    if a.device() != apxinf_core::Device::Musa(ctx.device_id())
        || out.device() != a.device()
        || w.scales.device() != a.device()
    {
        return Err(Error::Other(
            "W4 tensors must belong to the context device".into(),
        ));
    }
    #[cfg(apxinf_musa_available)]
    {
        let aa = MusaBuffer::from_tensor(a).map_err(Error::Other)?;
        let ss = MusaBuffer::from_tensor(w.scales).map_err(Error::Other)?;
        let cc = MusaBuffer::from_tensor(out).map_err(Error::Other)?;
        if w.scales.dtype() != DType::BF16 || w.scales.shape().dims() != [w.n, kp / group] {
            return Err(Error::Other("W4A16 shape/dtype mismatch".into()));
        }
        if w.packed.len()
            < w.n
                .checked_mul(kp / 2)
                .ok_or_else(|| Error::Other("W4 byte size overflow".into()))?
            || w.packed.device() != ctx.device_id()
        {
            return Err(Error::Other(
                "W4 packed storage size/device mismatch".into(),
            ));
        }
        // The decode kernels depend on the weight shape; the BF16 GEMM accepts
        // The vectorized G16 decoder accepts the actual number of Vision or language rows.
        if !native_shape(dims[0], w.n, w.k, group) {
            return Err(Error::Other(format!(
                "unsupported native W4A16 shape {}x{}x{} (group {})",
                dims[0], w.n, w.k, group
            )));
        }
        let scratch = crate::workspace::w4_session_buffer(ctx, w.n, w.k)?;
        crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
            m: dims[0],
            n: w.n,
            k: w.k,
        })?;
        unsafe {
            crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_w4_decode_kn(
                w.packed.ptr(),
                ss.ptr(),
                scratch.ptr(),
                w.n as i32,
                w.k as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
            crate::ffi::mublas::check_mublas(crate::ffi::mublas::apxinf_musa_bf16_gemm(
                aa.ptr(),
                scratch.ptr(),
                cc.ptr(),
                dims[0] as i32,
                w.n as i32,
                w.k as i32,
                1.,
                ctx.stream().handle(),
            ))
            .map_err(Error::Other)?;
        }
        Ok(())
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, w, kp);
        Err(Error::Other("W4A16 requires a native MUSA build".into()))
    }
}

fn native_shape(m: usize, n: usize, k: usize, group: usize) -> bool {
    group == 16
        && (1..=i32::MAX as usize).contains(&m)
        && matches!(
            (n, k),
            (32768, 2048)
                | (2048, 16384)
                | (3456, 1152)
                | (1152, 1152)
                | (4304, 1152)
                | (1152, 4304)
        )
}

#[cfg(test)]
mod tests {
    use super::native_shape;
    #[test]
    fn native_shapes_follow_weight_layout_not_prompt_length() {
        for m in [522, 565, 576] {
            assert!(native_shape(m, 32768, 2048, 16));
            assert!(native_shape(m, 2048, 16384, 16));
        }
        assert!(native_shape(512, 3456, 1152, 16));
        for (n, k) in [(3456, 1152), (1152, 1152), (4304, 1152), (1152, 4304)] {
            for m in [256, 512, 768] {
                assert!(native_shape(m, n, k, 16));
            }
            assert!(!native_shape(0, n, k, 16));
            assert!(!native_shape(i32::MAX as usize + 1, n, k, 16));
        }
        assert!(!native_shape(512, 3456, 1152, 128));
        assert!(!native_shape(0, 3456, 1152, 16));
    }
}
