//! W4A16 linear: the language MLP runs the mate G128 two-stage route (FP16
//! scales) and vision runs the G16 native pair (BF16 scales). Both decode the
//! packed weight into BF16 scratch with a custom KN kernel and hand it to
//! MuBLAS, with FP32 accumulation and BF16 boundaries.
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
        if group == 128 {
            // Language MLP G128: vectorized decode over mate's U4 packing
            // (FP16 scales, 128-wide groups) into BF16 [k][n] scratch, then a
            // MuBLAS BF16 GEMM. The fused TileLang route serialized its
            // in-warp unpack with the MMA, so the two-stage split wins even
            // though it streams the decoded weight through HBM.
            if w.k % 128 != 0 {
                return Err(Error::Other("mate W4 route requires K % 128 == 0".into()));
            }
            if w.scales.dtype() != DType::F16
                || w.scales.shape().dims() != [w.n, w.k / 128]
                || w.packed.len()
                    < w.n
                        .checked_mul(w.k / 8)
                        .ok_or_else(|| Error::Other("W4 byte size overflow".into()))?
                        * 4
                || w.packed.device() != ctx.device_id()
            {
                return Err(Error::Other("W4A16 mate format mismatch".into()));
            }
            crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
                m: dims[0],
                n: w.n,
                k: w.k,
            })?;
            let scratch = crate::workspace::w4_session_buffer(ctx, w.n, w.k)?;
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
            return Ok(());
        }
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
        // the actual number of input rows. Language MLP is served by the G128
        // branch above, so only the fixed 512-row Vision shapes remain here:
        // the KN dispatch keys on shape alone and would mis-decode a G16
        // language weight with the mate kernel.
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
        && match (n, k) {
            (3456, 1152) | (1152, 1152) | (4304, 1152) | (1152, 4304) => m == 512,
            _ => false,
        }
}

#[cfg(test)]
mod tests {
    use super::native_shape;
    #[test]
    fn native_shapes_follow_weight_layout_not_prompt_length() {
        assert!(native_shape(512, 3456, 1152, 16));
        assert!(!native_shape(768, 3456, 1152, 16));
        assert!(!native_shape(512, 3456, 1152, 128));
        assert!(!native_shape(522, 32768, 2048, 16));
    }
}
