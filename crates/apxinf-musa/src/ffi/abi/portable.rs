//! ABI declarations for the bring-up portable MUSA operators.
use super::super::musa::musaError_t;
use std::ffi::c_void;
#[cfg(apxinf_musa_available)]
extern "C" {
    pub fn apxinf_musa_portable_point(
        a: *const c_void,
        b: *const c_void,
        o: *mut c_void,
        n: i32,
        op: i32,
        factor: f32,
        stream: *mut c_void,
    ) -> musaError_t;
    pub fn apxinf_musa_portable_rope(
        a: *const c_void,
        o: *mut c_void,
        positions: *const c_void,
        n: i32,
        heads: i32,
        head_dim: i32,
        theta: f32,
        mode: i32,
        height_sections: i32,
        width_sections: i32,
        stream: *mut c_void,
    ) -> musaError_t;
    pub fn apxinf_musa_portable_attention(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        causal: i32,
        stream: *mut c_void,
    ) -> musaError_t;
    pub fn apxinf_musa_portable_embedding(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        rows: i32,
        width: i32,
        stream: *mut c_void,
    ) -> musaError_t;
}
