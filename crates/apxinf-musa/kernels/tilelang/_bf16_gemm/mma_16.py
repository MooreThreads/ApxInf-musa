"""Packed BF16 register-B MMA with optional K16 slices and FP32 accumulation."""

import tilelang
import tilelang.language as T
from tilelang.layout import (
    make_gemm_qy2_wmma_fragment_b,
    make_mp22_fragment_packed_shared_layout,
)

from .. import BuildResult
from .weight_pack import build_weight_packer


VECTOR_LOAD = r"""
__device__ __forceinline__ void load_weight_fragment(
    void* destination, const void* tile, int thread) {
  uint4* registers = reinterpret_cast<uint4*>(destination);
  const uint4* source = reinterpret_cast<const uint4*>(tile);
  #pragma unroll
  for (int packet = 0; packet < PACKETS; ++packet) {
    registers[packet] = source[packet * THREADS + thread];
  }
}
"""


def build_packed(op) -> BuildResult:
    """Build a packed register-B kernel with optional serial K16 MMA slices."""
    m, n, k = op["shape"]
    tile_m = 16
    tile = op["tile"]
    tile_n, tile_k, threads = tile["n"], tile["k"], tile["threads"]
    slice_k16 = tile["slice_k16"]
    assert m == 10 and n % tile_n == 0 and k % tile_k == 0
    packets = tile_n * tile_k // (threads * 8)
    vector_load = VECTOR_LOAD.replace("PACKETS", str(packets)).replace("THREADS", str(threads))
    a_layout = make_mp22_fragment_packed_shared_layout((16, 16, 16), "A", False, tile_m, tile_k)

    # A full weight tile stays prefetched; only the MMA operands are sliced.
    b_layout = make_gemm_qy2_wmma_fragment_b(
        tile_m, tile_n, tile_k, 16, 16, 16, True, (16, 16, 16)
    )

    @tilelang.jit(
        out_idx=[], target="musa", execution_backend="cython", compile_flags=op["flags"],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION: True,
        },
    )
    def compile_kernel():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "bfloat16"),
            B: T.Tensor((k // tile_k, n // tile_n, tile_n * tile_k), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(n // tile_n, threads=threads) as block:
                T.import_source(vector_load)
                a_tile = T.alloc_shared((tile_m, tile_k), "bfloat16")
                b_registers = T.alloc_fragment((tile_n, tile_k), "bfloat16")
                if slice_k16:
                    a_fragment = T.alloc_fragment((tile_m, 16), "bfloat16")
                    b_fragment = T.alloc_fragment((tile_n, 16), "bfloat16")
                accum = T.alloc_fragment((tile_m, tile_n), "float32")
                a_padded = T.alias_buffer(A, (tile_m, k))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k * 2)
                T.annotate_layout({a_tile: a_layout})
                if slice_k16:
                    T.annotate_layout({b_registers: b_layout})
                T.clear(accum)
                for ko in T.serial(k // tile_k):
                    T.sync_threads()
                    T.copy(
                        a_padded[0, ko * tile_k], a_tile,
                        src_robust_desc=a_bounds, force_async_copy=True,
                    )
                    # The helper writes the physical register order expected by MMA.
                    T.evaluate(T.call_extern(
                        "handle", "load_weight_fragment",
                        b_registers.access_ptr("w", offset=0, extent=tile_n * tile_k),
                        B.access_ptr(
                            "r", offset=(ko * (n // tile_n) + block) * tile_n * tile_k,
                            extent=tile_n * tile_k,
                        ),
                        T.get_thread_binding(),
                    ))
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    if slice_k16:
                        for ki in T.serial(tile_k // 16):
                            # Explicit loads avoid the MP22 shared-to-fragment LDSM path.
                            for row, column in T.Parallel(tile_m, 16):
                                a_fragment[row, column] = a_tile[row, ki * 16 + column]
                            T.copy(b_registers[0:tile_n, ki * 16:(ki + 1) * 16], b_fragment)
                            T.gemm(a_fragment, b_fragment, accum, transpose_B=True)
                    else:
                        T.gemm(a_tile, b_registers, accum, transpose_B=True)
                for row, column in T.Parallel(tile_m, tile_n):
                    if row < m:
                        C[row, block * tile_n + column] = T.cast(accum[row, column], "bfloat16")

        return kernel

    return BuildResult(
        {"part": compile_kernel(), "pack": build_weight_packer(op)},
        op["scratch_bytes"],
    )
