"""Groot small-M BF16 GEMM: two 16-row MMA tiles per scheduling block."""

import tilelang
import tilelang.language as T
from tilelang.layout import (
    make_gemm_qy2_wmma_fragment_b,
    make_mp22_fragment_packed_shared_layout,
)

from .. import BuildResult
from .mma_16 import VECTOR_LOAD
from .weight_pack import _complete_musa_wrapper, build_weight_packer


def build_block_rows(op) -> BuildResult:
    """Build the readable packed-B Groot kernel used by M=40/41 cases."""
    m, n, k = op["shape"]
    packed_n = op.get("packed_n", n)
    tile = op["tile"]
    tile_m, tile_n, tile_k = 32, tile["n"], tile["k"]
    threads = tile["threads"]
    k_padded = ((k + tile_k - 1) // tile_k) * tile_k
    assert m in (1, 40, 41) and packed_n % tile_n == 0 and k_padded % tile_k == 0
    m_padded = ((m + tile_m - 1) // tile_m) * tile_m
    packets = tile_n * tile_k // (threads * 8)
    vector_load = VECTOR_LOAD.replace("PACKETS", str(packets)).replace("THREADS", str(threads))
    a_layout = make_mp22_fragment_packed_shared_layout(
        (16, 16, 16), "A", False, tile_m, tile_k
    )
    b_layout = make_gemm_qy2_wmma_fragment_b(
        16, tile_n, tile_k, 16, 16, 16, True, (16, 16, 16)
    )
    b_fragment_layout = make_gemm_qy2_wmma_fragment_b(
        16, tile_n, 16, 16, 16, 16, True, (16, 16, 16)
    )

    @tilelang.jit(
        out_idx=[], target="musa", compile_flags=op["flags"],
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
            B: T.Tensor((k_padded // tile_k, packed_n // tile_n, tile_n * tile_k), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(m_padded // tile_m, packed_n // tile_n, threads=threads) as (
                block_row, block_col
            ):
                T.import_source(vector_load)
                a_tile = T.alloc_shared((tile_m, tile_k), "bfloat16")
                b_registers = T.alloc_fragment((tile_n, tile_k), "bfloat16")
                b_fragment = T.alloc_fragment((tile_n, 16), "bfloat16")
                a_fragment = T.alloc_fragment((16, 16), "bfloat16")
                accum0 = T.alloc_fragment((16, tile_n), "float32")
                accum1 = T.alloc_fragment((16, tile_n), "float32")
                a_padded = T.alias_buffer(A, (m_padded, k_padded))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k * 2)
                T.annotate_layout({
                    a_tile: a_layout,
                    b_registers: b_layout,
                    b_fragment: b_fragment_layout,
                })
                T.clear(accum0)
                T.clear(accum1)
                for ko in T.serial(k_padded // tile_k):
                    if k_padded == k:
                        T.copy(
                            a_padded[block_row * tile_m, ko * tile_k], a_tile,
                            src_robust_desc=a_bounds, force_async_copy=True,
                        )
                    else:
                        for row, column in T.Parallel(tile_m, tile_k):
                            a_tile[row, column] = T.if_then_else(
                                block_row * tile_m + row < m and ko * tile_k + column < k,
                                a_padded[block_row * tile_m + row, ko * tile_k + column],
                                0,
                            )
                    T.evaluate(T.call_extern(
                        "handle", "load_weight_fragment",
                        b_registers.access_ptr("w", offset=0, extent=tile_n * tile_k),
                        B.access_ptr(
                            "r", offset=(ko * (packed_n // tile_n) + block_col) * tile_n * tile_k,
                            extent=tile_n * tile_k,
                        ),
                        T.get_thread_binding(),
                    ))
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    for ki in T.serial(tile_k // 16):
                        for row, column in T.Parallel(16, 16):
                            a_fragment[row, column] = a_tile[row, ki * 16 + column]
                        T.copy(b_registers[0:tile_n, ki * 16:(ki + 1) * 16], b_fragment)
                        T.gemm(
                            a_fragment, b_fragment, accum0,
                            transpose_B=True, policy=T.GemmWarpPolicy.FullRow,
                        )
                        for row, column in T.Parallel(16, 16):
                            a_fragment[row, column] = a_tile[16 + row, ki * 16 + column]
                        T.gemm(
                            a_fragment, b_fragment, accum1,
                            transpose_B=True, policy=T.GemmWarpPolicy.FullRow,
                        )
                for row, column in T.Parallel(16, tile_n):
                    if block_row * tile_m + row < m and block_col * tile_n + column < n:
                        C[block_row * tile_m + row, block_col * tile_n + column] = T.cast(
                            accum0[row, column], "bfloat16"
                        )
                    if block_row * tile_m + 16 + row < m and block_col * tile_n + column < n:
                        C[block_row * tile_m + 16 + row, block_col * tile_n + column] = T.cast(
                            accum1[row, column], "bfloat16"
                        )

        return kernel

    return BuildResult(
        {"part": _complete_musa_wrapper(compile_kernel()),
         "pack": build_weight_packer(op)},
        op["scratch_bytes"],
    )
