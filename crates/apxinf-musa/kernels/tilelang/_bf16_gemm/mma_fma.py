import tilelang
import tilelang.language as T
from tilelang.layout import (
    make_gemm_qy2_wmma_fragment_b,
    make_mp22_fragment_packed_shared_layout,
)

from .. import BuildResult
from .weight_pack import _complete_musa_wrapper, build_weight_packer
from .mma_16 import VECTOR_LOAD


def build_hybrid(op):
    """One CTA per N tile: 16 MMA rows and one FMA row, without split-K."""
    m, n, k = op["shape"]
    packed_n = op["packed_n"]
    tile_n, tile_k = op["tile"]["n"], op["tile"]["k"]
    threads = op["tile"]["threads"]
    # The packed shared layout needs complete 16-row atoms. Only 17 input
    # rows exist; robust async copies zero the rest without an external pad.
    a_layout = make_mp22_fragment_packed_shared_layout(
        (16, 16, 16), "A", False, 32, tile_k
    )
    vector_load = VECTOR_LOAD.replace("PACKETS", str(tile_n * tile_k // threads // 8)).replace("THREADS", str(threads))
    weight_layout = make_gemm_qy2_wmma_fragment_b(16, tile_n, tile_k, 16, 16, 16, True, (16, 16, 16))
    fragment_layout = make_gemm_qy2_wmma_fragment_b(16, tile_n, 16, 16, 16, 16, True, (16, 16, 16))
    assert m == 17 and n % tile_n == 0 and k % tile_k == 0
    assert op["splits"] == 1, "M=17 hybrid must produce its output in one kernel"

    @tilelang.jit(
        out_idx=[], target="musa", compile_flags=op["flags"],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True},
    )
    def compile_kernel():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "bfloat16"),
            B: T.Tensor((k // tile_k, packed_n // tile_n, tile_n * tile_k), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(n // tile_n, threads=threads) as block:
                T.import_source(vector_load)
                activation = T.alloc_shared((32, tile_k), "bfloat16")
                a_padded = T.alias_buffer(A, (32, k))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k * 2)
                weight = T.alloc_fragment((tile_n, tile_k), "bfloat16")
                accum = T.alloc_fragment((16, tile_n), "float32")
                activation_fragment = T.alloc_fragment((16, 16), "bfloat16")
                weight_fragment = T.alloc_fragment((tile_n, 16), "bfloat16")
                tail = T.alloc_fragment((tile_n, tile_k), "float32")
                total = T.alloc_fragment((tile_n,), "float32")
                T.annotate_layout({weight: weight_layout, weight_fragment: fragment_layout, tail: weight_layout})
                T.annotate_layout({activation: a_layout})
                T.clear(accum)
                T.clear(tail)
                for chunk in T.serial(k // tile_k):
                    T.copy(
                        a_padded[0, chunk * tile_k], activation,
                        src_robust_desc=a_bounds, force_async_copy=True,
                    )
                    T.evaluate(T.call_extern(
                        "handle", "load_weight_fragment",
                        weight.access_ptr("w", offset=0, extent=tile_n * tile_k),
                        B.access_ptr("r", offset=(chunk * (packed_n // tile_n) + block) * tile_n * tile_k, extent=tile_n * tile_k),
                        T.get_thread_binding(),
                    ))
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    for inner_block in T.unroll(tile_k // 16):
                        for row, inner in T.Parallel(16, 16):
                            activation_fragment[row, inner] = activation[row, inner_block * 16 + inner]
                        T.copy(weight[0:tile_n, inner_block * 16:(inner_block + 1) * 16], weight_fragment)
                        T.gemm(
                            activation_fragment, weight_fragment, accum, transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    for column, inner in T.Parallel(tile_n, tile_k):
                        tail[column, inner] += (
                            T.cast(activation[16, inner], "float32")
                            * T.cast(weight[column, inner], "float32")
                        )
                    T.sync_threads()
                for row, column in T.Parallel(16, tile_n):
                    C[row, block * tile_n + column] = T.cast(accum[row, column], "bfloat16")
                T.reduce_sum(tail, total, dim=1)
                for column in T.Parallel(tile_n):
                    C[16, block * tile_n + column] = T.cast(total[column], "bfloat16")
        return kernel

    kernels = {
        "part": _complete_musa_wrapper(compile_kernel()),
        "pack": build_weight_packer(op),
    }
    return BuildResult(kernels, 0)
