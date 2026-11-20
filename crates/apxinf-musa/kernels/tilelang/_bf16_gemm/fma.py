import tilelang
import tilelang.language as T

from .. import BuildResult
from .weight_pack import _complete_musa_wrapper, build_weight_packer


def build_fma(op):
    m, n, k = op["shape"]
    tile = op.get("tile", {})
    tile_n = tile.get("n", op.get("fma_tile_n", 128))
    tile_k = tile.get("k", op.get("fma_tile_k", 64))
    threads = tile.get("threads", op.get("fma_threads", 512))
    assert m == 1 and k % tile_k == 0 and n % tile_n == 0

    @tilelang.jit(out_idx=[], target="musa", compile_flags=op["flags"])
    def compile_kernel():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "bfloat16"),
            B: T.Tensor((n * k,), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(n // tile_n, threads=threads) as block:
                staged = T.alloc_shared((k,), "bfloat16")
                accum = T.alloc_fragment((tile_n, tile_k), "float32")
                total = T.alloc_fragment((tile_n,), "float32")
                T.copy(A[0, 0:k], staged)
                T.sync_threads()
                T.clear(accum)
                for chunk in T.serial(k // tile_k):
                    for column, inner in T.Parallel(tile_n, tile_k):
                        output_column = block * tile_n + column
                        weight_inner = chunk * tile_k + inner
                        accum[column, inner] += (
                            T.cast(staged[weight_inner], "float32")
                            * T.cast(B[output_column * k + weight_inner], "float32")
                        )
                T.reduce_sum(accum, total, dim=1)
                for column in T.Parallel(tile_n):
                    C[0, block * tile_n + column] = T.cast(total[column], "bfloat16")
        return kernel

    return BuildResult(
        {"part": _complete_musa_wrapper(compile_kernel()),
         "pack": build_weight_packer(op)},
        0,
    )
