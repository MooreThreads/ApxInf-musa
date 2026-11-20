"""Small-K BF16 MMA."""

import tilelang
import tilelang.language as T

from .. import BuildResult
from .weight_pack import build_weight_packer


def build(op) -> BuildResult:
    return BuildResult({"part": build_input_projection(op), "pack": build_weight_packer(op)}, op["scratch_bytes"])


def build_input_projection(op):
    """One warp handles a 16-column tile, with A and B directly in registers."""
    m, n, k = op["shape"]
    tile_m, tile_n = 16, 16

    @tilelang.jit(
        out_idx=[],
        target="musa",
        execution_backend="cython",
        compile_flags=op["flags"],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION: True,
        },
    )
    def compile_kernel():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "bfloat16"),
            B: T.Tensor((n, k), "bfloat16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(n // tile_n, threads=32) as block:
                a_registers = T.alloc_fragment((tile_m, k), "bfloat16")
                b_registers = T.alloc_fragment((tile_n, k), "bfloat16")
                accum = T.alloc_fragment((tile_m, tile_n), "float32")
                # This virtual padded view allocates no memory. The descriptor
                # bounds reads to the real ten rows and supplies zero afterward.
                a_padded = T.alias_buffer(A, (tile_m, k))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k * 2)
                T.clear(accum)
                T.copy(a_padded, a_registers, src_robust_desc=a_bounds)
                T.copy(B[block * tile_n, 0], b_registers)
                T.gemm(a_registers, b_registers, accum, transpose_B=True)
                for row, column in T.Parallel(tile_m, tile_n):
                    if row < m:
                        C[row, block * tile_n + column] = T.cast(accum[row, column], "bfloat16")

        return kernel

    return compile_kernel()
