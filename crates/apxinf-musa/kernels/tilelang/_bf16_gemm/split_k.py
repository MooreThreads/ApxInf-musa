"""Split-K BF16 MMA with FP32 reduction."""

import tilelang
import tilelang.language as T
from tilelang.layout import make_mp22_fragment_packed_shared_layout

from .. import BuildResult
from .weight_pack import build_weight_packer


def build(op) -> BuildResult:
    return BuildResult({"part": build_output_partials(op), "reduce": build_output_reduction(op), "pack": build_weight_packer(op)}, op["scratch_bytes"])


def build_output_partials(op):
    """Eight K splits give the narrow output projection one CTA per MP."""
    m, n, k = op["shape"]
    splits = op["splits"]
    tile_m, chunk = 16, k // splits
    layout_a = make_mp22_fragment_packed_shared_layout((16, 16, 16), "A", False, tile_m, chunk)
    layout_b = make_mp22_fragment_packed_shared_layout((16, 16, 16), "B", True, n, chunk)

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
            B: T.Tensor((splits, n, chunk), "bfloat16"),
            P: T.Tensor((splits, m, n), "float32"),
        ):
            with T.Kernel(splits, threads=64) as split:
                a_tile = T.alloc_shared((tile_m, chunk), "bfloat16")
                b_tile = T.alloc_shared((n, chunk), "bfloat16")
                accum = T.alloc_fragment((tile_m, n), "float32")
                a_padded = T.alias_buffer(A, (tile_m, k))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k * 2)
                T.annotate_layout({a_tile: layout_a, b_tile: layout_b})
                T.clear(accum)
                T.copy(a_padded[0, split * chunk], a_tile, src_robust_desc=a_bounds)
                T.copy(B[split, 0, 0], b_tile)
                T.gemm(a_tile, b_tile, accum, transpose_B=True)
                for row, column in T.Parallel(tile_m, n):
                    if row < m:
                        P[split, row, column] = accum[row, column]

        return kernel

    return compile_kernel()


def build_output_reduction(op):
    """Sum FP32 partials and round once to BF16."""
    m, n, _ = op["shape"]
    splits = op["splits"]

    @tilelang.jit(
        out_idx=[], target="musa", execution_backend="cython", compile_flags=op["flags"]
    )
    def compile_kernel():
        @T.prim_func
        def kernel(P: T.Tensor((splits, m, n), "float32"), C: T.Tensor((m, n), "bfloat16")):
            with T.Kernel(T.ceildiv(m * n, 256), threads=256) as block:
                index = block * 256 + T.get_thread_binding()
                if index < m * n:
                    row, column = index // n, index % n
                    total = T.alloc_local((1,), "float32")
                    total[0] = 0
                    for split in T.unroll(splits):
                        total[0] += P[split, row, column]
                    C[row, column] = T.cast(total[0], "bfloat16")

        return kernel

    return compile_kernel()
