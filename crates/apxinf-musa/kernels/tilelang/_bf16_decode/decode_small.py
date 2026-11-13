"""Readable MP22 kernels for pi0.5's two small decode projections.

Input projection:  A[10, 32]   @ B[32, 1024].
Output projection: A[10, 1024] @ B[1024, 32].

Weights remain bit-for-bit BF16. The GPU weight packer prepares them once.
The input projection uses one kernel; the output projection uses a partial GEMM
followed by an FP32 reduction. Both launches belong in any
reported output-projection GEMM time.
"""

import tilelang
import tilelang.language as T
from tilelang.layout import make_mp22_fragment_packed_shared_layout

from .. import BuildResult
from .weight_pack import build_weight_packer


def build(op) -> BuildResult:
    """Build the selected small-projection algorithm from registry metadata."""
    if op["algorithm"] == "input-register":
        kernels = {"part": build_input_projection(op)}
    elif op["algorithm"] == "output-split-k":
        kernels = {"part": build_output_partials(op), "reduce": build_output_reduction(op)}
    else:
        raise ValueError(f"Unsupported small decode algorithm: {op['algorithm']}")
    kernels["pack"] = build_weight_packer(op)
    return BuildResult(kernels, op["scratch_bytes"])


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
