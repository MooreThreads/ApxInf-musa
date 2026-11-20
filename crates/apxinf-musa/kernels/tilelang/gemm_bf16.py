"""Small-M BF16 matrix multiplication with FP32 accumulation."""

from ._bf16_gemm.registry import OPS

__all__ = ["OPS", "build"]


def build(op, entries):
    """Build the algorithm selected by the registry."""
    algorithm = op["algorithm"]
    if algorithm == "small-k":
        from ._bf16_gemm.small_k import build
    elif algorithm == "split-k":
        from ._bf16_gemm.split_k import build
    elif algorithm == "mma_16":
        from ._bf16_gemm.mma_16 import build_packed as build
    elif algorithm == "mma_16x2":
        from ._bf16_gemm.mma_16x2 import build_block_rows as build
    elif algorithm == "mma_fma":
        from ._bf16_gemm.mma_fma import build_hybrid as build
    elif algorithm == "fma":
        from ._bf16_gemm.fma import build_fma as build
    else:
        raise ValueError(f"Unsupported BF16 GEMM algorithm: {algorithm}")
    return build(op)
