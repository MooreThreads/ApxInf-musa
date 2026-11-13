"""Selected M=10 BF16 projections: exact BF16 weights, FP32 accumulation."""

from ._bf16_decode.registry import OPS

__all__ = ["OPS", "build"]


def build(op, entries):
    """Dispatch a registry entry; each algorithm returns the shared BuildResult."""
    if op["algorithm"] == "packed-register":
        from ._bf16_decode.decode_packed import build_packed

        return build_packed(op)
    from ._bf16_decode.decode_small import build as build_small

    return build_small(op)
