"""One-time BF16 permutations used by the existing offline TileLang loader."""
import tilelang
import tilelang.language as T
from tilelang.jit.adapter.musa_wrapper import TLMUSASourceWrapper
from tvm.target import Target


def _complete_musa_wrapper(kernel):
    adapter = kernel.adapter
    device_source = adapter.get_kernel_source(kernel_only=True)
    with Target("musa"):
        wrapper = TLMUSASourceWrapper(
            adapter.ir_module, device_source, Target("musa")
        )
    host_source = wrapper.lib_code[len(wrapper.source):]
    adapter.get_device_source = (
        lambda: "#include <tl_templates/musa/gemm_mp22.h>\n" + device_source
    )
    adapter.get_host_source = lambda: host_source
    return kernel


def build_weight_packer(op):
    _, n, k = op["shape"]
    packed_n = op.get("packed_n", n)
    if op["algorithm"] not in ("mma_16", "mma_16x2", "mma_fma"):
        split_k = op["algorithm"] == "split-k"
        chunk = k // op["splits"]
        @tilelang.jit(out_idx=[], target="musa")
        def compile_small():
            @T.prim_func
            def pack(Weight: T.Tensor((k, n), "bfloat16"),
                     Packed: T.Tensor((packed_n * k,), "bfloat16")):
                with T.Kernel(T.ceildiv(packed_n * k, 256), threads=128) as block:
                    for local in T.Parallel(256):
                        index = block * 256 + local
                        if index < packed_n * k:
                            if split_k:
                                split = index // (n * chunk)
                                row = (index // chunk) % n
                                column = index % chunk
                                Packed[index] = Weight[split * chunk + column, row]
                            else:
                                Packed[index] = T.if_then_else(
                                    index // k < n, Weight[index % k, index // k], 0
                                )
            return pack
        return _complete_musa_wrapper(compile_small())

    tile = op["tile"]
    tile_n, tile_k, threads = tile["n"], tile["k"], tile["threads"]
    k_padded = ((k + tile_k - 1) // tile_k) * tile_k
    # The MMA fragment stores eight BF16 words per thread, then advances
    # to the next group of eight registers. Invert the public B layout so
    # every loop iteration writes one unique, contiguous physical word.
    @tilelang.jit(out_idx=[], target="musa")
    def compile_packed():
        @T.prim_func
        def pack(Weight: T.Tensor((k, n), "bfloat16"),
                 Packed: T.Tensor((k_padded // tile_k, packed_n // tile_n, tile_n * tile_k), "bfloat16")):
            with T.Kernel(k_padded // tile_k, packed_n // tile_n, threads=128) as (ko, no):
                for physical in T.Parallel(tile_n * tile_k):
                    lane = physical % (threads * 8)
                    thread = lane // 8
                    register = (physical // (threads * 8)) * 8 + lane % 8
                    row = (thread // 32) * 16 + (thread % 32) // 4 + (register % 8) // 4 * 8
                    column = (register // 8) * 16 + (register % 4) // 2 * 8 + (thread % 4) * 2 + register % 2
                    Packed[ko, no, physical] = T.if_then_else(
                        no * tile_n + row < n and ko * tile_k + column < k,
                        Weight[ko * tile_k + column, no * tile_n + row],
                        0,
                    )
        return pack
    return _complete_musa_wrapper(compile_packed())
