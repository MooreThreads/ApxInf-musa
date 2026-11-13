"""One-time BF16 permutations used by the existing offline TileLang loader."""
import tilelang
import tilelang.language as T


def build_weight_packer(op):
    _, n, k = op["shape"]
    if op["algorithm"] != "packed-register":
        split_k = op["algorithm"] == "output-split-k"
        chunk = k // op["splits"]
        @tilelang.jit(out_idx=[], target="musa", execution_backend="cython")
        def compile_small():
            @T.prim_func
            def pack(Weight: T.Tensor((k, n), "bfloat16"),
                     Packed: T.Tensor((n * k,), "bfloat16")):
                with T.Kernel(T.ceildiv(n * k, 256), threads=128) as block:
                    for local in T.Parallel(256):
                        index = block * 256 + local
                        if index < n * k:
                            if split_k:
                                split = index // (n * chunk)
                                row = (index // chunk) % n
                                column = index % chunk
                                Packed[index] = Weight[split * chunk + column, row]
                            else:
                                Packed[index] = Weight[index % k, index // k]
            return pack
        return compile_small()

    tile = op["tile"]
    tile_n, tile_k, threads = tile["n"], tile["k"], tile["threads"]
    # The MMA fragment stores eight BF16 words per thread, then advances
    # to the next group of eight registers. Invert the public B layout so
    # every loop iteration writes one unique, contiguous physical word.
    @tilelang.jit(out_idx=[], target="musa", execution_backend="cython")
    def compile_packed():
        @T.prim_func
        def pack(Weight: T.Tensor((k, n), "bfloat16"),
                 Packed: T.Tensor((k // tile_k, n // tile_n, tile_n * tile_k), "bfloat16")):
            with T.Kernel(k // tile_k, n // tile_n, threads=128) as (ko, no):
                for physical in T.Parallel(tile_n * tile_k):
                    thread = (physical // 8) % threads
                    register = (physical // (threads * 8)) * 8 + physical % 8
                    row = (thread // 32) * 16 + (thread % 32) // 4 + (register % 8) // 4 * 8
                    column = (register // 8) * 16 + (register % 4) // 2 * 8 + (thread % 4) * 2 + register % 2
                    Packed[ko, no, physical] = Weight[ko * tile_k + column, no * tile_n + row]
        return pack
    return compile_packed()
