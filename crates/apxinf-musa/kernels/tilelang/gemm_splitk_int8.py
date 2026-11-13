"""gemm-splitk-int8: TileLang INT8 split-K GEMM (part + reduce).

算子契约（与 kernels/gemm/tilelang.rs 的 extern 声明一致）：
  part(a, b, partials, stream) -> i32
  reduce(partials, a_scale, b_scale, out, stream) -> i32
  <entry>_init() -> i32

布局：A 行主 INT8 [m, k]；B 为 KN（行主 [k, n]）INT8——需要模型侧预打包的
KN 权重副本；partials INT32 [splits, m, n]；scales FP16（激活按行、权重按列）；
输出 BF16 [m, n]。
"""

from . import BuildResult


def _operator(name, k):
    m, n, splits = 10, 1024, 4
    return {
        "id": name,
        "type": "gemm-splitk-int8",
        "supported_arches": ["mp_22"],
        "shape": (m, n, k),
        "tile": {"bm": 32, "bn": 128, "bk": 32, "splits": splits},
        "flags": ["-Od3", "-O2"],
        "status": "enabled",
        "stages": ["part", "reduce"],
        "scratch_bytes": splits * m * n * 4,
    }


OPS = [
    _operator("pi05-int8-down", 4096),
    _operator("pi05-int8-attn-o", 2048),
]


def build(op, entries):
    """编译单个算子，返回统一的 BuildResult。

    入口符号唯一性由 tools/tilelang/gen.py 在生成源码上改名保证
    （call/init/get_last_error 与 kernel_kernel → {entry}_*），此处无需感知。
    tile 配置实测依据：dev217 单算子调优（down 53-54us vs native 72-75us）。
    """
    import tilelang
    import tilelang.language as T

    m, n, k = op["shape"]
    splits = op["tile"]["splits"]
    bm, bn, bk = op["tile"]["bm"], op["tile"]["bn"], op["tile"]["bk"]
    flags = op["flags"]

    @tilelang.jit(out_idx=[], target="musa", execution_backend="cython", compile_flags=flags)
    def make_part():
        @T.prim_func
        def kernel(
            A: T.Tensor((m, k), "int8"),
            B: T.Tensor((k, n), "int8"),
            P: T.Tensor((splits, m, n), "int32"),
        ):
            with T.Kernel(T.ceildiv(n, bn), splits, threads=256) as (bx, split):
                aa = T.alloc_shared((bm, bk), "int8")
                bb = T.alloc_shared((bk, bn), "int8")
                cc = T.alloc_fragment((bm, bn), "int32")
                padded_a = T.alias_buffer(A, (bm, k))
                a_bounds = T.make_robust_desc(T.address_of(A[0, 0]), m * k)
                T.clear(cc)
                for i in T.Pipelined(T.ceildiv(k // splits, bk), num_stages=1):
                    T.copy(padded_a[0, split * (k // splits) + i * bk], aa, src_robust_desc=a_bounds)
                    T.copy(B[split * (k // splits) + i * bk, bx * bn], bb)
                    T.gemm(aa, bb, cc, transpose_B=False)
                for row, column in T.Parallel(bm, bn):
                    if row < m:
                        P[split, row, bx * bn + column] = cc[row, column]
        return kernel

    @tilelang.jit(out_idx=[], target="musa", execution_backend="cython", compile_flags=flags)
    def make_reduce():
        @T.prim_func
        def kernel(
            P: T.Tensor((splits, m, n), "int32"),
            a_scale: T.Tensor((m,), "float16"),
            b_scale: T.Tensor((n,), "float16"),
            C: T.Tensor((m, n), "bfloat16"),
        ):
            with T.Kernel(T.ceildiv(m * n, 256), threads=256) as bx:
                tx = T.get_thread_binding()
                index = bx * 256 + tx
                row = index // n
                col = index % n
                if index < m * n:
                    value = T.alloc_local((1,), "int32")
                    value[0] = 0
                    for split in T.serial(splits):
                        value[0] += P[split, row, col]
                    row_scaled = T.Cast("float32", value[0]) * T.Cast("float32", a_scale[row])
                    C[row, col] = T.Cast("bfloat16", row_scaled * T.Cast("float32", b_scale[col]))
        return kernel

    return BuildResult({"part": make_part(), "reduce": make_reduce()}, splits * m * n * 4)
