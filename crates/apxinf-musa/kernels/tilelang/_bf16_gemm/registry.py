"""Single source for small-M BF16 GEMM shapes, algorithms and offline build metadata.

Adding a projection using an existing algorithm requires only one registry row.
Builders and the weight packer consume the same complete operator description.
"""


def _gemm(m, n, k, algorithm, *, tile=None, splits=1, flags=None, padded=False):
    op = {
        "id": f"bf16-m{m}-n{n}-k{k}",
        "type": "gemm-bf16",
        "supported_arches": ["mp_22"],
        "shape": (m, n, k),
        "algorithm": algorithm,
        "tile": tile,
        "splits": splits,
        "flags": flags if flags is not None else ["-Od3", "-O2"],
        "status": "enabled",
        "stages": ["part", "pack", "reduce"] if splits > 1 else ["part", "pack"],
        "scratch_bytes": splits * m * n * 4 if splits > 1 else 0,
    }
    if padded:
        op.update(packed_n=((n + 63) // 64) * 64, packed_k=((k + 63) // 64) * 64)
    return op


def _block_rows(m, n, k, *, tile=None, algorithm="mma_16x2"):
    return _gemm(m, n, k, algorithm, padded=True,
                 tile=tile or dict(n=64, k=64, threads=128),
                 flags=["-Od3", "-O2", "-ffp-contract=fast"])


OPS = [
    _gemm(10, 1024, 32, "small-k",
                flags=["-Od3", "-O2", "-ffp-contract=fast"]),
    _gemm(10, 32, 1024, "split-k", splits=8),
    _gemm(10, 2560, 1024, "mma_16",
                tile=dict(n=128, k=128, threads=256, slice_k16=False)),
    _gemm(10, 1024, 2048, "mma_16",
                tile=dict(n=64, k=256, threads=128, slice_k16=True)),
    _gemm(10, 8192, 1024, "mma_16",
                tile=dict(n=128, k=128, threads=256, slice_k16=False)),
    _gemm(10, 1024, 4096, "mma_16",
                tile=dict(n=64, k=128, threads=128, slice_k16=True)),
    *[
        _block_rows(m, n, k)
        for m in (40, 41)
        for n, k in (
            (1024, 1024), (1536, 1024), (1536, 1536),
            (6144, 1536), (1536, 4608), (1536, 6144),
        )
    ],
    _block_rows(40, 1536, 3072),
    _block_rows(40, 1024, 1536),
    _block_rows(40, 4608, 1536),
    _block_rows(41, 132, 1024),
    _block_rows(41, 1024, 1536),
    _block_rows(41, 4608, 1536),
    _block_rows(1, 1536, 1024, algorithm="fma"),
    _block_rows(17, 1536, 2048, tile=dict(n=64, k=128, threads=128),
                algorithm="mma_fma"),
]
