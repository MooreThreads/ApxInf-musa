"""Single source for BF16 decode shapes, algorithms and offline build metadata.

Adding a projection using an existing algorithm requires only one registry row.
Builders and the weight packer consume the same complete operator description.
"""


def _projection(name, n, k, algorithm, *, tile=None, splits=1, flags=None):
    m = 10
    return {
        "id": f"pi05-bf16-{name}",
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


OPS = [
    _projection("input", 1024, 32, "input-register",
                flags=["-Od3", "-O2", "-ffp-contract=fast"]),
    _projection("output", 32, 1024, "output-split-k", splits=8),
    _projection("qkv", 2560, 1024, "packed-register",
                tile=dict(n=128, k=128, threads=256, slice_k16=False)),
    _projection("out", 1024, 2048, "packed-register",
                tile=dict(n=64, k=256, threads=128, slice_k16=True)),
    _projection("gate-up", 8192, 1024, "packed-register",
                tile=dict(n=128, k=128, threads=256, slice_k16=False)),
    _projection("down", 1024, 4096, "packed-register",
                tile=dict(n=64, k=128, threads=128, slice_k16=True)),
]
