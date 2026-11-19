# MP31 native FP8 epilogues

`geglu.cuh` implements the verified M=10, K=1024, N=8192 E4M3 gate/up
projection, FP32 accumulation, actual alpha, BF16 intermediate rounding,
tanh GeGLU and static E4M3 output. It uses BM16/BN64/BK256, one pipeline stage,
128 producer and 128 consumer threads. Barrier initialization uses the full first warp. Producer and consumer
loops have separate warp-uniform scopes, every used stage is drained, and all
threads rendezvous before and after the epilogue. The gate/up pairs have width 32.

The BM16 SQMMA/TME producer/consumer schedule derives from muDNN commit
`aaa2127d1`, `3rdparty/tilelang_kern_gen/gen_kernels/benchmark_gemv_cases.py`,
`small_m_bm16_sqmma_tme_pc_transb`. The local wrapper only uses the already
vendored MATE 0.2.7 MUTLASS headers; no external checkout or TileLang installation
is required to build.

`prepare_fp8_weight` retains the immutable canonical `[K,N]` tensor and packs
its E4M3 bytes to `[N,K]` once, on the context stream. It allocates K*N bytes per
registered weight. The context owns both allocations and the host TME descriptor.
Activation descriptors are created during eager preparation; a missing descriptor
is rejected during capture. Replay launches one kernel, with no runtime pack or
intermediate BF16 gate/up allocation. Zero alpha follows muDNN's positive-zero
output. Runtime calls require the same immutable registered weight.

The existing `ops::gemm_geglu` interface selects this epilogue for the exact
verified geometry and `GateUpPairs { pair_width: 32 }`. Other shapes, layouts,
output types and targets use the existing path. The adapter is compiled only by
the existing MP31 vendor gate. This header, adapter, build flags and vendored
dependencies participate in the kernel build fingerprint.

## Fused action QKV producer

`direct.cuh` packs the immutable `[1024,2560]` action qkv weight into 512-byte
16x32 register atoms before capture. One 80-CTA launch computes both RoPE
halves, applies the optional bias, rounds to BF16, and writes q plus the K/V
cache rows, so the intermediate `[10,2560]` BF16 tensor and the separate rope
launch leave the action graph. Preparation adds 2.5 MiB per qkv weight (45 MiB
for eighteen layers). The epilogue uses the accumulator's actual atom layout;
treating each lane's registers as a contiguous output row is incorrect. BF16
GEMM rounding, bias rounding, RoPE rounding and the q/destination layout match
the separate path.

The same build uses the MP31 fast transcendental intrinsics in the native
attention softmax/merge, the sigmoid/GELU helpers and the RoPE `sincos` pair.
The D256 attention uses five KV partitions, and the action `partial_norm`
keeps the FP32 intermediate for its M=10 rows instead of re-reading the
BF16-rounded value. The Rust workspace and the adapter launch both size for
the same five-way split.

On S5000, the unchanged fixed L0 graph benchmark measures 23.30 ms p50 /
23.53 ms p95 with cosine 0.99946348 and max absolute error 0.0424036, and
graph replay is deterministic. The 16.67 ms / 60 Hz target remains open at
about 43 Hz.

The original benchmark, calibration and workload remain unchanged. A device
test checks alpha, scale, byte transpose, poisoned output, boundary guards,
eager/graph parity, 1000 bitwise replays per case and T=10 -> 21 -> 10 fallback. Run on MP31:

```
APXINF_MUSA_BENCH_DEVICE=5 cargo test --release -p apxinf-musa --lib \
  prepared_geglu_alpha_coverage_graph_and_shape_switch -- --ignored --nocapture
```

The source manifest and full-model reports are kept outside the repository.

## Projection with adaptive residual normalization

`projection.cuh` and `projection_norm.cuh` cover static FP8 projections with
M=10, N=1024 and K=2048/4096. The producer uses BM16/BN64/BK256 and split-K=8;
the consumer reduces FP32 partials in fixed order, applies actual alpha and BF16
rounding, then computes the existing gate/residual/AdaRMSNorm/E4M3 chain.
O projections encode BF16 activation into shared E4M3 while loading; down
projections consume the preceding GeGLU E4M3 output. The final decoder layer
retains its original down/residual/final-norm handling.

The safe `gemm_ada_gate_residual_rms_norm` operator checks matrix, output, style,
device, alpha, scale and workspace contracts. `Fp8Bf16Activation { scale }`
means activation encoding only; alpha already includes both effective scales.
The norm style has two segments and gate style has three. Unsupported shapes
and constrained workspaces use the original composed operators. Preparation
retains immutable weight allocations keyed by pointer **and shape**, and graph
capture requires descriptors and arena storage to be ready.

Both `o` and `down` sites are enabled by default; the selected consumer uses
320 KiB of transient workspace per invocation.

With the revised GeGLU schedule, the original benchmark's same-card ABBA
measured off 27.357/27.315 ms versus on 26.623/26.583 ms, with cosine 0.99948796.
Both candidates beat both controls; the 0.733 ms mean gain exceeds 0.042 ms control
drift. Earlier timings with the previous GeGLU schedule are archived separately;
60 Hz remains unachieved. Artifacts are kept outside the repository. Run the
poison, alpha/scale, shape-switch and 18000-replay test with:

```
APXINF_MUSA_BENCH_DEVICE=5 cargo test --release -p apxinf-musa --lib \
  projection_norm_alpha_scales_poison_graph_and_shapes -- --ignored --nocapture
```
