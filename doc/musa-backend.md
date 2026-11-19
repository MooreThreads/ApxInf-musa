# MUSA backend

`crates/apxinf-musa` runs ApxInf on Moore Threads MUSA devices. Its runtime was
ported from the former `apxinf-cuda` backend, preserving its allocation,
tensor and workspace contracts while using MUSA's own APIs (`musaMalloc`,
`musart` and `mublas`). The current CUDA runtime is `apxinf-cuda-new`; the two
backends evolve independently.

PI0.5 and GR00T reach it through `crates/apxinf-model/src/musa_pi05/` and
`crates/apxinf-model/src/musa_gr00t/`, backend-native ports that **copy** what
they need instead of sharing the CUDA families' code. That placement is the
rule in [model-layer-architecture](model-layer-architecture.md#where-backend-native-model-code-lives),
and it is what makes an upstream change to a CUDA family unable to alter the
MUSA port's numbers.

What is measured on M1000 (mp_22), PI0.5 BF16, 2 views, H = 10, 10 flow steps,
against the `torch-eager-gold.v1` bundle recorded on Orin:

| | captured graph | eager |
|---|---|---|
| p50, 15 samples after 3 warm-ups | 507.6 ± 1.3 ms | 512.6 ± 9.5 ms |
| vs Orin `raw_actions` | cosine 0.99995867, max abs 0.0128611 | same |

Both policies are bit-identical to each other, and repeated replays are
bit-identical to the first. Read the numbers with
[the benchmark caveats](#measuring-a-change) below; they move with host load,
which is why every figure here names the run it came from.

## Enabling it

`musa` is an additive feature. It coexists with `cuda` and `hip` in one binary:
each family owns its own top-level port, so nothing has to choose between them
at compile time.

```bash
export PATH="$HOME/.cargo/bin:$PATH"          # if cargo is not already on PATH
cargo build -p apxinf-musa                    # the backend alone

CARGO_TARGET_DIR=target/wheel maturin build --release --features musa \
  --auditwheel skip -m crates/apxinf-py/Cargo.toml
.venv/bin/pip install --force-reinstall --no-deps target/wheel/wheels/apxinf_py-*.whl
```

That wheel carries the backend without the MTTX range markers. Build with
`--features musa-mttx` instead to link them; `apxinf_musa::MTTX_AVAILABLE` says
which build a process got, and [profiling](pi05-mttx.md) covers the rest.

A device is `musa` or `musa:<index>` anywhere a device string is taken —
`ModelRunner.load(..., device="musa:0")`, `--device`, the Python API. The CLI
(`src/main.rs`) and the binding (`crates/apxinf-py/src/lib.rs`) parse it
separately and agree on what they accept.

The model name is `pi05`. `pi05-musa` is accepted as the backend-qualified
spelling, the way `pi05-cuda` and `pi05-hip` are, and for the same reason: a
caller that names the backend it means still reaches the family. The name is
admission only — it does not bind the device, which is what decides which
backend runs.

`ModelRunner.load(..., execution=...)` accepts `eager`, `prefer-graph` and
`require-graph` for MUSA PI0.5. Omitting it keeps `prefer-graph`. The separate
`weight_layout_budget_bytes` option is implemented by CUDA PI0.5 BF16/FP8;
it does not control MUSA allocations.

`crates/apxinf-musa` builds without a toolkit: with `MUSA_PATH` unset or
pointing nowhere, the build compiles the no-device configuration, warns, and
exposes `MUSA_AVAILABLE = false`, which is how a caller can tell. That is
deliberate — it is the configuration a CI job or a checkout on another machine
gets, and it still type-checks every feature combination, which is the one
property that would otherwise only be checked on a machine that has the
toolkit.

## Offload target

The kernels are compiled for one generation, chosen at build time:
`APXINF_MUSA_ARCH` if set, else a probe of the installed toolkit. A machine can
hold two generations (M1000 and an S5000, say), and launching one generation's
image on another is not a degraded run — it is a driver-level failure with no
attribution, or a plausible wrong answer.

So the build **refuses to guess**: an `APXINF_MUSA_ARCH` value that is neither
blank nor a known name is an error naming the setting, rather than silently
falling back to detection and surfacing the mismatch at run time. The probe
itself is advisory — when it cannot run, the build warns and keeps the default,
because a probe that fails hard turns a toolkit problem into a disabled backend.

`MusaContext::new` re-checks at run time: a binary built for `mp_22` opened
against a different generation is refused with both values in the message.

## Generations

[`MusaArchFamily`](../crates/apxinf-musa/src/device_caps.rs) classifies a
compute-capability number: `Mp22` (M1000; BF16 MMA uses a `32x32x16` atom, no
TMA or cluster unit), `Mp31` (S5000/S4000, which add TME and SQMMA), and
`Other(n)`.

`Other` is not "unknown hardware". It is the arm a generation lands in before
this backend has measured what it can run, and its capability questions answer
`false` on purpose: the conservative answer, and the **silent** one. Opening a
context on such a device prints one line per process naming the generation, so
that "a kernel that exists for this generation is unused rather than absent"
stops being invisible.

## Relationship to the CUDA backends

The legacy `apxinf-cuda` tree was removed when all CUDA families migrated to
`apxinf-cuda-new`. Its source at
[`01232a9a`](https://github.com/infinigence/ApxInf/tree/01232a9a8220977a76094907eaed6da2ef1ba0db/crates/apxinf-cuda)
is the historical origin of MUSA's runtime, not a second live dependency.
The current relationship is:

- **Surface serves the MUSA port.** The typed `ops/` contract, `ExecutionSession`
  and `graph::capture` were shaped after `apxinf-cuda-new`. They cover the
  independent `musa_pi05` BF16/INT8 paths; CUDA-only additions such as
  `RepackedWeight` do not automatically become MUSA requirements.
- **Interior follows legacy CUDA.** `kernels/`, `ffi/`, `buffer`/`context`/
  `stream`/`device_caps` and the flat `adapters/*.mu` are ported from
  `apxinf-cuda`, which is the tree this backend was taken from.
- **Kernels are a subset, rewritten where the hardware forced it.** 39 device
  kernels against the original CUDA tree's 221; the 16 that keep the same name
  are the ones PI0.5 uses. Their bodies differ where M1000 made the upstream shape wrong:
  row-grid launches (64-bit integer division is a software routine here), a
  precomputed RoPE frequency table, register-cached W8A8 quantization, FP16
  scales with a 512-entry zero pad for the `s8mma` tiles, and `copysign`
  rounding in place of `roundf`. Each rewrite states the measurement behind it.
- **The comment density is deliberate.** These files carry roughly three times
  the comment-to-code ratio of the CUDA originals. That prose is what makes a
  rewritten kernel reviewable against its origin; it is not something to trim
  into line with upstream.

Relative to that historical source, the `kernels/custom` files correspond
one-for-one to files in `apxinf-cuda`'s set, except `graph.cuh` (a MUSA capture
anchor) and `norm_quantization.cuh` (the fused normalization-plus-quantization
producers), which have no CUDA counterpart; `custom_kernels.mu` uses the same
anonymous-namespace plus single-C-ABI translation-unit pattern, and every ported
`extern "C"` signature
has the same parameter order and meaning. Where a signature does differ —
`quantize_rows_bf16_int8` grew the per-channel equalization vector — the adapter
README says so.

`scripts/check_ops_contract_surface.py` checks the operators imported by the
MUSA model port, including its INT8 path. `crates/apxinf-musa/tests/backend_parity.rs`
checks literal environment reads in the current CUDA Rust/native sources and
MUSA backend/model sources. Its table documents the current differences;
removed legacy CUDA allocation switches are not MUSA counterparts.

## PI0.5 on MUSA

`crates/apxinf-model/src/musa_pi05/` is self-contained: 22 files, no import of
`crate::pi05`. `scripts/check_model_family_boundaries.sh` enforces that in both
directions. Differences from the upstream file a copy came from belong to the
port; upstream changes to `pi05/` are audited against each copy's import
watermark with [copy-drift-audit](../skills/copy-drift-audit/SKILL.md), so a fix
upstream cannot pass the port by unnoticed.

The BF16 layers follow `pi05/model/blocks/bf16.rs` operator for operator, with
the same rounding points, so a stage-by-stage comparison is meaningful. The
static-FP8 route follows `pi05/model/blocks/fp8_static.rs` in operator order
but keeps BF16 runtime intermediates (see the FP8 section below), so compare
it within its own route rather than against `bf16.rs`. Differences that do not
change a value:

- Per-step modulation is a constant of the model (the flow schedule comes from
  config, the projections from the checkpoint), so it is built once at load and
  held by the model. That is also what makes it safe to capture: the tensors
  live as long as the graph does, unlike a temporary built before the region.
- Decode GEMMs run on the MP22 assembly tiles, and the encode path takes the
  registered variant for its shapes. Both are prepared at load.
- Immutable weights are registered once at load (`prepare_assembly_weights`),
  held for the context's lifetime, and must not be mutated afterwards. There is
  no per-call candidate cache, so there is no weight-version token to key one.
- The prefix cache reserves the action rows, so language layers write K/V
  straight into it.

### INT8 (W8A8)

`model_variant=int8_dynamic` and `int8_dynamic_no_encode` run the route upstream
and the integration branch define. `auto` never selects INT8: whether a
projection can be quantized is a per-shape question, and MP22's `mp` number is
not a compute capability. On `mp_22` it stays BF16; on `mp_31` it selects
`fp8_static` only when FP8 activation scales were supplied (a calibration
profile or a `uniform:<scale>` bring-up scale). What the port adds to the
device path:

- The tiles read their dequantization scales as **FP16 with a 512-entry zero
  pad**, not f32, so the weight packer renders that vector once at load and the
  row quantizer writes the activation half in the same form. Nothing converts
  scales per GEMM.
- `geglu(gate_up)` and its row quantization run as one kernel for the down
  projection, reproducing the materialized chain's intermediate BF16 rounding in
  registers. The fused output is byte-identical; a device test checks it across
  150..16392 columns and falls back for views the kernel cannot take.
- The language layers fuse their normalization into the same row quantizer:
  qkv's input RMSNorm (`prepare_norm`) and the post-attention bias + residual +
  RMSNorm (`prepare_residual_norm`) hand the packed INT8 rows straight to the
  GEMM. The vision layers do the same for LayerNorm -- including the post-norm
  that feeds the multimodal projector -- and for FC1's bias + GELU,
  and the action expert's adaptive RMSNorm and gated residual RMSNorm +
  quantization share the contract. A producer is used only when the consuming
  projection is quantized and the width/alignment matches; anything else keeps
  the materialized BF16 chain. Device-only tests compare every fused producer
  with the split chain byte-for-byte (`w8a8_norm_tests.rs`,
  `w8a8_affine_norm_tests.rs`); they are `#[ignore]`d and no CI job runs a
  MUSA device, so run them manually with
  `cargo test -p apxinf-musa --lib -- --ignored kernels::gemm::w8a8
  --test-threads=1`. `APXINF_MUSA_BF16_RMS=scalar` forces only the language
  qkv input RMSNorm's fallback onto the scalar two-pass kernel for accuracy
  runs; the other norm producers have no scalar form.
- The down projection's `(522, 2048, 16384)` shape may take the `256x128`
  **M-fast** tile. Its assembly image is derived from `qint8_256x128.asm` by
  `aot/export_qint8_mfast.py`, with the source hash pinned in the script and
  the asm README; the tile is routed only for that measured shape, and other
  `m` values keep the shape band's traversal order.
- The action traversal marks itself as quantized decode, and its short-row GEMMs
  take NN/NT images rendered from the BF16 GEMM template; gate/up's KN weight
  copy is packed once at load, so no transpose runs per call.
- A projection no tile serves keeps BF16 arithmetic — the ViT patch embedding
  (`k = 588`), the action input and the action head — and the ViT MLP is zero
  padded from 4304 to 4352 so that it does not.

Full-encode quantization needs per-channel equalization to clear the 0.999 gate:
the encode layers' shared residual stream has outlier channels one `amax/127`
row scale cannot cover. `APXINF_INT8_EQUALIZATION` names the checked-in
`configs/pi05_int8_equalization_q50_g60_d60.json`; the fold costs one multiply
per lane inside the quantizer. That table is built by
`scripts/build_pi05_int8_equalization.py` from `APXINF_PI05_INT8_STATS` row
levels, and the script's docstring records how to rebuild it and why a schedule
change invalidates it. `int8_dynamic_no_encode` keeps those four language
projections in BF16 instead and needs no calibration.

Measured on M1000, two views / ten PaliGemma tokens / ten flow steps, against the
Orin bundle (3 warm-up, 15 samples, equalization set, frozen checkpoint copy):

| variant | execution | L1 p50 | cosine |
|---|---|---:|---|
| `bf16` | eager | 512.6 ms | 0.99995867 |
| `bf16` | graph | 507.6 ms | 0.99995867 |
| `int8_dynamic` | eager | 361.5 ms | 0.99936157 |
| `int8_dynamic` | graph | 363.2 ms | 0.99936157 |
| `int8_dynamic_no_encode` | eager | 502.2 ms | 0.99963635 |
| `int8_dynamic_no_encode` | graph | 502.9 ms | 0.99963635 |

The table above is the port's original measurement. The fused branch was
compared against `musa-dev` with an ABBA sequence on the same machine
(`int8_dynamic`, graph capture, warm-up 3 / samples 15, equalization set):

| run | branch | L1 p50 |
|---|---|---:|
| base-a1 | `musa-dev` | 317.21 ms |
| base-a2 | `musa-dev` | 318.67 ms |
| pr-b1 | fused branch | 312.05 ms |
| pr-b2 | fused branch | 312.23 ms |

The medians are 317.94 ms (base) and 312.14 ms (fused), a 5.80 ms / 1.83%
reduction. The exported actions were bitwise identical in every round, and all
four runs scored cosine 0.99936157 / max_abs 0.0504164 against the Orin bundle
(gate 0.999 PASS). Device validation: `cargo test -p apxinf-musa --lib --
--ignored kernels::gemm::w8a8 --test-threads=1` passes 14/14 (device-only; see
the INT8 note above), including the 2*cols style and scalar-RMS regressions;
`cargo test -p apxinf-model --features musa --lib musa_pi05::` passes 36.

The integration branch on the same machine measures 513.4 / 510.1 ms for `bf16`, 406.4 /
363.2 for `int8_dynamic` and 545.1 / 502.9 for `int8_dynamic_no_encode` (eager /
graph), with the same cosine to the last digit in every row. Both captured routes
are at parity. Eager is 11% ahead on `int8_dynamic` for one reason: the port
builds every step's modulation once, when the model is assembled, because the
flow schedule and the projection weights are constants and so is what they
project to. A traversal that recomputes it pays 390 `m=1` GEMMs — and inside a
captured graph it paid them on every replay, which is what the graph column used
to trail dev by.

`int8_dynamic_no_encode` is the slower of the two INT8 routes because its BF16
language layers do not reuse the BF16 route's assembly registrations; it lands
where plain BF16 does (502.9 against 507.6 ms captured) and exists for accuracy,
not speed. The capture's arena is not CUDA's either — a measured INT8 capture
touches 3.84 GB, so the reservation is the BF16 one rather than the doubled one
upstream keeps for its INT32 accumulator and verifier.

### FP8 (static E4M3)

`model_variant=fp8_static` is the `mp_31` route. `mp_22` refuses it before
weights load: its 8-bit tensor-core path is the INT8 `s8mma` assembly, it has no
E4M3 MMA, and the port has no emulation fallback. The build publishes
`FP8_ROUTE` (`mudnn` on `mp_31`, `unsupported` elsewhere). The model loader
rejects `fp8_static` when that route is `unsupported` and when the device
family is not `mp_31`; the FP8 kernel additionally requires
`MusaArchFamily::has_native_fp8`.

The GEMM is muDNN's FP8 MatMul: E4M3 operands, f32 accumulation, one `alpha`
(`input_scale * weight_scale`) and a BF16 result. `libmudnn_xmma.so` carries the
`MutlassKernelGemm<float_e4m3_t, float_e4m3_t, bfloat16_t>` instantiations.
On S5000, muDNN measured 1.6-2.7x the muBLAS FP8 time on qkv, o and down,
equal on up, and matched at the test's cosine 0.9999 gate (not byte equality);
muBLAS is not part of the route.

Static-FP8 preparation keeps the CUDA artifact contract:

- weights: one per-tensor absmax scale, host F32 amax -> F16 rounding ->
  `FixedScaleE4m3` on device;
- activations: one calibrated per-tensor scale per site from `calibration.json`;
  `uniform:<scale>` is accepted for bring-up but does not clear the accuracy
  gate;
- runtime intermediates stay **BF16** (the rest of this port), so the GEMM's one
  rounding is BF16 rather than CUDA's F16. This is the deliberate numerical
  difference; the calibration schema and site names are copied unchanged, so a
  CUDA-generated profile is readable here.

The producer quantizers are fused where the profile showed the extra pass
mattered: GeGLU and the adaptive-gate residual RMSNorm write E4M3 directly
(`kernels/custom/activation.cuh`, `fused.cuh`), rounding to BF16 before the
conversion so the bytes match the composed pair. That removes 557 of the 919
per-GEMM quantize launches. The attention-output quantize and the prefix/vision
norm producers still compose.

Decode attention on `mp_31` uses the unpacked `TileM=16` render of the vendored
MATE kernel for 9–16 query rows (`mp31_decode_unpacked_tile16`). The packed
render launches `(1, 1, 1)` for the model's ten-query pass — one CTA over all
532 keys — while the unpacked one gives each of the eight heads a block,
`(1, 8, 1)`. Measured on an idle S5000: 22.3 µs vs 32.5 µs per call.
`APXINF_MUSA_FA_UNPACKED=0` keeps the packed kernel for an A/B.

The end-to-end measurement on S5000 (L0, `require-graph`, 2 views, T=10) is
**29.45 ms** against **34.71 ms** for tuned BF16 on the same idle card
(1.18x), with the muDNN FP8 route and a one-observation calibration profile
(cosine 0.999295 against the Orin reference; the accuracy gate is 0.999). The
pre-optimization FP8 build measured 33.36 ms against the then-current 37.45 ms
tuned BF16 (1.12x); the BF16 baseline also moved because the unpacked decode
attention is shared by both routes. The measurement uses a one-observation
profile for bring-up; a deployment should generate its profile from its own
data:

```bash
python3 scripts/calibrate_pi05.py --model-dir <checkpoint> --device musa:0 \
  --input-dir <observation-npz-dir> --output <checkpoint>/calibration.json
```

`scripts/bench_pi05_musa.py --model-variant fp8_static --calibration <json> \
  --reference <gold-bundle>`
runs the same L0 call the gate scores.

## GR00T N1.7 on MUSA

`crates/apxinf-model/src/musa_gr00t/` is the independent MUSA port of the
GR00T N1.7 model core. Its validated slice is BF16 on MP22/M1000 and
MP31/S5000, and calibrated static FP8 on MP31/S5000, with batch 1, two camera
grids, official preprocessed tensors, explicit initial noise, and dynamic text
sequence lengths. FP8 needs native E4M3 hardware and the muDNN route
(`FP8_ROUTE == "mudnn"`); MP22 and route-less builds reject it at load instead
of selecting another precision. INT8 is rejected on MUSA.

The FP8 executor mirrors the CUDA static-FP8 boundaries: each backbone,
vision and action-head linear gets a per-tensor E4M3 weight scale
(`max_abs / 448`) and a per-consumer activation scale from the shared
`apxinf.fp8-calibration.v1` profile, the GEMM's single `alpha` is
`activation_scale * weight_scale`, and biases, normalization, attention and
all intermediates stay BF16. The profile is produced on the target by the
model-neutral Python `CalibrationRunner` through
`ModelRunner._calibrate_preprocessed`; its checkpoint identity is the same
versioned value the Rust loader recomputes. The MUSA E4M3 producers
(`rms`/`layer`/`adaptive` norm, bias+GELU, SiLU-mul) are used where the
executor's shape predicates admit them; the rest compose the BF16 producer
with `quantize_fixed_e4m3`, which is byte-identical by construction.

Measured on an idle S5000 (GPU 4, two views, 145 tokens, `require-graph`,
Orin gold fixture): BF16 p50 **23.1 ms** (warmup 10 / 100 samples; cosine
0.9999960, after regenerating the shipped `mtt-s5000-mp31` tactics database
for the current kernel build) and static FP8 p50 **18.7 ms** (warmup 10 / 100
samples; cosine 0.9999611, relative L2 0.008815, max_abs 0.0469;
gates 0.997 / 0.10 / 0.125) after the 2026-11-17 fourth pass. FP8 started at
29.36 ms; the first pass reached 24.36 ms with the duplicated-bias fix, the
FFN `bias_gelu_bf16_e4m3_into` producer, concatenated DiT/vl QKV, E4M3
LayerNorm widths 1024/1536/2048, and the dual-output adaptive LayerNorm. The
second pass added:

- a dual-output `layer_bf16_e4m3` producer: the FF input's BF16 and E4M3
  values in one pass, removing the discarded second LayerNorm (1.35 ms);
- `uses_quantized_output` so the FFN consumes that E4M3 tensor directly
  instead of re-quantizing the BF16 producer;
- a `[41,1536]` bias + residual + LayerNorm + E4M3 producer for the attention
  output, using the same two BF16 roundings as `bias_bf16` followed by `add`;
- a register-cached adaptive LayerNorm that loads the 41x1536 row once
  (11.3 -> 6.7 us per block-step);
- a four-pair vectorized weight+bias+RoPE producer for the vision QKV
  (the pre-vectorization kernel measured 49.4 us per call; the vectorized
  version removed about 0.3 ms end-to-end).

The third pass added:

- MP31 batched vision attention: `try_noncausal_batched_equal` no longer
  refuses `mp_31`, and the FP8 execution enables
  `USE_BATCHED_VISION_ATTENTION`. Each vision block's attention is one
  `[2,256,16,64]` call instead of six row gathers, two segmented calls and a
  concat (FP8 21.1-21.6 -> 20.7 ms; BF16 also benefits, 24.2-24.8 -> 23.7 ms);
- muDNN's bias epilogue (`MatMul::RunWithBiasAdd`) for the fused self-QKV
  projection: one GEMM + bias call instead of GEMM + `bias_bf16` (0.70 ms);
- muDNN's bias + residual epilogue for the DiT FF down projection: one
  GEMM + bias + residual call instead of GEMM + `bias_then_residual_bf16`
  (0.39 ms);
- a warp-per-row `qk_rms_norm_mrope_precomputed_bf16_warp_kernel`: one warp
  owns one `(token, head)` row, the reduction is a warp shuffle and the
  normalized row stays in registers (29.4 -> ~15 us per text layer, 0.26 ms);
- the width-1024 vision residual+LayerNorm reuses the existing
  `bias_residual_layer_norm_bf16_warp_kernel` template instead of the
  block-per-row kernel (21.9 -> ~7 us per vision block, 0.32 ms).

The fourth pass added four producer/launch fusions (FP8 19.48 -> 18.70 ms,
p50, warmup 10 / 100; accuracy unchanged at cosine 0.9999611, relative L2
0.008815, max_abs 0.0469):

- the DiT cross-attention query projection uses the fused bias epilogue
  (`forward_reusable_quantized_bias`) instead of a separate `bias_bf16`
  launch (64 launches removed, 0.31 ms);
- the text Qwen gated-MLP epilogue is one `silu_mul_bf16_e4m3` producer
  instead of `silu_mul_bf16` + `quantize_fixed_e4m3`, removing the
  `[145,6144]` BF16 round trip per text layer (16 launches);
- the vision FC1 bias+GELU is written directly as E4M3 and the following
  FC2 consumes it through `forward_quantized_tensor`, removing the
  `[512,4096]` BF16 round trip and 28 quantize launches (0.35 ms);
- the fixed-scale E4M3 quantizer processes eight BF16 per thread (16-byte
  load / 8-byte store) when the buffers allow it, byte-identical to the
  scalar form, which covers the remaining quantize calls.

A BM64 small-M native FP8 GEMM (BM64, BN in {16,32,64,128}, BK256,
SQMMA/TME) was implemented and measured during this pass and rejected: its
output is bit-identical to muDNN, but it is slower on all four DiT M=41
shapes (native/muDNN 1.10-1.20x against the 0.66x acceptance gate), and an
e2e check with the plain `o`+`up` routes enabled measured 20.22 ms against
18.71 ms. The measurement ledger is in `devlocal/gr00t/reports/`.

FP8 is now about 23% faster than tuned BF16.

The 2026-11-17 second pass moved FP8 18.743 -> 18.541 ms (p50, same fixture;
cosine 0.9999610, relative L2 0.008837, max_abs 0.0469) with two changes: the
adaptive LayerNorm's 1536-column form switched from `<128,3>` to `<384,1>`
(probe 5.81 -> 4.45 us/call; E4M3 output byte-identical, BF16 within one ulp),
and the text backbone's
attention residual add and FF-input RMSNorm+E4M3 producer collapsed into
`bias_residual_rms_e4m3_into` with `bias = None` (the fused kernel writes the
same `bf16(projection + residual)` bytes the composed `add` did; the FP8
weights expose it as `residual_rms_norm_quantized`). A native DiT noncausal
attention kernel was written and rejected on measurement: three layouts ran
85-95 us/call warm against muDNN's 12.02 us at (41 rows, 41/17/128 keys, 32
heads, 48 dims), with the phase probe attributing 35 us to QK and 39 us to PV
-- a SIMT implementation cannot approach the vendor's tensor-core SDPA (the
gate was 8.4 us), so the attention route stops there. The text QKV wide-GEMM
path (one `[145,4096,2048]` projection plus a one-launch Q|K|V split, with a
group-max weight scale and no recalibration) replaced the three per-projection
GEMMs: it also makes the text layer's fused RMS-norm + E4M3 producer
reachable, so a layer emits one projection and one normalization kernel
instead of three and two. Paired interleaved A/B on MUSA4 (2026-11-18) put it
at 16.6234/16.6530 ms against 16.8673/16.6839 ms on the three-GEMM path; the
three-GEMM form still runs for BF16 and for a calibration whose Q/K/V
activation scales differ. The per-stage ledger and probes are
in `devlocal/gr00t/`.

The current dev-Yi pass moved FP8 18.541 -> **17.46 ms** (p50, four runs
17.44-17.75, Orin fixture, cosine 0.9999584, relative L2 0.009127, max_abs
0.0591; gates unchanged) with six producer/launch changes:

- the DiT attention-output producer's 1536-column form switched from
  `<128,3>` to `<384,1>` (probe 6.90 -> 4.13 us/call; `hidden` and the E4M3
  output byte-identical, BF16 within one ulp; 128 calls, 0.39 ms);
- the cross-attention K/V projections use the fused bias epilogue
  (`linear_quantized_bias`) instead of `bias_bf16` (32 launches removed;
  the ledger's `bias_activation_bf16_vector8` family falls 63 -> 18 calls);
- the action head's state/action-encoder outputs, output projection and
  decoder output quantize and then use the same fused bias epilogue, removing
  one bias launch per site;
- the text gated-MLP `silu_mul_bf16_e4m3` producer is vectorized eight lanes
  per thread (byte-identical; 12.7 -> 12.2 us/call);
- the vision QKV+bias+RoPE producer is specialized to the tower's fixed
  16 heads x 64 dim: eight rotary pairs per thread with 32-bit shift/mask
  indexing instead of the generic runtime-division form (probe 19.45 ->
  7.22 us/call, q/k/v byte-identical; 24 calls, 0.29 ms);
- the vision MLP boundary fuses residual + LayerNorm + E4M3 + the following
  projection: `bias_residual_layer_norm_bf16_e4m3_block_kernel` is now
  templated on its block shape and also covers width 1024 (256 threads x one
  4-wide chunk; the 1152 form stays `<96,3>`), removing the BF16 normalized
  intermediate and 24 quantize launches. The fused form's reduction tree
  differs from the warp sibling, which is why the accuracy numbers moved by
  ~0.5% relative while staying well inside the gates.

A sixteenth-lane `bias_activation_bf16_e4m3` epilogue was measured and
rejected: 6.79 us/call against 5.94 for the eight-lane form on [41,6144]
(probe and msys ledger agree), so the launcher keeps the eight-lane kernel.
The post-pass ledger is 1,861 kernels / 17.79 ms busy with the same
803-call / 10.4 ms vendor FP8 GEMM and ~172-call / 2.2 ms attention budget;
the remaining producer cost is ~5.0 ms across norm, quantize, activation,
rope and residual launches.

The remaining cost is the 803 vendor FP8 GEMMs (10.4 ms, 524 of them M=41) and
the 196 muDNN attention calls (2.6 ms). A native MP31 plain-GEMM prototype
(BM16/BN64/BK256 SQMMA/TME, `[N,K]` prepack) was measured and rejected: muDNN
22.04 ms versus native 23.74 ms end-to-end, because the vendor's ~10.3 us
small-M floor is a latency floor that a one/two-stage schedule does not beat;
the `1f7cb0e` GeGLU win came from its fused epilogue, not the plain
contraction. 60 Hz (16.67 ms) therefore remains out of reach on the current
vendor stack. The next honest lever is the vendor's small-M FP8 GEMM (the
measured shape table is in
`devlocal/gr00t/reports/2026-11-16-ledger/native-gemm.md`), not
another plain kernel route. Accuracy acceptance is unchanged by any of this
work; it only moves where the quantizer runs.

The 2026-11-17 pass extended the same measurement to two more native routes and
rejected both: the tall-M vision/text/merger shapes (512/145 rows on the
2D-grid kernel) ran at 2-5x muDNN's time (vision fc1 96 vs 19 us, text down
73 vs 24 us), and a bias + GeLU + E4M3 epilogue for the DiT up projection
measured 19.97 ms end-to-end against 19.51 ms for muDNN plus the
`bias_gelu_bf16_e4m3` producer. The epilogue itself is not the cost -- the
BM16/BN64/BK256 schedule is -- so the fused variant was removed after the
measurement and the native plain-GEMM prototype was dropped; the default route
remains muDNN.

The 2026-11-18 pass measured the remaining precision-for-speed levers and
rejected both. muDNN does expose a vendor GELU+bias epilogue, but only through
`BatchMatMul::RunLt` with a `MatMulLtParam` (the `MatMul` class has no epilogue
parameter), and FP8 output stays BF16, so the diT up projection still needs an
E4M3 quantize after it. Measured on the DiT and vision shapes, the GELU-epilogue
GEMM is ~2.6 us slower than the plain GEMM, which cancels the
`bias_gelu_bf16_e4m3` producer it replaces (41x6144: 14.86 us composed versus
14.88 us vendor; vision fc1: 26.18 versus 27.76). The E4M3 bytes were identical
in all three shapes. Attention-output projections in BF16 were also rejected:
muDNN BF16 is 10.7 us against 14.0 us for FP8 plus quantize in a hot-weight
probe, but end to end it regressed 17.46 -> 17.87 ms, because keeping the BF16
weight doubles that GEMM's per-layer-step weight traffic (2.36 -> 4.72 MB) and
the shape is latency-bound. W4A16/FP4 have no MP31 vendor route (`arch == QY2`
for the former, `>= PH1S` for the latter; MP31 is PH1), so the accuracy margin
is not spendable on 4-bit weights.

Kept from the pass: the prepared-input staging path (the only path now. The
1.57 MB upload measures 203 us pageable/synchronous versus 73 us
pinned/asynchronous; a 2026-11-18 ABBA on a loaded host measured -0.41 and
-0.06 ms, median -0.13 ms, and the earlier quiet-host pass measured -0.05
median / -0.16 mean), an
eight-lane `add_bf16` (byte-identical; 5.86 ->
5.30 us/call), and an eight-lane 1024-column vision residual LayerNorm
(10.64 -> 8.37 us/call; the four-warp cross-warp sum moves the rounding
boundaries by less than the gates). The full node ledger moved 17.786 ->
17.748 ms; reverting the neutral quantize split leaves ~0.067 ms of
kernel-time reduction plus the input-upload saving. The producer kernels are
within one to two microseconds of the 48-CTA launch floor, so the remaining
producer space is ~0.05 ms, not the 0.2-0.3 ms an earlier estimate suggested.
Two wider-lane sweeps were measured and reverted: eight-lane 192-thread forms
of the two DiT dual-LayerNorm kernels (bias_then_residual 5.94 vs 5.41 us,
adaptive 5.46 vs 5.54) and a sixteen-pair vision QKV+RoPE form whose halved
grid costs more than it saves (19.70 vs 10.70 us/call). A muBLAS-LT FP8 probe
was also rejected: the OP_N column-major route measured 13-26x slower than
muDNN on every DiT M=41 shape (133-266 us versus 8-15 us) and left the output
zero, so the path is not usable on this toolkit.

The 2026-11-18 attention pass replaced the DiT and vision SDPA with vendored
MATE instantiations that write the output projection's E4M3 input directly.
The DiT self-attention reads the fused QKV projection through a 4608-wide row
stride (no split), the cross-attention runs the same kernel on separate Q/K/V,
and the cross K/V input is quantized once per (source, scale) instead of once
per layer (16 launches drop to 2). The vision tower uses a batch-2 x 256-token
D64 sibling. The MATE epilogue gained `output_scale` (1.0f is an exact no-op,
so the existing D256/BF16 paths are unchanged); the new instantiations pass
`1/activation_scale` and store E4M3, which removes the 128 DiT and 24 vision
post-attention quantizes.

Paired A/B on GPU4 (145 tokens, require-graph, Orin gold): 17.45 ->
17.15-17.18 ms; cosine 0.9999585, relative L2 0.0091, max_abs 0.0508 (better
than stock's 0.9999528 / 0.0097 / 0.0625). The node ledger moves 17.668 ->
17.530 ms busy and quantize 227 -> 98 calls (0.789 -> 0.387 ms); the MATE D48
kernel itself measures 12.29 us against muDNN's ~12.0-12.5 us, so the gain is
the fused quantize, not the attention math. The E4M3 text attention measured
slower (17.18 -> 17.24 ms) and was removed. The DiT and vision variants are
now selected unconditionally; every shape or precision the kernels do not own
still falls through to muDNN, and a launch error leaves the SDPA path in
place. (The text-QKV wide GEMM first measured 17.18 -> 17.57 ms and was left
off; the 2026-11-18 pass re-measured it as 16.8673/16.6839 ms on the
three-GEMM path against 16.6234/16.6530 ms fused, so it is now the FP8 path.)
With prepared-input staging, the fixed-input run measures 17.04-17.11 ms on a
loaded host (the 2026-11-18 ABBA pair: 17.04 vs 17.46 ms pageable).

The same round fused the four VL self-attention blocks' two LayerNorms with the
following projection's static-E4M3 quantizer (the 145x2048 scalar LayerNorm
measured 15.15 us and each projection added a 3.2 us quantize). The fused
producer writes normalized BF16 and E4M3 in one launch and the projection reads
the E4M3 directly; BF16 and non-static-FP8 linears still take the composed path.
Two paired runs measured -0.253 and +0.069 ms (median -0.09), so the kernel
mechanism - eight 18.4 us producers replaced by four ~6 us dual producers -
explains the mean; accuracy moves from cosine 0.99995849 / relative L2 0.00911
/ max_abs 0.0508 to 0.99995668 / 0.00931 / 0.0547, still inside the gates.

The backend keeps exact-shape acceleration behind model-neutral dispatchers.
Measured GEMM variants, the MP22 scalar M=1 GEMV, muDNN attention, fused
residual/norm operations, and precomputed text/vision rotary tables are used
only when their complete shape predicates match; otherwise the existing
generic device path remains the fallback. The GEMV route is limited to the two
measured steady-state shapes `(1,1024,132)` and `(1,1536,1024)`. In particular,
the kernel predicates cover 129/141/145/255/256 tokens, while end-to-end
validation is limited to the 145-token profile. Not validated: S5000 (MP31),
single-camera input, and other sequence lengths.

Use `scripts/gr00t/bench_apxinf_musa_rust.py` to launch the native Rust benchmark and apply
the BF16 parity gate. The script does not execute the model in Python. See
[GR00T N1.7](gr00t-n1.7.md#musa-bf16) for the fixed-input contract and command.
The MUSA GR00T loader accepts a tactics JSON file through `--tactics`.
For direct in-process use of the installed Python wheel, use
`scripts/gr00t/bench_apxinf_musa_wheel.py`; its timing also includes NumPy/PyO3
conversion. Both scripts accept the same preprocessed fixture.
Use `--autotune --tactics path_to/tactics.json` to measure missing BF16 GEMM
choices before Graph capture. Exact assembly routes retain their existing
dispatch; other supported BF16 shapes tune between muBLAS and muDNN.
Capture itself suppresses tuning. Device and library fingerprints prevent
reusing incompatible tactic databases.

## Serving on MUSA

The websocket policy server runs unchanged on this backend: `WebsocketPolicyServer`
is transport for any `Policy`, and `--device musa:0` reaches the same
`ModelRunner.load` everything else does.

One caveat about this repository's checkpoint. `pi05_libero_pytorch` is an OpenPI
export whose `config.json` has no `type` field and which carries no `metadata.pt`,
so `scripts/pi05_openpi_websocket_server.py`'s preflight refuses it under either
`--ckpt-format` — it cannot tell what the directory is. The example server has no
preflight and serves the same model:

```bash
.venv/bin/python python/apxinf/examples/openpi_server.py \
  --model-dir <checkpoint> --device musa:0 --model-variant bf16 \
  --tokenizer <path-to-paligemma_tokenizer.model> \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state \
  --policy-options '{"model_type":"pi05","num_views":2,
     "norm_stats":"<checkpoint>/assets/physical-intelligence/libero/norm_stats.json"}'
```

Every flag there is load-bearing on this checkpoint: without `model_type` the
policy cannot be dispatched; the tokenizer is not in the checkpoint; without
`norm_stats` the statistics resolve to identity and actions come back in the
normalized domain, with only a warning; and the checkpoint declares three
cameras, so `num_views` and the key list have to agree with what the client
sends.

Three things about the server are worth knowing before deploying it, and none of
them are visible from the wire.

**Inference is serial, one request at a time.** The engine handle is
`unsendable` — its context belongs to the thread that built it — so the server
runs the policy on its event loop and concurrent clients queue. One server per
card; `--device musa:N` picks the card.

**Consecutive identical requests do not return identical actions.** The engine
draws a fresh flow-matching latent per request from its own RNG stream, which is
a stochastic policy behaving normally. To make a call reproducible, send the
latent: it travels in the observation under `"noise"`, a key `Pi05Policy` already
reads, so the wire needs no extension and a client that sends none is unaffected.
With a fixed latent two calls must come back **bit-identical** — that is the one
wire-integrity check this path has.

**The device index reaches every layer.** `--device musa:N` sets the driver
context, and every allocation and launch takes its device from that context;
`cargo run -p apxinf-musa --example device_report N` answers "would this build
serve card N?" without loading a model.

For the far side of a split deployment — render and drive on a robot host,
infer on the card — `scripts/pi05_remote_client.py` is the client. It is
self-contained (numpy, msgpack, websockets, plus a byte-identical copy of the
`apxinf.serving` codec pinned by `tests/test_pi05_remote_client.py`), so it runs
on a host with no `openpi_client` and no checkout. Its `RemotePolicy` reads the
server's metadata at connect and sends exactly the keys the server published, and
`--noise-seed` makes a request reproducible:

```bash
.venv/bin/python scripts/pi05_remote_client.py --host <server> --port 8000 \
  --noise-seed 0 --requests 5 --dump-actions actions.npz
```

The benchmark's L3 attaches to a running server through the same client and times
one round trip per call. It cannot share a 32 GB host with the server when
in-process layers are also selected — that is a second model in a second process.

Measured on M1000, `bf16`, two views, 224×224 synthetic frames, one warm-up and
ten samples: round trip p50 **514.0 ms**, of which the model is 510.8, the
policy's own processor chain 0.75 and the socket 1.97.

**Bit-identity across processes is not something this path promises.** Inside one
server, two requests carrying the same latent come back identical, and an
in-process load of the same checkpoint reproduced the second server's actions
byte for byte. A first server process of the same build returned actions
differing by 3.3e-3 — the size of a provider-selection difference, not of a
transport one. Read the seeded comparison as "the same server reproduces
itself", not as "every process computes the same bits"; a cross-host comparison
needs the tolerance the accuracy gate uses, not equality.

## Capture

Capture is the contract to read before changing any operator. Vendor state
(per-stream handles, per-shape descriptors and plans) is created on first use
and **only outside capture**, so a graph records a traversal that has already
run once on the same stream. `ExecutionSession::prepare` is that pass;
`ExecutionSession::run` refuses a GEMM or attention shape the prepare pass never
reached, naming the shape, and `prepared_shapes()` lists what it covered.

The stream is the context's own, and that is what keeps warm-up and capture
together: `ExecutionSession::prepare` runs on `MusaContext::stream` and so does
the capture that follows, so there is no second stream for a prepared resource
to be launched into.

A capture that fails is not a recoverable failure on this device: the runtime
leaves the stream in `INVALIDATED`, where every later launch is refused and even
`musaDeviceSynchronize` fails for as long as it exists, and the error slot keeps
a status that every adapter would report as its own verdict. Nothing clears that
stream state. `graph::capture` therefore hands the context a fresh stream when
leaving the region fails, and empties the error slot with it, so a caller that
falls back to eager — which is what `PreferGraph` asks for — has a stream to
fall back onto.

## Tuning database

A GEMM shape's provider is chosen from a MUSA tactics database, looked up in
two steps: the toolkit-qualified directory wins when it holds a `tactics.json`,
and the hardware-only directory is the fallback.

```text
configs/tuning/musa/<device-name-lowercased>-mp<mp>/
├── tactics.json
├── tuning_report.json
└── musa<major.minor>-mublaslt<major.minor>/
    ├── tactics.json
    └── tuning_report.json
```

`TuningPaths::resolve_for_device` derives that name from the running device's
caps and library versions, and resolves `configs/tuning` against the **process
working directory** — run from the repository root. A database is written by
`scripts/bench_pi05_musa.py --autotune` (or `ModelRunner.load(autotune=True)`),
and `--tactics DIR` names one explicitly — the directory, since the loader
appends `tactics.json` itself.

Records are bound to the kernel build id, the MP generation, the device name and
the toolkit version numbers. The load path decides whether the machine-local
vendor library fingerprints are also compared: the unversioned
`<device>-mp<N>` database is a hardware-class baseline, so it loads as
`portable` and only the version numbers bind it (the file mtimes differ between
machines, the versions do not). A version-qualified subdirectory or an explicit
`--tactics` directory loads strict and requires the file fingerprints. A
database that does not match is ignored with a printed reason rather than
silently used: a stale one measures another build.

The MTT S5000 `mp_31` database is checked in under
`configs/tuning/musa/mtt-s5000-mp31/`, written by `--autotune`. M1000's shapes
are already covered closely enough that autotuning produces no measurable gain,
so a M1000 directory, if present, is an artifact of a local tuning run rather
than a shipped file. A checkout without a directory for the running device is
normal: every shape uses the safe default, and
`cargo test -p apxinf-musa --lib` validates a database only when one is present.

Tuning itself is a load-time setting: `--tactics` and `--autotune` live in
`LoadOptions` and are installed once before prepare or capture. They are not
part of the per-call operator policy.

## Operator policies

`GemmPolicy` and `AttentionPolicy` are enforced at the operator entry point,
not carried and ignored:

- `accumulation_dtype` must be F32 — the adapter's `MUBLAS_COMPUTE_32F`
  contract. Anything else is refused.
- `workspace_limit` bounds the bytes the operator allocates for itself: none for
  a plain GEMM or attention, `rows × gate_cols × 2` for `gemm_geglu`. Vendor
  libraries allocate their own internal workspaces outside this accounting.
- `allow_fallback = false` requires a tactic record measured at the exact shape.
  A bucket extrapolation or the muBLAS default is refused; autotuning may
  measure the shape and write the record. On attention it requires the packed
  decode route for a decode-shaped call, per build: the assembly build's packed
  decode kernel, or the MUTLASS build's packed tile 128 (9..=16 query rows).
  The general D256 assembly, and the MUTLASS unpacked tile 192 for 17..=256
  rows, are slower substitutes and are refused; from 257 rows the unpacked tile
  is MATE's own choice, and a prefix pass uses the general route by
  construction. The assembly GEMM route is a platform optimisation and does not
  count as a fallback.
- `deterministic = true` is refused: there is no bitwise-determinism contract
  across the available providers.
- `graph_safe` is accepted either way: MUSA's routes are capture-safe
  unconditionally, so `false` is a weaker request that is satisfied too.

## Measuring a change

```bash
# Latency and the accuracy gate in one run: L0 is both the timed and the scored
# call, so the cosine and the p50 describe one configuration.
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 \
  --reference <gold-bundle> --execution eager
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 \
  --reference <gold-bundle> --execution require-graph

# Latency only, no bundle: synthetic zero inputs at the loaded checkpoint's shapes.
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1 --views 2 --token-count 10
```

`execution=` on `ModelRunner.load` — the same policy the benchmark selects with
`--execution` — is read only by this port. The CUDA and HIP runners hardcode
`PreferGraph`, so a non-default value is silently ignored there; set it only
when the device is `musa`.

The benchmark's own `--execution` defaults to `prefer-graph`, so a run without
the flag times the captured path when capture succeeds; `execution.fallback` in
the report says whether it did. Pass `--execution eager` for the eager baseline
and `--execution require-graph` to turn a capture failure into a refusal.

```bash

# L2 needs a tokenizer no checkpoint ships, and LIBERO's own statistics
.venv/bin/python scripts/bench_pi05_musa.py --layer l0,l1,l2 \
  --reference <gold-bundle> \
  --tokenizer <path-to-paligemma_tokenizer.model> \
  --norm-stats <path-to-pi05_libero_base>/assets/physical-intelligence/libero/norm_stats.json
```

The gate scores the engine's `raw_actions` against a `torch-eager-gold.v1`
bundle named by `--reference` — a directory holding `inputs.npz`, `debug.npz`
and `manifest.json`. Bundles are not committed, so each host records one with
`python/apxinf_ref/pi05/generate_pi05_torch_gold.py` and passes it on every
scored call. `--reference` decides both what is measured and whether it is
scored: its frames are the workload, which is why omitting it switches the run
to synthetic zero inputs (`--views`, `--token-count`, and the checkpoint's own
solver schedule) and to a report that says only how long the layers took — the
console prints `accuracy: not scored (no --reference)` and the exit status is 0.
The gate refuses a bundle it cannot compare against — a checkpoint whose
bytes differ from the one the bundle records, a flow schedule other than the
one the engine derives, padded tokens, a mis-shaped noise tensor, or repeats
that are not bit-identical — instead of scoring it. Exit status is 0
measured-and-passed (or measured only), 1 compared-and-failed (cosine <
`--gate`, default 0.999), 2 refused.

L0 is the scored layer when a bundle is given, because it is the call the bundle
fixes exactly; without one nothing is scored and only the timings are reported.
With a bundle, L1 and L2 take its frames through a lossy float32 -> uint8
requantization, which is what a client sends but not what was recorded, so they
are latency only: scoring them would measure the requantization rather than the
engine. Their deltas are not boundary promises either — L0 marshals a host patch
buffer while L1 copies uint8, so L1−L0 can come out negative. A synthetic run
keeps the same shapes and dtypes, so its deltas mean the same thing; its actions
mean nothing, which is why `--export` is refused without `--reference`.
The cosine is computed in float64 over the whole chunk (the artifact module's
own), so a value recorded at float32 precision can differ in its eighth digit
without the actions differing; the band and the gate are what it is read for.

Two caveats for any latency number. First, **eager and captured must be measured
in the same session and compared as p50 with their spread**, because this device
is memory-bound and its spread moves with host load: runs an hour apart can
differ by several percent with no code change. Second, the `.venv` wheel has to
be rebuilt after any change to `crates/apxinf-musa` or the port — a stale wheel
measures the previous build, and nothing else tells you that.

## GR00T BF16 batched vision attention

The MUSA GR00T BF16 path batches two independent 256-token vision segments
with 16 heads and head dimension 64 into one muDNN Flash call. The output
is contiguous `[512, 1024]`, avoiding the two-call path's final concatenation.
The backend enables this route for `mp_22`; other architectures
and unmeasured shapes retain the existing segmented path. Vendor errors are
propagated rather than silently changing algorithms.

The backend's legacy BF16 SDPA mode bypasses batching. Keep prepared arena
enabled, allocation cache disabled,
and use W10/R50 with required Graph execution for whole-model comparisons.
For the fixed `libero-two-view` fixture, use the Orin normalized-action gold
under `path_to/orin_gold/debug.npz`
with `scripts/gr00t/compare_outputs.py --reference`; changed inputs need their
own matching reference. Candidate and baseline should additionally match exactly.

## Verifying the port's own tests

```bash
export MUSA_PATH=<path-to-musa-toolkit>
cargo test -p apxinf-model --features musa --lib musa_pi05::
APXINF_PI05_TEST_CHECKPOINT=<checkpoint dir> \
  cargo test -p apxinf-model --features musa --lib -- --ignored
```

The first runs without a checkpoint. The second needs a real one: it includes
the capture acceptance test, which builds an eager and a captured plan from one
loaded model and compares raw action bits under a changed latent.
