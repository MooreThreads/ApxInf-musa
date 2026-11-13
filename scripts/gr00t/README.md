# GR00T scripts

Run these commands from the repository root. Keep generated results under
`devlocal/gr00t-n1d7/`; scripts do not automatically run other benchmarks.

The MUSA gates check cosine, relative L2 and `max_abs <= 0.125`, and require
the reference noise (`--reference-noise`, for example the Orin `inputs.npz`)
unless `--allow-missing-noise` is explicit. All MUSA paths require warmup >= 1.
The three GPU lifecycle tests are `#[ignore]`; see
[doc/gr00t-n1.7.md](../../doc/gr00t-n1.7.md#musa-bf16) for the serial commands.

| Script | Purpose |
| --- | --- |
| `bench_apxinf_musa_rust.py` | Native MUSA benchmark, W10/R50, strict Graph by default; optional explicit `--reference` |
| `bench_apxinf_musa_wheel.py` | In-process MUSA Python wheel benchmark using `apxinf_py.ModelRunner`, W10/R50, BF16 |
| `../bench_gr00t.py` | Existing native CUDA benchmark, kept at its original location |
| `bench_torch_eager_musa.py` | Optional NVIDIA PyTorch MUSA reference exporter and benchmark |
| `compare_outputs.py` | Compare native benchmark JSON or normalized-action NPZ against an explicit reference |
| `check_gr00t_module_boundaries.py` | Check MUSA model module isolation |

## Reference selection

The Rust script launches an executable. The wheel script calls the existing
`ModelRunner.load("gr00t", ...)` and `_infer_preprocessed` bindings directly;
it does not launch Cargo or a Rust subprocess. Both use the same fixture and
explicit initial noise. The wheel converts fixture BF16 bits to exact float32
NumPy values before timing. Its measured boundary includes per-call NumPy/PyO3
conversion, synchronized model execution, action D2H and NumPy output, but not
the official processor or action decoding. Do not treat it as the identical
timing boundary to the Rust executable.

Build/install the wheel and run it as documented in
[GR00T N1.7](../../doc/gr00t-n1.7.md#musa-python-wheel).

Prefer acceptance reference values generated **on Orin** using
[`python/apxinf_ref/GR00T-N1.7-LIBERO`](../../python/apxinf_ref/GR00T-N1.7-LIBERO/README.md)
and its `generate_GR00T_N1_7_LIBERO_torch_gold.py` script. Use the exact MUSA
preprocessed fixture as the generator's `--input`, and compare against the
generated `debug.npz` (normalized actions) or `reference.json`. Verify fixture
bytes and checkpoint hashes against the generated `manifest.json` before
comparison; see the linked reference README for the Orin generation command.

The optional M1000 PyTorch eager exporter is a diagnostic tool, not the preferred
Orin acceptance reference. No Orin result is bundled or implicitly selected.
Pass the actual reference explicitly:

```bash
python scripts/gr00t/compare_outputs.py \
  --apxinf path_to/musa_output.json \
  --reference path_to/orin_gold/debug.npz \
  --reference-noise path_to/orin_gold/inputs.npz \
  --precision bf16 \
  --output path_to/parity.json
```

NPY contains normalized actions directly. JSON requires `output.values` and
`output.shape`; NPZ requires `normalized_actions`, with optional `actions` and
`noise`. Compare normalized
actions to normalized actions, not decoded actions. The checkpoint, backbone,
processor, input images/prompt/state, embodiment, precision, denoising settings
and initial noise must match. Verify their provenance separately; this tool
checks shapes, finite values, numerical gates and noise equality. Noise is
required by default; `--allow-missing-noise` is an explicit exemption. A
passing comparison alone does not establish matching input provenance.

## Shared numeric formats

The MUSA wheel benchmark, Rust benchmark wrapper, PyTorch MUSA reference script
and comparison CLI use `formats.py` for their numeric inputs. These rules do not
change the CUDA benchmark or the bit-preserving official gold artifact writer.

| Input | Storage / accepted shape | Loaded representation |
| --- | --- | --- |
| Fixture directory | `manifest.json` plus little-endian raw `.bin` tensors; existing `apxinf.gr00t-n1.7.preprocessed-fixture.v1` schema | BF16 tensors decoded to contiguous float32; IDs remain uint32, mask remains uint8 |
| `--noise-input` NPY | Floating array or uint16 BF16 bits, `[40,132]` or `[1,40,132]` | Contiguous float32 `[1,40,132]` |
| `--noise-input` NPZ | Required `noise` array, same encodings/shapes as NPY | Same as NPY; official `inputs.npz` is supported |
| Reference NPY / NPZ `normalized_actions` | Floating array or uint16 BF16 bits, `[horizon,dim]` or `[1,horizon,dim]` | Contiguous float32 `[horizon,dim]` |
| Benchmark/reference JSON | `output.shape` and numeric `output.values`; numbers represent values, not BF16 bit patterns | Float32 normalized actions, optional singleton batch removed |
| Optional reference `noise` | Same array encodings and shapes as `--noise-input` | Decoded and batch-normalized before equality comparison |

For floating tensors, uint16 always means **raw BF16 bits**, not integer values:
`(values.astype(np.uint32) << 16).view(np.float32)`. Thus `0x3f80` becomes `1.0`,
not `16256.0`. Float16/32/64 inputs are converted numerically to float32. Other
integer types, bool, complex, object arrays, empty arrays and non-finite values
are rejected; NumPy pickle loading is disabled. JSON integer numbers are normal
numeric values, not BF16 storage. Token IDs and attention masks are not passed
through the floating-tensor decoder.

The fixture reader checks declared dtype, exact byte count, shape and path
containment before inference. Wheel and Rust wrapper validate available reference
noise against fixture noise before loading/running the model. A reference without
noise remains usable, but cannot prove matching initial noise.

All reference readers require explicit normalized-action semantics. The Rust
wrapper no longer guesses NPZ keys `final_action`, `output`, or `actions`, nor
accepts bare JSON lists or `actions` aliases. Convert such legacy files to the
canonical `normalized_actions` NPZ or `output.shape`/`output.values` JSON only
after confirming the data is normalized, not decoded robot actions.

Wheel/Rust benchmark reports use `.json`; the PyTorch exporter requires `.npz`
and writes decoded floating arrays. Its exported `noise` records the values
actually consumed after the model's BF16 conversion, not the raw uint16 words
or an unrounded float32 input. `debug.npz` need not contain noise: use the matching
`inputs.npz` for `--noise-input` when necessary. Raw `noise.bin` alone has no shape
metadata and is not accepted by `--noise-input`.

CPU-only format regression tests (no checkpoint or GPU required):

```bash
python -m unittest discover -s scripts/gr00t/tests -v
```

## Optional M1000 PyTorch reference

`bench_torch_eager_musa.py` exports a diagnostic reference on MUSA.
It needs the NVIDIA GR00T package, compatible torch/torch_musa, and this
repository's Python processor adapter. Use `--help` for the required checkpoint,
backbone, images, prompt, eight state values, fixed noise and output paths.
Its output NPZ contains `normalized_actions`, decoded `actions`, `noise`, timing
samples and a report. Pass that NPZ via `compare_outputs.py --reference` for
optional diagnostics; `--torch-eager` remains an alias for compatibility.
As with Orin gold capture, `--repeat` must be at least 2. All measured outputs
must be finite and exactly equal across runs (both normalized and decoded
actions); otherwise the script fails before writing the reference NPZ. On
success it exports the first measured output and records `repeat_exact: true`.

## Benchmark options

- The native Rust benchmark and `bench_apxinf_musa_rust.py` require positive
  warmup and iteration counts. Zero warmup is rejected because execution mode
  is checked after warmup, when the prepared plan must already exist.
- The MUSA wrapper requires `--precision` and accepts optional
  `--calibration`, `--tactics`, `--autotune`, `--output`, and `--reference`.
  `--execution` defaults to strict Graph execution.
  `--autotune` requires an explicit `--tactics` output path.
- The current MUSA GR00T loader supports BF16 only. It loads the explicit
  `--tactics` JSON file or the device/checkpoint default. A missing explicit
  file is an error unless `--autotune` requests its creation. Tuning runs on
  real request tensors before Graph capture; capture itself suppresses tuning.
  This uses the current MUSA backend candidate set: exact assembly routes stay
  on their existing dispatch, while remaining BF16 GEMMs tune muBLAS vs muDNN.
- The comparison tool accepts backend-neutral `--reference`;
  `--torch-eager` remains an alias for compatibility.
- Each benchmark runs independently; no combined benchmark launcher is provided.
