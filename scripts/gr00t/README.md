# GR00T scripts

Run these commands from the repository root. Keep generated results under
`devlocal/gr00t-port/`; scripts do not automatically run other benchmarks.

| Script | Purpose |
| --- | --- |
| `bench_apxinf_musa.py` | Native MUSA benchmark, W10/R50, strict Graph by default; optional explicit `--reference` |
| `bench_apxinf_cuda.py` | Existing native CUDA benchmark, relocated without behavioral changes |
| `bench_torch_eager_musa.py` | Optional NVIDIA PyTorch MUSA reference exporter and benchmark |
| `compare_outputs.py` | Compare native benchmark JSON or normalized-action NPZ against an explicit reference |
| `check_gr00t_module_boundaries.py` | Check MUSA model module isolation |

## Reference selection

The intended acceptance reference is an output exported on Orin, not an
automatically launched M1000 PyTorch eager run. No Orin result is bundled or
implicitly selected. Pass the actual reference explicitly:

```bash
python scripts/gr00t/compare_outputs.py \
  --apxinf path_to/musa_output.json \
  --reference path_to/orin_gold/debug.npz \
  --precision bf16 \
  --output path_to/parity.json
```

JSON requires `output.values` and `output.shape`; NPZ requires
`normalized_actions`, with optional `actions` and `noise`. Compare normalized
actions to normalized actions, not decoded actions. The checkpoint, backbone,
processor, input images/prompt/state, embodiment, precision, denoising settings
and initial noise must match. Verify their provenance separately; this tool
checks shapes, finite values, numerical gates and noise equality only when
both archives contain noise. A passing comparison alone does not establish
matching input provenance.

## Optional M1000 PyTorch reference

`bench_torch_eager_musa.py` exports a diagnostic reference on MUSA.
It needs the NVIDIA GR00T package, compatible torch/torch_musa, and this
repository's Python processor adapter. Use `--help` for the required checkpoint,
backbone, images, prompt, eight state values, fixed noise and output paths.
Its output NPZ contains `normalized_actions`, decoded `actions`, `noise`, timing
samples and a report. Pass that NPZ via `compare_outputs.py --reference` for
optional diagnostics; `--torch-eager` remains an alias for compatibility.

## Benchmark options

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
