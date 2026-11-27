# Fast-WAM MUSA wheel benchmark

`bench_apxinf_musa_wheel.py` is the fixed-profile benchmark and correctness
checker for the installed `apxinf_py` wheel. It intentionally has no prompt
option: the current profile consumes the cached `context` and `context_mask`
captured by the official Torch eager reference.

The script accepts all three execution paths (`fast`, `idm`, `joint`), verifies
that fixture/reference inputs and fixed noise are identical, discards warmups,
reports the Rust-reported synchronized action latency, checks exact repeated
outputs, and compares only the complete request's final normalized actions.
Video latent/decoded-video values may remain in reference artifacts for
diagnostics, but are not correctness gates.

The Fast-WAM model and Python binding are implemented for the fixed BF16
profile. The binding exposes:

```python
model = apxinf_py.ModelRunner.load("fastwam", checkpoint, ...)
model._infer_fastwam_preprocessed(
    input_image, context, context_mask, proprio, action_noise,
    video_noise=video_noise_or_none,
)
```

The return value may be actions, `(actions, video)`, or a mapping containing
`normalized_actions` and an optional `video_latents`/`decoded_video`. All
returned arrays must be synchronized NumPy arrays; otherwise the wall-clock
measurement would only cover asynchronous launch.

After the binding lands, a joint run using the existing MUSA eager reference is:

```bash
python scripts/fastwam/bench_apxinf_musa_wheel.py \
  --checkpoint /path/to/fastwam-checkpoint \
  --config-json /path/to/fastwam-joint-apxinf.json \
  --mode joint \
  --fixture devlocal/reference/joint-musa-uncond \
  --reference devlocal/reference/joint-musa-uncond \
  --device 0 --execution eager \
  --output devlocal/fastwam-musa/results/joint-musa-wheel.json
```

Run the CPU-only parser/comparison tests with the ApxInf environment:

```bash
/home/mccxadmin/youyang/Apxinf/apxinf_shaonianyou/bin/python \
  -m unittest discover -s scripts/fastwam/tests -v
```

The action gates match the existing BF16 GR00T wheel benchmark. The wheel
benchmark prepares inputs once on MUSA and reports timing from Rust, so the timed interval
includes action D2H just like the official `infer_action()` return path, but excludes
PyO3 input conversion, H2D, Python call overhead, NumPy materialization, and video
auxiliary transfers. Set `MUSA_VISIBLE_DEVICES` to
one currently idle physical GPU and keep `--device 0`; `musa:0` is the process-local
logical device, not a request for physical GPU0.

The prepared plan captures the Wan VAE encoder and DiT separately. The VAE
implements zero padding by concatenating zero rows cached during preflight.
`--execution require-graph` requires both captures to succeed; `prefer-graph`
allows eager fallback, and `eager` disables capture. Use MUSA 5.3.0 consistently for Torch
reference generation and ApxInf execution; references generated with MUSA 5.1/5.2
must be regenerated before parity or performance comparison.

Convert the two official pickle checkpoints once in the trusted Torch eager
environment before loading them from Rust:

```bash
source /home/mccxadmin/youyang/Apxinf/activate-gr00t-torch-eager.sh
python scripts/fastwam/convert_checkpoint.py \
  --model /data/yrRobotics/fastwam/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --vae /data/yrRobotics/fastwam/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --output /path/to/fastwam-apxinf
```

The converter rejects topology/dtype drift, verifies the released model hash,
transposes only verified Linear matrices to ApxInf's `[input,output]` layout,
casts the official F32 VAE source to the official BF16 inference dtype, and
writes separate indexed `model/` and `vae/` SafeTensors assets plus
`fastwam-conversion.json`. Product inference never evaluates PyTorch pickle.

The timing boundary also follows the official Fast-WAM
[`scripts/dryrun_fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/scripts/dryrun_fastwam.py):
warm up first, synchronize before and after each inference, and report repeated
host-observed completion times. The official script uses random Torch/CUDA
inputs; this wheel script adds fixed MUSA eager artifacts and parity checks.
