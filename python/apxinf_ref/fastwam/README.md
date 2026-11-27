# Fast-WAM Torch eager reference

Independent batch-one, two-camera correctness reference for the official
Fast-WAM implementation. It supports three model-core inference paths:

- `fast`: first-frame action inference (`FastWAM.infer_action`);
- `idm`: future-video denoising followed by action inference
  (`FastWAMIDM.infer_action`);
- `joint`: joint video/action denoising (`infer_joint`).

The callable contracts follow the official
[`fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py),
[`fastwam_idm.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam_idm.py),
and [`fastwam_joint.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam_joint.py).

The reference calls the upstream methods directly. It does not reproduce the
Wan scheduler, MoT attention, VAE, or action loop. The source checkout, resolved
model config, checkpoint, runtime, fixed stochastic inputs, and generator files
are identified in `manifest.json`.

## Input fixture

`--input` is an `.npz` file (or an earlier artifact directory) containing:

| name | contract |
| --- | --- |
| `camera_primary` | uint8 `[H,W,3]` or `[1,H,W,3]` |
| `camera_wrist` | uint8 `[H,W,3]` or `[1,H,W,3]` |
| `context` | float32/float16 or raw BF16 uint16, `[L,4096]` or `[1,L,4096]` |
| `context_mask` | bool/uint8, `[L]` or `[1,L]` |
| `proprio` | float32/float16 or raw BF16 uint16, `[D]` or `[1,D]` |

Each camera is center-crop resized independently and the two results are
concatenated horizontally, matching the official LIBERO evaluator. Defaults are
224x224 per view, producing `[1,3,224,448]`. The model input is normalized to
`[-1,1]` after conversion to the selected model dtype.

The released configs normally use cached T5 context (`load_text_encoder=false`),
so the fixture deliberately uses `context/context_mask` instead of a prompt.

## Generate

Use a matching model config and checkpoint for each mode. The config can be an
official standalone `configs/model/fastwam*.yaml` or a composed task config with
a top-level `model` section. Model assets must already be available locally;
the generator sets Hugging Face and Transformers offline mode.

```bash
python generate_fastwam_torch_gold.py \
  --source /path/to/FastWAM \
  --config /path/to/FastWAM/configs/model/fastwam.yaml \
  --checkpoint /path/to/libero_uncond_2cam224.pt \
  --input /path/to/two_camera_context_fixture.npz \
  --mode fast --device cuda:0 \
  --output /path/to/devlocal/fastwam-gold/fast-cuda

python generate_fastwam_torch_gold.py \
  --source /path/to/FastWAM \
  --config /path/to/FastWAM/configs/model/fastwam_idm.yaml \
  --checkpoint /path/to/idm_checkpoint.pt \
  --input /path/to/two_camera_context_fixture.npz \
  --mode idm --device cuda:0 \
  --output /path/to/devlocal/fastwam-gold/idm-cuda

python generate_fastwam_torch_gold.py \
  --source /path/to/FastWAM \
  --config /path/to/FastWAM/configs/model/fastwam_joint.yaml \
  --checkpoint /path/to/joint_checkpoint.pt \
  --input /path/to/two_camera_context_fixture.npz \
  --mode joint --device cuda:0 \
  --output /path/to/devlocal/fastwam-gold/joint-cuda
```

All three release-profile references default to `--sigma-shift 5.0`, matching
the original released checkpoint and the ApxInf LIBERO two-camera profile.
For a different checkpoint, pass its evaluation shift explicitly.

The default is eager BF16, one discarded warmup and two exact repeated captures.
`joint` disables the upstream `test_action_with_infer_action` side run so the
artifact represents exactly one joint generation.

When `--input` is an earlier artifact, its stored `action_noise` and
`video_noise` are reused bit-for-bit; otherwise they are generated on CPU from
`--seed` and then frozen into the new artifact.

## Artifact

- `inputs.npz`: the two raw cameras plus exact canonical image, context,
  proprio, action noise, and (for IDM/joint) video noise;
- `debug.npz`: VAE first-frame latent, selected video K/V cache boundaries,
  every scheduler flow/latent step, final normalized action, and video latent or
  decoded frames where applicable;
- `reference.json`: compact action reference and optional video digest;
- `manifest.json`: source/config/checkpoint/runtime identity, tensor hashes,
  preprocessing contract, and exact-repeat checks.

Existing output directories are never replaced. The artifact is staged, read
back, hashed, and atomically published. Its artifact-side tests require only
NumPy and can run without torch:

```bash
python -m unittest discover -s tests -v
```

Here, "torch-free" applies only to artifact inspection and the tests above.
Capturing a real Fast-WAM reference runs the upstream model and therefore
requires torch; a MUSA capture additionally requires a matching `torch_musa`.
