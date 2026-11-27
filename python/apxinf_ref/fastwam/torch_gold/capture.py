"""Drive official Fast-WAM eager inference and record deterministic boundaries."""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from .artifact import array_digest, read_fixture, sha256, write_artifact


def synchronize(device: torch.device) -> None:
    if device.type == "cpu":
        return
    getattr(torch, device.type).synchronize(device)


def tensor_array(tensor: torch.Tensor) -> np.ndarray:
    source_device = tensor.device
    synchronize(source_device)
    host = tensor.detach().contiguous().to("cpu")
    synchronize(source_device)
    if host.dtype == torch.bfloat16:
        return host.view(torch.uint16).numpy().copy()
    return host.numpy().copy()


def tensor_from_array(array: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.ascontiguousarray(array)
    if array.dtype == np.uint16:
        tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(array.copy())
    return tensor.to(device=device, dtype=dtype)


def file_identity(root: Path, patterns: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    root = Path(root).resolve()
    paths = sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()})
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    }


def source_revision(source: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(source)}
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        record["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None
        record["git_dirty"] = None
    record["files"] = file_identity(source / "src/fastwam", ("*.py",))
    return record


def resolve_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type not in ("cuda", "musa"):
        raise ValueError("Fast-WAM gold requires an explicit cuda:N or musa:N device")
    module = getattr(torch, device.type, None)
    if module is None or not module.is_available():
        raise RuntimeError(f"requested {device.type}, but torch reports it unavailable")
    index = 0 if device.index is None else device.index
    if index < 0 or index >= module.device_count():
        raise RuntimeError(f"requested {device}, but only {module.device_count()} devices are visible")
    return torch.device(device.type, index)


def _center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    source_width, source_height = pil_image.size
    scale = max(width / source_width, height / source_height)
    resized = pil_image.resize(
        (round(source_width * scale), round(source_height * scale)),
        resample=Image.BILINEAR,
    )
    resized_width, resized_height = resized.size
    left = max((resized_width - width) // 2, 0)
    top = max((resized_height - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def prepare_inputs(arrays, args, device, dtype):
    primary = _center_crop_resize(
        arrays["camera_primary"], width=args.camera_width, height=args.camera_height
    )
    wrist = _center_crop_resize(
        arrays["camera_wrist"], width=args.camera_width, height=args.camera_height
    )
    rgb = np.ascontiguousarray(np.concatenate([primary, wrist], axis=1))
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    image = image * (2.0 / 255.0) - 1.0
    context = tensor_from_array(arrays["context"], device=device, dtype=dtype)
    context_mask = torch.from_numpy(arrays["context_mask"].copy()).to(device=device, dtype=torch.bool)
    proprio = tensor_from_array(arrays["proprio"], device=device, dtype=dtype)
    return image, context, context_mask, proprio


@contextlib.contextmanager
def source_working_directory(source: Path):
    previous = Path.cwd()
    os.chdir(source)
    try:
        yield
    finally:
        os.chdir(previous)


def load_model(args, arrays, device, dtype):
    if not args.config.is_file():
        raise FileNotFoundError(f"model config not found: {args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    sys.path.insert(0, str(args.source / "src"))
    sys.path.insert(0, str(args.source))
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict

    loaded = OmegaConf.load(args.config)
    model_config = loaded.model if "model" in loaded else loaded
    if "_target_" not in model_config:
        raise ValueError("--config must be a model config or contain a top-level model section")
    with open_dict(model_config):
        model_config.proprio_dim = int(arrays["proprio"].shape[-1])
        model_config.video_dit_config.action_dim = int(args.action_dim)
        model_config.action_dit_config.action_dim = int(args.action_dim)
        # The required fine-tuned checkpoint replaces both DiT experts. Avoid
        # loading redundant pretrained backbones; this is also the upstream
        # released-checkpoint MUSA smoke-test setup in the supplied checkout.
        model_config.skip_dit_load_from_pretrain = True
        model_config.action_dit_pretrained_path = None
        # The local Wan asset is the original Wan2.2_VAE.pth layout rather than
        # the redirected DiffSynth converted-safetensors repository.
        model_config.redirect_common_files = False
        gradient_checkpointing = bool(model_config.get("mot_checkpoint_mixed_attn", False))
        model_config.video_dit_config.use_gradient_checkpointing = gradient_checkpointing
        model_config.action_dit_config.use_gradient_checkpointing = gradient_checkpointing
    OmegaConf.resolve(model_config)
    resolved = OmegaConf.to_container(model_config, resolve=True)
    target = str(resolved["_target_"])
    allowed_targets = {
        "fast": ("create_fastwam", "create_fastwam_optional_idm"),
        "idm": ("create_fastwam_idm", "create_fastwam_optional_idm"),
        "joint": ("create_fastwam_joint",),
    }
    if not target.endswith(allowed_targets[args.mode]):
        raise ValueError(
            f"mode {args.mode!r} is incompatible with model target {target!r}; "
            f"expected one of {allowed_targets[args.mode]}"
        )
    with source_working_directory(args.source):
        model = instantiate(model_config, model_dtype=dtype, device=str(device))
        model.load_checkpoint(str(args.checkpoint))
    model = model.to(device).eval()
    if int(model.action_expert.action_dim) != args.action_dim:
        raise ValueError(
            f"loaded action dim {model.action_expert.action_dim} does not match --action-dim {args.action_dim}"
        )
    return model, resolved


def make_fixed_noise(model, args, arrays, image, device, dtype):
    action_shape = (1, args.action_horizon, args.action_dim)
    if "action_noise" in arrays:
        action = tensor_from_array(arrays["action_noise"], device=device, dtype=dtype)
        action_source = "fixture"
    else:
        action_generator = torch.Generator(device="cpu").manual_seed(args.seed)
        action = torch.randn(action_shape, generator=action_generator, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        action_source = "seed"
    if tuple(action.shape) != action_shape:
        raise ValueError(f"action_noise must have shape {action_shape}, got {tuple(action.shape)}")
    video = None
    video_source = None
    if args.mode in ("idm", "joint"):
        latent_t = (args.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
        latent_h = image.shape[-2] // model.vae.upsampling_factor
        latent_w = image.shape[-1] // model.vae.upsampling_factor
        video_shape = (1, model.vae.model.z_dim, latent_t, latent_h, latent_w)
        if "video_noise" in arrays:
            video = tensor_from_array(arrays["video_noise"], device=device, dtype=dtype)
            video_source = "fixture"
        else:
            video_generator = torch.Generator(device="cpu").manual_seed(args.seed)
            video = torch.randn(
                video_shape,
                generator=video_generator,
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)
            video_source = "seed"
        if tuple(video.shape) != video_shape:
            raise ValueError(f"video_noise must have shape {video_shape}, got {tuple(video.shape)}")
    return action, video, {"action": action_source, "video": video_source}


@contextlib.contextmanager
def record_boundaries(model, enabled: bool):
    captured: dict[str, Any] = {}
    if not enabled:
        yield captured
        return
    originals: list[tuple[Any, str, Any]] = []

    def replace(owner, name, replacement):
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    original_encode = model._encode_input_image_latents_tensor

    def encode_wrapper(*args, **kwargs):
        value = original_encode(*args, **kwargs)
        captured["first_frame_latents"] = value.detach().clone()
        return value

    replace(model, "_encode_input_image_latents_tensor", encode_wrapper)

    original_action_step = model.infer_action_scheduler.step
    action_index = 0

    def action_step_wrapper(flow, delta, latent):
        nonlocal action_index
        output = original_action_step(flow, delta, latent)
        captured[f"action_flow_{action_index:02d}"] = flow.detach().clone()
        captured[f"action_latent_{action_index:02d}"] = output.detach().clone()
        action_index += 1
        return output

    replace(model.infer_action_scheduler, "step", action_step_wrapper)

    original_video_step = model.infer_video_scheduler.step
    video_index = 0

    def video_step_wrapper(flow, delta, latent):
        nonlocal video_index
        output = original_video_step(flow, delta, latent)
        captured[f"video_flow_{video_index:02d}"] = flow.detach().clone()
        captured[f"video_latent_{video_index:02d}"] = output.detach().clone()
        video_index += 1
        return output

    replace(model.infer_video_scheduler, "step", video_step_wrapper)

    original_prefill = model.mot.prefill_video_cache_tensor

    def prefill_wrapper(*args, **kwargs):
        keys, values = original_prefill(*args, **kwargs)
        if keys:
            captured["video_cache_k_first"] = keys[0].detach().clone()
            captured["video_cache_v_first"] = values[0].detach().clone()
            captured["video_cache_k_last"] = keys[-1].detach().clone()
            captured["video_cache_v_last"] = values[-1].detach().clone()
        return keys, values

    replace(model.mot, "prefill_video_cache_tensor", prefill_wrapper)
    try:
        yield captured
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def decoded_video_array(frames) -> np.ndarray:
    arrays = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        else:
            array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"decoded frame must be HWC RGB, got {array.shape}")
        arrays.append(array)
    if not arrays:
        raise ValueError("joint inference returned no decoded video frames")
    return np.stack(arrays)


def infer_once(model, args, image, context, context_mask, proprio, action_noise, video_noise, capture):
    action_shape = tuple(action_noise.shape)
    video_shape = None if video_noise is None else tuple(video_noise.shape)
    calls = {"action": 0, "video": 0}

    def fixed_randn(*positional, **keywords):
        shape = keywords.get("size", positional[0] if positional else ())
        shape = tuple(shape) if isinstance(shape, (tuple, list, torch.Size)) else ()
        requested_device = torch.device(keywords.get("device", "cpu"))
        requested_dtype = keywords.get("dtype", torch.float32)
        if shape == action_shape:
            calls["action"] += 1
            return action_noise.to(device=requested_device, dtype=requested_dtype).clone()
        if video_noise is not None and shape == video_shape:
            calls["video"] += 1
            return video_noise.to(device=requested_device, dtype=requested_dtype).clone()
        raise RuntimeError(f"unexpected torch.randn request during Fast-WAM inference: {shape}")

    kwargs = {
        "prompt": None,
        "input_image": image,
        "action_horizon": args.action_horizon,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "negative_prompt": None,
        "text_cfg_scale": 1.0,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
        "rand_device": "cpu",
        "tiled": False,
        "compile_action_infer": False,
    }
    if args.mode in ("idm", "joint"):
        kwargs["num_video_frames"] = args.num_video_frames
    method = model.infer_joint if args.mode == "joint" else model.infer_action
    signature = inspect.signature(method)
    if "action_infer_mode" in signature.parameters:
        kwargs["action_infer_mode"] = "first_frame" if args.mode == "fast" else "idm"
    if args.mode == "joint" and "test_action_with_infer_action" in signature.parameters:
        kwargs["test_action_with_infer_action"] = False
    with record_boundaries(model, capture) as boundaries, patch.object(
        torch, "randn", side_effect=fixed_randn
    ), torch.inference_mode():
        result = method(**kwargs)
    synchronize(image.device)
    expected = {"action": 1, "video": 0 if args.mode == "fast" else 1}
    if calls != expected:
        raise RuntimeError(f"fixed-noise call mismatch: expected {expected}, got {calls}")
    debug = {name: tensor_array(value) for name, value in boundaries.items()}
    actions = result["action"].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    debug["normalized_actions"] = actions
    if "video_latents" in result:
        debug["returned_video_latents"] = tensor_array(result["video_latents"])
    if "video" in result:
        debug["decoded_video"] = decoded_video_array(result["video"])
    return debug


def check_repeats(captures: list[dict[str, np.ndarray]]) -> None:
    reference = captures[0]
    for run_index, candidate in enumerate(captures[1:], start=1):
        if set(candidate) != set(reference):
            raise RuntimeError(f"capture {run_index} has different debug tensor names")
        for name, expected in reference.items():
            if not np.array_equal(expected, candidate[name]):
                raise RuntimeError(f"repeat capture differs at {name}; refusing to publish gold")


def run(args) -> int:
    arrays = read_fixture(args.input)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model, resolved_config = load_model(args, arrays, device, dtype)
    image, context, context_mask, proprio = prepare_inputs(arrays, args, device, dtype)
    if int(context.shape[-1]) != int(model.text_dim):
        raise ValueError(f"context width {context.shape[-1]} does not match model text_dim {model.text_dim}")
    action_noise, video_noise, noise_source = make_fixed_noise(
        model, args, arrays, image, device, dtype
    )
    for _ in range(args.warmup_runs):
        infer_once(
            model, args, image, context, context_mask, proprio, action_noise, video_noise, False
        )
    captures = [
        infer_once(
            model, args, image, context, context_mask, proprio, action_noise, video_noise, True
        )
        for _ in range(args.capture_runs)
    ]
    check_repeats(captures)
    canonical_inputs = {
        "camera_primary": arrays["camera_primary"],
        "camera_wrist": arrays["camera_wrist"],
        "input_image": tensor_array(image),
        "context": tensor_array(context),
        "context_mask": tensor_array(context_mask),
        "proprio": tensor_array(proprio),
        "action_noise": tensor_array(action_noise),
    }
    if video_noise is not None:
        canonical_inputs["video_noise"] = tensor_array(video_noise)
    stack = getattr(torch, device.type)
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hydra_core": importlib.metadata.version("hydra-core"),
        "omegaconf": importlib.metadata.version("omegaconf"),
        "device": str(device),
        "device_name": stack.get_device_name(device),
        "precision": args.precision,
        "attention": "upstream eager/SDPA",
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "cudnn_enabled": getattr(getattr(torch.backends, "cudnn", None), "enabled", None),
        "cudnn_benchmark": getattr(getattr(torch.backends, "cudnn", None), "benchmark", None),
        "cudnn_deterministic": getattr(
            getattr(torch.backends, "cudnn", None), "deterministic", None
        ),
        "tf32_matmul": getattr(
            getattr(getattr(torch.backends, "cuda", None), "matmul", None),
            "allow_tf32",
            None,
        ),
        "tf32_cudnn": getattr(getattr(torch.backends, "cudnn", None), "allow_tf32", None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    provenance = {
        "invocation": {
            "argv": [sys.executable, *sys.argv],
            "cwd": os.getcwd(),
            "environment": {
                key: os.environ[key]
                for key in (
                    "MUSA_HOME", "MUSA_PATH", "MUSA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
                    "LD_PRELOAD", "DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_SKIP_DOWNLOAD",
                )
                if key in os.environ
            },
        },
        "source": source_revision(args.source),
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": sha256(args.checkpoint),
        },
        "config": {
            "path": str(args.config),
            "sha256": sha256(args.config),
            "resolved_model": resolved_config,
        },
        "generator": file_identity(Path(__file__).resolve().parents[1], ("*.py",)),
        "runtime": runtime,
        "inference": {
            "mode": args.mode,
            "action_horizon": args.action_horizon,
            "action_dim": args.action_dim,
            "num_video_frames": None if args.mode == "fast" else args.num_video_frames,
            "num_inference_steps": args.num_inference_steps,
            "sigma_shift": args.sigma_shift,
            "compile_action_infer": False,
            "tiled": False,
            "text_cfg_scale": 1.0,
            "fixed_noise": True,
            "noise_source": noise_source,
        },
        "preprocessing": {
            "batch": 1,
            "camera_views": 2,
            "per_camera_hw": [args.camera_height, args.camera_width],
            "concat": "horizontal",
            "model_image_shape": list(image.shape),
            "image_range": "uint8 -> model dtype -> x*(2/255)-1",
        },
        "checks": {
            "finite_actions": bool(np.isfinite(captures[0]["normalized_actions"]).all()),
            "repeat_exact": True,
            "warmup_runs": args.warmup_runs,
            "capture_runs": args.capture_runs,
            "input_digest": {name: array_digest(value) for name, value in canonical_inputs.items()},
        },
        "boundary": "two uint8 cameras + cached T5 context + proprio + fixed noise to normalized action/video",
    }
    write_artifact(
        args.output,
        mode=args.mode,
        inputs=canonical_inputs,
        debug=captures[0],
        provenance=provenance,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
    )
    print(f"Verified Fast-WAM {args.mode} artifact: {args.output}")
    return 0
