"""Torch-free Fast-WAM fixture validation and atomic artifact publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import MODEL_NAME, MODES, SCHEMA


REQUIRED_INPUTS = ("camera_primary", "camera_wrist", "context", "context_mask", "proprio")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bf16_values(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.uint16:
        raise ValueError(f"raw BF16 storage must be uint16, got {array.dtype}")
    return (array.astype(np.uint32) << 16).view(np.float32)


def require_finite(name: str, array: np.ndarray) -> None:
    values = bf16_values(array) if array.dtype == np.uint16 else array
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def array_metadata(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    metadata: dict[str, Any] = {
        "shape": list(array.shape),
        "storage_dtype": str(array.dtype),
        "sha256": array_digest(array),
    }
    if array.dtype == np.uint16:
        metadata.update(dtype="torch.bfloat16", encoding="bfloat16-bits")
    else:
        metadata.update(dtype=str(array.dtype), encoding="native")
    return metadata


def _unbatch_camera(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must have batch size one, got {value.shape}")
        value = value[0]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be HWC RGB or batch-one NHWC, got {value.shape}")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {value.dtype}")
    return np.ascontiguousarray(value)


def read_fixture(path: Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser().resolve()
    archive_path = path / "inputs.npz" if path.is_dir() else path
    if not archive_path.is_file():
        raise FileNotFoundError(f"fixture not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_INPUTS if name not in archive]
        if missing:
            raise ValueError(f"fixture is missing required arrays: {missing}")
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["camera_primary"] = _unbatch_camera("camera_primary", arrays["camera_primary"])
    arrays["camera_wrist"] = _unbatch_camera("camera_wrist", arrays["camera_wrist"])
    context = arrays["context"]
    if context.ndim == 2:
        context = context[None]
    if context.ndim != 3 or context.shape[0] != 1:
        raise ValueError(f"context must be [L,D] or [1,L,D], got {context.shape}")
    if context.dtype not in (np.float16, np.float32, np.uint16):
        raise ValueError(f"context must be float16, float32, or raw BF16 uint16, got {context.dtype}")
    require_finite("context", context)
    mask = arrays["context_mask"]
    if mask.ndim == 1:
        mask = mask[None]
    if mask.ndim != 2 or mask.shape[0] != 1 or mask.shape[1] != context.shape[1]:
        raise ValueError(f"context_mask must match context [1,L], got {mask.shape} vs {context.shape}")
    proprio = arrays["proprio"]
    if proprio.ndim == 1:
        proprio = proprio[None]
    if proprio.ndim != 2 or proprio.shape[0] != 1:
        raise ValueError(f"proprio must be [D] or [1,D], got {proprio.shape}")
    if proprio.dtype not in (np.float16, np.float32, np.uint16):
        raise ValueError(f"proprio must be float16, float32, or raw BF16 uint16, got {proprio.dtype}")
    require_finite("proprio", proprio)
    for name in ("action_noise", "video_noise"):
        if name in arrays:
            value = arrays[name]
            if value.dtype not in (np.float16, np.float32, np.uint16):
                raise ValueError(f"{name} must be float16, float32, or raw BF16 uint16")
            require_finite(name, value)
    arrays["context"] = np.ascontiguousarray(context)
    arrays["context_mask"] = np.ascontiguousarray(mask.astype(np.bool_))
    arrays["proprio"] = np.ascontiguousarray(proprio)
    return arrays


def _validate_output(mode: str, debug: dict[str, np.ndarray], action_horizon: int, action_dim: int) -> None:
    if mode not in MODES:
        raise ValueError(f"unsupported Fast-WAM mode: {mode}")
    if "normalized_actions" not in debug:
        raise ValueError("debug output is missing normalized_actions")
    actions = np.asarray(debug["normalized_actions"])
    if actions.shape != (action_horizon, action_dim):
        raise ValueError(
            f"normalized_actions must be [{action_horizon},{action_dim}], got {actions.shape}"
        )
    if actions.dtype != np.float32 or not np.isfinite(actions).all():
        raise ValueError("normalized_actions must be finite float32")
    for name, value in debug.items():
        require_finite(name, np.asarray(value))
    if mode == "idm" and "returned_video_latents" not in debug:
        raise ValueError("IDM capture must include returned_video_latents")
    if mode == "joint":
        frames = debug.get("decoded_video")
        if frames is None or frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("joint capture must include uint8 decoded_video [T,H,W,3]")


def write_artifact(
    output: Path,
    *,
    mode: str,
    inputs: dict[str, np.ndarray],
    debug: dict[str, np.ndarray],
    provenance: dict[str, Any],
    action_horizon: int,
    action_dim: int,
) -> None:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    _validate_output(mode, debug, action_horizon, action_dim)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        metadata: dict[str, Any] = {}
        for filename, arrays in (("inputs.npz", inputs), ("debug.npz", debug)):
            arrays = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
            np.savez_compressed(staging / filename, **arrays)
            metadata[filename] = {name: array_metadata(value) for name, value in arrays.items()}
            with np.load(staging / filename, allow_pickle=False) as restored:
                if set(restored.files) != set(arrays):
                    raise ValueError(f"artifact member mismatch in {filename}")
                for name, value in arrays.items():
                    if not np.array_equal(value, restored[name]):
                        raise ValueError(f"artifact roundtrip mismatch: {filename}:{name}")
        actions = debug["normalized_actions"]
        reference: dict[str, Any] = {
            "model": MODEL_NAME,
            "mode": mode,
            "output": {"shape": list(actions.shape), "values": actions.reshape(-1).tolist()},
        }
        if "decoded_video" in debug:
            reference["decoded_video"] = array_metadata(debug["decoded_video"])
        if "returned_video_latents" in debug:
            reference["video_latents"] = array_metadata(debug["returned_video_latents"])
        (staging / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
        files = {
            name: sha256(staging / name)
            for name in ("inputs.npz", "debug.npz", "reference.json")
        }
        manifest = {
            "schema": SCHEMA,
            "model": MODEL_NAME,
            "mode": mode,
            "profile": {
                "batch": 1,
                "camera_views": 2,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
            },
            "provenance": provenance,
            "arrays": metadata,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        for name, digest in files.items():
            if sha256(staging / name) != digest:
                raise ValueError(f"artifact hash mismatch: {name}")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
