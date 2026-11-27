"""Safe NumPy readers for fixed-profile Fast-WAM benchmark artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_SCHEMA = "apxinf.fastwam.torch-eager-gold.v1"
MODEL_NAME = "yuantianyuan01/FastWAM"
MODES = ("fast", "idm", "joint")


def decode_float_array(values: np.ndarray, name: str) -> np.ndarray:
    """Decode float storage or little-endian raw BF16 bits to finite float32."""
    values = np.asarray(values)
    if values.dtype.kind == "u" and values.dtype.itemsize == 2:
        values = (values.astype(np.uint32) << 16).view(np.float32)
    elif values.dtype.kind == "f":
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
    else:
        raise ValueError(
            f"{name} must contain floating values or raw BF16 uint16 bits, "
            f"got {values.dtype}"
        )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain nonempty, finite values")
    return np.ascontiguousarray(values)


def _array(archive: Any, name: str, *, floating: bool = True) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"fixture is missing {name}")
    values = archive[name]
    if floating:
        return decode_float_array(values, name)
    return np.ascontiguousarray(values)


def _expect_shape(values: np.ndarray, expected: tuple[int | None, ...], name: str) -> None:
    if values.ndim != len(expected) or any(
        wanted is not None and actual != wanted
        for actual, wanted in zip(values.shape, expected)
    ):
        rendered = ["*" if size is None else str(size) for size in expected]
        raise ValueError(f"{name} must have shape [{','.join(rendered)}], got {values.shape}")


def load_fixture(path: Path, mode: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load an ``inputs.npz`` file or a complete eager-reference directory."""
    if mode not in MODES:
        raise ValueError(f"unsupported Fast-WAM mode {mode!r}")
    path = path.expanduser().resolve()
    metadata: dict[str, Any] = {}
    if path.is_dir():
        manifest_path = path / "manifest.json"
        path = path / "inputs.npz"
        if not manifest_path.is_file():
            raise ValueError(f"fixture directory has no manifest.json: {manifest_path.parent}")
        metadata = json.loads(manifest_path.read_text())
        if metadata.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"unsupported fixture schema {metadata.get('schema')!r}")
        if metadata.get("model") != MODEL_NAME:
            raise ValueError(f"unexpected fixture model {metadata.get('model')!r}")
        if metadata.get("mode") != mode:
            raise ValueError(
                f"fixture mode {metadata.get('mode')!r} does not match --mode {mode!r}"
            )
    if path.suffix != ".npz" or not path.is_file():
        raise ValueError(f"fixture must be an inputs .npz or artifact directory: {path}")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            "input_image": _array(archive, "input_image"),
            "context": _array(archive, "context"),
            "context_mask": _array(archive, "context_mask", floating=False),
            "proprio": _array(archive, "proprio"),
            "action_noise": _array(archive, "action_noise"),
        }
        if mode in ("idm", "joint"):
            arrays["video_noise"] = _array(archive, "video_noise")

    _expect_shape(arrays["input_image"], (1, 3, 224, 448), "input_image")
    _expect_shape(arrays["context"], (1, None, 4096), "context")
    _expect_shape(arrays["context_mask"], (1, arrays["context"].shape[1]), "context_mask")
    _expect_shape(arrays["proprio"], (1, None), "proprio")
    _expect_shape(arrays["action_noise"], (1, 32, 7), "action_noise")
    if mode in ("idm", "joint"):
        _expect_shape(arrays["video_noise"], (1, 48, 3, 14, 28), "video_noise")
    if arrays["context_mask"].dtype.kind not in "bu" or not np.isin(
        arrays["context_mask"], (0, 1)
    ).all():
        raise ValueError("context_mask must be boolean/uint8 and contain only 0 or 1")
    arrays["context_mask"] = np.ascontiguousarray(arrays["context_mask"].astype(np.uint8))
    return metadata, arrays


def _checked_json_array(document: dict[str, Any], key: str) -> np.ndarray:
    try:
        item = document[key]
        shape = item["shape"]
        values = item["values"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"reference requires {key}.shape and {key}.values") from error
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError(f"{key}.shape must contain positive integers")
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or values.size != math.prod(shape):
        raise ValueError(f"{key}.values must be numeric and match {key}.shape")
    return decode_float_array(values.reshape(shape), key)


def load_reference(path: Path, mode: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load action and optional video outputs from an artifact, JSON, or NPZ."""
    path = path.expanduser().resolve()
    metadata: dict[str, Any] = {}
    artifact_root: Path | None = None
    if path.is_dir():
        artifact_root = path
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"reference directory has no manifest.json: {path}")
        metadata = json.loads(manifest_path.read_text())
        path = path / "reference.json"

    result: dict[str, np.ndarray]
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            action_key = "normalized_actions" if "normalized_actions" in archive else "output"
            result = {"normalized_actions": _array(archive, action_key)}
            for key in ("video_latents", "decoded_video"):
                if key in archive:
                    result[key] = (
                        decode_float_array(archive[key], key)
                        if key == "video_latents"
                        else np.ascontiguousarray(archive[key])
                    )
    elif path.suffix == ".json":
        document = json.loads(path.read_text())
        metadata = {**document, **metadata}
        result = {"normalized_actions": _checked_json_array(document, "output")}
        # Large outputs are intentionally stored only in debug.npz. JSON carries
        # their shape/hash so accidental substitution is still detectable.
        if artifact_root is not None and (artifact_root / "debug.npz").is_file():
            with np.load(artifact_root / "debug.npz", allow_pickle=False) as debug:
                for key in ("returned_video_latents", "decoded_video"):
                    if key in debug:
                        result["video_latents" if key == "returned_video_latents" else key] = (
                            decode_float_array(debug[key], key)
                            if key == "returned_video_latents"
                            else np.ascontiguousarray(debug[key])
                        )
    else:
        raise ValueError(f"reference must be an artifact directory, .json, or .npz: {path}")

    if metadata.get("model") not in (None, MODEL_NAME):
        raise ValueError(f"unexpected reference model {metadata.get('model')!r}")
    if metadata.get("mode") not in (None, mode):
        raise ValueError(
            f"reference mode {metadata.get('mode')!r} does not match --mode {mode!r}"
        )
    actions = result["normalized_actions"]
    if actions.shape == (1, 32, 7):
        actions = actions[0]
    _expect_shape(actions, (32, 7), "normalized_actions")
    result["normalized_actions"] = np.ascontiguousarray(actions)
    return metadata, result


def exact_input_mismatches(
    actual: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> list[str]:
    """Name stochastic/model inputs that are not bit-identical after decoding."""
    mismatches = []
    for name in ("input_image", "context", "context_mask", "proprio", "action_noise", "video_noise"):
        if name in actual or name in reference:
            if name not in actual or name not in reference or not np.array_equal(
                actual[name], reference[name]
            ):
                mismatches.append(name)
    return mismatches
